import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import shutil
from dotenv import load_dotenv
import uuid

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the `mycode` directory
# This must be done BEFORE importing other modules that need the env vars.
env_path = Path(__file__).parent.parent / '.env'
print(f"Attempting to load .env file from: {env_path.resolve()}")
if not env_path.exists():
    print("!!! CRITICAL: .env file not found at the expected path. !!!")
    print("Please create a .env file in the 'mycode' directory with your API keys.")
    raise FileNotFoundError(f".env file not found at {env_path.resolve()}")

load_dotenv(dotenv_path=env_path)
print(".env file loaded successfully.")

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from transcription_agent import TranscriptionAgent, TranscriptionRequest, TranscriptionResponse
from supabase_client import get_supabase_manager
from chat_client import get_chat_manager
from rag_handler import VertexAIRAGAgent

app = FastAPI(
    title="AI Transcription and RAG Agent",
    description="An AI agent that transcribes media, ingests documents, and provides RAG-based chat.",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agents
transcription_agent = TranscriptionAgent()
rag_agent = VertexAIRAGAgent()
chat_manager = get_chat_manager()
supabase_manager = get_supabase_manager()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Define supported file types
MEDIA_FORMATS = {".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
TEXT_FORMATS = {".pdf", ".txt", ".md", ".json", ".csv", ".html", ".htm", ".xml", ".log", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".sql", ".sh"}

class TranscriptionAPIRequest(BaseModel):
    """API request model for transcription"""
    language: Optional[str] = "en"
    prompt: Optional[str] = ""
    temperature: Optional[float] = 0.0
    chunk_duration: Optional[int] = 300
    overlap_duration: Optional[int] = 10

class TranscriptionAPIResponse(BaseModel):
    """API response model for transcription"""
    success: bool
    text: str
    timestamps: list
    segments: list
    language: str
    duration: float
    error: Optional[str] = None
    file_path: Optional[str] = None
    chunked: bool = False
    num_chunks: int = 1
    file_size_mb: Optional[float] = None
    transcription_id: Optional[str] = None
    chunks: List[Dict[str, Any]] = []
    filename: Optional[str] = None

class StorageRequest(BaseModel):
    """Request model for storing transcription"""
    transcription_data: Dict[str, Any]
    user_id: Optional[str] = None
    store_in_supabase: bool = True

class UploadResponse(BaseModel):
    success: bool
    message: str
    file_id: Optional[str] = None
    filename: str
    file_type: str

class QueryRequest(BaseModel):
    query: str
    file_ids: List[str]
    chat_id: str

class SuggestedQuestionsRequest(BaseModel):
    file_ids: List[str]

class ChatCreationRequest(BaseModel):
    user_id: Optional[str] = None
    title: str

# --- Helper Functions ---
def get_valid_user_id(user_id: str) -> str:
    """Checks if a user_id is a valid UUID, otherwise generates a new one."""
    try:
        uuid.UUID(user_id)
        return user_id
    except (ValueError, TypeError):
        logger.warning(f"Invalid UUID for user_id: '{user_id}'. Generating a new one.")
        return str(uuid.uuid4())

@app.get("/")
async def root():
    """Redirects to the main dashboard."""
    return RedirectResponse(url="/app/dashboard.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    supabase_status = "configured" if get_supabase_manager() else "not_configured"
    rag_status = rag_agent.get_agent_status()
    return {
        "status": "healthy", 
        "transcription_agent_ready": True,
        "rag_agent_status": rag_status,
        "supabase": supabase_status
    }

@app.post("/transcribe/file", response_model=TranscriptionAPIResponse)
async def transcribe_file_path(
    file_path: str = Form(...),
    language: Optional[str] = Form("en"),
    prompt: Optional[str] = Form(""),
    temperature: Optional[float] = Form(0.0),
    chunk_duration: Optional[int] = Form(300),
    overlap_duration: Optional[int] = Form(10)
):
    """Transcribe a file by providing its path"""
    try:
        request = TranscriptionRequest(
            file_path=file_path,
            language=language,
            prompt=prompt,
            temperature=temperature,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration
        )
        
        result = await transcription_agent.transcribe_file(request)
        
        # Get file size for response
        file_size_mb = None
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        return TranscriptionAPIResponse(
            success=result.success,
            text=result.text,
            timestamps=result.timestamps,
            segments=result.segments,
            language=result.language,
            duration=result.duration,
            error=result.error,
            file_path=file_path,
            chunked=result.chunked,
            num_chunks=result.num_chunks,
            file_size_mb=file_size_mb,
            chunks=result.chunks
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/upload", response_model=TranscriptionAPIResponse)
async def transcribe_uploaded_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
    prompt: Optional[str] = Form(""),
    temperature: Optional[float] = Form(0.0),
    chunk_duration: Optional[int] = Form(300),
    overlap_duration: Optional[int] = Form(10)
):
    """Upload and transcribe a file"""
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in transcription_agent.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(transcription_agent.supported_formats)}"
            )
        
        # Save uploaded file
        file_path = UPLOADS_DIR / f"{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Create transcription request
        request = TranscriptionRequest(
            file_path=str(file_path),
            language=language,
            prompt=prompt,
            temperature=temperature,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration
        )
        
        # Perform transcription
        result = await transcription_agent.transcribe_file(request)
        
        # Save detailed JSON result
        if result.success:
            detailed_result = {
                "text": result.text,
                "timestamps": result.timestamps,
                "segments": result.segments,
                "language": result.language,
                "duration": result.duration,
                "chunked": result.chunked,
                "num_chunks": result.num_chunks,
                "file_size_mb": file_size_mb,
                "metadata": {
                    "original_filename": file.filename,
                    "content_type": file.content_type,
                    "language": language,
                    "prompt": prompt,
                    "temperature": temperature
                }
            }
            
            json_path = UPLOADS_DIR / f"{Path(file.filename).stem}_transcription.json"
            with open(json_path, "w") as f:
                json.dump(detailed_result, f, indent=4)
        
        return TranscriptionAPIResponse(
            success=result.success,
            text=result.text,
            timestamps=result.timestamps,
            segments=result.segments,
            language=result.language,
            duration=result.duration,
            error=result.error,
            file_path=str(file_path),
            chunked=result.chunked,
            num_chunks=result.num_chunks,
            file_size_mb=file_size_mb,
            chunks=result.chunks,
            filename=file.filename
        )
        
    except Exception as e:
        logger.error(f"Error in /transcribe/upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcriptions")
async def list_transcriptions(limit: int = 50, offset: int = 0):
    """List all stored transcriptions"""
    try:
        supabase_manager = get_supabase_manager()
        if not supabase_manager:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        transcriptions = await supabase_manager.list_transcriptions(limit=limit, offset=offset)
        
        return {
            "transcriptions": [transcription.dict() for transcription in transcriptions],
            "total": len(transcriptions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: str):
    """Get a specific transcription by ID"""
    try:
        supabase_manager = get_supabase_manager()
        if not supabase_manager:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        transcription = await supabase_manager.get_transcription(transcription_id)
        
        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")
        
        return transcription.dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/natural")
async def transcribe_natural_language(user_request: str = Form(...)):
    """Process a natural language transcription request"""
    try:
        result = await transcription_agent.process_transcription_request(user_request)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": transcription_agent.supported_formats,
        "description": "These are the file formats supported for transcription",
        "chunking_info": {
            "max_file_size": "25MB (free tier) / 100MB (dev tier)",
            "chunking_automatic": "Files larger than 25MB are automatically chunked",
            "chunk_duration": "5 minutes (300 seconds) by default",
            "overlap_duration": "10 seconds by default"
        }
    }

@app.get("/chunking-info")
async def get_chunking_info():
    """Returns information about the current chunking settings"""
    return transcription_agent.get_chunking_parameters()

@app.post("/chats")
async def create_chat(request: ChatCreationRequest):
    """Create a new chat for a user. If user_id is not provided, a new user is created."""
    try:
        # This handles None, invalid UUIDs, or existing UUIDs
        user = await chat_manager.get_or_create_user(user_id=request.user_id)
        
        new_chat = await chat_manager.create_chat(user_id=user['id'], title=request.title)
        
        # The create_chat function in chat_manager returns the full chat object
        # which now includes the user_id, essential for new users.
        return new_chat
        
    except Exception as e:
        logger.error(f"Error creating chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{user_id}")
async def get_user_chats(user_id: str):
    """Get all chats for a specific user"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        # If the user_id from the path is not a valid UUID, we cannot fetch chats.
        # The frontend should use the UUID returned upon chat creation.
        # We'll return an empty list to prevent a server crash.
        try:
            uuid.UUID(user_id)
        except (ValueError, TypeError):
            return {"chats": []}

        chats = await chat_manager.get_user_chats(user_id)
        return {"chats": chats}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/chat/{chat_id}")
async def get_chat(chat_id: str):
    """Get a specific chat by ID"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        chat = await chat_manager.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return chat.dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, title: str = Form(...)):
    """Update the title of a chat"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        success = await chat_manager.update_chat_title(chat_id, title)
        return {"success": success}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        success = await chat_manager.delete_chat(chat_id)
        return {"success": success}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats/{chat_id}/responses")
async def add_chat_response(
    chat_id: str,
    user_message: str = Form(...),
    ai_response: str = Form(...),
    response_type: str = Form("text"),
    metadata: str = Form("{}")
):
    """Add a response to a chat"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        metadata_dict = json.loads(metadata) if metadata else {}
        response_id = await chat_manager.add_chat_response(
            chat_id, user_message, ai_response, response_type, metadata_dict
        )
        return {"success": True, "response_id": response_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}/responses")
async def get_chat_responses(chat_id: str):
    """Get all responses for a chat"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        responses = await chat_manager.get_chat_responses(chat_id)
        return {"responses": responses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats/{chat_id}/files")
async def add_file_to_chat(chat_id: str, transcription_id: str = Form(...)):
    """Add a file to a chat"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        file_id = await chat_manager.add_file_to_chat(chat_id, transcription_id)
        return {"success": True, "file_id": file_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}/files/{transcription_id}")
async def remove_file_from_chat(chat_id: str, transcription_id: str):
    """Remove a file from a chat"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        success = await chat_manager.remove_file_from_chat(chat_id, transcription_id)
        return {"success": success}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}/files")
async def get_chat_files(chat_id: str):
    """Get all files associated with a chat"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        files = await chat_manager.get_chat_files(chat_id)
        return {"files": files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}/responses/saved")
async def get_saved_responses_for_chat(chat_id: str):
    """Gets all saved responses for a specific chat."""
    try:
        responses = await chat_manager.get_saved_responses_for_chat(chat_id)
        return {"saved_responses": responses}
    except Exception as e:
        logger.error(f"Error fetching saved responses for chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch saved responses.")

@app.post("/upload", response_model=UploadResponse)
async def upload_and_process_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    chat_id: str = Form(...),
    language: str = Form("en")
):
    """
    Handles file uploads, performs transcription, stores the result,
    ingests the content for RAG, and links it to a chat.
    This is the primary endpoint for adding new files to the system.
    """
    logger.info(f"Received upload request for user '{user_id}' and chat '{chat_id}'")
    
    # 1. Ensure user and chat exist
    try:
        user = await chat_manager.get_or_create_user(user_id)
        current_user_id = user['id']
        logger.info(f"Confirmed user: {current_user_id}")

        chat_exists = await chat_manager.check_chat_exists(chat_id)
        if not chat_exists:
            logger.error(f"Upload failed: Chat with ID '{chat_id}' does not exist.")
            raise HTTPException(status_code=404, detail=f"Chat with id {chat_id} not found.")

    except Exception as e:
        logger.error(f"Error verifying user/chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to verify user and chat.")

    # 2. Save uploaded file temporarily
    file_ext = Path(file.filename).suffix.lower()
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"File '{file.filename}' saved to temp path: {file_path}")

        # 3. Perform Transcription
        transcription_result = None
        if file_ext in MEDIA_FORMATS:
            logger.info(f"Starting transcription for media file: {file.filename}")
            request = TranscriptionRequest(file_path=str(file_path), language=language)
            transcription_result = await transcription_agent.transcribe_file(request)
            if not transcription_result.success:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {transcription_result.error}")
            logger.info(f"Transcription successful for {file.filename}")
            text_to_ingest = transcription_result.text
        elif file_ext in TEXT_FORMATS:
             # For non-media files, we just read the text content for RAG
            logger.info(f"File '{file.filename}' is a text document. Reading content for RAG.")
            # Use the RAG agent's own document reader
            doc_data = rag_agent._read_document(str(file_path))
            if "error" in doc_data:
                raise HTTPException(status_code=400, detail=doc_data["error"])
            text_to_ingest = doc_data["content"]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")

        # 4. Store record in Supabase
        logger.info(f"Storing file record for '{file.filename}'")
        
        # Consolidate record data for both media and text files
        record_data = {
            "user_id": current_user_id,
            "filename": file.filename,
            "file_size_mb": file_size_mb,
            "text": text_to_ingest, # Use the ingested text for all types
            "language": transcription_result.language if transcription_result else file_ext.replace('.', ''),
            "duration": transcription_result.duration if transcription_result else 0,
            "segments": transcription_result.segments if transcription_result else [],
            "timestamps": transcription_result.timestamps if transcription_result else [],
            "chunked": transcription_result.chunked if transcription_result else False,
            "num_chunks": transcription_result.num_chunks if transcription_result else 1,
        }
        
        insert_result = await supabase_manager.store_transcription(record_data, user_id=current_user_id)
        if not insert_result or not insert_result.data:
            raise HTTPException(status_code=500, detail="Failed to store transcription in database.")
        
        transcription_id = insert_result.data[0]['id']
        logger.info(f"Transcription stored with ID: {transcription_id}")

        # 5. Add file to chat
        await chat_manager.add_file_to_chat(chat_id, transcription_id)
        logger.info(f"Linked transcription {transcription_id} to chat {chat_id}")

        # 6. Ingest into RAG Agent
        try:
            rag_agent.ingest_text(
                file_name=file.filename,
                text=text_to_ingest,
                file_id=transcription_id
            )
            logger.info(f"Successfully started ingestion for file '{file.filename}' (ID: {transcription_id}).")
        except Exception as e:
            logger.error(f"RAG ingestion failed for {file.filename}: {e}", exc_info=True)
            # Don't fail the whole request, but log the error.
            # The file is uploaded and transcribed, can be re-ingested later.

        return UploadResponse(
            success=True,
            message=f"File '{file.filename}' processed and stored successfully.",
            file_id=transcription_id,
            filename=file.filename,
            file_type=file_ext
        )

    except Exception as e:
        logger.error(f"Error during file upload and processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        # 7. Cleanup
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temp directory: {temp_dir}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question and file context.
    Streams the response back.
    """
    try:
        # Use the RAG agent to get a streaming response
        return StreamingResponse(
            rag_agent.query_documents_stream(
                query=request.query, 
                file_ids=request.file_ids
            ), 
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest-questions")
async def suggest_questions(request: SuggestedQuestionsRequest):
    """
    Generates suggested questions for a given set of files.
    """
    if not request.file_ids:
        return {"suggested_questions": []}
        
    try:
        # This is a synchronous call, but FastAPI runs it in a worker thread
        questions = rag_agent.generate_suggested_questions(file_ids=request.file_ids)
        return {"suggested_questions": questions}
    except Exception as e:
        logger.error(f"Error generating suggested questions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate suggested questions.")

@app.get("/responses/{response_id}")
async def get_response_by_id(response_id: str):
    """Gets a single chat response by its ID."""
    try:
        response = await chat_manager.get_chat_response_by_id(response_id)
        if not response:
            raise HTTPException(status_code=404, detail="Response not found.")
        return response
    except Exception as e:
        logger.error(f"Error fetching response {response_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch response.")

@app.post("/responses/{response_id}/save")
async def save_response(response_id: str):
    """Marks a specific chat response as 'saved'."""
    try:
        success = await chat_manager.set_response_saved_status(response_id, is_saved=True)
        if not success:
            raise HTTPException(status_code=404, detail="Response not found or failed to update.")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error saving response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save response.")

@app.post("/responses/{response_id}/unsave")
async def unsave_response(response_id: str):
    """Marks a specific chat response as 'not saved'."""
    try:
        success = await chat_manager.set_response_saved_status(response_id, is_saved=False)
        if not success:
            raise HTTPException(status_code=404, detail="Response not found or failed to update.")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error unsaving response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to unsave response.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Mount static files last to ensure it doesn't override API endpoints.
# The `html=True` argument tells FastAPI to serve index.html for root requests.
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "app", html=True), name="app") 