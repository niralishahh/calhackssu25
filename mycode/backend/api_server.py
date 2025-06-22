import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from transcription_agent import TranscriptionAgent, TranscriptionRequest, TranscriptionResponse
from supabase_client import get_supabase_manager
from chat_client import get_chat_manager

app = FastAPI(
    title="AI Transcription Agent",
    description="An AI agent that transcribes audio/video files using Groq's Whisper model with chunking support for large files and Supabase storage",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the transcription agent
agent = TranscriptionAgent()

# Create uploads directory
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Transcription Agent API",
        "version": "1.0.0",
        "features": [
            "Multi-format audio/video transcription",
            "Large file chunking support (up to 100MB+)",
            "Word-level and segment-level timestamps",
            "Multi-language support",
            "Context prompts for better accuracy",
            "Supabase storage for transcriptions"
        ],
        "endpoints": {
            "/transcribe/file": "POST - Transcribe a file by path",
            "/transcribe/upload": "POST - Upload and transcribe a file",
            "/store-transcription": "POST - Store transcription in Supabase",
            "/transcriptions": "GET - List stored transcriptions",
            "/transcriptions/{id}": "GET - Get specific transcription",
            "/health": "GET - Health check",
            "/supported-formats": "GET - List supported formats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    supabase_status = "configured" if get_supabase_manager() else "not_configured"
    return {
        "status": "healthy", 
        "agent_ready": True,
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
        
        result = await agent.transcribe_file(request)
        
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
        if file_ext not in agent.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(agent.supported_formats)}"
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
        result = await agent.transcribe_file(request)
        
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
                    "file_size": file_path.stat().st_size,
                    "upload_timestamp": str(file_path.stat().st_mtime),
                    "chunk_duration": chunk_duration,
                    "overlap_duration": overlap_duration,
                    "prompt": prompt,
                    "temperature": temperature
                }
            }
            
            json_path = UPLOADS_DIR / f"{Path(file.filename).stem}_transcription.json"
            with open(json_path, "w") as f:
                json.dump(detailed_result, f, indent=2, default=str)
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store-transcription")
async def store_transcription(storage_request: StorageRequest):
    """Store a transcription in Supabase"""
    try:
        print(f"🔍 Debug: Received storage request with data keys: {list(storage_request.transcription_data.keys())}")
        
        supabase_manager = get_supabase_manager()
        if not supabase_manager:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        print(f"🔍 Debug: Supabase manager created successfully")
        
        # Store the actual transcription data passed from frontend
        transcription_id = await supabase_manager.store_transcription(
            storage_request.transcription_data, 
            storage_request.user_id
        )
        
        print(f"🔍 Debug: Transcription stored with ID: {transcription_id}")
        
        return {"success": True, "transcription_id": transcription_id}
        
    except Exception as e:
        print(f"❌ Error in store_transcription: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcriptions")
async def list_transcriptions(limit: int = 50, offset: int = 0):
    """List stored transcriptions"""
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
        result = await agent.process_transcription_request(user_request)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": agent.supported_formats,
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
    """Get information about the chunking feature"""
    return {
        "chunking_enabled": True,
        "max_file_size_mb": 25.0,
        "default_chunk_duration_seconds": 300,
        "default_overlap_duration_seconds": 10,
        "supported_formats_for_chunking": agent.supported_formats,
        "description": "Large audio/video files are automatically split into chunks for processing, then merged back together with proper timestamp alignment."
    }

# Chat-specific endpoints
@app.post("/chats")
async def create_chat(user_id: str = Form(...), title: str = Form("New Chat")):
    """Create a new chat for a user"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        chat_id = await chat_manager.create_chat(user_id, title)
        return {"success": True, "chat_id": chat_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{user_id}")
async def get_user_chats(user_id: str):
    """Get all chats for a user"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
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

@app.get("/users/{user_id}/files")
async def get_user_files(user_id: str):
    """Get all files uploaded by a user"""
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not configured")
        
        files = await chat_manager.get_user_files(user_id)
        return {"files": files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Load environment variables from parent directory
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    
    # Look for .env in parent directory (root)
    parent_env = Path(__file__).parent.parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env)
    else:
        # Fallback to current directory
        load_dotenv()
    
    # Check for required API keys
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found in environment variables")
    
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not found in environment variables")
    
    # Check for Supabase configuration
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_ANON_KEY"):
        print("Warning: Supabase configuration not found. Storage features will be disabled.")
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 