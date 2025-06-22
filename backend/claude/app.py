from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv
from claude import ClaudeAgent, DocumentSummary
import traceback
import tempfile
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager
import time
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance
agent = None

# Configure upload settings
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.pdf'}

# Get the absolute path to the uploads directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_ABS = os.path.join(SCRIPT_DIR, UPLOAD_FOLDER)

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER_ABS, exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if not filename:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def initialize_agent():
    """Initialize the Claude agent"""
    global agent
    try:
        agent = ClaudeAgent()
        logger.info("Claude agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    initialize_agent()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI server...")

# Pydantic models for request bodies
class SummarizeRequest(BaseModel):
    file_path: str
    summary_type: str = "comprehensive"

class BatchSummarizeRequest(BaseModel):
    file_paths: List[str]
    summary_type: str = "comprehensive"

class AnalyzeRequest(BaseModel):
    text: str
    summary_type: str = "comprehensive"

class DocumentQARequest(BaseModel):
    file_path: str
    question: str

class QuoteExtractionRequest(BaseModel):
    file_path: str
    query: str

class AgenticRequest(BaseModel):
    file_path: str
    request: str

# Initialize FastAPI app
app = FastAPI(
    title="Claude AI Document Summarizer",
    description="An intelligent AI agent that uses Claude to summarize documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None
    }

@app.get("/status")
async def get_status():
    """Get agent status"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        status = agent.get_agent_status()
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Check if file is present
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename or file.filename == '':
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Save file
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER_ABS, filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded: {file_path}")
        
        return {
            "message": "File uploaded successfully",
            "file_path": file_path,
            "filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Check if files are present
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        uploaded_files = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(UPLOAD_FOLDER_ABS, filename)
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                uploaded_files.append(file_path)
                logger.info(f"File uploaded: {file_path}")
            else:
                logger.warning(f"Skipping invalid file: {file.filename if file else 'None'}")
        
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")
        
        return {
            "message": f"{len(uploaded_files)} files uploaded successfully",
            "file_paths": uploaded_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_document(request: SummarizeRequest):
    """Summarize a single document"""
    try:
        logger.info(f"Received summarize request: {request}")
        
        if agent is None:
            logger.error("Agent not initialized")
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        logger.info(f"Received summarize request for file: {request.file_path}")
        
        # Convert relative path to absolute path if needed
        file_path = request.file_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(UPLOAD_FOLDER_ABS, os.path.basename(file_path))
        
        logger.info(f"Using absolute file path: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Validate summary type
        valid_types = ['comprehensive', 'bullet_points', 'executive']
        if request.summary_type not in valid_types:
            logger.error(f"Invalid summary type: {request.summary_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid summary_type. Must be one of: {valid_types}"
            )
        
        # Summarize document
        logger.info(f"Starting summarization with type: {request.summary_type}")
        try:
            result = agent.summarize_document(file_path, request.summary_type)
            logger.info("Agent summarization completed successfully")
        except Exception as agent_error:
            logger.error(f"Agent summarization failed: {agent_error}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Summarization failed: {str(agent_error)}")
        
        # Convert to dict for JSON response
        try:
            response_data = {
                "original_text": result.original_text,
                "summary": result.summary,
                "key_points": result.key_points,
                "word_count": result.word_count,
                "summary_length": result.summary_length,
                "confidence_score": result.confidence_score,
                "file_path": file_path,
                "summary_type": request.summary_type
            }
            logger.info(f"Response data prepared successfully")
        except Exception as response_error:
            logger.error(f"Error preparing response data: {response_error}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error preparing response: {str(response_error)}")
        
        logger.info(f"Summarization completed successfully")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in summarize_document: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/batch-summarize")
async def batch_summarize(request: BatchSummarizeRequest):
    """Summarize multiple documents"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        if not isinstance(request.file_paths, list):
            raise HTTPException(status_code=400, detail="file_paths must be a list")
        
        # Validate summary type
        valid_types = ['comprehensive', 'bullet_points', 'executive']
        if request.summary_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid summary_type. Must be one of: {valid_types}"
            )
        
        # Batch summarize documents
        results = agent.batch_summarize(request.file_paths, request.summary_type)
        
        # Convert to list of dicts for JSON response
        response_data = []
        for result in results:
            response_data.append({
                "original_text": result.original_text,
                "summary": result.summary,
                "key_points": result.key_points,
                "word_count": result.word_count,
                "summary_length": result.summary_length,
                "confidence_score": result.confidence_score,
                "summary_type": request.summary_type
            })
        
        return {
            "results": response_data,
            "total_documents": len(results),
            "summary_type": request.summary_type
        }
        
    except Exception as e:
        logger.error(f"Error in batch_summarize: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_text(request: AnalyzeRequest):
    """Analyze text content"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Generate summary for the text
        summary_result = agent._generate_summary(request.text, request.summary_type)
        
        if "error" in summary_result:
            raise HTTPException(status_code=500, detail=summary_result["error"])
        
        # Create a DocumentSummary-like response
        response_data = {
            "original_text": request.text,
            "summary": summary_result["summary"],
            "key_points": summary_result["key_points"],
            "word_count": summary_result["word_count"],
            "summary_length": summary_result["summary_length"],
            "confidence_score": summary_result["confidence_score"],
            "summary_type": request.summary_type
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_text: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: DocumentQARequest):
    """Ask a question about a specific document"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        logger.info(f"Received question request for file: {request.file_path}")
        logger.info(f"Question: {request.question}")
        
        # Convert relative path to absolute path if needed
        file_path = request.file_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(UPLOAD_FOLDER_ABS, os.path.basename(file_path))
        
        logger.info(f"Using absolute file path: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Read the document
        doc_result = agent._read_document(file_path)
        if "error" in doc_result:
            raise HTTPException(status_code=500, detail=f"Failed to read document: {doc_result['error']}")
        
        original_text = doc_result["content"]
        
        # Create a context-aware prompt for the question
        qa_prompt = f"""
        You are an AI assistant helping a local journalist with a question about a city council meeting. The user has uploaded a document and is asking a question about it.
        
        Document content:
        {original_text}
        
        User's question: {request.question}
        
        Please provide a helpful, accurate answer based on the document content. If the question cannot be answered from the document, say so clearly. If you need to make any assumptions, state them explicitly.
        
        Provide your answer in a clear, well-structured format.
        """
        
        # Get response from Claude
        response = agent.claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": qa_prompt}]
        )
        
        # Extract the answer
        answer = ""
        for content_block in response.content:
            if content_block.type == 'text':
                answer += content_block.text
        
        response_data = {
            "question": request.question,
            "answer": answer,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "word_count": len(original_text.split()),
            "timestamp": time.time()
        }
        
        logger.info(f"Question answered successfully")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/extract-quotes")
async def extract_quotes(request: QuoteExtractionRequest):
    """Extract direct quotes with page numbers from a document"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        logger.info(f"Received quote extraction request for file: {request.file_path}")
        logger.info(f"Query: {request.query}")
        
        # Convert relative path to absolute path if needed
        file_path = request.file_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(UPLOAD_FOLDER_ABS, os.path.basename(file_path))
        
        logger.info(f"Using absolute file path: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Read the document
        doc_result = agent._read_document(file_path)
        if "error" in doc_result:
            raise HTTPException(status_code=500, detail=f"Failed to read document: {doc_result['error']}")
        
        original_text = doc_result["content"]
        
        # Extract quotes using the specialized method
        quotes_result = agent._extract_quotes_with_pages(original_text, request.query)
        
        if "error" in quotes_result:
            raise HTTPException(status_code=500, detail=f"Quote extraction failed: {quotes_result['error']}")
        
        response_data = {
            "query": request.query,
            "quotes": quotes_result["quotes"],
            "total_quotes_found": quotes_result["total_quotes_found"],
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "word_count": len(original_text.split()),
            "extraction_method": quotes_result["extraction_method"],
            "timestamp": time.time()
        }
        
        logger.info(f"Quote extraction completed successfully")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_quotes: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error extracting quotes: {str(e)}")

@app.post("/agentic-execute")
async def agentic_execute(request: AgenticRequest):
    """Execute a user request using the AI agent's planning capabilities"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        logger.info(f"Received agentic execution request for file: {request.file_path}")
        logger.info(f"User request: {request.request}")
        
        # Convert relative path to absolute path if needed
        file_path = request.file_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(UPLOAD_FOLDER_ABS, os.path.basename(file_path))
        
        logger.info(f"Using absolute file path: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Execute the request using the agent's planning capabilities
        logger.info("Starting agentic execution...")
        result = agent.execute_request(request.request, file_path)
        logger.info(f"Agentic execution completed. Result type: {type(result)}")
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Agentic execution failed: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Agentic execution failed: {result['error']}")
        
        response_data = {
            "request": request.request,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "result": result,
            "execution_method": "agentic_planning",
            "timestamp": time.time()
        }
        
        logger.info(f"Agentic execution completed successfully")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agentic_execute: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in agentic execution: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the API is working"""
    return {"message": "API is working", "agent_initialized": agent is not None}

@app.post("/test-agentic")
async def test_agentic():
    """Test the agentic execution with a simple request"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Test with a simple text analysis
        test_text = "This is a test document about AI and machine learning."
        result = agent._analyze_text(test_text, "summary")
        
        return {
            "test_type": "agentic_text_analysis",
            "result": result,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Test agentic failed: {e}")
        return {
            "test_type": "agentic_text_analysis",
            "error": str(e),
            "status": "failed"
        }

if __name__ == '__main__':
    # Initialize agent on startup
    if initialize_agent():
        logger.info("Starting FastAPI server...")
        uvicorn.run(app, host='0.0.0.0', port=5004)
    else:
        logger.error("Failed to initialize agent. Exiting.")
        exit(1) 