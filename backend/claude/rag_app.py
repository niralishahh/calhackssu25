import os
import json
import logging
import time
import traceback
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Import the RAG agent
from rag_agent import VertexAIRAGAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.pdf'}

# Get the absolute path to the uploads directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_ABS = os.path.join(SCRIPT_DIR, UPLOAD_FOLDER)

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER_ABS, exist_ok=True)

# Global agent instance
rag_agent: Optional[VertexAIRAGAgent] = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if not filename:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def initialize_agent():
    """Initialize the Vertex AI RAG agent"""
    global rag_agent
    try:
        rag_agent = VertexAIRAGAgent()
        logger.info("Vertex AI RAG agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI RAG agent: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    if initialize_agent():
        logger.info("RAG agent initialized successfully")
    else:
        logger.error("Failed to initialize RAG agent")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG server...")

# Create FastAPI app
app = FastAPI(
    title="RAG Document Analysis API",
    description="Scalable RAG-based document analysis using Supabase and Anthropic",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="../../app"), name="static")

# Serve the frontend
@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    return FileResponse("../../app/rag_index.html")

# Pydantic models
class IngestRequest(BaseModel):
    file_path: str

class QueryRequest(BaseModel):
    query: str
    file_ids: List[str]

class BatchIngestRequest(BaseModel):
    file_paths: List[str]

class FileInfo(BaseModel):
    file_id: str
    file_name: str
    chunks_created: int
    file_size: int
    file_type: str

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_initialized": rag_agent is not None,
        "timestamp": time.time()
    }

@app.get("/status")
async def get_status():
    """Get detailed status of the RAG system"""
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    
    try:
        status = rag_agent.get_agent_status()
        return {
            "status": "operational",
            "agent_info": status,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file"""
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    
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
        
        # Ingest the document into RAG system
        ingest_result = rag_agent.ingest_document(file_path)
        
        if "error" in ingest_result:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {ingest_result['error']}")
        
        return {
            "message": "File uploaded and ingested successfully",
            "file_path": file_path,
            "filename": filename,
            "ingestion_result": ingest_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    
    try:
        # Check if files are present
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        uploaded_files = []
        ingestion_results = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(UPLOAD_FOLDER_ABS, filename)
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                uploaded_files.append(file_path)
                logger.info(f"File uploaded: {file_path}")
                
                # Ingest the document
                ingest_result = rag_agent.ingest_document(file_path)
                ingestion_results.append(ingest_result)
                
            else:
                logger.warning(f"Skipping invalid file: {file.filename if file else 'None'}")
        
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")
        
        return {
            "message": f"{len(uploaded_files)} files uploaded and ingested successfully",
            "file_paths": uploaded_files,
            "ingestion_results": ingestion_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    """Ingest an existing file into the RAG system"""
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    
    try:
        logger.info(f"Received ingest request for file: {request.file_path}")
        
        # Check if file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Ingest the document
        result = rag_agent.ingest_document(request.file_path)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {result['error']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-ingest")
async def batch_ingest(request: BatchIngestRequest):
    """Ingest multiple existing files into the RAG system"""
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    
    try:
        if not isinstance(request.file_paths, list):
            raise HTTPException(status_code=400, detail="file_paths must be a list")
        
        results = []
        for file_path in request.file_paths:
            try:
                if not os.path.exists(file_path):
                    results.append({"file_path": file_path, "error": "File not found"})
                    continue
                
                result = rag_agent.ingest_document(file_path)
                results.append({"file_path": file_path, "result": result})
                
            except Exception as e:
                logger.error(f"Error ingesting {file_path}: {e}")
                results.append({"file_path": file_path, "error": str(e)})
        
        return {
            "results": results,
            "total_files": len(request.file_paths)
        }
        
    except Exception as e:
        logger.error(f"Error in batch_ingest: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query documents using RAG pipeline"""
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    
    try:
        logger.info(f"Received query request: {request.query}")
        logger.info(f"Querying files: {request.file_ids}")
        
        if not request.file_ids:
            raise HTTPException(status_code=400, detail="No file IDs provided")
        
        # Query the documents
        result = rag_agent.query_documents(request.query, request.file_ids)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Query failed: {result['error']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_ingested_files():
    """List all ingested files from the database"""
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    
    try:
        # Query Supabase to get unique files
        result = rag_agent.supabase.table("document_chunks").select("file_id, file_name").execute()
        
        # Group by file_id and get unique files
        files = {}
        for row in result.data:
            file_id = row["file_id"]
            if file_id not in files:
                files[file_id] = {
                    "file_id": file_id,
                    "file_name": row["file_name"]
                }
        
        return {
            "files": list(files.values()),
            "total_files": len(files)
        }
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file and all its chunks from the database"""
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    
    try:
        # Delete all chunks for this file
        result = rag_agent.supabase.table("document_chunks").delete().eq("file_id", file_id).execute()
        
        return {
            "message": f"File {file_id} deleted successfully",
            "chunks_deleted": len(result.data) if result.data else 0
        }
        
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the API is working"""
    return {
        "message": "RAG API is working", 
        "agent_initialized": rag_agent is not None,
        "timestamp": time.time()
    }

if __name__ == '__main__':
    # Initialize agent on startup
    if initialize_agent():
        logger.info("Starting RAG FastAPI server...")
        uvicorn.run(app, host='0.0.0.0', port=5005)
    else:
        logger.error("Failed to initialize RAG agent. Exiting.")
        exit(1) 