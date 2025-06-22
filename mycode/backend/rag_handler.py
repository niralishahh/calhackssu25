import os
import json
import logging
import uuid
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import anthropic
from supabase.client import create_client, Client
import requests
import numpy as np
from pydantic import BaseModel
from google.cloud import aiplatform
from google.cloud import storage
from google.auth import default

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PDF libraries not installed. PDF support will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunk(BaseModel):
    """Model for document chunk"""
    id: Optional[int] = None
    file_id: str
    file_name: str
    content: str
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    """Model for search result"""
    id: int
    file_id: str
    file_name: str
    content: str
    similarity: float

class VertexAIRAGAgent:
    """RAG-based AI Agent using Google Vertex AI and Supabase"""
    
    def __init__(self, 
                 anthropic_api_key: Optional[str] = None,
                 supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None,
                 google_project_id: Optional[str] = None,
                 google_location: Optional[str] = None):
        """
        Initialize the Vertex AI RAG Agent
        
        Args:
            anthropic_api_key: Anthropic API key for Claude access
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
            google_project_id: Google Cloud project ID
            google_location: Google Cloud location (e.g., 'us-central1')
        """
        # Initialize API keys
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.google_project_id = google_project_id or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.google_location = google_location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Validate required API keys
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials are required. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
        if not self.google_project_id:
            raise ValueError("Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT_ID environment variable.")
        
        # Initialize clients
        self.claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize Vertex AI
        try:
            aiplatform.init(project=self.google_project_id, location=self.google_location)
            logger.info(f"Vertex AI initialized for project: {self.google_project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
        
        # Initialize Google Cloud Storage client
        try:
            self.storage_client = storage.Client(project=self.google_project_id)
            logger.info("Google Cloud Storage client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Storage: {e}")
            raise
        
        # Vertex AI configuration
        self.embedding_model = "text-embedding-005"  # User-specified Vertex AI embedding model
        self.claude_model = "claude-3-5-sonnet-20241022"  # Valid Claude model for RAG
        
        # Configuration
        self.chunk_size = 300  # words per chunk
        self.chunk_overlap = 50  # words overlap between chunks
        self.max_search_results = 8  # number of chunks to retrieve
        
        logger.info("Vertex AI RAG agent initialized successfully")
        if PDF_SUPPORT:
            logger.info("PDF support enabled")
        else:
            logger.warning("PDF support disabled - install PyPDF2 and pdfplumber for PDF support")
    
    def _read_pdf(self, file_path: str) -> Dict[str, Any]:
        """Read PDF file and extract text content"""
        try:
            content = ""
            metadata = {}
            
            # Try pdfplumber first (better text extraction)
            try:
                with pdfplumber.open(file_path) as pdf:
                    metadata["pages"] = len(pdf.pages)
                    metadata["file_type"] = "PDF"
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            content += f"\n--- Page {page_num} ---\n{page_text}\n"
                    
                    logger.info(f"Successfully extracted text from PDF using pdfplumber")
                    
            except Exception as e:
                logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
                
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata["pages"] = len(pdf_reader.pages)
                    metadata["file_type"] = "PDF"
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            content += f"\n--- Page {page_num} ---\n{page_text}\n"
                    
                    logger.info(f"Successfully extracted text from PDF using PyPDF2")
            
            if not content.strip():
                return {"error": "No text content found in PDF"}
            
            return {
                "content": content.strip(),
                "file_name": Path(file_path).name,
                "file_size": Path(file_path).stat().st_size,
                "file_type": ".pdf",
                **metadata
            }
            
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return {"error": f"Failed to read PDF: {str(e)}"}
    
    def _read_document(self, file_path: str) -> Dict[str, Any]:
        """Read document and extract text content"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            # Handle PDF files
            if path.suffix.lower() == '.pdf':
                if not PDF_SUPPORT:
                    return {"error": "PDF support not available. Install PyPDF2 and pdfplumber."}
                return self._read_pdf(file_path)
            
            # Read text files
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                return {
                    "content": content,
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                    "file_type": path.suffix.lower()
                }
                
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                
                return {
                    "content": content,
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                    "file_type": path.suffix.lower()
                }
                
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return {"error": f"Failed to read document: {str(e)}"}
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            chunks.append(text)
        else:
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                if chunk_text.strip():
                    chunks.append(chunk_text)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _get_embedding_vertexai(self, text: str) -> List[float]:
        """Get embedding using Vertex AI with fallback to hash-based embedding"""
        try:
            # Use Vertex AI Text Embedding API
            from vertexai.language_models import TextEmbeddingModel
            
            model = TextEmbeddingModel.from_pretrained(self.embedding_model)
            embeddings = model.get_embeddings([text])
            
            if embeddings and len(embeddings) > 0:
                return embeddings[0].values
            else:
                logger.error("No embeddings returned from Vertex AI")
                return self._get_embedding_fallback(text)
                
        except Exception as e:
            logger.warning(f"Vertex AI embedding failed, using fallback: {e}")
            return self._get_embedding_fallback(text)
    
    def _get_embedding_fallback(self, text: str) -> List[float]:
        """Hash-based embedding as fallback when Vertex AI is not available"""
        try:
            import hashlib
            import struct
            
            # The target dimension for the embedding vector.
            target_dimension = 768

            # Create a simple hash-based embedding
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Convert hash to a vector
            embedding = []
            # Use chunks of 4 bytes to create floats
            for i in range(0, len(hash_bytes), 4):
                if len(embedding) >= target_dimension:
                    break
                chunk = hash_bytes[i:i+4]
                if len(chunk) == 4:
                    value, = struct.unpack('f', chunk)
                    embedding.append(value)
            
            # Pad with zeros if the vector is too short
            while len(embedding) < target_dimension:
                embedding.append(0.0)
            
            return embedding[:target_dimension]
            
        except Exception as e:
            logger.error(f"Fallback embedding also failed: {e}")
            # Return zero vector as last resort
            return [0.0] * 768
    
    def _store_chunks_in_supabase(self, file_id: str, file_name: str, chunks: List[str]) -> bool:
        """Store document chunks in Supabase with embeddings"""
        try:
            logger.info(f"Attempting to store {len(chunks)} chunks for file '{file_name}' (ID: {file_id}).")
            records_to_insert = []
            for i, chunk_content in enumerate(chunks):
                # Get embedding for chunk
                embedding = self._get_embedding_vertexai(chunk_content)
                
                if not embedding:
                    logger.warning(f"Failed to get embedding for chunk {i} of '{file_name}', skipping this chunk.")
                    continue
                
                record = {
                    "file_id": file_id,
                    "file_name": file_name,
                    "content": chunk_content,
                    "embedding": embedding,
                    "chunk_index": i
                }
                records_to_insert.append(record)
            
            if not records_to_insert:
                logger.error(f"No valid records could be generated for '{file_name}'. Ingestion failed.")
                return False

            logger.info(f"Inserting {len(records_to_insert)} records into Supabase for '{file_name}'.")
            result = self.supabase.table("document_chunks").insert(records_to_insert).execute()
            
            if result.data:
                logger.info(f"Successfully stored {len(result.data)} chunks in Supabase for '{file_name}'.")
                return True
            else:
                logger.error(f"Failed to store chunks in Supabase for '{file_name}'. Response: {result}")
                return False
            
        except Exception as e:
            logger.error(f"An exception occurred while storing chunks in Supabase: {e}", exc_info=True)
            return False
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest document into the RAG system"""
        logger.info(f"Starting document ingestion process for: {file_path}")
        try:
            # Read document
            doc_result = self._read_document(file_path)
            if "error" in doc_result:
                logger.error(f"Failed to read document: {doc_result['error']}")
                return {"error": doc_result["error"]}
            logger.info(f"Successfully read document '{doc_result['file_name']}'.")

            # Generate file ID
            file_id = str(uuid.uuid4())
            
            # Chunk the document
            chunks = self._chunk_text(doc_result["content"])
            if not chunks:
                logger.error("Failed to create document chunks, content might be empty.")
                return {"error": "Failed to create document chunks"}
            
            # Store chunks in Supabase
            if not self._store_chunks_in_supabase(file_id, doc_result["file_name"], chunks):
                logger.error(f"Failed to store document chunks for '{doc_result['file_name']}'.")
                return {"error": "Failed to store document chunks"}
            
            logger.info(f"Successfully ingested document '{doc_result['file_name']}' with file_id {file_id}.")
            return {
                "success": True,
                "file_id": file_id,
                "file_name": doc_result["file_name"],
                "chunks_created": len(chunks),
                "file_size": doc_result["file_size"],
                "file_type": doc_result["file_type"]
            }
            
        except Exception as e:
            logger.error(f"An unhandled exception occurred during ingestion: {e}", exc_info=True)
            return {"error": f"Failed to ingest document: {str(e)}"}
    
    def _search_chunks(self, query: str, file_ids: List[str]) -> List[SearchResult]:
        """Search for relevant chunks using Vertex AI embeddings and Supabase"""
        try:
            # Get query embedding
            query_embedding = self._get_embedding_vertexai(query)
            if not query_embedding:
                logger.error("Failed to generate a query embedding.")
                return []
            
            logger.info(f"Generated query embedding for: '{query[:50]}...'")

            # Search in Supabase, now passing the file_ids for DB-level filtering
            rpc_params = {
                'query_embedding': query_embedding,
                'match_threshold': 0.1,
                'match_count': self.max_search_results,
                'filter_file_ids': file_ids
            }
            logger.info(f"Executing Supabase RPC 'match_documents' with file_ids: {file_ids}")
            
            search_query = self.supabase.rpc('match_documents', rpc_params).execute()
            
            if not search_query.data:
                logger.warning("Supabase RPC returned no data.")
                return []

            logger.info(f"RPC returned {len(search_query.data)} results.")

            # The filtering is now done in the DB, so we can directly create the results.
            results = [
                SearchResult(
                    id=i,
                    file_id=row.get("file_id", "unknown"),
                    file_name=row.get("file_name", "unknown"),
                    content=row.get("content", ""),
                    similarity=row.get("similarity", 0.0)
                )
                for i, row in enumerate(search_query.data)
            ]
            
            logger.info(f"Found {len(results)} relevant results.")
            return results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}", exc_info=True)
            return []
    
    def query_documents(self, query: str, file_ids: List[str]) -> Dict[str, Any]:
        """Query documents using RAG"""
        try:
            logger.info(f"Processing query: {query}")
            
            # Search for relevant chunks
            search_results = self._search_chunks(query, file_ids)
            
            if not search_results:
                return {
                    "error": "No relevant documents found for the query",
                    "query": query,
                    "results": []
                }
            
            # Prepare context from search results
            context_parts = []
            for result in search_results:
                context_parts.append(f"From {result.file_name}:\n{result.content}\n")
            
            context = "\n".join(context_parts)
            
            # Generate response using Claude
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

Document Context:
{context}

User Question: {query}

Please provide a comprehensive answer based on the document context. If the information is not available in the context, say so clearly. Include specific details and quotes from the documents when relevant.

Answer:"""
            
            try:
                response = self.claude_client.messages.create(
                    model=self.claude_model,
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract text from response
                answer = ""
                for content_block in response.content:
                    if content_block.type == 'text':
                        answer += content_block.text
                
                return {
                    "success": True,
                    "query": query,
                    "answer": answer,
                    "sources": [
                        {
                            "file_name": result.file_name,
                            "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                            "similarity": result.similarity
                        }
                        for result in search_results
                    ],
                    "results_count": len(search_results)
                }
                
            except Exception as e:
                logger.error(f"Error generating response with Claude: {e}")
                return {
                    "error": f"Failed to generate response: {str(e)}",
                    "query": query,
                    "context": context
                }
                
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return {"error": f"Failed to query documents: {str(e)}"}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and configuration"""
        return {
            "agent_type": "Vertex AI RAG Agent",
            "anthropic_model": self.claude_model,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_search_results": self.max_search_results,
            "pdf_support": PDF_SUPPORT,
            "google_project_id": self.google_project_id,
            "google_location": self.google_location
        }

    def ingest_text(self, text: str, file_name: str, file_id: str) -> Dict[str, Any]:
        """
        Ingests raw text content directly.
        This bypasses file reading and is useful when text is pre-extracted.
        """
        try:
            logger.info(f"Starting text ingestion for file '{file_name}' (ID: {file_id}).")
            
            # 1. Chunk the text
            chunks = self._chunk_text(text)
            if not chunks:
                logger.warning(f"No chunks were generated for file '{file_name}'.")
                return {"error": "Text content resulted in zero chunks."}
            
            # 2. Store chunks and embeddings in Supabase
            success = self._store_chunks_in_supabase(file_id, file_name, chunks)
            if not success:
                return {"error": "Failed to store chunks in Supabase."}
            
            logger.info(f"Successfully ingested text from '{file_name}'.")
            return {
                "success": True,
                "file_id": file_id,
                "file_name": file_name,
                "chunks_created": len(chunks)
            }
        except Exception as e:
            logger.error(f"Error ingesting text for file '{file_name}': {e}", exc_info=True)
            return {"error": str(e)}

    def generate_suggested_questions(self, file_ids: List[str], num_chunks: int = 10, num_questions: int = 4) -> List[str]:
        """Generates suggested questions based on document content using Claude."""
        try:
            logger.info(f"Generating suggested questions for file_ids: {file_ids}")
            
            # 1. Retrieve a sample of chunks from the specified files
            response = self.supabase.table('document_chunks').select('content').in_('file_id', file_ids).limit(num_chunks).execute()
            
            if not response.data:
                logger.warning("No document chunks found for the given file IDs to generate questions.")
                return []

            # 2. Concatenate the content of the chunks to form a context
            context = "\\n---\\n".join([chunk['content'] for chunk in response.data])
            
            # 3. Create a prompt for Claude
            prompt = f"""Based on the following excerpts from one or more documents, please generate {num_questions} insightful questions a user might ask. The questions should be directly answerable from the document's content and encourage exploration.

Return your response *only* as a valid JSON object with a single key "questions" which contains a list of the question strings. Do not include any other text, explanation, or markdown formatting like ```json.

Example format:
{{"questions": ["What is the main topic of the document?", "How is concept X explained?"]}}

Here are the document excerpts:
---
{context[:4000]}
---
"""
            
            # 4. Call Claude to generate the questions
            claude_response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=500,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = claude_response.content[0].text
            
            # 5. Parse the JSON response from Claude
            try:
                questions_data = json.loads(response_text)
                questions = questions_data.get("questions", [])
                
                if not isinstance(questions, list):
                    logger.error(f"Claude returned questions in an invalid format: {questions}")
                    return []
                    
                logger.info(f"Successfully generated {len(questions)} suggested questions.")
                return questions

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse suggested questions from Claude's response: {e}")
                logger.error(f"Claude raw response: {response_text}")
                return []

        except Exception as e:
            logger.error(f"An unexpected error occurred while generating suggested questions: {e}", exc_info=True)
            return []

    def query_documents_stream(self, query: str, file_ids: List[str]):
        """
        Queries documents and streams the response from Claude.
        """
        try:
            logger.info(f"Streaming query for files: {file_ids}")
            context_chunks = self._search_chunks(query, file_ids)
            
            if not context_chunks:
                yield f"data: {json.dumps({'error': 'No relevant context found.'})}\\n\\n"
                return

            context_str = "\\n---\\n".join([chunk.content for chunk in context_chunks])
            
            system_prompt = """You are an expert AI assistant. Your task is to answer the user's question based *only* on the provided document context.

**Formatting Rules:**
- Structure your answer clearly using Markdown. Use headings, bold text, and bullet points (`* `) where appropriate.
- **Crucially, ensure there is a newline character (`\\n`) between all paragraphs and list items.** Your output must be highly readable.

**Example of a good response:**
This is a paragraph that answers part of the question.

Here is a list of key points:
* First key point. \n
* Second key point with more detail. \n
* Third key point.

This is another paragraph to conclude the answer.

**Your Task:**
Now, using the context below, please answer the user's question. If the answer is not in the context, you MUST state that you cannot find the answer in the provided documents.
"""
            
            messages = [
                {
                    "role": "user",
                    "content": f"Here is the context from the documents:\\n\\n---\\n{context_str}\\n---\\n\\nBased on this context, please answer the following question: {query}"
                }
            ]

            with self.claude_client.messages.stream(
                model=self.claude_model,
                system=system_prompt,
                messages=messages,
                max_tokens=1024
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'content': text})}\\n\\n"
        except Exception as e:
            logger.error(f"Error during streaming query: {e}", exc_info=True)
            error_message = json.dumps({"error": "An error occurred while processing your request."})
            yield f"data: {error_message}\\n\\n" 