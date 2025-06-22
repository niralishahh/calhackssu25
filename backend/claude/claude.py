import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import anthropic
from pydantic import BaseModel
import re

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

class DocumentSummary(BaseModel):
    """Model for document summary response"""
    original_text: str
    summary: str
    key_points: List[str]
    word_count: int
    summary_length: int
    confidence_score: float

class ClaudeAgent:
    """AI Agent that uses Claude to summarize documents"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """
        Initialize the Claude Agent
        
        Args:
            anthropic_api_key: Anthropic API key for Claude access
        """
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
        
        # Initialize Anthropic client
        self.claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        
        self.tools = self._define_tools()
        
        logger.info("Claude agent initialized successfully")
        if PDF_SUPPORT:
            logger.info("PDF support enabled")
        else:
            logger.warning("PDF support disabled - install PyPDF2 and pdfplumber for PDF support")
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available tools with descriptions and input schemas"""
        return [
            {
                "name": "summarize_document",
                "description": "Generate a comprehensive summary of a document with key points and analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the document file"},
                        "summary_type": {
                            "type": "string", 
                            "enum": ["comprehensive", "bullet_points", "executive"],
                            "description": "Type of summary to generate"
                        }
                    },
                    "required": ["file_path", "summary_type"]
                }
            },
            {
                "name": "extract_quotes",
                "description": "Extract direct quotes with page numbers from a document based on a specific query",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the document file"},
                        "query": {"type": "string", "description": "Specific query to find relevant quotes"}
                    },
                    "required": ["file_path", "query"]
                }
            },
            {
                "name": "answer_question",
                "description": "Answer specific questions about document content with contextual analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the document file"},
                        "question": {"type": "string", "description": "Question to answer about the document"}
                    },
                    "required": ["file_path", "question"]
                }
            },
            {
                "name": "analyze_text",
                "description": "Analyze and summarize raw text content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text content to analyze"},
                        "analysis_type": {
                            "type": "string",
                            "enum": ["summary", "key_points", "sentiment", "topics"],
                            "description": "Type of analysis to perform"
                        }
                    },
                    "required": ["text", "analysis_type"]
                }
            }
        ]
    
    def _read_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Read PDF file and extract text content
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF content and metadata
        """
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
        """
        Read document and extract text content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document content and metadata
        """
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
            elif path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            # For other file types, you might need additional processing
            else:
                return {"error": f"Unsupported file type: {path.suffix}. Supported types: .txt, .md, .py, .js, .html, .css, .json, .xml, .csv, .pdf"}
            
            return {
                "content": content,
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "file_type": path.suffix
            }
            
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return {"error": str(e)}
    
    def _extract_quotes_with_pages(self, text: str, query: str) -> Dict[str, Any]:
        """
        Extract direct quotes with page numbers based on a specific query
        
        Args:
            text: Text content to search through
            query: Specific query for finding quotes
            
        Returns:
            Dictionary containing quotes with page numbers and metadata
        """
        try:
            # Create a specialized prompt for quote extraction
            quote_prompt = f"""
            "Extract direct quotes relevant to: '{query}' from the document. Include page number and speaker if available. Explain relevance briefly."
            
            Document content:
            {text}

            If no relevant quotes are found, clearly state that.
            If the document doesn't have page numbers, indicate this and still provide the quotes.
            """
            
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                temperature=0.1,
                messages=[{"role": "user", "content": quote_prompt}]
            )
            
            # Extract the response
            quotes_text = ""
            for content_block in response.content:
                if content_block.type == 'text':
                    quotes_text += content_block.text
            
            return {
                "quotes": quotes_text,
                "query": query,
                "total_quotes_found": quotes_text.count("**Quote"),
                "extraction_method": "AI-powered quote extraction"
            }
            
        except Exception as e:
            logger.error(f"Error extracting quotes: {e}")
            return {"error": str(e)}
    
    def _analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze text content based on specified analysis type"""
        try:
            analysis_prompts = {
                "summary": "Provide a comprehensive summary of the following text:",
                "key_points": "Extract the key points from the following text in bullet format:",
                "sentiment": "Analyze the sentiment and tone of the following text:",
                "topics": "Identify the main topics and themes in the following text:"
            }
            
            prompt = f"{analysis_prompts.get(analysis_type, 'Analyze the following text:')}\n\n{text}"
            
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the response
            analysis_text = ""
            for content_block in response.content:
                if content_block.type == 'text':
                    analysis_text += content_block.text
            
            return {
                "analysis": analysis_text,
                "analysis_type": analysis_type,
                "original_text_length": len(text),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {"error": str(e)}
    
    def _answer_question(self, file_path: str, question: str) -> Dict[str, Any]:
        """Answer a specific question about document content"""
        try:
            # Read the document
            doc_result = self._read_document(file_path)
            if "error" in doc_result:
                return {"error": f"Failed to read document: {doc_result['error']}"}
            
            original_text = doc_result["content"]
            
            # Create a prompt for answering questions
            question_prompt = f"""
            "Answer the question based on the document content. Be specific, cite document facts, and say if the answer is not in the file."
            
            Document content:
            {original_text}
            
            Question: {question}
                        
            If the question cannot be answered based on the document content, clearly state that.
            """
            
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": question_prompt}]
            )
            
            # Extract the response
            answer_text = ""
            for content_block in response.content:
                if content_block.type == 'text':
                    answer_text += content_block.text
            
            return {
                "answer": answer_text,
                "question": question,
                "file_path": file_path,
                "word_count": len(original_text.split()),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"error": str(e)}

    def _generate_summary(self, text: str, summary_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Generate summary using Claude
        
        Args:
            text: Text to summarize
            summary_type: Type of summary (comprehensive, bullet_points, executive)
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            prompt = "Summarize the transcript for a journalist: list key topics discussed, decisions made, public comments, and any disagreements. Format clearly."

            full_prompt = f"""
            {prompt}
            
            Document:
            {text}
            
            Provide the summary in a clear, well-structured format.
            """
            
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            summary = ""
            for content_block in response.content:
                if content_block.type == 'text':
                    summary += content_block.text
            
            # Extract key points
            key_points_prompt = f"""
            Extract 5-7 key points from this summary:
            
            {summary}
            
            Return only the key points as a numbered list.
            """
            
            key_points_response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": key_points_prompt}]
            )
            
            key_points_text = ""
            for content_block in key_points_response.content:
                if content_block.type == 'text':
                    key_points_text += content_block.text
            
            # Clean up key points - split by newlines and filter out empty lines
            key_points = []
            for line in key_points_text.strip().split('\n'):
                line = line.strip()
                # Remove common prefixes like numbers, bullets, etc.
                if line:
                    # Remove leading numbers and dots (e.g., "1.", "•", "-")
                    cleaned_line = line.lstrip('0123456789.-• ')
                    if cleaned_line:
                        key_points.append(cleaned_line)
            
            # If no key points were extracted, create a simple list
            if not key_points:
                key_points = ["Key information extracted from the document"]
            
            return {
                "summary": summary,
                "key_points": key_points,
                "word_count": len(text.split()),
                "summary_length": len(summary.split()),
                "confidence_score": 0.85  # Add default confidence score
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}

    
    def summarize_document(self, file_path: str, summary_type: str = "comprehensive") -> DocumentSummary:
        """
        Main method to summarize a document
        
        Args:
            file_path: Path to the document file
            summary_type: Type of summary to generate
            
        Returns:
            DocumentSummary object with results
        """
        try:
            logger.info(f"Starting document summarization for: {file_path}")
            
            # Step 1: Read the document
            doc_result = self._read_document(file_path)
            if "error" in doc_result:
                raise ValueError(f"Failed to read document: {doc_result['error']}")
            
            original_text = doc_result["content"]
            
            # Step 2: Analyze the text
            analysis_result = self._analyze_text(original_text, "summary")
            if "error" in analysis_result:
                logger.warning(f"Text analysis failed: {analysis_result['error']}")
            
            # Step 3: Generate summary
            summary_result = self._generate_summary(original_text, summary_type)
            if "error" in summary_result:
                raise ValueError(f"Failed to generate summary: {summary_result['error']}")
            
            # Create DocumentSummary object
            document_summary = DocumentSummary(
                original_text=original_text,
                summary=summary_result["summary"],
                key_points=summary_result["key_points"],
                word_count=summary_result["word_count"],
                summary_length=summary_result["summary_length"],
                confidence_score=summary_result["confidence_score"]
            )
            
            logger.info(f"Document summarization completed successfully")
            return document_summary
            
        except Exception as e:
            logger.error(f"Error in summarize_document: {e}")
            raise
    
    def batch_summarize(self, file_paths: List[str], summary_type: str = "comprehensive") -> List[DocumentSummary]:
        """
        Summarize multiple documents in batch
        
        Args:
            file_paths: List of file paths to summarize
            summary_type: Type of summary to generate
            
        Returns:
            List of DocumentSummary objects
        """
        results = []
        
        for file_path in file_paths:
            try:
                summary = self.summarize_document(file_path, summary_type)
                results.append(summary)
                logger.info(f"Completed summary for: {file_path}")
            except Exception as e:
                logger.error(f"Failed to summarize {file_path}: {e}")
                # Continue with other files
                continue
        
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and configuration"""
        return {
            "agent_name": "claude-document-agent",
            "claude_model": "claude-3-5-sonnet-20241022",
            "status": "active",
            "tools_available": ["document_reader", "text_analyzer", "summary_generator"],
            "pdf_support": PDF_SUPPORT,
            "supported_formats": [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv", ".pdf"]
        }

    def _create_planner_prompt(self, user_request: str, document_info: Optional[Dict] = None) -> str:
        """Create a prompt for the planner to choose appropriate tools"""
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}" for tool in self.tools
        ])
        
        document_context = ""
        if document_info:
            document_context = f"\nDocument Context:\n- File: {document_info.get('file_name', 'Unknown')}\n- Type: {document_info.get('file_type', 'Unknown')}\n- Size: {document_info.get('file_size', 'Unknown')}"
        
        return f"""You are an AI planning agent that decides which tools to use based on user requests.

Available Tools:
{tools_description}

{document_context}

User Request: "{user_request}"

If multiple tools are needed, prioritize them by importance (1 = highest priority)."""

    def _plan_execution(self, user_request: str, document_info: Optional[Dict] = None) -> List[Dict]:
        """Plan which tools to execute based on user request"""
        try:
            planner_prompt = self._create_planner_prompt(user_request, document_info)
            
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": planner_prompt}]
            )
            
            # Extract the response
            plan_text = ""
            for content_block in response.content:
                if content_block.type == 'text':
                    plan_text += content_block.text
            
            # Parse the JSON response - handle verbose JSON responses
            plan_data = self._parse_json_response(plan_text)
            logger.info(f"Planner reasoning: {plan_data.get('reasoning', 'No reasoning provided')}")
            
            # Sort tools by priority
            tools = plan_data.get('tools', [])
            tools.sort(key=lambda x: x.get('priority', 999))
            
            return tools
            
        except Exception as e:
            logger.error(f"Error in planning: {e}")
            # Fallback to simple heuristic
            return self._fallback_plan(user_request)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling verbose responses"""
        try:
            # First, try to parse as-is
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            import re
            
            # Look for JSON object patterns
            json_patterns = [
                r'\{.*\}',  # Basic JSON object
                r'\[.*\]',  # JSON array
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, create a fallback response
            logger.warning(f"Could not parse JSON from response: {text[:200]}...")
            return {
                "reasoning": "Could not parse planner response",
                "tools": []
            }

    def _fallback_plan(self, user_request: str) -> List[Dict]:
        """Fallback planning using simple heuristics"""
        user_request_lower = user_request.lower()
        
        if any(word in user_request_lower for word in ['quote', 'extract', 'exact', 'direct']):
            return [{
                "name": "extract_quotes",
                "parameters": {"query": user_request},
                "priority": 1
            }]
        elif any(word in user_request_lower for word in ['summarize', 'summary', 'overview']):
            return [{
                "name": "summarize_document",
                "parameters": {"summary_type": "comprehensive"},
                "priority": 1
            }]
        else:
            return [{
                "name": "answer_question",
                "parameters": {"question": user_request},
                "priority": 1
            }]

    def execute_request(self, user_request: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute a user request using the appropriate tools"""
        try:
            # Get document info if file path provided
            document_info = None
            if file_path:
                document_info = {
                    "file_name": os.path.basename(file_path),
                    "file_type": os.path.splitext(file_path)[1],
                    "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
                }
            
            # Plan the execution
            logger.info(f"Planning execution for request: {user_request}")
            tools_to_execute = self._plan_execution(user_request, document_info)
            
            if not tools_to_execute:
                logger.warning("No tools selected, using fallback")
                tools_to_execute = self._fallback_plan(user_request)
            
            results = []
            for tool_plan in tools_to_execute:
                tool_name = tool_plan["name"]
                parameters = tool_plan["parameters"]
                
                # Add file_path to parameters if not present
                if file_path and "file_path" not in parameters:
                    parameters["file_path"] = file_path
                
                logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
                
                try:
                    # Execute the tool
                    if tool_name == "summarize_document":
                        # Ensure required parameters are present
                        if "summary_type" not in parameters:
                            parameters["summary_type"] = "comprehensive"
                        result = self.summarize_document(parameters["file_path"], parameters["summary_type"])
                    elif tool_name == "extract_quotes":
                        result = self._extract_quotes_from_file(parameters["file_path"], parameters["query"])
                    elif tool_name == "answer_question":
                        result = self._answer_question(parameters["file_path"], parameters["question"])
                    elif tool_name == "analyze_text":
                        # Ensure required parameters are present
                        if "analysis_type" not in parameters:
                            parameters["analysis_type"] = "summary"
                        result = self._analyze_text(parameters["text"], parameters["analysis_type"])
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    
                    results.append({
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result
                    })
                    
                except Exception as tool_error:
                    logger.error(f"Error executing tool {tool_name}: {tool_error}")
                    results.append({
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": {"error": f"Tool execution failed: {str(tool_error)}"}
                    })
            
            # Combine results
            if len(results) == 1:
                return results[0]["result"]
            else:
                return {
                    "multi_tool_results": results,
                    "summary": f"Executed {len(results)} tools to fulfill your request"
                }
                
        except Exception as e:
            logger.error(f"Error executing request: {e}")
            return {"error": str(e)}

    def _extract_quotes_from_file(self, file_path: str, query: str) -> Dict[str, Any]:
        """Extract quotes from a document file"""
        try:
            # Read the document
            doc_result = self._read_document(file_path)
            if "error" in doc_result:
                return {"error": f"Failed to read document: {doc_result['error']}"}
            
            original_text = doc_result["content"]
            
            # Use the text-based quote extraction method
            return self._extract_quotes_with_pages(original_text, query)
            
        except Exception as e:
            logger.error(f"Error extracting quotes from file: {e}")
            return {"error": str(e)}
