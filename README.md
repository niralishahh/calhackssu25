# 🤖 Claude AI Research Agent

An intelligent AI research agent powered by Anthropic's Claude that uses advanced planning and tool selection to analyze documents, extract direct quotes with page numbers, and answer questions through natural conversation.

## ✨ Features

- **🧠 Intelligent Planning**: AI agent that plans and chooses the best tools for each request
- **🔧 Tool-Based Architecture**: Specialized tools for different types of analysis
- **📄 Direct Quote Extraction**: Extract exact quotes with page numbers for citations
- **📋 Multiple Summary Types**: Comprehensive, bullet points, and executive summaries
- **🔍 Contextual Analysis**: Answer questions with deep document understanding
- **📊 Batch Processing**: Process multiple documents simultaneously
- **📝 Text Analysis**: Direct text input and analysis
- **💬 Natural Conversation**: Upload a document and ask questions naturally
- **📚 PDF Support**: Extract and analyze text from PDF documents
- **🔄 Multi-Format Support**: PDF, TXT, MD, PY, JS, HTML, CSS, JSON, XML, CSV
- **🎨 Modern Web Interface**: Beautiful, responsive UI with real-time feedback
- **🔌 RESTful API**: Full API for integration with other applications
- **⚡ Claude AI Integration**: Leverages Anthropic's Claude for intelligent document processing

## 🏗️ Architecture

```
calhackssu25/
├── backend/
│   └── claude/
│       ├── claude.py          # Main AI agent implementation
│       ├── app.py             # Flask API server
│       └── example.py         # Usage examples
├── app/
│   └── index.html             # Web interface
├── requirements.txt           # Python dependencies
├── start.sh                   # Startup script
└── README.md                 # This file
```

## 🤖 AI Agent Capabilities

This system functions as an intelligent AI research agent with advanced planning and tool selection capabilities:

### 🧠 Intelligent Planning System
The AI agent automatically plans and executes the best approach for each request:
- **Tool Selection**: Chooses the most appropriate tools based on user intent
- **Multi-Step Planning**: Can combine multiple tools for complex requests
- **Context Awareness**: Considers document type and content when planning
- **Fallback Logic**: Graceful degradation when planning fails

### 🔧 Available Tools
The agent has access to specialized tools for different tasks:

#### 📄 Document Summarization Tool
- **Purpose**: Generate comprehensive summaries with key points
- **Input**: Document path and summary type (comprehensive/bullet_points/executive)
- **Output**: Structured summary with analysis

#### 🗣️ Quote Extraction Tool
- **Purpose**: Extract direct quotes with page numbers and context
- **Input**: Document path and specific query
- **Output**: Formatted quotes with page numbers, speakers, and relevance

#### ❓ Question Answering Tool
- **Purpose**: Answer specific questions about document content
- **Input**: Document path and question
- **Output**: Contextual answer with document references

#### 📊 Text Analysis Tool
- **Purpose**: Analyze raw text content for various insights
- **Input**: Text content and analysis type (summary/key_points/sentiment/topics)
- **Output**: Structured analysis results

### 🔍 Intelligent Analysis
The AI agent can:
- **Plan Complex Requests**: Automatically determine which tools to use
- **Combine Multiple Tools**: Execute multi-step analyses when needed
- **Adapt to Context**: Choose different approaches based on document type
- **Provide Reasoning**: Explain why certain tools were chosen
- **Handle Ambiguity**: Use fallback strategies when planning is uncertain

### 💡 Natural Language Understanding
Users can ask questions naturally:
- "Summarize this document"
- "What are the main topics discussed?"
- "Give me quotes about budget discussions"
- "Who were the key speakers and what did they say?"
- "Analyze the decision-making process"

The agent will automatically:
1. **Analyze the request** to understand user intent
2. **Plan the execution** by selecting appropriate tools
3. **Execute the plan** using the chosen tools
4. **Combine results** into a coherent response

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Anthropic API Key** - Get one from [Anthropic Console](https://console.anthropic.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd calhackssu25
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

   Or create a `.env` file:
   ```env
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

### Running the Application

#### Option 1: Using the startup script (Recommended)
```bash
chmod +x start.sh
./start.sh
```

#### Option 2: Manual startup
1. **Start the backend server**
   ```bash
   cd backend/claude
   python app.py
   ```
   The API server will start on `http://localhost:5002`

2. **Open the web interface**
   Open `app/index.html` in your browser or serve it with a local server:
   ```bash
   cd app
   python -m http.server 8000
   ```
   Then visit `http://localhost:8000`

## 📖 Usage

### Web Interface

The web interface provides three main modes:

1. **Single Document**: Upload and summarize a single document
2. **Batch Processing**: Process multiple documents at once
3. **Direct Text**: Paste text directly for analysis

### API Endpoints

#### Health Check
```bash
GET http://localhost:5002/health
```

#### Get Agent Status
```bash
GET http://localhost:5002/status
```

#### Summarize Single Document
```bash
POST http://localhost:5002/summarize
Content-Type: application/json

{
  "file_path": "/path/to/document.pdf",
  "summary_type": "comprehensive"
}
```

#### Batch Summarize Documents
```bash
POST http://localhost:5002/batch-summarize
Content-Type: application/json

{
  "file_paths": ["/path/to/doc1.pdf", "/path/to/doc2.txt"],
  "summary_type": "bullet_points"
}
```

#### Analyze Text
```bash
POST http://localhost:5003/analyze
Content-Type: application/json

{
  "text": "Your text content here...",
  "summary_type": "executive"
}
```

#### Ask Questions About Document
```bash
POST http://localhost:5003/ask
Content-Type: application/json

{
  "file_path": "/path/to/document.pdf",
  "question": "What was the general consensus on the housing budget?"
}
```

### Python API

```python
from backend.claude.claude import ClaudeAgent

# Initialize the agent
agent = ClaudeAgent()

# Summarize a PDF document
result = agent.summarize_document("path/to/document.pdf", "comprehensive")
print(f"Summary: {result.summary}")
print(f"Key points: {result.key_points}")

# Batch summarize multiple documents (mix of formats)
results = agent.batch_summarize(["doc1.pdf", "doc2.txt", "doc3.md"], "bullet_points")
for result in results:
    print(f"Document summary: {result.summary}")
```

## 🔧 Configuration

### Summary Types

- **comprehensive**: Detailed summary with main ideas, supporting information, and context
- **bullet_points**: Concise bullet-point format for quick scanning
- **executive**: High-level summary suitable for executive review

### Supported File Types

- **`.pdf`** - PDF documents (requires PyPDF2 and pdfplumber)
- **`.txt`** - Plain text files
- **`.md`** - Markdown files
- **`.py`** - Python files
- **`.js`** - JavaScript files
- **`.html`** - HTML files
- **`.css`** - CSS files
- **`.json`** - JSON files
- **`.xml`** - XML files
- **`.csv`** - CSV files

### PDF Processing

The application uses two libraries for PDF processing:
- **pdfplumber**: Primary PDF text extraction (better quality)
- **PyPDF2**: Fallback PDF processing

PDF files are processed page by page, with page numbers included in the extracted text for better context.

## 🛠️ Development

### Running Examples

```bash
cd backend/claude
python example.py
```

### Project Structure

- **`backend/claude/claude.py`**: Core AI agent implementation using Claude API
- **`backend/claude/app.py`**: Flask API server with RESTful endpoints
- **`backend/claude/example.py`**: Usage examples and demonstrations
- **`app/index.html`**: Modern web interface with JavaScript

### Key Components

#### ClaudeAgent Class
- **`__init__()`**: Initialize with API key and check PDF support
- **`summarize_document()`**: Main method for single document summarization
- **`batch_summarize()`**: Process multiple documents
- **`_read_pdf()`**: PDF text extraction using pdfplumber and PyPDF2
- **`_generate_summary()`**: Core summarization logic using Claude
- **`_analyze_text()`**: Text analysis and key information extraction

#### Flask API
- **Health monitoring**: `/health` endpoint for system status
- **Document processing**: `/summarize` and `/batch-summarize` endpoints
- **Text analysis**: `/analyze` endpoint for direct text input
- **Error handling**: Comprehensive error handling and logging

## 🔒 Security Considerations

1. **API Key Management**: Store API keys securely using environment variables
2. **File Validation**: Validate file types and content before processing
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **Input Sanitization**: Sanitize all user inputs to prevent injection attacks
5. **PDF Security**: PDF processing libraries are used safely with proper error handling

## 🚀 Deployment

### Local Development
```bash
# Backend
cd backend/claude
python app.py

# Frontend (optional)
cd app
python -m http.server 8000
```

### Production Deployment

1. **Set up a production server** (e.g., AWS, Google Cloud, Heroku)
2. **Configure environment variables** for production
3. **Set up a reverse proxy** (nginx) for the Flask app
4. **Use a production WSGI server** (gunicorn, uwsgi)
5. **Enable HTTPS** for secure communication

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY app/ ./app/

EXPOSE 5002
CMD ["python", "backend/claude/app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **API Key Issues**: Verify your Anthropic API key is valid and has sufficient credits
3. **CORS Issues**: The Flask app includes CORS headers, but you may need to configure them for your domain
4. **Port Conflicts**: If port 5002 is in use, you can change it in `backend/claude/app.py`
5. **PDF Processing Issues**: Ensure PyPDF2 and pdfplumber are installed for PDF support

### PDF-Specific Issues

- **"PDF support not available"**: Install PDF libraries with `pip install PyPDF2 pdfplumber`
- **"No text content found in PDF"**: The PDF might be image-based or encrypted
- **PDF processing errors**: Try different PDF files or check if the PDF is corrupted

### Getting Help

- Check the logs in the backend console for detailed error messages
- Verify your environment variables are set correctly
- Test with simple text files first before trying PDFs

## 🎯 Future Enhancements

- [ ] Support for more document formats (DOCX, PPTX, etc.)
- [ ] Advanced text analysis features
- [ ] Custom summary templates
- [ ] Integration with cloud storage (Google Drive, Dropbox)
- [ ] Real-time collaboration features
- [ ] Advanced analytics and reporting
- [ ] OCR support for image-based PDFs
- [ ] Document comparison features