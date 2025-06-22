# Vertex AI RAG Document Analysis System

A scalable Retrieval-Augmented Generation (RAG) system built with Google Vertex AI, Supabase, and Anthropic Claude for efficient document analysis and question answering.

## 🚀 Features

- **Scalable Architecture**: Handles large documents by chunking them into smaller pieces
- **Vector Search**: Uses Google Vertex AI embeddings for semantic similarity search
- **Multi-Document Support**: Query across multiple documents simultaneously
- **Real-time Processing**: Fast response times with optimized vector search
- **PDF Support**: Full PDF text extraction and processing
- **Modern UI**: Clean, responsive web interface
- **Google Cloud Integration**: Leverages Vertex AI's powerful embedding models

## 🏗️ Architecture

### Phase 1: Document Ingestion
1. **File Upload**: Users upload documents through the web interface
2. **Text Extraction**: Documents are processed to extract text content
3. **Chunking**: Text is split into 300-word chunks with 50-word overlap
4. **Embedding Generation**: Each chunk is converted to a vector using Vertex AI
5. **Database Storage**: Chunks and embeddings are stored in Supabase with pgvector

### Phase 2: Query Processing
1. **Query Input**: Users ask questions and select relevant documents
2. **Query Embedding**: The question is converted to a vector using Vertex AI
3. **Semantic Search**: Vector similarity search finds relevant chunks
4. **Context Assembly**: Top 8 most relevant chunks are combined
5. **AI Response**: Claude generates answers based on the retrieved context

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: Supabase (PostgreSQL with pgvector)
- **Embeddings**: Google Vertex AI Text Embedding API
- **LLM**: Anthropic Claude 3 Haiku
- **Frontend**: HTML/CSS/JavaScript
- **PDF Processing**: pdfplumber, PyPDF2
- **Cloud Platform**: Google Cloud Platform

## 📋 Prerequisites

1. **Supabase Account**: Create a project and get your URL and service role key
2. **Google Cloud Account**: Set up a project and enable Vertex AI API
3. **Anthropic Account**: Get API key for Claude access
4. **Python 3.8+**: For running the backend
5. **Google Cloud Authentication**: Service account key or gcloud auth

## 🚀 Quick Start

### 1. Environment Setup

Set the required environment variables:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export SUPABASE_URL="your-supabase-project-url"
export SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
export GOOGLE_CLOUD_PROJECT_ID="your-google-cloud-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # Optional, defaults to us-central1
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"  # Optional if using gcloud auth
```

### 2. Google Cloud Setup

1. **Enable APIs**: Enable the Vertex AI API in your Google Cloud project
2. **Authentication**: Either:
   - Use service account: `export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"`
   - Or use gcloud: `gcloud auth application-default login`

### 3. Database Setup

Run the SQL setup script in your Supabase SQL editor:

```sql
-- Copy and paste the contents of backend/claude/supabase_setup.sql
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the System

```bash
chmod +x start_rag.sh
./start_rag.sh
```

Or manually:

```bash
cd backend/claude
python rag_app.py
```

### 6. Access the Interface

Open your browser and go to: `http://localhost:5005`

## 📁 Project Structure

```
calhackssu25/
├── app/
│   └── rag_index.html          # RAG frontend interface
├── backend/
│   └── claude/
│       ├── rag_agent.py        # Vertex AI RAG agent implementation
│       ├── rag_app.py          # FastAPI application
│       ├── supabase_setup.sql  # Database setup script
│       └── uploads/            # File upload directory
├── requirements.txt            # Python dependencies
├── start_rag.sh               # Startup script
└── RAG_README.md              # This file
```

## 🔧 Configuration

### Chunking Parameters

Edit `backend/claude/rag_agent.py` to adjust:

```python
self.chunk_size = 300          # Words per chunk
self.chunk_overlap = 50        # Words overlap between chunks
self.max_search_results = 8    # Number of chunks to retrieve
```

### Models

```python
self.embedding_model = "textembedding-gecko@003"  # Vertex AI embedding model
self.claude_model = "claude-3-haiku-20240307"    # Claude model for responses
```

## 📊 API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /status` - Detailed system status
- `GET /test` - Test endpoint

### File Management
- `POST /upload` - Upload and ingest single file
- `POST /upload-batch` - Upload and ingest multiple files
- `POST /ingest` - Ingest existing file
- `POST /batch-ingest` - Ingest multiple existing files
- `GET /files` - List all ingested files
- `DELETE /files/{file_id}` - Delete file and its chunks

### Query
- `POST /query` - Query documents with RAG pipeline

## 🔍 Usage Examples

### Upload and Query Documents

1. **Upload Documents**: Use the web interface to upload PDFs, text files, etc.
2. **Select Files**: Choose which documents to query from the file list
3. **Ask Questions**: Type your question and get AI-powered answers

### API Usage

```python
import requests

# Upload a file
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5005/upload', files=files)
    file_data = response.json()

# Query documents
query_data = {
    "query": "What are the main topics discussed?",
    "file_ids": ["file-uuid-1", "file-uuid-2"]
}
response = requests.post('http://localhost:5005/query', json=query_data)
result = response.json()
print(result['answer'])
```

## 🎯 Key Benefits

### Scalability
- **Token Limit Avoidance**: No more hitting API token limits
- **Large Document Support**: Process documents of any size
- **Efficient Storage**: Vector embeddings enable fast similarity search

### Performance
- **Fast Queries**: Vector search is much faster than full-text search
- **Relevant Results**: Semantic similarity finds truly relevant content
- **Optimized Context**: Only relevant chunks are sent to Claude

### Flexibility
- **Multi-Document Queries**: Ask questions across multiple files
- **Selective Search**: Choose which documents to include in queries
- **Real-time Updates**: Add new documents without restarting

## 🔧 Troubleshooting

### Common Issues

1. **Connection Errors**
   - Check if all environment variables are set
   - Verify Supabase project is active
   - Ensure Google Cloud project is properly configured

2. **Google Cloud Authentication Errors**
   - Verify service account key is valid and has proper permissions
   - Or run: `gcloud auth application-default login`
   - Check if Vertex AI API is enabled in your project

3. **Database Errors**
   - Run the SQL setup script in Supabase
   - Check if pgvector extension is enabled
   - Verify table permissions

4. **File Upload Issues**
   - Check file size limits
   - Verify supported file types
   - Ensure uploads directory exists

### Debug Mode

Enable detailed logging by setting:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

- **Streaming Responses**: Real-time streaming of AI responses
- **Document Metadata**: Store and search by document metadata
- **Advanced Filtering**: Filter by date, author, document type
- **Batch Processing**: Background processing for large document sets
- **User Management**: Multi-user support with document sharing
- **Analytics**: Query analytics and usage statistics

## 📝 License

This project is part of the CalHacks hackathon submission.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Test with the `/test` endpoint 