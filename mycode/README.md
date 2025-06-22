# AI Transcription Agent

An AI-powered transcription agent that uses Groq's Whisper model and Google's Gemini for intelligent analysis. Features a modern web interface with chat capabilities, file management, and Supabase integration.

## Features

- 🎙️ **Multi-format transcription**: Supports audio and video files (MP3, MP4, WAV, M4A, FLAC, OGG, WebM)
- 🧠 **AI-powered analysis**: Uses Google Gemini for intelligent chat and analysis
- 📊 **Dashboard management**: Organize conversations and files with an intuitive dashboard
- 💬 **Chat interface**: Three-panel interface similar to Google NotebookLM
- 🔄 **Large file support**: Automatic chunking for files over 300 seconds
- ☁️ **Supabase integration**: Persistent storage for transcriptions and conversations
- 🎯 **Word-level timestamps**: Precise timing information for each word
- 🌍 **Multi-language support**: Transcribe in multiple languages

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd calhackssu25

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

### 3. Setup Supabase Database

1. Go to your Supabase project SQL editor
2. Run the SQL schema from `supabase_chat_schema.sql`
3. This creates the necessary tables and policies

### 4. Start the Application

```bash
# Terminal 1: Start the backend API server
cd backend
python api_server.py

# Terminal 2: Start the frontend server (optional, for development)
cd app
python -m http.server 3000
```

### 5. Access the Application

- **Dashboard**: http://localhost:3000/dashboard.html
- **Chat Interface**: http://localhost:3000/chat_interface.html
- **API Documentation**: http://localhost:8000/docs

## Dashboard Features

The dashboard provides a central hub for managing your AI transcription conversations:

### 🏠 **Home Page**
- View all your conversations in a clean card layout
- See chat statistics (responses, files)
- Quick access to recent chats

### ➕ **Create New Chats**
- Click "New Chat" to start a fresh conversation
- Give your chat a meaningful title
- Automatically opens the chat interface

### 🗑️ **Manage Chats**
- Edit chat titles inline
- Delete conversations with confirmation
- View chat metadata and activity

### 🔍 **Search & Filter**
- Search through your conversations
- Filter by date and activity
- Quick navigation between chats

## Chat Interface

The three-panel chat interface provides a NotebookLM-style experience:

### 📝 **Left Panel - Saved Responses**
- View all AI responses for the current chat
- Click to expand and read full responses
- Organized by conversation flow

### 💬 **Middle Panel - Chat Assistant**
- Main conversation area with AI
- Upload files directly or select from right panel
- Real-time chat with transcription context

### 📁 **Right Panel - File Management**
- Upload new audio/video files
- View all your transcriptions
- Select files to include in chat context
- View transcription details

## File Management

### 📤 **Upload Files**
- Drag and drop or click to upload
- Supports multiple audio/video formats
- Automatic transcription processing
- Progress indicators for large files

### 📊 **View Transcriptions**
- Multiple viewing modes (segments, text, JSON)
- Word-level timestamps
- Duration and metadata display
- Search within transcriptions

### 🔗 **Chat Integration**
- Select files to include in chat context
- AI can reference specific parts of transcriptions
- Context-aware responses

## API Endpoints

### Transcription
- `POST /transcribe/upload` - Upload and transcribe files
- `POST /transcribe/file` - Transcribe by file path
- `POST /transcribe/natural` - Natural language requests

### Storage
- `POST /store-transcription` - Store in Supabase
- `GET /transcriptions` - List all transcriptions
- `GET /transcriptions/{id}` - Get specific transcription

### Chat Management
- `POST /chats` - Create new chat
- `GET /chats/{user_id}` - Get user's chats
- `PUT /chats/{chat_id}/title` - Update chat title
- `DELETE /chats/{chat_id}` - Delete chat

### Chat Content
- `POST /chats/{chat_id}/responses` - Add chat response
- `GET /chats/{chat_id}/responses` - Get chat responses
- `POST /chats/{chat_id}/files` - Add file to chat
- `GET /chats/{chat_id}/files` - Get chat files

## Configuration

### Supported File Formats
- Audio: MP3, WAV, M4A, FLAC, OGG
- Video: MP4, WebM

### Chunking Settings
- Default chunk duration: 300 seconds
- Overlap duration: 10 seconds
- Automatic for files > 300 seconds

### Environment Variables
- `GROQ_API_KEY`: Required for transcription
- `GOOGLE_API_KEY`: Required for chat analysis
- `SUPABASE_URL`: Required for storage
- `SUPABASE_KEY`: Required for storage

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'supabase'"**
   ```bash
   pip install supabase
   ```

2. **"Supabase not configured"**
   - Check your `.env` file exists in the root directory
   - Verify Supabase credentials are correct

3. **"Request Entity Too Large"**
   - Large files are automatically chunked
   - Check file size limits in your server configuration

4. **Port conflicts**
   - Backend runs on port 8000
   - Frontend runs on port 3000
   - Change ports in the respective files if needed

### File Upload Issues

1. **ffmpeg not found**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Large file processing**
   - Files are automatically chunked for processing
   - Progress is shown in the interface
   - Check browser console for detailed logs

## Development

### Project Structure
```
calhackssu25/
├── app/                    # Frontend files
│   ├── dashboard.html     # Chat management dashboard
│   ├── chat_interface.html # Main chat interface
│   ├── index.html         # Landing page
│   └── view_transcription.html # Transcription viewer
├── backend/               # Backend API
│   ├── api_server.py     # FastAPI server
│   ├── transcription_agent.py # Core transcription logic
│   ├── supabase_client.py # Database client
│   └── chat_client.py    # Chat management
├── uploads/              # Temporary file storage
└── requirements.txt      # Python dependencies
```

### Adding New Features

1. **Backend**: Add endpoints to `api_server.py`
2. **Frontend**: Update HTML/JS files in `app/`
3. **Database**: Update schema in `supabase_chat_schema.sql`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check browser console for error messages
4. Verify environment variables are set correctly