# Chat System Setup Guide

This guide explains how to set up the chat system for the AI Transcription Agent with user-specific chats, responses, and file management.

## Database Schema Setup

### 1. Run the SQL Schema

First, execute the SQL schema in your Supabase database. You can do this through the Supabase dashboard:

1. Go to your Supabase project dashboard
2. Navigate to the SQL Editor
3. Copy and paste the contents of `supabase_chat_schema.sql`
4. Execute the script

This will create:
- `users` table (if not exists)
- `chats` table for managing user conversations
- `chat_responses` table for storing AI responses
- `chat_files` table for associating files with chats
- Updated `transcriptions` table with user_id field
- Row Level Security (RLS) policies
- Helper functions for common operations

### 2. Environment Variables

Make sure your `.env` file includes:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## Backend Setup

### 1. Install Dependencies

The chat system uses the existing dependencies plus the new chat client:

```bash
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
cd backend
python api_server.py
```

The server will run on `http://localhost:8000` and includes all the new chat endpoints.

## Frontend Setup

### 1. Start the Frontend Server

```bash
cd app
python -m http.server 3000
```

### 2. Access the Chat Interface

Open your browser and go to:
- **Original Interface**: `http://localhost:3000/index.html`
- **New Chat Interface**: `http://localhost:3000/chat_interface.html`

## Chat System Features

### Three-Panel Layout

1. **Left Panel - Saved Responses**
   - Shows chat history for the current conversation
   - Click on any response to view it in the chat
   - Responses are automatically saved when you chat

2. **Middle Panel - Chat Interface**
   - Real-time chat with AI about your files
   - Requires files to be selected before chatting
   - Auto-saves all conversations

3. **Right Panel - File Management**
   - Upload new audio/video files
   - View all files uploaded by the current user
   - Add files to the current chat conversation
   - View transcription details

### User Management

Currently, the system uses a demo user ID (`demo-user-123`). In a production environment, you would:

1. Implement proper authentication (e.g., Supabase Auth)
2. Replace the hardcoded `currentUserId` with the actual authenticated user's ID
3. Update the RLS policies to work with Supabase Auth

### File Operations

- **Upload**: Files are automatically transcribed and stored with the user ID
- **Add to Chat**: Files can be added to specific chat conversations
- **View**: Click "View" to see detailed transcription with segments and timestamps
- **Remove**: Files can be removed from chat conversations

### Chat Features

- **Automatic Chat Creation**: A new chat is created when you first load the interface
- **Response History**: All AI responses are saved and can be reviewed
- **File Context**: The AI knows which files are selected for the conversation
- **Real-time Updates**: The interface updates automatically as you interact

## API Endpoints

### Chat Management
- `POST /chats` - Create a new chat
- `GET /chats/{user_id}` - Get all chats for a user
- `GET /chats/chat/{chat_id}` - Get a specific chat
- `PUT /chats/{chat_id}/title` - Update chat title
- `DELETE /chats/{chat_id}` - Delete a chat

### Chat Responses
- `POST /chats/{chat_id}/responses` - Add a response to a chat
- `GET /chats/{chat_id}/responses` - Get all responses for a chat

### File Management
- `POST /chats/{chat_id}/files` - Add a file to a chat
- `DELETE /chats/{chat_id}/files/{transcription_id}` - Remove a file from a chat
- `GET /chats/{chat_id}/files` - Get all files for a chat
- `GET /users/{user_id}/files` - Get all files for a user

## Database Structure

### Tables

1. **users**
   - `id` (UUID, Primary Key)
   - `email` (TEXT, Unique)
   - `created_at`, `updated_at` (Timestamps)

2. **chats**
   - `id` (UUID, Primary Key)
   - `user_id` (UUID, Foreign Key to users)
   - `title` (TEXT)
   - `created_at`, `updated_at` (Timestamps)

3. **chat_responses**
   - `id` (UUID, Primary Key)
   - `chat_id` (UUID, Foreign Key to chats)
   - `user_message` (TEXT)
   - `ai_response` (TEXT)
   - `response_type` (TEXT, default: 'text')
   - `metadata` (JSONB)
   - `created_at` (Timestamp)

4. **chat_files**
   - `id` (UUID, Primary Key)
   - `chat_id` (UUID, Foreign Key to chats)
   - `transcription_id` (UUID, Foreign Key to transcriptions)
   - `added_at` (Timestamp)

5. **transcriptions** (Updated)
   - All existing fields plus `user_id` (UUID, Foreign Key to users)

### Security

The system uses Row Level Security (RLS) to ensure users can only access their own data:

- Users can only see their own chats
- Users can only see responses in their own chats
- Users can only see files they've uploaded
- All operations are restricted to the authenticated user's data

## Usage Workflow

1. **Start a Chat**: The system automatically creates a new chat when you load the interface
2. **Upload Files**: Use the upload button to add audio/video files
3. **Add Files to Chat**: Click "Add" on files to include them in your conversation
4. **Start Chatting**: Type questions about your files and get AI responses
5. **Review History**: Click on any response in the left panel to review it
6. **View Details**: Click "View" on files to see detailed transcriptions

## Future Enhancements

- **Real AI Integration**: Replace placeholder responses with actual AI analysis
- **Authentication**: Implement proper user authentication
- **Multiple Chats**: Allow users to create and switch between multiple chats
- **File Sharing**: Allow sharing files between users
- **Advanced Search**: Search through chat history and file content
- **Export**: Export chat conversations and transcriptions

## Troubleshooting

### Common Issues

1. **"Supabase not configured"**: Check your environment variables
2. **"Chat manager not configured"**: Ensure Supabase credentials are correct
3. **Files not showing**: Check that files are being stored with the correct user_id
4. **Chat responses not saving**: Verify the chat_id is being set correctly

### Debug Mode

The backend includes extensive logging. Check the console output for:
- Debug messages starting with 🔍
- Error messages starting with ❌
- Success confirmations

### Database Queries

You can verify the setup by running these queries in Supabase:

```sql
-- Check if tables exist
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';

-- Check if functions exist
SELECT routine_name FROM information_schema.routines WHERE routine_schema = 'public';

-- Test user chats (replace with actual user_id)
SELECT * FROM get_user_chats('demo-user-123');
``` 