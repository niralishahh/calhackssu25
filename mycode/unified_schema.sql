-- Unified Chat and RAG System Schema for Supabase

--== Extensions ==--
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

--== Tables ==--

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Transcriptions table (from original app)
CREATE TABLE IF NOT EXISTS transcriptions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_size_mb DECIMAL(10,2) NOT NULL,
    duration DECIMAL(10,2) NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    text TEXT NOT NULL,
    segments JSONB NOT NULL DEFAULT '[]',
    timestamps JSONB NOT NULL DEFAULT '[]',
    chunked BOOLEAN NOT NULL DEFAULT FALSE,
    num_chunks INTEGER NOT NULL DEFAULT 1,
    chunk_duration INTEGER NOT NULL DEFAULT 300,
    overlap_duration INTEGER NOT NULL DEFAULT 10,
    prompt TEXT,
    temperature DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chats table
CREATE TABLE IF NOT EXISTS chats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chat responses table
CREATE TABLE IF NOT EXISTS chat_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    response_type TEXT DEFAULT 'text',
    metadata JSONB DEFAULT '{}',
    is_saved BOOLEAN NOT NULL DEFAULT FALSE,
    saved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Document Chunks table (from RAG app)
-- Note: We'll use the transcription_id as the file_id for consistency
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    file_id UUID NOT NULL, -- This will be the UUID from the 'transcriptions' table
    file_name TEXT,
    content TEXT NOT NULL,
    chunk_index INTEGER,
    embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chat-file associations table
CREATE TABLE IF NOT EXISTS chat_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
    transcription_id UUID REFERENCES transcriptions(id) ON DELETE CASCADE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(chat_id, transcription_id)
);


--== Indexes ==--
-- Original app indexes
CREATE INDEX IF NOT EXISTS idx_transcriptions_user_id ON transcriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_responses_chat_id ON chat_responses(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_files_chat_id ON chat_files(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_files_transcription_id ON chat_files(transcription_id);

-- RAG app indexes
CREATE INDEX IF NOT EXISTS document_chunks_file_id_idx ON document_chunks (file_id);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx ON document_chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chat_responses_is_saved ON chat_responses(is_saved);

--== Row Level Security (RLS) ==--
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE transcriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Policies for users
CREATE POLICY "Users can view their own data" ON users FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can insert their own data" ON users FOR INSERT WITH CHECK (auth.uid() = id);

-- Policies for transcriptions
CREATE POLICY "Users can manage their own transcriptions" ON transcriptions FOR ALL USING (auth.uid() = user_id);

-- Policies for chats
CREATE POLICY "Users can manage their own chats" ON chats FOR ALL USING (auth.uid() = user_id);

-- Policies for chat_responses
CREATE POLICY "Users can manage responses in their own chats" ON chat_responses FOR ALL USING (
    EXISTS (SELECT 1 FROM chats WHERE chats.id = chat_responses.chat_id AND chats.user_id = auth.uid())
);

-- Policies for chat_files
CREATE POLICY "Users can manage files in their own chats" ON chat_files FOR ALL USING (
    EXISTS (SELECT 1 FROM chats WHERE chats.id = chat_files.chat_id AND chats.user_id = auth.uid())
);

-- Policies for document_chunks
CREATE POLICY "Users can manage chunks for files they own" ON document_chunks FOR ALL USING (
    EXISTS (SELECT 1 FROM transcriptions WHERE transcriptions.id = document_chunks.file_id AND transcriptions.user_id = auth.uid())
);


--== Functions ==--

-- Function to match document chunks for RAG
CREATE OR REPLACE FUNCTION match_document_chunks(
  query_embedding VECTOR(768),
  match_threshold FLOAT,
  match_count INT,
  filter_file_ids UUID[] DEFAULT NULL
)
RETURNS TABLE (
  id BIGINT,
  file_id UUID,
  file_name TEXT,
  content TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  -- If no file IDs provided, search all documents (respecting RLS)
  IF filter_file_ids IS NULL OR array_length(filter_file_ids, 1) IS NULL THEN
    RETURN QUERY
    SELECT
      chunks.id,
      chunks.file_id,
      chunks.file_name,
      chunks.content,
      1 - (chunks.embedding <=> query_embedding) AS similarity
    FROM
      document_chunks AS chunks
    WHERE 1 - (chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY
      chunks.embedding <=> query_embedding
    LIMIT
      match_count;
  ELSE
    -- Filter by file IDs and then perform vector search
    RETURN QUERY
    SELECT
      chunks.id,
      chunks.file_id,
      chunks.file_name,
      chunks.content,
      1 - (chunks.embedding <=> query_embedding) AS similarity
    FROM
      document_chunks AS chunks
    WHERE
      chunks.file_id = ANY(filter_file_ids) AND 1 - (chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY
      chunks.embedding <=> query_embedding
    LIMIT
      match_count;
  END IF;
END;
$$;

-- Function to get user chats (from original app)
CREATE OR REPLACE FUNCTION get_user_chats(user_uuid UUID)
RETURNS TABLE (
    id UUID,
    title TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    response_count BIGINT,
    file_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.title,
        c.created_at,
        c.updated_at,
        (SELECT COUNT(*) FROM chat_responses cr WHERE cr.chat_id = c.id) as response_count,
        (SELECT COUNT(*) FROM chat_files cf WHERE cf.chat_id = c.id) as file_count
    FROM chats c
    WHERE c.user_id = user_uuid
    ORDER BY c.updated_at DESC;
END;
$$ LANGUAGE plpgsql; 