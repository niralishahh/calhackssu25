-- Chat System Schema for Supabase
-- This creates tables for managing chats, responses, and file associations

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (if not already exists)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
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
    response_type TEXT DEFAULT 'text', -- 'text', 'transcription_analysis', etc.
    metadata JSONB DEFAULT '{}',
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

-- Update existing transcriptions table to include user_id
ALTER TABLE transcriptions ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE CASCADE;

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_responses_chat_id ON chat_responses(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_files_chat_id ON chat_files(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_files_transcription_id ON chat_files(transcription_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_user_id ON transcriptions(user_id);

-- Row Level Security (RLS) policies
ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE transcriptions ENABLE ROW LEVEL SECURITY;

-- Chat policies
CREATE POLICY "Users can view their own chats" ON chats
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert their own chats" ON chats
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update their own chats" ON chats
    FOR UPDATE USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete their own chats" ON chats
    FOR DELETE USING (auth.uid()::text = user_id::text);

-- Chat responses policies
CREATE POLICY "Users can view responses in their chats" ON chat_responses
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM chats 
            WHERE chats.id = chat_responses.chat_id 
            AND chats.user_id::text = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert responses in their chats" ON chat_responses
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM chats 
            WHERE chats.id = chat_responses.chat_id 
            AND chats.user_id::text = auth.uid()::text
        )
    );

CREATE POLICY "Users can update responses in their chats" ON chat_responses
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM chats 
            WHERE chats.id = chat_responses.chat_id 
            AND chats.user_id::text = auth.uid()::text
        )
    );

CREATE POLICY "Users can delete responses in their chats" ON chat_responses
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM chats 
            WHERE chats.id = chat_responses.chat_id 
            AND chats.user_id::text = auth.uid()::text
        )
    );

-- Chat files policies
CREATE POLICY "Users can view files in their chats" ON chat_files
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM chats 
            WHERE chats.id = chat_files.chat_id 
            AND chats.user_id::text = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert files in their chats" ON chat_files
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM chats 
            WHERE chats.id = chat_files.chat_id 
            AND chats.user_id::text = auth.uid()::text
        )
    );

CREATE POLICY "Users can delete files from their chats" ON chat_files
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM chats 
            WHERE chats.id = chat_files.chat_id 
            AND chats.user_id::text = auth.uid()::text
        )
    );

-- Transcriptions policies (updated to include user_id)
CREATE POLICY "Users can view their own transcriptions" ON transcriptions
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert their own transcriptions" ON transcriptions
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update their own transcriptions" ON transcriptions
    FOR UPDATE USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete their own transcriptions" ON transcriptions
    FOR DELETE USING (auth.uid()::text = user_id::text);

-- Functions for common operations
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
        COALESCE(cr.response_count, 0) as response_count,
        COALESCE(cf.file_count, 0) as file_count
    FROM chats c
    LEFT JOIN (
        SELECT chat_id, COUNT(*) as response_count 
        FROM chat_responses 
        GROUP BY chat_id
    ) cr ON c.id = cr.chat_id
    LEFT JOIN (
        SELECT chat_id, COUNT(*) as file_count 
        FROM chat_files 
        GROUP BY chat_id
    ) cf ON c.id = cf.chat_id
    WHERE c.user_id = user_uuid
    ORDER BY c.updated_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION get_chat_responses(chat_uuid UUID)
RETURNS TABLE (
    id UUID,
    user_message TEXT,
    ai_response TEXT,
    response_type TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cr.id,
        cr.user_message,
        cr.ai_response,
        cr.response_type,
        cr.metadata,
        cr.created_at
    FROM chat_responses cr
    JOIN chats c ON cr.chat_id = c.id
    WHERE c.id = chat_uuid 
    AND c.user_id::text = auth.uid()::text
    ORDER BY cr.created_at ASC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION get_chat_files(chat_uuid UUID)
RETURNS TABLE (
    id UUID,
    filename TEXT,
    duration FLOAT,
    language TEXT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.id,
        t.filename,
        t.duration,
        t.language,
        t.created_at
    FROM chat_files cf
    JOIN transcriptions t ON cf.transcription_id = t.id
    JOIN chats c ON cf.chat_id = c.id
    WHERE c.id = chat_uuid 
    AND c.user_id::text = auth.uid()::text
    ORDER BY cf.added_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 