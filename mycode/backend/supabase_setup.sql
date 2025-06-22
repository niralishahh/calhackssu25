-- Supabase setup script for transcriptions table
-- Run this in your Supabase SQL editor

-- Create the transcriptions table
CREATE TABLE IF NOT EXISTS transcriptions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
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

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at ON transcriptions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transcriptions_language ON transcriptions(language);
CREATE INDEX IF NOT EXISTS idx_transcriptions_filename ON transcriptions(filename);
CREATE INDEX IF NOT EXISTS idx_transcriptions_text ON transcriptions USING gin(to_tsvector('english', text));

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_transcriptions_updated_at 
    BEFORE UPDATE ON transcriptions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE transcriptions ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (you can customize this based on your needs)
CREATE POLICY "Allow all operations" ON transcriptions
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Create a view for easier querying
CREATE OR REPLACE VIEW transcription_summary AS
SELECT 
    id,
    filename,
    file_size_mb,
    duration,
    language,
    LEFT(text, 200) as text_preview,
    chunked,
    num_chunks,
    created_at,
    updated_at
FROM transcriptions
ORDER BY created_at DESC;

-- Grant permissions (adjust based on your Supabase setup)
GRANT ALL ON transcriptions TO authenticated;
GRANT ALL ON transcription_summary TO authenticated;
GRANT USAGE ON SCHEMA public TO authenticated;

-- Insert sample data (optional)
INSERT INTO transcriptions (
    filename, 
    file_size_mb, 
    duration, 
    language, 
    text, 
    segments, 
    timestamps, 
    chunked, 
    num_chunks,
    prompt,
    temperature
) VALUES (
    'sample_audio.mp3',
    2.5,
    120.5,
    'en',
    'This is a sample transcription text for testing purposes.',
    '[{"start": 0.0, "end": 5.0, "text": "This is a sample", "words": [{"word": "This", "start": 0.0, "end": 1.0}, {"word": "is", "start": 1.0, "end": 2.0}, {"word": "a", "start": 2.0, "end": 3.0}, {"word": "sample", "start": 3.0, "end": 5.0}]}]',
    '[{"word": "This", "start": 0.0, "end": 1.0, "confidence": 0.95}, {"word": "is", "start": 1.0, "end": 2.0, "confidence": 0.98}, {"word": "a", "start": 2.0, "end": 3.0, "confidence": 0.99}, {"word": "sample", "start": 3.0, "end": 5.0, "confidence": 0.92}]',
    FALSE,
    1,
    'Sample context prompt',
    0.0
) ON CONFLICT DO NOTHING; 