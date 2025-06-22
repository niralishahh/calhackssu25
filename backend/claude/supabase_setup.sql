-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the document_chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    file_id UUID NOT NULL, -- The unique ID for the source file
    file_name TEXT, -- The original name of the file
    content TEXT NOT NULL, -- The text of this specific chunk
    chunk_index INTEGER, -- The index of this chunk in the document
    embedding VECTOR(768) -- Match the dimension of Google's text-embedding-004 model
);

-- Create an index for efficient vector search
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
ON document_chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

-- Create an index on file_id for faster filtering
CREATE INDEX IF NOT EXISTS document_chunks_file_id_idx 
ON document_chunks (file_id);

-- Create the function for filtered semantic search
CREATE OR REPLACE FUNCTION match_document_chunks(
  query_embedding VECTOR(768),
  match_count INT DEFAULT 8,
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
  -- If no file IDs provided, search all documents
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
      chunks.file_id = ANY(filter_file_ids) -- This line is critical for filtering
    ORDER BY
      chunks.embedding <=> query_embedding
    LIMIT
      match_count;
  END IF;
END;
$$;

-- Create a function to get file statistics
CREATE OR REPLACE FUNCTION get_file_stats()
RETURNS TABLE (
  file_id UUID,
  file_name TEXT,
  chunk_count BIGINT,
  total_content_length BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    dc.file_id,
    dc.file_name,
    COUNT(*) as chunk_count,
    SUM(LENGTH(dc.content)) as total_content_length
  FROM
    document_chunks dc
  GROUP BY
    dc.file_id, dc.file_name
  ORDER BY
    dc.file_name;
END;
$$;

-- Create a function to clean up orphaned chunks
CREATE OR REPLACE FUNCTION cleanup_orphaned_chunks()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  -- This function can be used to clean up chunks that might be orphaned
  -- For now, it just returns 0 as we don't have a specific cleanup scenario
  deleted_count := 0;
  
  -- You can add specific cleanup logic here if needed
  -- For example, deleting chunks older than a certain date
  
  RETURN deleted_count;
END;
$$;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT USAGE ON SCHEMA public TO your_role;
-- GRANT ALL ON document_chunks TO your_role;
-- GRANT EXECUTE ON FUNCTION match_document_chunks TO your_role;
-- GRANT EXECUTE ON FUNCTION get_file_stats TO your_role;
-- GRANT EXECUTE ON FUNCTION cleanup_orphaned_chunks TO your_role;

-- Create the match_documents function for similarity search
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(768),
    match_threshold FLOAT,
    match_count INT,
    filter_file_ids UUID[]
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
    RETURN QUERY
    SELECT
        dc.id,
        dc.file_id,
        dc.file_name,
        dc.content,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE 1 - (dc.embedding <=> query_embedding) > match_threshold
      AND (filter_file_ids IS NULL OR dc.file_id = ANY(filter_file_ids))
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$; 