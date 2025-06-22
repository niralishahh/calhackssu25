-- Migration to add the 'is_saved' feature to the chat_responses table

-- Add the 'is_saved' column with a default value of FALSE.
-- This ensures existing rows get a sensible default.
ALTER TABLE public.chat_responses
ADD COLUMN IF NOT EXISTS is_saved BOOLEAN NOT NULL DEFAULT FALSE;

-- Add the 'saved_at' column to track when a response was saved.
ALTER TABLE public.chat_responses
ADD COLUMN IF NOT EXISTS saved_at TIMESTAMP WITH TIME ZONE;

-- Create an index on the 'is_saved' column to speed up queries
-- for fetching saved responses.
CREATE INDEX IF NOT EXISTS idx_chat_responses_is_saved ON public.chat_responses(is_saved);

-- The Supabase schema cache might need to be reloaded. 
-- This is often done automatically, but if you still see errors,
-- you can notify PostgREST of schema changes by running this:
NOTIFY pgrst, 'reload schema';

-- Confirmation
SELECT 'Migration to add saved responses feature has been applied.' as status; 