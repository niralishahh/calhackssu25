#!/usr/bin/env python3
"""
Script to run the SQL setup for Supabase
"""

import os
from supabase.client import create_client

def run_sql_setup():
    """Run the SQL setup commands"""
    try:
        # Get Supabase credentials from environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase environment variables")
            return
        
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)
        
        # SQL to create the match_documents function
        sql = """
        CREATE OR REPLACE FUNCTION match_documents(
            query_embedding VECTOR(1536),
            match_threshold FLOAT DEFAULT 0.7,
            match_count INT DEFAULT 8
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
            ORDER BY dc.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        """
        
        # Try to execute the SQL
        try:
            result = supabase.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Successfully created match_documents function")
        except Exception as e:
            print(f"⚠️ RPC method failed: {e}")
            print("You may need to manually create the function in Supabase dashboard")
            print("Go to SQL Editor and run the function creation SQL")
            return
        
        # Test the function
        try:
            # Create a test embedding (all zeros)
            test_embedding = [0.0] * 1536
            result = supabase.rpc('match_documents', {
                'query_embedding': test_embedding,
                'match_threshold': 0.0,
                'match_count': 1
            }).execute()
            print("✅ Function test successful")
        except Exception as e:
            print(f"⚠️ Function test failed: {e}")
        
    except Exception as e:
        print(f"❌ Error running SQL setup: {e}")

if __name__ == "__main__":
    run_sql_setup() 