#!/usr/bin/env python3
"""
Script to fix the Supabase database schema by adding the missing chunk_index column
"""

import os
from supabase.client import create_client

def fix_schema():
    """Add the missing chunk_index column to the document_chunks table"""
    try:
        # Get Supabase credentials from environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase environment variables")
            print("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
            return
        
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)
        
        # SQL to add the missing column
        sql = """
        ALTER TABLE document_chunks 
        ADD COLUMN IF NOT EXISTS chunk_index INTEGER;
        """
        
        # Execute the SQL using RPC
        try:
            result = supabase.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Successfully added chunk_index column to document_chunks table")
        except Exception as e:
            print(f"⚠️ RPC method failed: {e}")
            print("Trying alternative approach...")
            
            # Try using the Supabase client to execute raw SQL
            try:
                # First, let's check if the column already exists
                result = supabase.table("document_chunks").select("chunk_index").limit(1).execute()
                print("✅ chunk_index column already exists")
                return
            except Exception as e2:
                print(f"❌ Column doesn't exist and can't be added via RPC: {e2}")
                print("You may need to manually add the column in Supabase dashboard")
                return
        
        # Verify the column exists
        try:
            result = supabase.table("document_chunks").select("chunk_index").limit(1).execute()
            print("✅ Column verification successful")
        except Exception as e:
            print(f"⚠️ Column verification failed: {e}")
        
    except Exception as e:
        print(f"❌ Error fixing schema: {e}")

if __name__ == "__main__":
    fix_schema() 