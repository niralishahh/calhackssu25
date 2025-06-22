#!/usr/bin/env python3
"""
Connects to Supabase and deletes all rows from the document_chunks table,
effectively clearing all ingested documents from the RAG system.
"""

import os
from supabase.client import create_client, Client

def clear_all_documents():
    """Connects to Supabase and deletes all rows from the document_chunks table."""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in your environment.")
            return

        print("Connecting to Supabase...")
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase.")

        print("Clearing all rows from the 'document_chunks' table...")
        
        # This command deletes all rows in the table. 
        # The filter `neq("id", -1)` is a way to apply the delete to all rows,
        # since the 'id' column will always be a positive number.
        response = supabase.table("document_chunks").delete().neq("id", -1).execute()

        if response.data:
            print(f"✅ Successfully deleted {len(response.data)} chunk(s). All indexed documents have been cleared.")
        else:
            print("✅ 'document_chunks' table appears to be empty already.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    clear_all_documents() 