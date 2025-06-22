import os
import logging
from dotenv import load_dotenv
from supabase.client import create_client, Client

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Parameters ---
# The File ID we want to inspect in the database
TARGET_FILE_ID = "5c8147d1-913a-40d2-a528-763b9c7d82e9"

def inspect_database_chunks():
    """Directly inspects the document_chunks table for a given file_id."""
    logger.info("--- Starting Database Chunk Inspection ---")

    # 1. Initialize Supabase Client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        logger.error("❌ FAILURE: Supabase credentials not found in .env file.")
        return
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("✅ SUCCESS: Supabase client initialized.")
    except Exception as e:
        logger.error(f"❌ FAILURE: Could not initialize Supabase client: {e}")
        return

    # 2. Query the document_chunks table directly
    logger.info(f"\n🔍 Querying 'document_chunks' table for file_id: {TARGET_FILE_ID}")
    try:
        # We select all columns to see the full picture
        response = supabase.table('document_chunks').select('*').eq('file_id', TARGET_FILE_ID).execute()
        
        if response.data:
            num_chunks = len(response.data)
            logger.info(f"✅ SUCCESS: Found {num_chunks} chunk(s) for this file_id.")
            
            # Inspect the first chunk as a sample
            first_chunk = response.data[0]
            logger.info("\n--- Sample Chunk Data ---")
            logger.info(f"  Chunk ID: {first_chunk.get('id')}")
            logger.info(f"  File Name: {first_chunk.get('file_name')}")
            
            # Check the embedding
            embedding = first_chunk.get('embedding')
            if embedding and isinstance(embedding, list):
                logger.info(f"  Embedding: Found a {len(embedding)}-dimensional vector. (e.g., {str(embedding[:5])[:-1]}...])")
            elif isinstance(embedding, str):
                 logger.warning(f"  Embedding: Found a string, not a vector! Length: {len(embedding)}. This is likely the problem.")
            else:
                logger.error("  Embedding: ❌ CRITICAL: The embedding is missing or null. This is the cause of the issue.")

            # Check the content
            content = first_chunk.get('content', '')
            logger.info(f"  Content Preview: '{content[:100]}...'")

        else:
            logger.error(f"❌ CRITICAL: No chunks found for file_id '{TARGET_FILE_ID}'. The file was likely never ingested correctly.")

    except Exception as e:
        logger.error(f"❌ FAILURE: An error occurred while querying the table: {e}", exc_info=True)
        logger.error("➡️ This could be a permissions issue or a problem with the table name. Please verify 'document_chunks' exists.")

    logger.info("\n--- Inspection Complete ---")

if __name__ == "__main__":
    inspect_database_chunks() 