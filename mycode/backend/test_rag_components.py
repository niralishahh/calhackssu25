import os
import logging
from dotenv import load_dotenv
from rag_handler import VertexAIRAGAgent

# --- Configuration ---
# Load environment variables from .env file.
# Make sure you have a .env file in the root of the project.
load_dotenv()

# Set up basic logging to see the output clearly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Test Parameters (customize as needed) ---
TEST_QUERY = "summarize?"
# This is the file ID from your logs that we are testing against
TEST_FILE_ID = "5c8147d1-913a-40d2-a528-763b9c7d82e9"

def run_tests():
    """Runs diagnostic tests on the RAG agent components."""
    logger.info("--- Starting RAG Component Test ---")

    # 1. Initialize Agent
    # This checks if your environment variables are loaded correctly.
    try:
        agent = VertexAIRAGAgent()
        logger.info("✅ SUCCESS: VertexAIRAGAgent initialized.")
    except Exception as e:
        logger.error(f"❌ FAILURE: Could not initialize VertexAIRAGAgent: {e}")
        logger.error("➡️ Please ensure SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GOOGLE_CLOUD_PROJECT_ID, and ANTHROPIC_API_KEY are set in your .env file.")
        return

    # 2. Test Embedding Generation
    logger.info("\n--- Testing Step 1: Embedding Generation ---")
    embedding = None
    try:
        embedding = agent._get_embedding_vertexai(TEST_QUERY)
        if embedding and isinstance(embedding, list):
            logger.info(f"✅ SUCCESS: Generated a {len(embedding)}-dimensional embedding.")
        else:
            # This could happen if the API returns an unexpected response or fails silently
            logger.error("❌ FAILURE: Embedding generation did not return a valid list.")
            # We'll try the fallback to see if that works
            embedding = agent._get_embedding_fallback(TEST_QUERY)
            logger.info(f"➡️ Trying fallback embedding. Success: {isinstance(embedding, list)}")
    except Exception as e:
        logger.error(f"❌ FAILURE: _get_embedding_vertexai failed with an error: {e}", exc_info=True)

    if not embedding:
        logger.error("🛑 Cannot proceed to RPC test without a valid embedding. Please check Google Cloud credentials and API configuration.")
        return

    # 3. Test Supabase RPC Calls
    logger.info("\n--- Testing Step 2: Supabase RPC Calls ---")
    rpc_params = {
        'query_embedding': embedding,
        'match_threshold': 0.1,  # Using a low threshold for testing to maximize chances of a match
        'match_count': 5,
        'filter_file_ids': [TEST_FILE_ID]
    }

    # Test 'match_documents' (the one currently in your code)
    try:
        logger.info("\n📞 Testing RPC call to 'match_documents'...")
        response = agent.supabase.rpc('match_documents', rpc_params).execute()
        if response.data:
            logger.info(f"✅ SUCCESS: 'match_documents' returned {len(response.data)} results.")
        else:
            logger.warning("⚠️ WARNING: 'match_documents' returned no data. This is likely the source of your error.")
    except Exception as e:
        logger.error(f"❌ FAILURE: RPC call to 'match_documents' failed entirely: {e}")

    # Test 'match_document_chunks' (the one from the SQL schema)
    try:
        logger.info("\n📞 Testing RPC call to 'match_document_chunks'...")
        response = agent.supabase.rpc('match_document_chunks', rpc_params).execute()
        if response.data:
            logger.info(f"✅ SUCCESS: 'match_document_chunks' returned {len(response.data)} results.")
        else:
            logger.warning("⚠️ WARNING: 'match_document_chunks' returned no data. If this also fails, the issue might be with the stored data itself.")
    except Exception as e:
        logger.error(f"❌ FAILURE: RPC call to 'match_document_chunks' failed entirely. This indicates the function does not exist in your DB schema under this name, or there's a permissions issue.")

    logger.info("\n--- Test Complete ---")

if __name__ == "__main__":
    run_tests() 