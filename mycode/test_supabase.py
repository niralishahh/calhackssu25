#!/usr/bin/env python3
"""
Test script to debug Supabase connection
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from parent directory
parent_env = Path(__file__).parent / ".env"
if parent_env.exists():
    load_dotenv(parent_env)
else:
    load_dotenv()

def test_supabase_config():
    """Test Supabase configuration"""
    print("🔍 Testing Supabase Configuration...")
    
    # Check environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    print(f"SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Not set'}")
    print(f"SUPABASE_ANON_KEY: {'✅ Set' if supabase_key else '❌ Not set'}")
    
    if supabase_url:
        print(f"URL: {supabase_url}")
    if supabase_key:
        print(f"Key: {supabase_key[:20]}...{supabase_key[-20:]}")
    
    # Test Supabase connection
    if supabase_url and supabase_key:
        try:
            from supabase import create_client
            client = create_client(supabase_url, supabase_key)
            
            # Test connection by trying to select from transcriptions table
            result = client.table('transcriptions').select('count').limit(1).execute()
            print("✅ Supabase connection successful!")
            print(f"✅ Table 'transcriptions' exists and is accessible")
            
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            print("\nPossible issues:")
            print("1. Database table 'transcriptions' doesn't exist")
            print("2. Row Level Security (RLS) is blocking access")
            print("3. API key doesn't have proper permissions")
            print("\nTo fix:")
            print("1. Run the SQL setup script in Supabase dashboard")
            print("2. Check RLS policies")
            print("3. Verify API key permissions")
    else:
        print("❌ Missing Supabase configuration")
        print("\nTo fix:")
        print("1. Make sure your .env file has SUPABASE_ANON_KEY (not SUPABASE_KEY)")
        print("2. Check that the .env file is in the correct location")

if __name__ == "__main__":
    test_supabase_config() 