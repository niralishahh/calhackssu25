#!/usr/bin/env python3
"""
Test script for Vertex AI RAG Document Analysis System
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5005"
TEST_FILE = "backend/claude/sample.txt"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_status():
    """Test status endpoint"""
    print("\n🔍 Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status check passed:")
            print(f"   Agent type: {data['agent_info']['agent_type']}")
            print(f"   Embedding model: {data['agent_info']['embedding_model']}")
            print(f"   Claude model: {data['agent_info']['anthropic_model']}")
            print(f"   Google project: {data['agent_info']['google_project_id']}")
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_file_upload():
    """Test file upload and ingestion"""
    print("\n🔍 Testing file upload...")
    
    if not os.path.exists(TEST_FILE):
        print(f"❌ Test file not found: {TEST_FILE}")
        return False
    
    try:
        with open(TEST_FILE, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"✅ File upload successful:")
            print(f"   File: {data['filename']}")
            print(f"   Chunks created: {data['ingestion_result']['chunks_created']}")
            return data['ingestion_result']['file_id']
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return None

def test_query(file_id):
    """Test document querying"""
    print(f"\n🔍 Testing document query with file ID: {file_id}")
    
    query_data = {
        "query": "What is this document about?",
        "file_ids": [file_id]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=query_data)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query successful:")
            print(f"   Query: {data['query']}")
            print(f"   Answer: {data['answer'][:200]}...")
            print(f"   Sources: {len(data['sources'])}")
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

def test_files_list():
    """Test files listing"""
    print("\n🔍 Testing files list...")
    
    try:
        response = requests.get(f"{BASE_URL}/files")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Files list successful:")
            print(f"   Files found: {len(data['files'])}")
            for file in data['files']:
                print(f"   - {file['file_name']} (ID: {file['file_id']})")
            return True
        else:
            print(f"❌ Files list failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Files list error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Vertex AI RAG System Tests...")
    print("=" * 50)
    
    # Check if server is running
    if not test_health():
        print("\n❌ Server is not running. Please start the server first:")
        print("   ./start_rag.sh")
        return
    
    # Test status
    if not test_status():
        print("\n❌ Status check failed. Check your configuration.")
        return
    
    # Test file upload
    file_id = test_file_upload()
    if not file_id:
        print("\n❌ File upload failed. Check your setup.")
        return
    
    # Wait a moment for processing
    print("\n⏳ Waiting for document processing...")
    time.sleep(2)
    
    # Test query
    if not test_query(file_id):
        print("\n❌ Query failed. Check your Vertex AI setup.")
        return
    
    # Test files list
    test_files_list()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("\n🎉 Your Vertex AI RAG system is working correctly!")
    print(f"🌐 Access the web interface at: {BASE_URL}")

if __name__ == "__main__":
    main() 