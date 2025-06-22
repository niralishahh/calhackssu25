#!/usr/bin/env python3
"""
Simple test script to verify the FastAPI server
"""

import requests
import json
import time

def test_server():
    base_url = "http://localhost:5003"
    
    print("Testing FastAPI server...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/status")
        print(f"Status check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Status check failed: {e}")
        return False
    
    # Test upload endpoint with a simple text file
    try:
        # Create a simple test file
        with open("test.txt", "w") as f:
            f.write("This is a test document for summarization.")
        
        with open("test.txt", "rb") as f:
            files = {"file": f}
            response = requests.post(f"{base_url}/upload", files=files)
        
        print(f"Upload test: {response.status_code}")
        if response.status_code == 200:
            upload_data = response.json()
            print(f"Upload response: {upload_data}")
            
            # Test summarize endpoint
            summarize_data = {
                "file_path": upload_data["file_path"],
                "summary_type": "comprehensive"
            }
            
            response = requests.post(
                f"{base_url}/summarize",
                headers={"Content-Type": "application/json"},
                data=json.dumps(summarize_data)
            )
            
            print(f"Summarize test: {response.status_code}")
            if response.status_code == 200:
                print("✅ Summarization successful!")
                result = response.json()
                print(f"Summary: {result['summary'][:100]}...")
            else:
                print(f"❌ Summarization failed: {response.text}")
                return False
        else:
            print(f"Upload failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Upload/summarize test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_server()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!") 