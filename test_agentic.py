#!/usr/bin/env python3
"""
Test script for the agentic AI system
"""

import requests
import json

def test_agentic_execution():
    """Test the agentic execution endpoint"""
    base_url = "http://localhost:5003"
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test 2: Test agentic text analysis
    try:
        response = requests.post(f"{base_url}/test-agentic")
        print(f"\nAgentic test: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Agentic test failed: {e}")
    
    # Test 3: Test with a simple question (if you have a test file)
    # This would require uploading a file first
    print("\nTo test with a document, upload a file and then use the agentic-execute endpoint")

if __name__ == "__main__":
    test_agentic_execution() 