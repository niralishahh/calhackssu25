#!/usr/bin/env python3
"""
Example script demonstrating how to use the Claude AI Agent for document summarization
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from claude import ClaudeAgent

def main():
    """Main example function"""
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable is required")
        print("Please set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        print("❌ Error: GOOGLE_CLOUD_PROJECT environment variable is required")
        print("Please set your Google Cloud project ID:")
        print("export GOOGLE_CLOUD_PROJECT='your-project-id'")
        return
    
    try:
        print("🚀 Initializing Claude AI Agent...")
        
        # Initialize the agent
        agent = ClaudeAgent()
        
        print("✅ Agent initialized successfully!")
        print(f"📊 Agent Status: {agent.get_agent_status()}")
        
        # Example document paths (relative to project root)
        sample_files = [
            "../app/sample.txt",
            "../sample2.txt"
        ]
        
        # Check if sample files exist
        existing_files = []
        for file_path in sample_files:
            if Path(file_path).exists():
                existing_files.append(file_path)
                print(f"📄 Found sample file: {file_path}")
            else:
                print(f"⚠️  Sample file not found: {file_path}")
        
        if not existing_files:
            print("❌ No sample files found. Please create some text files to test with.")
            return
        
        # Example 1: Single document summarization
        print("\n" + "="*50)
        print("📝 EXAMPLE 1: Single Document Summarization")
        print("="*50)
        
        file_path = existing_files[0]
        print(f"Summarizing: {file_path}")
        
        # Comprehensive summary
        result = agent.summarize_document(file_path, "comprehensive")
        
        print(f"\n📊 Summary Statistics:")
        print(f"   Original word count: {result.word_count}")
        print(f"   Summary word count: {result.summary_length}")
        print(f"   Confidence score: {result.confidence_score:.2f}")
        
        print(f"\n📋 Key Points:")
        for i, point in enumerate(result.key_points, 1):
            print(f"   {i}. {point}")
        
        print(f"\n📄 Summary:")
        print(result.summary)
        
        # Example 2: Different summary types
        print("\n" + "="*50)
        print("📝 EXAMPLE 2: Different Summary Types")
        print("="*50)
        
        summary_types = ["bullet_points", "executive"]
        
        for summary_type in summary_types:
            print(f"\n🔹 {summary_type.upper()} Summary:")
            result = agent.summarize_document(file_path, summary_type)
            print(result.summary)
        
        # Example 3: Batch summarization (if multiple files exist)
        if len(existing_files) > 1:
            print("\n" + "="*50)
            print("📝 EXAMPLE 3: Batch Summarization")
            print("="*50)
            
            print(f"Summarizing {len(existing_files)} documents...")
            results = agent.batch_summarize(existing_files, "comprehensive")
            
            for i, result in enumerate(results):
                print(f"\n📄 Document {i+1}: {existing_files[i]}")
                print(f"   Word count: {result.word_count}")
                print(f"   Summary length: {result.summary_length}")
                print(f"   Key points: {len(result.key_points)}")
        
        print("\n" + "="*50)
        print("✅ All examples completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 