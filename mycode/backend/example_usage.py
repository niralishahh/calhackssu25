#!/usr/bin/env python3
"""
Example usage of the AI Transcription Agent
Demonstrates how to use the agent programmatically
"""

import asyncio
import os
from pathlib import Path
from transcription_agent import TranscriptionAgent, TranscriptionRequest

async def example_basic_transcription():
    """Example of basic transcription"""
    print("🎵 Example 1: Basic Transcription")
    print("-" * 40)
    
    agent = TranscriptionAgent()
    
    # Create a transcription request
    request = TranscriptionRequest(
        file_path="path/to/your/audio.mp3",  # Replace with actual file path
        language="en",
        prompt="This is a conversation about technology",
        temperature=0.0
    )
    
    print(f"Requesting transcription for: {request.file_path}")
    print(f"Language: {request.language}")
    print(f"Prompt: {request.prompt}")
    
    # Perform transcription
    result = await agent.transcribe_file(request)
    
    if result.success:
        print(f"✅ Transcription successful!")
        print(f"Text: {result.text[:100]}...")
        print(f"Duration: {result.duration:.2f} seconds")
        print(f"Words: {len(result.timestamps)}")
        print(f"Segments: {len(result.segments)}")
    else:
        print(f"❌ Transcription failed: {result.error}")

async def example_natural_language():
    """Example of natural language processing"""
    print("\n🤖 Example 2: Natural Language Processing")
    print("-" * 40)
    
    agent = TranscriptionAgent()
    
    # Natural language request
    user_request = "Please transcribe the audio file at /path/to/audio.mp3 in Spanish with context about medical terminology"
    
    print(f"User request: {user_request}")
    
    # Process the request
    result = await agent.process_transcription_request(user_request)
    
    print(f"Agent response: {result}")

async def example_file_validation():
    """Example of file validation"""
    print("\n🔍 Example 3: File Validation")
    print("-" * 40)
    
    agent = TranscriptionAgent()
    
    # Test different file types
    test_files = [
        "audio.mp3",
        "video.mp4", 
        "document.pdf",
        "nonexistent.wav"
    ]
    
    for file_path in test_files:
        is_valid = agent.validate_file(file_path)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"{file_path}: {status}")

async def example_supported_formats():
    """Example of checking supported formats"""
    print("\n📋 Example 4: Supported Formats")
    print("-" * 40)
    
    agent = TranscriptionAgent()
    
    print("Supported file formats:")
    for format_ext in agent.supported_formats:
        print(f"  • {format_ext}")
    
    print(f"\nTotal supported formats: {len(agent.supported_formats)}")

async def example_error_handling():
    """Example of error handling"""
    print("\n⚠️  Example 5: Error Handling")
    print("-" * 40)
    
    agent = TranscriptionAgent()
    
    # Try to transcribe a non-existent file
    request = TranscriptionRequest(
        file_path="/path/to/nonexistent/file.mp3",
        language="en"
    )
    
    print("Attempting to transcribe non-existent file...")
    result = await agent.transcribe_file(request)
    
    if not result.success:
        print(f"❌ Expected error: {result.error}")
    else:
        print("✅ Unexpected success!")

async def main():
    """Run all examples"""
    print("🎵 AI Transcription Agent - Example Usage")
    print("=" * 50)
    
    # Check if API keys are available
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not found")
        print("   Some examples may not work without proper API keys")
    
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not found")
        print("   Transcription examples will not work without this key")
    
    print()
    
    # Run examples
    await example_supported_formats()
    await example_file_validation()
    await example_error_handling()
    
    # Only run transcription examples if we have API keys
    if os.getenv("GROQ_API_KEY"):
        await example_basic_transcription()
        await example_natural_language()
    else:
        print("\n🔑 Skipping transcription examples (no GROQ_API_KEY)")
        print("   Set your GROQ_API_KEY to test transcription functionality")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run examples
 