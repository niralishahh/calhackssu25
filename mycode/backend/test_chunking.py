#!/usr/bin/env python3
"""
Test script for audio chunking functionality
"""

import asyncio
import os
from pathlib import Path
from transcription_agent import TranscriptionAgent, TranscriptionRequest
from audio_chunker import AudioChunker

async def test_chunking():
    """Test the chunking functionality"""
    print("🧪 Testing Audio Chunking Functionality")
    print("=" * 50)
    
    # Initialize components
    agent = TranscriptionAgent()
    chunker = AudioChunker()
    
    # Test file path (replace with actual test file)
    test_file = "test_audio.mp3"  # Replace with your test file
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Please provide a test audio file to verify chunking functionality")
        return
    
    # Get file info
    file_size_mb = chunker.get_file_size_mb(test_file)
    needs_chunking = chunker.should_chunk(test_file, max_size_mb=25.0)
    
    print(f"📁 Test file: {test_file}")
    print(f"📏 File size: {file_size_mb:.2f} MB")
    print(f"🔧 Needs chunking: {needs_chunking}")
    
    if needs_chunking:
        print("\n🔍 Testing chunking process...")
        
        # Get audio info
        try:
            audio_info = chunker.get_audio_info(test_file)
            print(f"⏱️  Duration: {audio_info['duration']:.2f} seconds")
            print(f"🎵 Sample rate: {audio_info['sample_rate']} Hz")
            print(f"🔊 Channels: {audio_info['channels']}")
        except Exception as e:
            print(f"❌ Error getting audio info: {e}")
            return
        
        # Test preprocessing
        print("\n🔄 Testing audio preprocessing...")
        try:
            preprocessed_path = chunker.preprocess_audio(test_file)
            print(f"✅ Preprocessed: {preprocessed_path}")
        except Exception as e:
            print(f"❌ Error preprocessing: {e}")
            return
        
        # Test chunking
        print("\n✂️  Testing chunk creation...")
        try:
            chunks = chunker.create_chunks(preprocessed_path)
            print(f"✅ Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                chunk_size = chunker.get_file_size_mb(chunk)
                print(f"   Chunk {i+1}: {chunk_size:.2f} MB")
                
        except Exception as e:
            print(f"❌ Error creating chunks: {e}")
            return
        
        # Test transcription (if API keys are available)
        if os.getenv("GROQ_API_KEY"):
            print("\n🎤 Testing transcription with chunking...")
            try:
                request = TranscriptionRequest(
                    file_path=test_file,
                    language="en",
                    chunk_duration=300,
                    overlap_duration=10
                )
                
                result = await agent.transcribe_file(request)
                
                if result.success:
                    print(f"✅ Transcription successful!")
                    print(f"📝 Text length: {len(result.text)} characters")
                    print(f"🔢 Words: {len(result.timestamps)}")
                    print(f"📊 Segments: {len(result.segments)}")
                    print(f"🔧 Chunked: {result.chunked}")
                    print(f"📦 Number of chunks: {result.num_chunks}")
                    print(f"⏱️  Duration: {result.duration:.2f} seconds")
                    
                    # Show first 200 characters of transcription
                    preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
                    print(f"\n📄 Preview: {preview}")
                else:
                    print(f"❌ Transcription failed: {result.error}")
            except Exception as e:
                print(f"❌ Error during transcription: {e}")
        else:
            print("\n🔑 Skipping transcription test (no GROQ_API_KEY)")
    
    else:
        print("\n📝 File is small enough for direct transcription")
        print("Try with a larger file (>25MB) to test chunking functionality")
    
    print("\n✅ Chunking test completed!")

def test_audio_chunker():
    """Test the AudioChunker class directly"""
    print("\n🔧 Testing AudioChunker class...")
    
    chunker = AudioChunker(chunk_duration=60, overlap_duration=5)  # 1-minute chunks
    
    print(f"Default chunk duration: {chunker.chunk_duration} seconds")
    print(f"Default overlap duration: {chunker.overlap_duration} seconds")
    
    # Test file size calculation
    test_files = ["test_audio.mp3", "nonexistent.mp3"]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            size_mb = chunker.get_file_size_mb(test_file)
            should_chunk = chunker.should_chunk(test_file)
            print(f"📁 {test_file}: {size_mb:.2f} MB, should chunk: {should_chunk}")
        else:
            print(f"📁 {test_file}: File not found")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run tests
    test_audio_chunker()
    asyncio.run(test_chunking()) 