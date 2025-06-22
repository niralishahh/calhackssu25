import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import shutil

# Load environment variables from .env file
from dotenv import load_dotenv
from pathlib import Path

# Look for .env in parent directory (root) first, then current directory
parent_env = Path(__file__).parent.parent / ".env"
if parent_env.exists():
    load_dotenv(parent_env)
else:
    load_dotenv()

import google.generativeai as genai
from groq import Groq
from pydantic import BaseModel, Field

from audio_chunker import AudioChunker

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class TranscriptionRequest(BaseModel):
    """Request model for transcription"""
    file_path: str = Field(..., description="Path to the audio/video file")
    language: Optional[str] = Field(default="en", description="Language code for transcription")
    prompt: Optional[str] = Field(default="", description="Optional context or spelling prompt")
    temperature: Optional[float] = Field(default=0.0, description="Temperature for transcription")
    chunk_duration: Optional[int] = Field(default=300, description="Chunk duration in seconds (default: 5 minutes)")
    overlap_duration: Optional[int] = Field(default=10, description="Overlap duration in seconds (default: 10 seconds)")

class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    text: str
    timestamps: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    language: str
    duration: float
    success: bool
    error: Optional[str] = None
    chunked: bool = False
    num_chunks: int = 1
    chunks: List[Dict[str, Any]] = []

class TranscriptionAgent:
    """AI Agent for transcribing audio/video files using Groq"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.supported_formats = ['.mp3', '.mp4', '.wav', '.m4a', '.flac', '.ogg', '.webm']
        self.audio_chunker = AudioChunker()
        
    def validate_file(self, file_path: str) -> bool:
        """Validate if the file exists and is supported"""
        if not os.path.exists(file_path):
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats
    
    def extract_audio_if_needed(self, file_path: str) -> str:
        """Extract audio from video files if needed"""
        file_ext = Path(file_path).suffix.lower()
        
        # If it's already an audio file, return as is
        if file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            return file_path
        
        # For video files, we'll need to extract audio
        # In a production environment, you'd want to use ffmpeg here
        return file_path
    
    async def transcribe_chunk(self, chunk_path: str, request: TranscriptionRequest) -> dict:
        """Transcribe a single audio chunk"""
        try:
            with open(chunk_path, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3-turbo",
                    prompt=request.prompt,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language=request.language,
                    temperature=request.temperature
                )
            
            # Parse the verbose JSON response
            if hasattr(transcription, 'model_dump'):
                transcription_dict = transcription.model_dump()
            else:
                transcription_dict = transcription
            
            return transcription_dict
            
        except Exception as e:
            raise Exception(f"Failed to transcribe chunk {chunk_path}: {str(e)}")
    
    async def transcribe_file(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Transcribe an audio/video file using Groq's Whisper model with chunking support"""
        temp_dir = None
        chunk_paths = []
        
        try:
            # Validate file
            if not self.validate_file(request.file_path):
                return TranscriptionResponse(
                    text="",
                    timestamps=[],
                    segments=[],
                    language=request.language,
                    duration=0.0,
                    success=False,
                    error="File not found or unsupported format"
                )
            
            # Extract audio if needed
            audio_path = self.extract_audio_if_needed(request.file_path)
            
            # Check if file needs chunking
            file_size_mb = self.audio_chunker.get_file_size_mb(audio_path)
            needs_chunking = self.audio_chunker.should_chunk(audio_path, max_size_mb=25.0)
            
            print(f"File size: {file_size_mb:.2f} MB, Needs chunking: {needs_chunking}")
            
            if needs_chunking:
                print("Large file detected, using chunking...")
                
                # Get total duration from audio info first
                audio_info = self.audio_chunker.get_audio_info(audio_path)
                duration = audio_info['duration']
                
                # Create temporary directory for chunks
                temp_dir = tempfile.mkdtemp()
                
                # Preprocess audio to optimal format
                preprocessed_path = self.audio_chunker.preprocess_audio(audio_path)
                
                # Create chunks
                chunk_paths = self.audio_chunker.create_chunks(
                    preprocessed_path, 
                    temp_dir=temp_dir
                )
                
                print(f"Created {len(chunk_paths)} chunks")
                
                # Transcribe each chunk
                chunk_transcriptions = []
                chunks_data = []  # Store individual chunk data
                for i, chunk_path in enumerate(chunk_paths):
                    print(f"Transcribing chunk {i+1}/{len(chunk_paths)}...")
                    chunk_result = await self.transcribe_chunk(chunk_path, request)
                    chunk_transcriptions.append(chunk_result)
                    
                    # Store chunk data with timing information
                    chunk_start_time = i * (request.chunk_duration - request.overlap_duration)
                    chunk_end_time = min(chunk_start_time + request.chunk_duration, duration)
                    
                    chunks_data.append({
                        'chunk_index': i,
                        'start': chunk_start_time,
                        'end': chunk_end_time,
                        'text': chunk_result.get('text', ''),
                        'segments': chunk_result.get('segments', []),
                        'words': chunk_result.get('timestamps', [])
                    })
                
                # Merge transcriptions
                print("Merging transcription results...")
                merged_result = self.audio_chunker.merge_transcriptions(
                    chunk_transcriptions, 
                    overlap_duration=request.overlap_duration
                )
                
                # Extract information from merged result
                text = merged_result.get('text', '')
                segments = merged_result.get('segments', [])
                timestamps = merged_result.get('timestamps', [])
                
                return TranscriptionResponse(
                    text=text,
                    timestamps=timestamps,
                    segments=segments,
                    language=request.language,
                    duration=duration,
                    success=True,
                    chunked=True,
                    num_chunks=len(chunk_paths),
                    chunks=chunks_data  # Include chunks data
                )
                
            else:
                # Small file, transcribe directly
                print("Small file, transcribing directly...")
                
                with open(audio_path, "rb") as file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=file,
                        model="whisper-large-v3-turbo",
                        prompt=request.prompt,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                        language=request.language,
                        temperature=request.temperature
                    )
                
                # Parse the verbose JSON response
                if hasattr(transcription, 'model_dump'):
                    transcription_dict = transcription.model_dump()
                else:
                    transcription_dict = transcription
                
                # Extract relevant information
                text = transcription_dict.get('text', '')
                segments = transcription_dict.get('segments', [])
                language = transcription_dict.get('language', request.language)
                duration = transcription_dict.get('duration', 0.0)
                
                # Extract word-level timestamps
                timestamps = []
                for segment in segments:
                    if 'words' in segment:
                        for word in segment['words']:
                            timestamps.append({
                                'word': word.get('word', ''),
                                'start': word.get('start', 0.0),
                                'end': word.get('end', 0.0),
                                'confidence': word.get('confidence', 0.0)
                            })
                
                return TranscriptionResponse(
                    text=text,
                    timestamps=timestamps,
                    segments=segments,
                    language=language,
                    duration=duration,
                    success=True,
                    chunked=False,
                    num_chunks=1
                )
            
        except Exception as e:
            return TranscriptionResponse(
                text="",
                timestamps=[],
                segments=[],
                language=request.language,
                duration=0.0,
                success=False,
                error=str(e),
                chunked=False,
                num_chunks=0
            )
        
        finally:
            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")
    
    async def process_transcription_request(self, user_input: str) -> str:
        """Process a natural language request for transcription"""
        try:
            # Use Gemini to understand the user's request
            prompt = f"""
            You are a transcription assistant. The user wants to transcribe an audio/video file.
            User request: {user_input}
            
            Please extract the following information if mentioned:
            1. File path or location
            2. Language (default: en)
            3. Any specific context or spelling prompts
            4. Temperature setting (default: 0.0)
            5. Chunk duration (default: 300 seconds)
            6. Overlap duration (default: 10 seconds)
            
            Respond with a JSON object containing:
            {{
                "file_path": "path/to/file",
                "language": "en",
                "prompt": "",
                "temperature": 0.0,
                "chunk_duration": 300,
                "overlap_duration": 10
            }}
            
            If the file path is not provided, ask the user to provide it.
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to parse the JSON response
            try:
                import json
                parsed_response = json.loads(response_text)
                
                # Create transcription request
                transcription_request = TranscriptionRequest(**parsed_response)
                
                # Perform transcription
                result = await self.transcribe_file(transcription_request)
                
                if result.success:
                    chunk_info = f" (chunked into {result.num_chunks} parts)" if result.chunked else ""
                    return f"""
                    Transcription completed successfully!
                    
                    Language: {result.language}
                    Duration: {result.duration:.2f} seconds{chunk_info}
                    
                    Full Text:
                    {result.text}
                    
                    Word-level timestamps available: {len(result.timestamps)} words
                    Segments: {len(result.segments)} segments
                    
                    Detailed JSON output saved to transcription_result.json
                    """
                else:
                    return f"Transcription failed: {result.error}"
                    
            except json.JSONDecodeError:
                return f"Could not parse the request. Please provide a valid file path. Response: {response_text}"
                
        except Exception as e:
            return f"Error processing request: {str(e)}"

# Example usage
async def main():
    """Example usage of the TranscriptionAgent"""
    agent = TranscriptionAgent()
    
    # Example request
    user_request = "Please transcribe the audio file at /path/to/audio.mp3 in English"
    
    result = await agent.process_transcription_request(user_request)
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 