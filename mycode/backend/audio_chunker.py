import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import wave
import math

class AudioChunker:
    """Handles audio file chunking for large files"""
    
    def __init__(self, chunk_duration: int = 300, overlap_duration: int = 10):
        """
        Initialize the audio chunker
        
        Args:
            chunk_duration: Duration of each chunk in seconds (default: 5 minutes)
            overlap_duration: Overlap between chunks in seconds (default: 10 seconds)
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
    def get_audio_info(self, file_path: str) -> dict:
        """Get audio file information using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # Extract audio stream info
            audio_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise ValueError("No audio stream found")
            
            duration = float(info['format']['duration'])
            sample_rate = int(audio_stream.get('sample_rate', 16000))
            channels = int(audio_stream.get('channels', 1))
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'format': info['format']['format_name']
            }
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to get audio info: {e}")
        except Exception as e:
            raise ValueError(f"Error processing audio file: {e}")
    
    def preprocess_audio(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Preprocess audio to 16kHz mono FLAC format for optimal transcription
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file (optional)
            
        Returns:
            Path to preprocessed audio file
        """
        if output_path is None:
            output_path = str(Path(input_path).with_suffix('.flac'))
        
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',  # Sample rate: 16kHz
                '-ac', '1',      # Channels: mono
                '-map', '0:a',   # Map audio stream
                '-c:a', 'flac',  # Codec: FLAC
                '-y',            # Overwrite output file
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to preprocess audio: {e}")
    
    def create_chunks(self, audio_path: str, temp_dir: Optional[str] = None) -> List[str]:
        """
        Create audio chunks from a large audio file
        
        Args:
            audio_path: Path to audio file
            temp_dir: Temporary directory for chunks (optional)
            
        Returns:
            List of paths to audio chunks
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        # Get audio information
        audio_info = self.get_audio_info(audio_path)
        duration = audio_info['duration']
        
        # If file is small enough, return the original
        if duration <= self.chunk_duration:
            return [audio_path]
        
        # Calculate number of chunks
        num_chunks = math.ceil(duration / (self.chunk_duration - self.overlap_duration))
        chunk_paths = []
        
        for i in range(num_chunks):
            start_time = i * (self.chunk_duration - self.overlap_duration)
            end_time = min(start_time + self.chunk_duration, duration)
            
            # Create chunk filename
            chunk_filename = f"chunk_{i:03d}.flac"
            chunk_path = os.path.join(temp_dir, chunk_filename)
            
            # Extract chunk using ffmpeg
            cmd = [
                'ffmpeg',
                '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'flac',
                '-y',
                chunk_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    
    def merge_transcriptions(self, transcriptions: List[dict], overlap_duration: int = 10) -> dict:
        """
        Merge multiple transcription results, handling overlaps
        
        Args:
            transcriptions: List of transcription results from chunks
            overlap_duration: Overlap duration in seconds
            
        Returns:
            Merged transcription result
        """
        if not transcriptions:
            return {'text': '', 'segments': [], 'timestamps': []}
        
        if len(transcriptions) == 1:
            return transcriptions[0]
        
        merged_text = ""
        merged_segments = []
        merged_timestamps = []
        
        current_time_offset = 0
        
        for i, transcription in enumerate(transcriptions):
            # Adjust timestamps for this chunk
            adjusted_segments = []
            adjusted_timestamps = []
            
            for segment in transcription.get('segments', []):
                adjusted_segment = segment.copy()
                adjusted_segment['start'] += current_time_offset
                adjusted_segment['end'] += current_time_offset
                
                # Adjust word timestamps
                if 'words' in adjusted_segment:
                    for word in adjusted_segment['words']:
                        word['start'] += current_time_offset
                        word['end'] += current_time_offset
                
                adjusted_segments.append(adjusted_segment)
            
            for timestamp in transcription.get('timestamps', []):
                adjusted_timestamp = timestamp.copy()
                adjusted_timestamp['start'] += current_time_offset
                adjusted_timestamp['end'] += current_time_offset
                adjusted_timestamps.append(adjusted_timestamp)
            
            # Handle overlap with previous chunk
            if i > 0 and overlap_duration > 0:
                # Remove overlapping segments from previous chunk
                overlap_start = current_time_offset - overlap_duration
                
                # Remove segments that overlap
                merged_segments = [
                    seg for seg in merged_segments 
                    if seg['end'] <= overlap_start
                ]
                
                # Remove timestamps that overlap
                merged_timestamps = [
                    ts for ts in merged_timestamps 
                    if ts['end'] <= overlap_start
                ]
            
            # Add text with space
            if merged_text and transcription.get('text', '').strip():
                merged_text += " "
            merged_text += transcription.get('text', '').strip()
            
            # Add segments and timestamps
            merged_segments.extend(adjusted_segments)
            merged_timestamps.extend(adjusted_timestamps)
            
            # Update time offset for next chunk
            if transcription.get('segments'):
                current_time_offset = adjusted_segments[-1]['end']
            else:
                current_time_offset += self.chunk_duration - self.overlap_duration
        
        return {
            'text': merged_text.strip(),
            'segments': merged_segments,
            'timestamps': merged_timestamps
        }
    
    def cleanup_chunks(self, chunk_paths: List[str]):
        """Clean up temporary chunk files"""
        for chunk_path in chunk_paths:
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except Exception as e:
                print(f"Warning: Could not remove chunk file {chunk_path}: {e}")
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in megabytes"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def should_chunk(self, file_path: str, max_size_mb: float = 25.0) -> bool:
        """Determine if a file should be chunked based on size"""
        file_size_mb = self.get_file_size_mb(file_path)
        return file_size_mb > max_size_mb 