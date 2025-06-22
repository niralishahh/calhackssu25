import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from supabase import create_client, Client
from pydantic import BaseModel

class TranscriptionRecord(BaseModel):
    """Model for transcription records in Supabase"""
    id: Optional[str] = None
    filename: str
    file_size_mb: float
    duration: float
    language: str
    text: str
    segments: List[Dict[str, Any]]
    timestamps: List[Dict[str, Any]]
    chunked: bool
    num_chunks: int
    chunk_duration: int
    overlap_duration: int
    prompt: Optional[str] = None
    temperature: float
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class SupabaseManager:
    """Manages Supabase operations for transcription storage"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
    
    def create_tables(self):
        """Create the necessary tables in Supabase (run once)"""
        # This would typically be done via Supabase dashboard or migrations
        # For now, we'll assume the table exists with the correct schema
        pass
    
    async def store_transcription(self, transcription_data: Dict[str, Any], user_id: str = None) -> str:
        """Store a transcription record in Supabase"""
        try:
            print(f"🔍 Debug: Processing transcription data with keys: {list(transcription_data.keys())}")
            
            # Create transcription record
            record = TranscriptionRecord(
                filename=transcription_data.get('filename', 'unknown'),
                file_size_mb=transcription_data.get('file_size_mb', 0.0),
                duration=transcription_data.get('duration', 0.0),
                language=transcription_data.get('language', 'en'),
                text=transcription_data.get('text', ''),
                segments=transcription_data.get('segments', []),
                timestamps=transcription_data.get('timestamps', []),
                chunked=transcription_data.get('chunked', False),
                num_chunks=transcription_data.get('num_chunks', 1),
                chunk_duration=transcription_data.get('metadata', {}).get('chunk_duration', 300),
                overlap_duration=transcription_data.get('metadata', {}).get('overlap_duration', 10),
                prompt=transcription_data.get('metadata', {}).get('prompt'),
                temperature=transcription_data.get('metadata', {}).get('temperature', 0.0),
                user_id=user_id,
                created_at=datetime.utcnow().isoformat()
            )
            
            print(f"🔍 Debug: Created record with filename: {record.filename}")
            print(f"🔍 Debug: Record duration: {record.duration}")
            print(f"🔍 Debug: Record text length: {len(record.text)}")
            
            # Insert into Supabase
            record_dict = record.dict()
            # Remove id field since it should be auto-generated
            if 'id' in record_dict:
                del record_dict['id']
            
            result = self.client.table('transcriptions').insert(record_dict).execute()
            
            print(f"🔍 Debug: Supabase insert result: {result}")
            
            if result.data:
                return result.data[0]['id']
            else:
                raise Exception("Failed to insert transcription record")
                
        except Exception as e:
            print(f"❌ Error in store_transcription: {str(e)}")
            print(f"❌ Error type: {type(e)}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to store transcription: {str(e)}")
    
    async def get_transcription(self, transcription_id: str) -> Optional[TranscriptionRecord]:
        """Retrieve a transcription record by ID"""
        try:
            result = self.client.table('transcriptions').select('*').eq('id', transcription_id).execute()
            
            if result.data:
                return TranscriptionRecord(**result.data[0])
            return None
            
        except Exception as e:
            raise Exception(f"Failed to retrieve transcription: {str(e)}")
    
    async def list_transcriptions(self, limit: int = 50, offset: int = 0) -> List[TranscriptionRecord]:
        """List all transcriptions with pagination"""
        try:
            result = self.client.table('transcriptions').select('*').order('created_at', desc=True).range(offset, offset + limit - 1).execute()
            
            return [TranscriptionRecord(**record) for record in result.data]
            
        except Exception as e:
            raise Exception(f"Failed to list transcriptions: {str(e)}")
    
    async def search_transcriptions(self, query: str, limit: int = 50) -> List[TranscriptionRecord]:
        """Search transcriptions by text content"""
        try:
            # Use full-text search if available, otherwise filter by text content
            result = self.client.table('transcriptions').select('*').ilike('text', f'%{query}%').limit(limit).execute()
            
            return [TranscriptionRecord(**record) for record in result.data]
            
        except Exception as e:
            raise Exception(f"Failed to search transcriptions: {str(e)}")
    
    async def delete_transcription(self, transcription_id: str) -> bool:
        """Delete a transcription record"""
        try:
            result = self.client.table('transcriptions').delete().eq('id', transcription_id).execute()
            return len(result.data) > 0
            
        except Exception as e:
            raise Exception(f"Failed to delete transcription: {str(e)}")

# Global instance
supabase_manager: Optional[SupabaseManager] = None

def get_supabase_manager() -> SupabaseManager:
    """Get or create the global Supabase manager instance"""
    global supabase_manager
    
    if supabase_manager is None:
        try:
            supabase_manager = SupabaseManager()
        except ValueError as e:
            print(f"Warning: Supabase not configured: {e}")
            return None
    
    return supabase_manager 