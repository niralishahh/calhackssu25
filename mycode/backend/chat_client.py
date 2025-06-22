import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from supabase import create_client, Client
from pydantic import BaseModel

class ChatRecord(BaseModel):
    """Model for chat records in Supabase"""
    id: Optional[str] = None
    user_id: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ChatResponseRecord(BaseModel):
    """Model for chat response records in Supabase"""
    id: Optional[str] = None
    chat_id: str
    user_message: str
    ai_response: str
    response_type: str = "text"
    metadata: Dict[str, Any] = {}
    created_at: Optional[str] = None

class ChatFileRecord(BaseModel):
    """Model for chat file associations in Supabase"""
    id: Optional[str] = None
    chat_id: str
    transcription_id: str
    added_at: Optional[str] = None

class ChatManager:
    """Manages chat operations with Supabase"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
    
    async def create_chat(self, user_id: str, title: str = "New Chat") -> str:
        """Create a new chat for a user"""
        try:
            record = ChatRecord(
                user_id=user_id,
                title=title,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            
            record_dict = record.dict()
            if 'id' in record_dict:
                del record_dict['id']
            
            result = self.client.table('chats').insert(record_dict).execute()
            
            if result.data:
                return result.data[0]['id']
            else:
                raise Exception("Failed to create chat")
                
        except Exception as e:
            raise Exception(f"Failed to create chat: {str(e)}")
    
    async def get_user_chats(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a user with response and file counts"""
        try:
            result = self.client.rpc('get_user_chats', {'user_uuid': user_id}).execute()
            return result.data if result.data else []
            
        except Exception as e:
            raise Exception(f"Failed to get user chats: {str(e)}")
    
    async def get_chat(self, chat_id: str) -> Optional[ChatRecord]:
        """Get a specific chat by ID"""
        try:
            result = self.client.table('chats').select('*').eq('id', chat_id).execute()
            
            if result.data:
                return ChatRecord(**result.data[0])
            return None
            
        except Exception as e:
            raise Exception(f"Failed to get chat: {str(e)}")
    
    async def update_chat_title(self, chat_id: str, title: str) -> bool:
        """Update the title of a chat"""
        try:
            result = self.client.table('chats').update({
                'title': title,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', chat_id).execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            raise Exception(f"Failed to update chat title: {str(e)}")
    
    async def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its associated data"""
        try:
            result = self.client.table('chats').delete().eq('id', chat_id).execute()
            return len(result.data) > 0
            
        except Exception as e:
            raise Exception(f"Failed to delete chat: {str(e)}")
    
    async def add_chat_response(self, chat_id: str, user_message: str, ai_response: str, 
                               response_type: str = "text", metadata: Dict[str, Any] = None) -> str:
        """Add a response to a chat"""
        try:
            record = ChatResponseRecord(
                chat_id=chat_id,
                user_message=user_message,
                ai_response=ai_response,
                response_type=response_type,
                metadata=metadata or {},
                created_at=datetime.utcnow().isoformat()
            )
            
            record_dict = record.dict()
            if 'id' in record_dict:
                del record_dict['id']
            
            result = self.client.table('chat_responses').insert(record_dict).execute()
            
            if result.data:
                # Update chat's updated_at timestamp
                await self.update_chat_title(chat_id, (await self.get_chat(chat_id)).title)
                return result.data[0]['id']
            else:
                raise Exception("Failed to add chat response")
                
        except Exception as e:
            raise Exception(f"Failed to add chat response: {str(e)}")
    
    async def get_chat_responses(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all responses for a chat"""
        try:
            result = self.client.rpc('get_chat_responses', {'chat_uuid': chat_id}).execute()
            return result.data if result.data else []
            
        except Exception as e:
            raise Exception(f"Failed to get chat responses: {str(e)}")
    
    async def add_file_to_chat(self, chat_id: str, transcription_id: str) -> str:
        """Add a file to a chat"""
        try:
            record = ChatFileRecord(
                chat_id=chat_id,
                transcription_id=transcription_id,
                added_at=datetime.utcnow().isoformat()
            )
            
            record_dict = record.dict()
            if 'id' in record_dict:
                del record_dict['id']
            
            result = self.client.table('chat_files').insert(record_dict).execute()
            
            if result.data:
                return result.data[0]['id']
            else:
                raise Exception("Failed to add file to chat")
                
        except Exception as e:
            raise Exception(f"Failed to add file to chat: {str(e)}")
    
    async def remove_file_from_chat(self, chat_id: str, transcription_id: str) -> bool:
        """Remove a file from a chat"""
        try:
            result = self.client.table('chat_files').delete().eq('chat_id', chat_id).eq('transcription_id', transcription_id).execute()
            return len(result.data) > 0
            
        except Exception as e:
            raise Exception(f"Failed to remove file from chat: {str(e)}")
    
    async def get_chat_files(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all files associated with a chat"""
        try:
            result = self.client.rpc('get_chat_files', {'chat_uuid': chat_id}).execute()
            return result.data if result.data else []
            
        except Exception as e:
            raise Exception(f"Failed to get chat files: {str(e)}")
    
    async def get_user_files(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all files uploaded by a user"""
        try:
            result = self.client.table('transcriptions').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
            return result.data if result.data else []
            
        except Exception as e:
            raise Exception(f"Failed to get user files: {str(e)}")

# Global instance
chat_manager: Optional[ChatManager] = None

def get_chat_manager() -> ChatManager:
    """Get or create the global chat manager instance"""
    global chat_manager
    
    if chat_manager is None:
        try:
            chat_manager = ChatManager()
        except ValueError as e:
            print(f"Warning: Chat manager not configured: {e}")
            return None
    
    return chat_manager 