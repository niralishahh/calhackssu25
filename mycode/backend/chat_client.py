import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
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
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
    
    async def get_or_create_user(self, user_id: Optional[str]) -> Dict[str, Any]:
        """
        Retrieves a user by ID. If user_id is None or not found, creates a new user.
        This prevents foreign key violations and handles new user sessions.
        """
        try:
            # If a user_id is provided, try to find them.
            if user_id:
                existing_user = self.client.table('users').select('id').eq('id', user_id).execute()
                if existing_user.data:
                    return existing_user.data[0]

            # If user_id is None, or if the provided user_id was not found, create a new user.
            new_user_id = str(uuid.uuid4())
            new_user_data = {
                'id': new_user_id,
                'email': f"{new_user_id}@placeholder.email"
            }
            
            creation_result = self.client.table('users').insert(new_user_data).execute()
            
            if creation_result.data:
                return creation_result.data[0]
            else:
                # This case should ideally not be reached if the DB is healthy.
                raise Exception("Failed to create a new user in the database.")
                
        except Exception as e:
            # Re-raise with a more informative message
            raise Exception(f"Failed to get or create user (ID: {user_id}): {str(e)}")
    
    async def check_chat_exists(self, chat_id: str) -> bool:
        """Check if a chat with the given ID exists."""
        try:
            result = self.client.table('chats').select('id').eq('id', chat_id).limit(1).execute()
            return bool(result.data)
        except Exception as e:
            # In case of error, assume it doesn't exist to be safe.
            print(f"Error checking if chat exists: {e}")
            return False
    
    async def create_chat(self, user_id: str, title: str = "New Chat") -> Dict[str, Any]:
        """Create a new chat for a user and return the full chat object."""
        try:
            record = {
                "user_id": user_id,
                "title": title
            }
            
            result = self.client.table('chats').insert(record).execute()
            
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to create chat in database.")
                
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
    
    async def get_chat_response_by_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Gets a single chat response by its ID and includes the chat title and source filenames."""
        try:
            # First, get the response data
            response_result = self.client.table('chat_responses').select('*').eq('id', response_id).limit(1).execute()
            if not response_result.data:
                return None
            
            response_data = response_result.data[0]
            chat_id = response_data.get('chat_id')

            # Now, get the corresponding chat title
            chat_title = "Untitled Chat"
            if chat_id:
                chat_result = self.client.table('chats').select('title').eq('id', chat_id).limit(1).execute()
                if chat_result.data:
                    chat_title = chat_result.data[0].get('title', chat_title)
            
            # Add the chat title to the response object
            response_data['chat_title'] = chat_title
            
            # Now, get the source filenames
            file_ids = response_data.get('metadata', {}).get('file_ids', [])
            source_files = []
            if file_ids:
                files_result = self.client.table('transcriptions').select('id, filename').in_('id', file_ids).execute()
                if files_result.data:
                    source_files = files_result.data
            
            response_data['source_files'] = source_files

            return response_data
        except Exception as e:
            raise Exception(f"Failed to get chat response {response_id}: {e}")
    
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
        """
        Gets all files associated with a specific chat.
        This method was updated to avoid an RPC call that was causing a type mismatch
        (numeric vs double precision) when joining with the transcriptions table.
        By using a direct query and casting DECIMAL columns, we resolve the issue.
        """
        try:
            select_str = (
                "id, user_id, filename, file_size_mb::float8, duration::float8, language, text, "
                "segments, timestamps, chunked, num_chunks, chunk_duration, overlap_duration, "
                "prompt, temperature::float8, created_at, updated_at"
            )
            
            query = self.client.table("chat_files").select(
                f"transcription_id, transcriptions({select_str})"
            ).eq("chat_id", chat_id)
            
            result = query.execute()
            
            if result.data:
                return [
                    item['transcriptions'] for item in result.data 
                    if 'transcriptions' in item and item['transcriptions'] is not None
                ]
            return []
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Failed to get chat files: {e}")
    
    async def get_user_files(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all files uploaded by a user"""
        try:
            result = self.client.table('transcriptions').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
            return result.data if result.data else []
            
        except Exception as e:
            raise Exception(f"Failed to get user files: {str(e)}")

    async def set_response_saved_status(self, response_id: str, is_saved: bool) -> bool:
        """Sets the 'is_saved' status of a chat response."""
        try:
            update_data = {
                "is_saved": is_saved,
                "saved_at": datetime.utcnow().isoformat() if is_saved else None
            }
            result = self.client.table('chat_responses').update(update_data).eq('id', response_id).execute()
            return len(result.data) > 0
        except Exception as e:
            raise Exception(f"Failed to update saved status for response {response_id}: {e}")

    async def get_saved_responses(self, user_id: str) -> List[Dict[str, Any]]:
        """Gets all saved responses for a given user."""
        try:
            # We need to join tables to get responses for a specific user
            # user -> chats -> chat_responses
            user_chats = self.client.table('chats').select('id').eq('user_id', user_id).execute()
            if not user_chats.data:
                return []
            
            chat_ids = [chat['id'] for chat in user_chats.data]
            
            saved_responses = self.client.table('chat_responses').select('*').in_('chat_id', chat_ids).eq('is_saved', True).order('saved_at', desc=True).execute()
            
            return saved_responses.data if saved_responses.data else []
        except Exception as e:
            raise Exception(f"Failed to get saved responses for user {user_id}: {e}")

    async def get_saved_responses_for_chat(self, chat_id: str) -> List[Dict[str, Any]]:
        """Gets all saved responses for a specific chat."""
        try:
            query = self.client.table("chat_responses").select(
                "id, chat_id, user_message, ai_response, created_at"
            ).eq("chat_id", chat_id).eq("is_saved", True).order("created_at", desc=True)
            
            result = query.execute()
            
            if result.data:
                return result.data
            return []
        except Exception as e:
            print(f"Error fetching saved responses for chat {chat_id}: {e}")
            return []

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