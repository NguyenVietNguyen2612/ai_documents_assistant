from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, Any]]] = []
    session_id: Optional[str] = "default"
    checkpoint_id: Optional[str] = None
    rollback_from_message_id: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    session_id: Optional[str] = "default"
    checkpoint_id: Optional[str] = None

class SessionItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "Cuộc trò chuyện mới"

class UpdateSessionRequest(BaseModel):
    title: str

class RollbackRequest(BaseModel):
    session_id: str
    checkpoint_id: str
    message_id: Optional[str] = None

class RollbackResponse(BaseModel):
    status: str = "success"
    session_id: str
    checkpoint_id: str
    messages: List[Dict[str, Any]]

class CheckpointInfo(BaseModel):
    checkpoint_id: str
    parent_checkpoint_id: Optional[str] = None
    step: Optional[int] = None
    timestamp: Optional[str] = None
    summary: Optional[str] = None
    message_count: Optional[int] = None