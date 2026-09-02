from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, Any]]] = []


class ChatResponse(BaseModel):
    question: str
    answer: str