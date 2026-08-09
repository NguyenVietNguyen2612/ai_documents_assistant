from datetime import datetime

from pydantic import BaseModel


class Document(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    created_at: datetime