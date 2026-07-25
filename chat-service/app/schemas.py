from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ChatSessionBase(BaseModel):
    title: Optional[str] = "New Chat"

class ChatSessionUpdate(BaseModel):
    title: str

class ChatSessionResponse(ChatSessionBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessageBase(BaseModel):
    role: str
    content: str
    session_id: str

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessageResponse(ChatMessageBase):
    id: str
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True
