from pydantic import BaseModel
from datetime import datetime

class ChatMessageBase(BaseModel):
    role: str
    content: str

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessageResponse(ChatMessageBase):
    id: str
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True
