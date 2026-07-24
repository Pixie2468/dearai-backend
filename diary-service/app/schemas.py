from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DiaryEntryBase(BaseModel):
    title: str
    content: str

class DiaryEntryCreate(DiaryEntryBase):
    pass

class DiaryEntryResponse(DiaryEntryBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
