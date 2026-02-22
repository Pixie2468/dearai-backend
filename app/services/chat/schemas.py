from uuid import UUID

from pydantic import BaseModel

from app.services.emotion.schemas import EmotionResult


class TextChatRequest(BaseModel):
    conversation_id: UUID
    content: str


class TextChatResponse(BaseModel):
    message_id: UUID
    content: str
    is_crisis: bool = False


class VoiceChatRequest(BaseModel):
    conversation_id: UUID


class VoiceChatResponse(BaseModel):
    message_id: UUID
    content: str
    audio_url: str | None = None
    emotion: EmotionResult | None = None
    is_crisis: bool = False
