from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class ConversationType(str, Enum):
    friend = "friend"
    therapy = "therapy"


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class MessageType(str, Enum):
    text = "text"
    voice = "voice"


# Conversation schemas
class ConversationCreate(BaseModel):
    id: str | None = None
    title: str | None = None
    type: ConversationType = ConversationType.therapy


class ConversationUpdate(BaseModel):
    title: str | None = None


class ConversationResponse(BaseModel):
    id: UUID
    title: str | None
    type: ConversationType
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int


# Message schemas
class MessageCreate(BaseModel):
    content: str
    type: MessageType = MessageType.text
    audio_url: str | None = None
    metadata: dict | None = None


class MessageResponse(BaseModel):
    id: UUID
    role: MessageRole
    content: str
    type: MessageType
    audio_url: str | None
    metadata: dict | None = Field(None, validation_alias="msg_metadata")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class ConversationWithMessages(ConversationResponse):
    messages: list[MessageResponse]


# Summary schemas
class SummaryResponse(BaseModel):
    id: UUID
    content: str
    last_message_id: UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}
