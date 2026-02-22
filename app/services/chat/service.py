from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.chat.handlers.text import handle_text_chat
from app.services.chat.handlers.voice import handle_voice_chat
from app.services.chat.schemas import (
    TextChatRequest,
    TextChatResponse,
    VoiceChatResponse,
)


async def process_text_chat(
    db: AsyncSession,
    user_id: UUID,
    request: TextChatRequest,
) -> TextChatResponse:
    """Process a text chat request."""
    return await handle_text_chat(
        db=db,
        user_id=user_id,
        conversation_id=request.conversation_id,
        content=request.content,
    )


async def process_voice_chat(
    db: AsyncSession,
    user_id: UUID,
    conversation_id: UUID,
    audio_data: bytes,
) -> tuple[VoiceChatResponse, bytes | None]:
    """Process a voice chat request."""
    return await handle_voice_chat(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        audio_data=audio_data,
    )
