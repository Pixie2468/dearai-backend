from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Conversation, Message
from app.services.conversations.schemas import ConversationCreate, ConversationUpdate


async def create_conversation(
    db: AsyncSession, user_id: UUID, data: ConversationCreate
) -> Conversation:
    conversation = Conversation(id=data.id, user_id=user_id, title=data.title, type=data.type)
    db.add(conversation)
    await db.flush()
    await db.refresh(conversation)
    return conversation


async def get_conversations(
    db: AsyncSession, user_id: UUID, skip: int = 0, limit: int = 50
) -> tuple[list[Conversation], int]:
    # Get total count
    count_result = await db.execute(
        select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
    )
    total = count_result.scalar() or 0

    # Get conversations
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all()), total


async def get_conversation_by_id(
    db: AsyncSession, conversation_id: UUID, user_id: UUID
) -> Conversation:
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


async def update_conversation(
    db: AsyncSession, conversation_id: UUID, user_id: UUID, data: ConversationUpdate
) -> Conversation:
    conversation = await get_conversation_by_id(db, conversation_id, user_id)
    if data.title is not None:
        conversation.title = data.title
    await db.flush()
    await db.refresh(conversation)
    return conversation


async def delete_conversation(db: AsyncSession, conversation_id: UUID, user_id: UUID) -> None:
    result = await db.execute(
        delete(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        )
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")


async def add_message(
    db: AsyncSession,
    conversation_id: UUID,
    user_id: UUID,
    role: str,
    content: str,
    msg_type: str = "text",
    audio_url: str | None = None,
    metadata: dict | None = None,
) -> Message:
    # Verify conversation belongs to user
    await get_conversation_by_id(db, conversation_id, user_id)

    message = Message(
        conv_id=conversation_id,
        role=role,
        content=content,
        type=msg_type,
        audio_url=audio_url,
        msg_metadata=metadata,
    )
    db.add(message)
    await db.flush()
    await db.refresh(message)
    return message


async def get_conversation_messages(
    db: AsyncSession, conversation_id: UUID, user_id: UUID, limit: int = 50
) -> list[Message]:
    # Verify conversation belongs to user
    await get_conversation_by_id(db, conversation_id, user_id)

    result = await db.execute(
        select(Message)
        .where(Message.conv_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))
