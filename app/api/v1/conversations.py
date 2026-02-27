from uuid import UUID

from fastapi import APIRouter, Query

from app.core.dependencies import CurrentUser, DbSession
from app.services.conversations.schemas import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
    ConversationWithMessages,
)
from app.services.conversations.service import (
    create_conversation,
    delete_conversation,
    get_conversation_by_id,
    get_conversations,
    update_conversation,
)

router = APIRouter()


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_new_conversation(
    db: DbSession, current_user: CurrentUser, data: ConversationCreate
):
    return await create_conversation(db, current_user.id, data)


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    db: DbSession,
    current_user: CurrentUser,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    conversations, total = await get_conversations(db, current_user.id, skip, limit)
    return ConversationListResponse(
        conversations=[ConversationResponse.model_validate(c) for c in conversations], total=total
    )


@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(db: DbSession, current_user: CurrentUser, conversation_id: UUID):
    return await get_conversation_by_id(db, conversation_id, current_user.id)


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation_title(
    db: DbSession,
    current_user: CurrentUser,
    conversation_id: UUID,
    data: ConversationUpdate,
):
    return await update_conversation(db, conversation_id, current_user.id, data)


@router.delete("/{conversation_id}", status_code=204)
async def remove_conversation(db: DbSession, current_user: CurrentUser, conversation_id: UUID):
    await delete_conversation(db, conversation_id, current_user.id)
