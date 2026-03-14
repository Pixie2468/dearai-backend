import asyncio
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import async_session_maker
from app.core.dependencies import CurrentUser, DbSession
from app.models import Conversation, ConversationType, Message, Summary, User
from app.services.cache.rag_cache import get_cached_rag_response_stream
from app.services.chat.schemas import (
    TextChatRequest,
    TextChatResponse,
    VoiceChatResponse,
)
from app.services.chat.service import process_text_chat, process_voice_chat
from app.services.llm.base import LLMMessage
from app.services.llm.factory import get_layer1_llm
from app.services.llm.prompts import LAYER1_PROMPT

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# REST endpoints (unchanged)
# ---------------------------------------------------------------------------


@router.post("/text", response_model=TextChatResponse)
async def text_chat(db: DbSession, current_user: CurrentUser, request: TextChatRequest):
    """Send a text message and receive a text response."""
    return await process_text_chat(db, current_user.id, request)


@router.post("/voice", response_model=VoiceChatResponse)
async def voice_chat(
    db: DbSession,
    current_user: CurrentUser,
    conversation_id: UUID = Form(...),
    audio: UploadFile = File(...),
):
    """Send a voice message and receive a text response."""
    audio_data = await audio.read()
    response, _ = await process_voice_chat(db, current_user.id, conversation_id, audio_data)
    return response


@router.post("/voice/audio")
async def voice_chat_with_audio(
    db: DbSession,
    current_user: CurrentUser,
    conversation_id: UUID = Form(...),
    audio: UploadFile = File(...),
):
    """Send a voice message and receive an audio response."""
    audio_data = await audio.read()
    _, audio_response = await process_voice_chat(db, current_user.id, conversation_id, audio_data)

    if audio_response:
        return Response(content=audio_response, media_type="audio/mpeg")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------

_LAYER1_TIMEOUT_SECS = 5


async def _authenticate_ws(websocket: WebSocket) -> User | None:
    """Wait for an auth message on the WebSocket and validate the JWT.

    Returns the authenticated ``User`` or ``None`` on failure.
    """
    try:
        data = await asyncio.wait_for(websocket.receive_json(), timeout=10)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        return None

    if data.get("type") != "auth" or not data.get("token"):
        await websocket.send_json({"type": "error", "content": "Expected auth message"})
        return None

    token: str = data["token"]
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise JWTError("Missing sub")
    except JWTError:
        await websocket.send_json({"type": "error", "content": "Invalid token"})
        return None

    async with async_session_maker() as db:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

    if user is None:
        await websocket.send_json({"type": "error", "content": "User not found"})
    return user


async def _get_or_create_conversation(
    db: AsyncSession, user_id: UUID, conversation_id: UUID | None = None
) -> Conversation:
    """Return an existing conversation or create a new one."""
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        conv = result.scalar_one_or_none()
        if conv:
            return conv

    # Auto-create
    conv = Conversation(
        id=conversation_id or uuid4(),
        user_id=user_id,
        type=ConversationType.friend,
    )
    db.add(conv)
    await db.flush()
    return conv


async def _get_summary_context(db: AsyncSession, conversation_id: UUID) -> str:
    """Fetch the most recent conversation summary."""
    result = await db.execute(
        select(Summary)
        .where(Summary.conv_id == conversation_id)
        .order_by(Summary.created_at.desc())
        .limit(1)
    )
    summary = result.scalar_one_or_none()
    if summary:
        return f"\n\nPrevious conversation summary:\n{summary.content}"
    return ""


async def _get_conversation_history(
    db: AsyncSession, conversation_id: UUID, limit: int = 20
) -> list[LLMMessage]:
    """Fetch the last N messages and convert to LLMMessage objects."""
    result = await db.execute(
        select(Message)
        .where(Message.conv_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = list(reversed(result.scalars().all()))
    return [LLMMessage(role=m.role.value, content=m.content) for m in rows]


async def _save_message(db: AsyncSession, conversation_id: UUID, role: str, content: str) -> None:
    """Persist a single message to the database."""
    from app.models import MessageRole, MessageType

    msg = Message(
        conv_id=conversation_id,
        role=MessageRole(role),
        content=content,
        type=MessageType.text,
    )
    db.add(msg)
    await db.flush()


# ---------------------------------------------------------------------------
# Layer tasks
# ---------------------------------------------------------------------------


async def _send_immediate(
    websocket: WebSocket,
    user_message: str,
    summary_context: str,
) -> None:
    """Layer 1: stream a fast empathetic acknowledgement via gemini-2.0-flash-lite."""
    try:
        async with asyncio.timeout(_LAYER1_TIMEOUT_SECS):
            llm = get_layer1_llm()

            system_prompt = LAYER1_PROMPT
            if summary_context:
                system_prompt = f"{LAYER1_PROMPT}{summary_context}"

            messages = [LLMMessage(role="user", content=user_message)]

            await websocket.send_json({"layer": "immediate", "type": "stream_start"})

            async for chunk in llm.chat_stream(messages, system_prompt):
                await websocket.send_json(
                    {"layer": "immediate", "type": "stream_chunk", "content": chunk}
                )

            await websocket.send_json({"layer": "immediate", "type": "stream_end"})

    except asyncio.CancelledError:
        raise
    except TimeoutError:
        logger.warning("Layer 1 timed out after %ds", _LAYER1_TIMEOUT_SECS)
        # Still send stream_end so frontend knows Layer 1 is done
        try:
            await websocket.send_json({"layer": "immediate", "type": "stream_end"})
        except Exception:
            pass
    except Exception:
        logger.exception("Layer 1 failed.")


async def _send_rag(
    websocket: WebSocket,
    user_message: str,
    conversation_history: list[LLMMessage],
) -> str:
    """Layer 2: run Graph RAG pipeline + stream the fine-tuned model response.

    Returns the full assembled response text (for persistence).
    """
    collected: list[str] = []
    try:
        await websocket.send_json({"layer": "rag", "type": "stream_start"})

        async for chunk in get_cached_rag_response_stream(user_message, conversation_history):
            collected.append(chunk)
            await websocket.send_json({"layer": "rag", "type": "stream_chunk", "content": chunk})

        await websocket.send_json({"layer": "rag", "type": "stream_end", "final": True})

    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Layer 2 (RAG) failed.")

    return "".join(collected)


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------


def _cancel_tasks(tasks: list[asyncio.Task]) -> None:
    """Request cancellation for every task in the list."""
    for task in tasks:
        if not task.done():
            task.cancel()


async def _await_tasks(tasks: list[asyncio.Task]) -> None:
    """Await all tasks, suppressing CancelledError."""
    for task in tasks:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """Two-layer concurrent WebSocket endpoint with authentication.

    Protocol:
        1. Client connects, server accepts.
        2. Client sends ``{"type": "auth", "token": "<jwt>"}``
           (optionally include ``"conversation_id": "<uuid>"``).
        3. Server validates JWT, creates/loads conversation, replies
           ``{"type": "auth_success", "conversation_id": "..."}``
        4. Client sends ``{"type": "message", "content": "..."}``
           (or ``{"content": "..."}`` for backward compat).
        5. Server streams Layer 1 (immediate) and Layer 2 (rag) in parallel.
        6. Layer 2 ``stream_start`` signals the frontend to replace Layer 1.
    """
    await websocket.accept()

    # --- Step 1: Authenticate ---
    user = await _authenticate_ws(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # --- Step 2: Create / load conversation ---
    async with async_session_maker() as db:
        conv = await _get_or_create_conversation(db, user.id)
        await db.commit()
        conversation_id = conv.id

    await websocket.send_json(
        {
            "type": "auth_success",
            "conversation_id": str(conversation_id),
        }
    )

    # --- Step 3: Message loop ---
    active_tasks: list[asyncio.Task] = []
    rag_result_holder: list[str] = []  # to capture Layer 2 result for persistence

    try:
        while True:
            data = await websocket.receive_json()
            user_message: str = data.get("content", "")

            if not user_message:
                continue

            # Cancel any in-flight tasks from the previous turn
            _cancel_tasks(active_tasks)
            await _await_tasks(active_tasks)
            active_tasks.clear()
            rag_result_holder.clear()

            # Fetch context from DB for this turn
            async with async_session_maker() as db:
                summary_ctx = await _get_summary_context(db, conversation_id)
                history = await _get_conversation_history(db, conversation_id)

            # Launch both layers concurrently
            async def _rag_wrapper():
                result = await _send_rag(websocket, user_message, history)
                rag_result_holder.append(result)

            immediate_task = asyncio.create_task(
                _send_immediate(websocket, user_message, summary_ctx)
            )
            rag_task = asyncio.create_task(_rag_wrapper())
            active_tasks.extend([immediate_task, rag_task])

            # Wait for both to finish before persisting
            await asyncio.gather(*active_tasks, return_exceptions=True)
            active_tasks.clear()

            # Persist messages
            rag_response = rag_result_holder[0] if rag_result_holder else ""
            if rag_response:
                async with async_session_maker() as db:
                    await _save_message(db, conversation_id, "user", user_message)
                    await _save_message(db, conversation_id, "assistant", rag_response)
                    await db.commit()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (user=%s).", user.id)
    except Exception:
        logger.exception("WebSocket error (user=%s).", user.id)
    finally:
        _cancel_tasks(active_tasks)
        await _await_tasks(active_tasks)
