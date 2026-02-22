import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.services.chat.schemas import (
    TextChatRequest,
    TextChatResponse,
    VoiceChatResponse,
)
from app.services.chat.service import process_text_chat, process_voice_chat
from app.services.cache.rag_cache import get_cached_rag_response
from app.services.llm.vertex import generate_immediate_response
from app.core.dependencies import CurrentUser, DbSession

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/text", response_model=TextChatResponse)
async def text_chat(db: DbSession, current_user: CurrentUser, request: TextChatRequest):
    """
    Send a text message and receive a text response.
    """
    return await process_text_chat(db, current_user.id, request)


@router.post("/voice", response_model=VoiceChatResponse)
async def voice_chat(
    db: DbSession,
    current_user: CurrentUser,
    conversation_id: UUID = Form(...),
    audio: UploadFile = File(...),
):
    """
    Send a voice message and receive a text response.

    The audio file should be in a supported format (wav, mp3, webm, etc.)
    """
    audio_data = await audio.read()
    response, _ = await process_voice_chat(
        db, current_user.id, conversation_id, audio_data
    )
    return response


@router.post("/voice/audio")
async def voice_chat_with_audio(
    db: DbSession,
    current_user: CurrentUser,
    conversation_id: UUID = Form(...),
    audio: UploadFile = File(...),
):
    """
    Send a voice message and receive an audio response.

    Returns the audio response directly as bytes.
    """
    audio_data = await audio.read()
    _, audio_response = await process_voice_chat(
        db, current_user.id, conversation_id, audio_data
    )

    if audio_response:
        return Response(content=audio_response, media_type="audio/mpeg")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# WebSocket – Two-Layer Concurrent Chat
# ---------------------------------------------------------------------------


async def _send_immediate(websocket: WebSocket, user_message: str) -> None:
    """Run the fast Vertex AI acknowledgement and send over WebSocket."""
    try:
        content = await asyncio.wait_for(
            generate_immediate_response(user_message),
            timeout=3.0,
        )
        await websocket.send_json(
            {
                "layer": "immediate",
                "content": content,
                "final": False,
            }
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("Immediate response timed out, skipping.")
    except Exception:
        logger.exception("Immediate response failed.")


async def _send_rag(websocket: WebSocket, user_message: str) -> None:
    """Run Graph RAG (with Redis cache) and send the result over WebSocket."""
    try:
        content = await get_cached_rag_response(user_message)
        await websocket.send_json(
            {
                "layer": "rag",
                "content": content,
                "final": True,
            }
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("RAG response failed.")


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


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """
    Two-layer concurrent WebSocket endpoint.

    For every incoming message the server spawns two tasks:
      • **immediate_task** – fast Vertex AI acknowledgement (≤ 3 s timeout)
      • **rag_task** – full Graph RAG response

    A new message cancels any still-running tasks from the previous turn.
    All task state is scoped to this single connection (no global state).
    """
    await websocket.accept()

    active_tasks: list[asyncio.Task] = []

    try:
        while True:
            data = await websocket.receive_json()
            user_message: str = data.get("content", "")

            if not user_message:
                continue

            # 1. Cancel any in-flight tasks from the previous message
            _cancel_tasks(active_tasks)
            await _await_tasks(active_tasks)
            active_tasks.clear()

            # 2. Launch both layers concurrently
            immediate_task = asyncio.create_task(
                _send_immediate(websocket, user_message)
            )
            rag_task = asyncio.create_task(_send_rag(websocket, user_message))

            active_tasks.extend([immediate_task, rag_task])

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception:
        logger.exception("WebSocket error.")
    finally:
        # Ensure cleanup on any exit path
        _cancel_tasks(active_tasks)
        await _await_tasks(active_tasks)
