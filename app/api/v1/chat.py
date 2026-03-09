import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import async_session_maker
from app.models import Message, Summary, User
from app.services.chat.schemas import (
    TextChatRequest,
    TextChatResponse,
    VoiceChatResponse,
)
from app.services.chat.service import process_text_chat, process_voice_chat
from app.services.context.graph_rag import (
    ChatState,
    StateGraph,
    extract_entities_node,
    graph_retrieval_node,
    graph_update_node,
    router_node,
)
from app.services.conversations.service import add_message
from app.services.llm.vertex import generate_immediate_response, generate_rag_continuation
from app.core.dependencies import CurrentUser, DbSession

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# REST endpoints (unchanged)
# ---------------------------------------------------------------------------


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
# WebSocket – Two-Layer Concurrent Chat (with JWT auth & persistence)
# ---------------------------------------------------------------------------


async def _authenticate_ws(websocket: WebSocket) -> User | None:
    """Authenticate a WebSocket connection via JWT token.

    The client must send the token as:
      1. Query param: ``?token=<jwt>``
      2. OR in the first JSON message: ``{"type": "auth", "token": "<jwt>"}``

    Returns the User on success, or None (and closes the socket) on failure.
    """
    # Try query parameter first
    token = websocket.query_params.get("token")

    if not token:
        # Accept temporarily to receive the auth message
        await websocket.accept()
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            if data.get("type") == "auth":
                token = data.get("token")
            if not token:
                await websocket.send_json(
                    {"type": "error", "content": "Missing authentication token."}
                )
                await websocket.close(code=4001, reason="Missing token")
                return None
        except (asyncio.TimeoutError, Exception):
            await websocket.close(code=4001, reason="Auth timeout")
            return None
    else:
        await websocket.accept()

    # Validate JWT
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise JWTError("Missing sub claim")
    except JWTError:
        await websocket.send_json(
            {"type": "error", "content": "Invalid or expired token."}
        )
        await websocket.close(code=4003, reason="Invalid token")
        return None

    # Look up the user
    async with async_session_maker() as db:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

    if user is None:
        await websocket.send_json(
            {"type": "error", "content": "User not found."}
        )
        await websocket.close(code=4003, reason="User not found")
        return None

    await websocket.send_json({"type": "auth_ok", "user_id": str(user.id)})
    return user


async def _get_conversation_summary(db: AsyncSession, conversation_id: UUID) -> str:
    """Fetch the latest summary for a conversation (empty string if none)."""
    result = await db.execute(
        select(Summary)
        .where(Summary.conv_id == conversation_id)
        .order_by(Summary.created_at.desc())
        .limit(1)
    )
    summary = result.scalar_one_or_none()
    return summary.content if summary else ""


async def _get_recent_history(
    db: AsyncSession, conversation_id: UUID, limit: int = 20
) -> list[dict[str, str]]:
    """Fetch recent messages as plain dicts for LLM consumption."""
    result = await db.execute(
        select(Message)
        .where(Message.conv_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = list(reversed(result.scalars().all()))
    return [{"role": m.role, "content": m.content} for m in rows]


async def _run_rag_retrieval(
    user_message: str,
    user_id: str,
    conversation_history: list[dict[str, str]],
) -> list[str]:
    """Run only the retrieval portion of Graph RAG (no LLM generation).

    Returns the list of context strings from FalkorDB without invoking the
    LLM generation node—because Layer 2 handles generation separately via
    ``generate_rag_continuation``.
    """
    state = ChatState(
        session_id="ws",
        user_id=user_id,
        user_message=user_message,
        conversation_history=conversation_history,
    )

    # Build a mini-pipeline: router → extract → retrieve (no LLM gen)
    workflow = StateGraph()
    workflow.add_node("router", router_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("graph_retrieval", graph_retrieval_node)

    workflow.add_conditional_edge("router", lambda s: s.next_node)
    workflow.add_edge("extract_entities", "graph_retrieval")

    # If router sends to "llm_generation" (direct path), we redirect to stop
    async def _stop_node(s: ChatState) -> ChatState:
        s.should_stop = True
        return s

    workflow.add_node("llm_generation", _stop_node)

    final = await workflow.execute(start_node="router", state=state)
    return final.graph_context


async def _fire_graph_update(
    user_message: str,
    user_id: str,
    extracted_entities: dict,
) -> None:
    """Fire-and-forget: update the user's graph with extracted entities."""
    try:
        state = ChatState(
            session_id="ws",
            user_id=user_id,
            user_message=user_message,
            extracted_entities=extracted_entities,
        )
        await graph_update_node(state)
    except Exception:
        logger.warning("Background graph update failed.", exc_info=True)


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
    """Two-layer concurrent WebSocket endpoint with JWT auth and persistence.

    **Connection flow**:
      1. Client connects with ``?token=<jwt>`` query param, OR sends
         ``{"type": "auth", "token": "<jwt>"}`` as the first message.
      2. Server responds ``{"type": "auth_ok", "user_id": "..."}`` on success.

    **Message flow** (per user message):
      1. Client sends ``{"type": "message", "conversation_id": "<uuid>", "content": "..."}``
      2. Server persists the user message to the database.
      3. Server launches two concurrent pipelines:
         - **Layer 1** (fast): ``gemini-2.0-flash-lite`` produces a 1-2 sentence
           empathetic acknowledgement. Sent as
           ``{"type": "layer1", "content": "...", "conversation_id": "..."}``
         - **Graph RAG retrieval**: FalkorDB context retrieval runs in parallel
           with Layer 1.
      4. Once **both** Layer 1 and RAG retrieval finish, the server calls
         ``generate_rag_continuation`` to produce a context-rich continuation.
      5. The continuation is sent as
         ``{"type": "layer2", "content": "...", "conversation_id": "..."}``
      6. The full response (layer1 + layer2) is persisted as an assistant message.
      7. Graph update runs in the background.

    **Voice support**:
      Client sends ``{"type": "voice", "conversation_id": "<uuid>", "audio": "<base64>"}``
      Server transcribes via Sarvam STT, then runs the same two-layer flow.
      After Layer 2, TTS is generated and sent as
      ``{"type": "audio", "audio": "<base64-wav>", "conversation_id": "..."}``
    """
    # --- Authenticate ---
    user = await _authenticate_ws(websocket)
    if user is None:
        return

    user_id = str(user.id)
    active_tasks: list[asyncio.Task] = []

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")

            # ---------------------------------------------------------------
            # Voice messages: transcribe first, then treat as text
            # ---------------------------------------------------------------
            if msg_type == "voice":
                import base64
                conversation_id_str = data.get("conversation_id")
                audio_b64 = data.get("audio", "")
                if not conversation_id_str or not audio_b64:
                    await websocket.send_json(
                        {"type": "error", "content": "voice requires conversation_id and audio"}
                    )
                    continue
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    from app.services.speech.stt import get_stt
                    stt = get_stt()
                    transcript, lang = await stt.transcribe(audio_bytes)
                    # Notify client of transcription
                    await websocket.send_json({
                        "type": "transcription",
                        "content": transcript,
                        "language": lang or "",
                        "conversation_id": conversation_id_str,
                    })
                    # Re-use as text message
                    data = {
                        "type": "message",
                        "conversation_id": conversation_id_str,
                        "content": transcript,
                        "voice": True,  # flag for TTS on response
                    }
                    msg_type = "message"
                except Exception:
                    logger.exception("Voice transcription failed.")
                    await websocket.send_json(
                        {"type": "error", "content": "Transcription failed."}
                    )
                    continue

            # ---------------------------------------------------------------
            # Text / transcribed messages
            # ---------------------------------------------------------------
            if msg_type != "message":
                continue

            conversation_id_str = data.get("conversation_id")
            user_message: str = data.get("content", "").strip()
            wants_voice: bool = data.get("voice", False)

            if not user_message or not conversation_id_str:
                await websocket.send_json(
                    {"type": "error", "content": "content and conversation_id required"}
                )
                continue

            try:
                conversation_id = UUID(conversation_id_str)
            except ValueError:
                await websocket.send_json(
                    {"type": "error", "content": "Invalid conversation_id format"}
                )
                continue

            # Cancel any in-flight tasks from a previous turn
            _cancel_tasks(active_tasks)
            await _await_tasks(active_tasks)
            active_tasks.clear()

            # ---------------------------------------------------------------
            # Persist user message & fetch context (in a DB session)
            # ---------------------------------------------------------------
            async with async_session_maker() as db:
                try:
                    await add_message(
                        db, conversation_id, user.id, "user", user_message, "text"
                    )
                    await db.commit()
                except Exception:
                    await db.rollback()
                    logger.exception("Failed to persist user message.")

            async with async_session_maker() as db:
                summary = await _get_conversation_summary(db, conversation_id)
                conversation_history = await _get_recent_history(
                    db, conversation_id, limit=20
                )

            # ---------------------------------------------------------------
            # Layer 1 (fast ack) + RAG retrieval -- run concurrently
            # ---------------------------------------------------------------
            layer1_task = asyncio.create_task(
                generate_immediate_response(
                    user_message,
                    summary=summary,
                    conversation_history=conversation_history,
                )
            )
            rag_task = asyncio.create_task(
                _run_rag_retrieval(
                    user_message,
                    user_id=user_id,
                    conversation_history=conversation_history,
                )
            )

            active_tasks.extend([layer1_task, rag_task])

            # Wait for Layer 1 to finish, then send immediately
            layer1_response = ""
            try:
                layer1_response = await asyncio.wait_for(layer1_task, timeout=5.0)
                await websocket.send_json({
                    "type": "layer1",
                    "content": layer1_response,
                    "conversation_id": conversation_id_str,
                })
            except asyncio.TimeoutError:
                logger.warning("Layer 1 timed out.")
                layer1_response = (
                    "I hear you, and I want you to know that your feelings are valid."
                )
                await websocket.send_json({
                    "type": "layer1",
                    "content": layer1_response,
                    "conversation_id": conversation_id_str,
                })
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Layer 1 failed.")
                layer1_response = (
                    "I hear you, and I want you to know that your feelings are valid."
                )

            # Wait for RAG retrieval to finish
            graph_context: list[str] = []
            try:
                graph_context = await asyncio.wait_for(rag_task, timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning("RAG retrieval timed out.")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("RAG retrieval failed.")

            # ---------------------------------------------------------------
            # Layer 2: Context-rich continuation
            # ---------------------------------------------------------------
            layer2_response = ""
            try:
                layer2_response = await generate_rag_continuation(
                    user_message=user_message,
                    layer1_response=layer1_response,
                    graph_context=graph_context,
                    summary=summary,
                    conversation_history=conversation_history,
                )
                await websocket.send_json({
                    "type": "layer2",
                    "content": layer2_response,
                    "conversation_id": conversation_id_str,
                })
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Layer 2 generation failed.")
                layer2_response = ""

            # ---------------------------------------------------------------
            # Persist the full assistant response
            # ---------------------------------------------------------------
            full_response = layer1_response
            if layer2_response:
                full_response += " " + layer2_response

            async with async_session_maker() as db:
                try:
                    await add_message(
                        db,
                        conversation_id,
                        user.id,
                        "assistant",
                        full_response,
                        "text",
                    )
                    await db.commit()
                except Exception:
                    await db.rollback()
                    logger.exception("Failed to persist assistant message.")

            # ---------------------------------------------------------------
            # Voice: generate TTS and send audio back
            # ---------------------------------------------------------------
            if wants_voice and layer2_response:
                try:
                    from app.services.speech.tts import get_tts
                    import base64 as b64

                    tts = get_tts()
                    audio_bytes = await tts.synthesize(full_response)
                    if audio_bytes:
                        await websocket.send_json({
                            "type": "audio",
                            "audio": b64.b64encode(audio_bytes).decode(),
                            "conversation_id": conversation_id_str,
                        })
                except Exception:
                    logger.exception("TTS synthesis failed.")

            # ---------------------------------------------------------------
            # Background: update graph + maybe generate summary
            # ---------------------------------------------------------------
            # Fire-and-forget graph update (don't block the next message)
            asyncio.create_task(
                _fire_graph_update(user_message, user_id, {})
            )

            # Fire-and-forget summary generation check
            async def _maybe_summarize(cid: UUID) -> None:
                try:
                    from app.services.context.summary_generator import (
                        maybe_generate_summary,
                    )
                    async with async_session_maker() as _db:
                        await maybe_generate_summary(_db, cid)
                        await _db.commit()
                except Exception:
                    logger.debug("Background summary generation skipped.", exc_info=True)

            asyncio.create_task(_maybe_summarize(conversation_id))

            active_tasks.clear()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for user %s.", user_id)
    except Exception:
        logger.exception("WebSocket error for user %s.", user_id)
    finally:
        _cancel_tasks(active_tasks)
        await _await_tasks(active_tasks)
