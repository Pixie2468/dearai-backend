"""FastAPI entrypoint for Dear AI WebSocket chat."""

import asyncio
import base64
import contextlib
import json
import logging
import os
import httpx
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL", "http://chat_service:8000")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from app.auth.dependencies import verify_websocket_handshake
from app.auth.paseto import verify_internal_token
from app.services.context.graphrag import (
    evict_idle_graphs,
    retrieve_context,
    schedule_ingestion,
)
from app.services.llm.generate_output import stream_response
from app.services.stt.stt import transcribe_audio
from app.services.tts.tts import synthesize_speech
from app.services.safety.check import check_safety, check_relevance
from app.utils.llm_setup import setup_llm
from app.utils.setup_client import get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: pre-warm singletons + periodic cache cleanup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-warm heavy singletons at startup and run periodic cache cleanup."""
    # Eagerly initialize the GenAI client and LiteLLM so the first
    # request doesn't pay the cold-start penalty.
    logger.info("Pre-warming GenAI client…")
    get_client()
    logger.info("Pre-warming LiteLLM + embedder…")
    setup_llm()
    logger.info("Startup pre-warming complete.")

    # Background task to evict idle GraphRAG instances
    async def _eviction_loop() -> None:
        while True:
            await asyncio.sleep(5 * 60)  # check every 5 minutes
            await evict_idle_graphs()

    eviction_task = asyncio.create_task(_eviction_loop())

    yield

    # Shutdown: cancel the eviction loop
    eviction_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await eviction_task


app = FastAPI(title="Dear AI", lifespan=lifespan)


@dataclass
class ConnectionState:
    """Tracks the active task and request id for a socket."""

    active_task: asyncio.Task | None = None
    request_id: int = 0


@app.get("/health")
async def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/voice/tts")
async def tts_endpoint(request: Request) -> Response:
    """Convert text to speech using Google Cloud TTS.

    Expects X-Internal-Auth header (PASETO) and JSON body: {"text": "..."}.
    Returns audio/mpeg bytes.
    """
    # --- Auth: verify the PASETO token injected by the gateway ---
    token = request.headers.get("x-internal-auth")
    if not token:
        return JSONResponse(content={"error": "missing_auth"}, status_code=401)

    user_id = verify_internal_token(token)
    if not user_id:
        return JSONResponse(content={"error": "invalid_auth"}, status_code=401)

    # --- Parse body ---
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "invalid_json"}, status_code=400)

    text = body.get("text", "").strip()
    if not text:
        return JSONResponse(content={"error": "missing_text"}, status_code=400)

    # Cap text length to prevent abuse
    if len(text) > 5000:
        text = text[:5000]

    voice = body.get("voice", "en-US-Journey-F")

    logger.info("TTS request from user %s (%d chars)", user_id, len(text))

    try:
        audio_bytes = await synthesize_speech(text, voice=voice)
    except Exception as exc:
        logger.exception("TTS synthesis failed: %s", exc)
        return JSONResponse(content={"error": "tts_failed"}, status_code=500)

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/voice/stt")
async def stt_endpoint(request: Request) -> dict:
    """Convert speech to text using Google Cloud STT.

    Expects X-Internal-Auth header (PASETO) and JSON body: {"audio": "<base64>"}.
    Returns {"transcript": "..."}.
    """
    # --- Auth ---
    token = request.headers.get("x-internal-auth")
    if not token:
        return JSONResponse(content={"error": "missing_auth"}, status_code=401)

    user_id = verify_internal_token(token)
    if not user_id:
        return JSONResponse(content={"error": "invalid_auth"}, status_code=401)

    # --- Parse body ---
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "invalid_json"}, status_code=400)

    audio_b64 = body.get("audio", "")
    if not audio_b64:
        return JSONResponse(content={"error": "missing_audio"}, status_code=400)

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return JSONResponse(content={"error": "invalid_audio_encoding"}, status_code=400)

    # Cap audio size to 10 MB
    if len(audio_bytes) > 10 * 1024 * 1024:
        return JSONResponse(content={"error": "audio_too_large"}, status_code=400)

    logger.info("STT request from user %s (%d bytes)", user_id, len(audio_bytes))

    try:
        transcript = await transcribe_audio(audio_bytes)
    except Exception as exc:
        logger.exception("STT transcription failed: %s", exc)
        return JSONResponse(content={"error": "stt_failed"}, status_code=500)

    return {"transcript": transcript}


async def _cancel_active(state: ConnectionState) -> None:
    """Cancel any in-flight task and wait for cleanup."""
    if state.active_task and not state.active_task.done():
        state.active_task.cancel()
        try:
            await state.active_task
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Active task cleanup failed: %s", exc)


async def _safe_send_json(
    websocket: WebSocket, state: ConnectionState, request_id: int, payload: dict
) -> None:
    """Send JSON only when this request id is still active."""
    if request_id != state.request_id:
        return
    try:
        await websocket.send_json(payload)
    except Exception as exc:
        logger.debug("WebSocket send failed: %s", exc)


async def _fetch_tts_audio_b64(text: str, voice: str) -> str:
    """Synthesize speech and return it as base64."""
    audio_bytes = await synthesize_speech(text, voice=voice)
    return base64.b64encode(audio_bytes).decode("utf-8")


def _get_internal_token(user_id: str) -> str:
    import datetime
    from pyseto import encode
    from app.auth.paseto import PASETO_KEY
    now = datetime.datetime.now(datetime.timezone.utc)
    exp = (now + datetime.timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    payload = {
        "iss": os.getenv("PASETO_ISSUER", "dear-ai-gateway"),
        "aud": os.getenv("PASETO_AUDIENCE", "dear-ai-python-backend"),
        "sub": user_id,
        "exp": exp
    }
    return encode(PASETO_KEY, payload).decode("utf-8")


async def _auto_title_session(user_id: str, session_id: str, first_message: str):
    try:
        genai_client, model = get_client()
        prompt = f"Generate a very short title (max 5 words) for a chat that starts with this message. Return ONLY the title string, no quotes.\n\nMessage: {first_message}"
        response = await genai_client.aio.models.generate_content(
            model=str(model),
            contents=prompt,
        )
        title = response.text.strip().replace('"', '')
        token = _get_internal_token(user_id)
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.patch(
                f"{CHAT_SERVICE_URL}/sessions/{session_id}",
                json={"title": title},
                headers={"X-Internal-Auth": token}
            )
    except Exception as e:
        logger.error(f"Auto-title failed: {e}")


async def _handle_message(
    websocket: WebSocket,
    state: ConnectionState,
    user_id: str,
    token: str,
    content: str | None,
    audio_b64: str | None,
    voice_mode: bool,
    voice: str,
    request_id: int,
    session_id: str | None,
) -> None:
    """Run GraphRAG retrieval + LLM streaming for a single user message.

    Ingestion of the new message into the graph runs as a fire-and-forget
    background task so it never blocks the response.
    """
    try:
        # --- STT ---
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                content = await transcribe_audio(audio_bytes)
                if content and content.strip():
                    await _safe_send_json(
                        websocket, state, request_id, 
                        {"layer": "transcript", "content": content.strip(), "final": True}
                    )
            except Exception as exc:
                logger.error(f"[{request_id}] Failed to decode or transcribe audio: {exc}")
                await _safe_send_json(websocket, state, request_id, {"error": "stt_failed"})
                return

        if not content or not content.strip():
            # Nothing to process
            return

        content = content.strip()

        # --- Safety Check ---
        if not check_safety(content):
            logger.warning(f"[{request_id}] Safety check failed for user {user_id}. Halting generation.")
            await _safe_send_json(
                websocket,
                state,
                request_id,
                {
                    "layer": "emergency",
                    "content": "Emergency: We detected that you might be in distress. If you are experiencing a crisis, please contact emergency services or a crisis helpline immediately. Help is available.",
                    "final": False,
                },
            )
            await _safe_send_json(websocket, state, request_id, {"layer": "emergency", "content": "", "final": True})
            return

        # --- Relevance Check ---
        if not check_relevance(content):
            logger.warning(f"[{request_id}] Relevance check failed for user {user_id}. Halting generation.")
            await _safe_send_json(
                websocket,
                state,
                request_id,
                {
                    "layer": "irrelevant",
                    "content": "I am a friendly chatbot and I am not designed to help with coding or unrelated technical tasks. Let's chat about something else!",
                    "final": False,
                },
            )
            await _safe_send_json(websocket, state, request_id, {"layer": "irrelevant", "content": "", "final": True})
            return

        internal_token = _get_internal_token(user_id)
        
        is_new_session = False
        if not session_id:
            # Create a new session
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{CHAT_SERVICE_URL}/sessions",
                    json={"title": "New Chat"},
                    headers={"X-Internal-Auth": internal_token}
                )
                if resp.status_code == 200:
                    session_id = resp.json().get("id")
                    is_new_session = True
                else:
                    logger.error(f"Failed to create session: {resp.text}")
                    raise Exception("Could not create chat session")

        # Notify client of the active session ID so it can resume/continue
        await _safe_send_json(
            websocket, state, request_id,
            {"layer": "session_id", "content": session_id, "final": False}
        )

        if is_new_session:
            asyncio.create_task(_auto_title_session(user_id, session_id, content))

        await _safe_send_json(
            websocket,
            state,
            request_id,
            {
                "layer": "immediate",
                "content": "Thanks for sharing - give me a moment to think.",
                "final": False,
            },
        )

        history = []
        if not is_new_session:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{CHAT_SERVICE_URL}/chats/{session_id}?limit=20",
                    headers={"X-Internal-Auth": internal_token}
                )
                if resp.status_code == 200:
                    history = resp.json()

        # --- Fast path: retrieve existing context (no write) ---
        logger.info(f"[{request_id}] Retrieving graph context…")
        graph_context = await retrieve_context(user_id, content)
        logger.info(
            f"[{request_id}] Graph context retrieved! Starting LLM stream…"
        )

        # --- Stream the LLM response ---
        ai_response_chunks = []
        current_sentence = []
        import re
        sentence_end_pattern = re.compile(r'([.?!])\s+')
        
        tts_queue = asyncio.Queue()
        tts_worker = None
        
        if voice_mode:
            async def _tts_sender():
                while True:
                    item = await tts_queue.get()
                    if item is None:
                        break
                    try:
                        audio_b64 = await item
                        if audio_b64 and request_id == state.request_id:
                            await _safe_send_json(
                                websocket, state, request_id,
                                {
                                    "layer": "audio",
                                    "audio": audio_b64,
                                    "final": False,
                                }
                            )
                    except Exception as exc:
                        logger.error(f"[{request_id}] TTS task failed: {exc}")
            
            tts_worker = asyncio.create_task(_tts_sender())

        async for chunk in stream_response(content, graph_context, history):
            ai_response_chunks.append(chunk)
            await _safe_send_json(
                websocket,
                state,
                request_id,
                {
                    "layer": "rag",
                    "content": chunk,
                    "final": False,
                },
            )

            if voice_mode:
                current_sentence.append(chunk)
                # Basic sentence detection: look for ., ?, ! followed by space
                joined_sentence = "".join(current_sentence)
                match = sentence_end_pattern.search(joined_sentence)
                if match:
                    # Split at the punctuation
                    split_idx = match.end()
                    sentence_to_speak = joined_sentence[:split_idx].strip()
                    current_sentence = [joined_sentence[split_idx:]]
                    
                    if sentence_to_speak:
                        # Fire off TTS task for this sentence
                        task = asyncio.create_task(_fetch_tts_audio_b64(sentence_to_speak, voice))
                        tts_queue.put_nowait(task)

        # Handle any remaining text for TTS
        if voice_mode and current_sentence:
            sentence_to_speak = "".join(current_sentence).strip()
            if sentence_to_speak:
                task = asyncio.create_task(_fetch_tts_audio_b64(sentence_to_speak, voice))
                tts_queue.put_nowait(task)

        ai_content = "".join(ai_response_chunks)

        # --- Fire-and-forget: ingest the full interaction in the background ---
        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ingest_text = (
            f"Date: {now_str}\n"
            f"Session: {session_id}\n"
            f"User: {content}\n"
            f"AI: {ai_content}"
        )
        schedule_ingestion(user_id, ingest_text)

        # Wait for all background TTS tasks to finish before signaling completion
        if voice_mode:
            tts_queue.put_nowait(None)
            await tts_worker
            await _safe_send_json(
                websocket,
                state,
                request_id,
                {
                    "layer": "audio",
                    "audio": "",
                    "final": True,
                },
            )
        else:
            await _safe_send_json(
                websocket,
                state,
                request_id,
                {
                    "layer": "rag",
                    "content": "",
                    "final": True,
                },
            )
        
        # --- Save chat to chat-service ---
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"X-Internal-Auth": internal_token}
                # Save user message
                await client.post(
                    f"{CHAT_SERVICE_URL}/chats",
                    json={"role": "user", "content": content, "session_id": session_id},
                    headers=headers,
                )
                # Save AI message
                await client.post(
                    f"{CHAT_SERVICE_URL}/chats",
                    json={"role": "ai", "content": ai_content, "session_id": session_id},
                    headers=headers,
                )
        except Exception as e:
            logger.error("Failed to save chat to chat-service: %s", e)

    except asyncio.CancelledError:
        logger.info("Cancelled in-flight request %s", request_id)
        raise
    except Exception as exc:
        logger.exception("Request %s failed: %s", request_id, exc)
        await _safe_send_json(
            websocket,
            state,
            request_id,
            {
                "layer": "rag",
                "content": "Something went wrong while processing your request.",
                "final": True,
            },
        )


@app.websocket("/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """WebSocket chat handler with cancellation on new message."""
    auth_result = await verify_websocket_handshake(websocket)
    if auth_result is None:
        return
    user_id, token = auth_result

    await websocket.accept()

    state = ConnectionState()

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid_json"})
                continue

            content = payload.get("content")
            audio_b64 = payload.get("audio")
            session_id = payload.get("session_id")
            voice_mode = payload.get("voice_mode", False)
            voice = payload.get("voice", "en-US-Journey-F")
            
            if not content and not audio_b64:
                await websocket.send_json({"error": "missing_content_or_audio"})
                continue

            await _cancel_active(state)

            state.request_id += 1
            current_id = state.request_id

            state.active_task = asyncio.create_task(
                _handle_message(websocket, state, user_id, token, content, audio_b64, voice_mode, voice, current_id, session_id)
            )
    except WebSocketDisconnect:
        await _cancel_active(state)
