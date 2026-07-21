"""FastAPI entrypoint for Dear AI WebSocket chat."""

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

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


@app.post("/tts")
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


@app.post("/stt")
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


async def _handle_message(
    websocket: WebSocket,
    state: ConnectionState,
    user_id: str,
    content: str,
    request_id: int,
) -> None:
    """Run GraphRAG retrieval + LLM streaming for a single user message.

    Ingestion of the new message into the graph runs as a fire-and-forget
    background task so it never blocks the response.
    """
    try:
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

        # --- Fast path: retrieve existing context (no write) ---
        logger.info(f"[{request_id}] Retrieving graph context…")
        graph_context = await retrieve_context(user_id, content)
        logger.info(
            f"[{request_id}] Graph context retrieved! Starting LLM stream…"
        )

        # --- Fire-and-forget: ingest the new message in the background ---
        schedule_ingestion(user_id, content)

        # --- Stream the LLM response ---
        async for chunk in stream_response(content, graph_context):
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
    user_id = await verify_websocket_handshake(websocket)
    if user_id is None:
        return

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
            if not isinstance(content, str) or not content.strip():
                await websocket.send_json({"error": "missing_content"})
                continue

            await _cancel_active(state)

            state.request_id += 1
            current_id = state.request_id

            state.active_task = asyncio.create_task(
                _handle_message(websocket, state, user_id, content.strip(), current_id)
            )
    except WebSocketDisconnect:
        await _cancel_active(state)
