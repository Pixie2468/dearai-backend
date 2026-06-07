"""FastAPI entrypoint for Dear AI WebSocket chat."""

import asyncio
import json
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.auth.dependencies import verify_websocket_handshake
from app.services.context.graphrag import DearAIGraphService
from app.services.llm.generate_output import stream_response

logger = logging.getLogger(__name__)

app = FastAPI(title="Dear AI")


@dataclass
class ConnectionState:
    """Tracks the active task and request id for a socket."""

    active_task: asyncio.Task | None = None
    request_id: int = 0


@app.get("/health")
async def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


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
    """Run GraphRAG + LLM streaming for a single user message."""
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

        async with DearAIGraphService(user_id) as graph_service:
            logger.info(f"[{request_id}] Querying graph database...")
            graph_context = await graph_service.execute_graph_pipeline(content)
            logger.info(
                f"[{request_id}] Graph context retrieved! Starting LLM stream..."
            )

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
