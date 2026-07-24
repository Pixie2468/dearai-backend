"""POST /episodic-intelligence/analyze — run the LangGraph agent and return structured results."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AnalysisRun
from app.schemas import AnalyzeRequest, AnalyzeResponse
from engine.graph import build_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/episodic-intelligence", tags=["analysis"])

# Build the graph once at module level (stateless; state lives in the checkpointer).
_graph = build_graph()


def _build_task_string(request: AnalyzeRequest) -> str:
    """Build the task string from the request fields."""
    task_parts: list[str] = [request.story_idea]
    if request.genre:
        task_parts.append(f"Genre: {request.genre}")
    if request.target_audience:
        task_parts.append(f"Target audience: {request.target_audience}")
    if request.tone:
        task_parts.append(f"Tone: {request.tone}")
    if request.episode_count_preference:
        task_parts.append(
            f"Preferred episode count: {request.episode_count_preference}"
        )
    return "\n".join(task_parts)


def _build_initial_state(request: AnalyzeRequest) -> dict:
    """Build the initial LangGraph state dict from a request."""
    return {
        "task": _build_task_string(request),
        "episode_planner": None,
        "emotional_arc": None,
        "retention_analysis": None,
        "cliffhanger_analysis": None,
        "optimization_report": None,
        # Loop controls — wire the user's max_revisions to both loops
        "story_revision_number": 1,
        "max_story_revisions": request.max_revisions,
        "pipeline_revision_number": 1,
        "max_pipeline_revisions": request.max_revisions,
    }


def _serialize_state_value(val: object) -> object:
    """Serialize a Pydantic model or pass through plain values."""
    if hasattr(val, "model_dump"):
        return val.model_dump(mode="json")  # type: ignore[union-attr]
    return val


def _build_response(
    request: AnalyzeRequest, final_state: dict
) -> tuple[AnalyzeResponse, uuid.UUID, datetime]:
    """Build an AnalyzeResponse from the final graph state."""
    run_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    return (
        AnalyzeResponse(
            run_id=run_id,
            story_idea=request.story_idea,
            revisions_completed=final_state.get("pipeline_revision_number", 1) - 1,
            episode_planner=final_state["episode_planner"],
            episode_scripts=final_state["episode_scripts"],
            emotional_arc=final_state["emotional_arc"],
            retention_analysis=final_state["retention_analysis"],
            cliffhanger_analysis=final_state["cliffhanger_analysis"],
            optimization_report=final_state["optimization_report"],
            created_at=now,
        ),
        run_id,
        now,
    )


def _persist_run(
    db: Session,
    run_id: uuid.UUID,
    request: AnalyzeRequest,
    response: AnalyzeResponse,
    now: datetime,
) -> None:
    """Persist an analysis run to the database (best-effort)."""
    run = AnalysisRun(
        id=run_id,
        story_idea=request.story_idea,
        request_payload=request.model_dump(mode="json"),
        response_payload=response.model_dump(mode="json"),
        created_at=now,
    )
    try:
        db.add(run)
        db.commit()
        logger.info("Persisted analysis run %s", run_id)
    except Exception:
        db.rollback()
        logger.exception("Failed to persist analysis run %s", run_id)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_story(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    """Run the Episodic Intelligence Engine on a story idea.

    This is a synchronous endpoint — it blocks until the full LangGraph
    pipeline (decompose -> analyse x3 in parallel -> optimise -> repeat)
    completes.  Typical wall-clock time: 1-3 minutes depending on model
    speed and the number of revision loops.
    """
    initial_state = _build_initial_state(request)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "Starting analysis for story=%r  max_revisions=%d  thread=%s",
        request.story_idea[:80],
        request.max_revisions,
        thread_id,
    )

    try:
        final_state = _graph.invoke(initial_state, config)
    except Exception:
        logger.exception("LangGraph pipeline failed")
        raise HTTPException(
            status_code=502,
            detail="The AI analysis pipeline encountered an error. Please try again.",
        )

    # Extract structured results from the final state.
    episode_planner = final_state.get("episode_planner")
    episode_scripts = final_state.get("episode_scripts")
    emotional_arc = final_state.get("emotional_arc")
    retention_analysis = final_state.get("retention_analysis")
    cliffhanger_analysis = final_state.get("cliffhanger_analysis")
    optimization_report = final_state.get("optimization_report")

    if not all(
        [
            episode_planner,
            episode_scripts,
            emotional_arc,
            retention_analysis,
            cliffhanger_analysis,
            optimization_report,
        ]
    ):
        raise HTTPException(
            status_code=502,
            detail="The analysis pipeline completed but produced incomplete results.",
        )

    response, run_id, now = _build_response(request, final_state)

    _persist_run(db, run_id, request, response, now)

    return response


# ---------------------------------------------------------------------------
# Helpers — thinking extraction
# ---------------------------------------------------------------------------


def _extract_thinking(content: str | list[dict]) -> str | None:  # type: ignore[type-arg]
    """Return thinking text from an AIMessageChunk's content, or None.

    With ``include_thoughts=True``, Gemini thinking blocks arrive in the
    ``content`` field as a *list* containing dicts of the form::

        {"type": "thinking", "thinking": "…actual reasoning text…"}

    Regular text blocks look like ``{"type": "text", "text": "…"}`` (or the
    content is a plain ``str`` when the model isn't thinking).
    """
    if isinstance(content, str):
        return None

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text = block.get("thinking", "")
                if text:
                    parts.append(text)
        return "\n".join(parts) if parts else None

    return None


# ---------------------------------------------------------------------------
# Streaming endpoint — SSE with node-level progress events
# ---------------------------------------------------------------------------


@router.post("/analyze/stream")
async def analyze_story_stream(
    request: AnalyzeRequest,
    http_request: Request,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Stream the analysis pipeline as Server-Sent Events.

    Emits the following SSE event types:

    - ``event: progress`` — fired when a graph node starts or completes.
      Payload: ``{"node": "<name>", "status": "started"|"completed"}``

    - ``event: thinking`` — fired when the LLM emits a thinking/reasoning
      chunk.  Payload: ``{"node": "<name>", "text": "<thinking content>"}``

    - ``event: node_insight`` — fired when a pipeline node produces a
      reasoning/feedback field of interest.  Payload:
      ``{"field": "<state_key>.<attr>", "label": "<friendly label>", "text": "<content>"}``
      Currently emits for: ``input_classification.reasoning``,
      ``story_validation.feedback``, ``final_validation.replan_instructions``.

    - ``event: complete`` — fired once after the entire pipeline finishes.
      Payload: the full ``AnalyzeResponse`` JSON.

    - ``event: error`` — fired if the pipeline fails.
      Payload: ``{"detail": "<error message>"}``
    """

    initial_state = _build_initial_state(request)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "Starting streaming analysis for story=%r  max_revisions=%d  thread=%s",
        request.story_idea[:80],
        request.max_revisions,
        thread_id,
    )

    async def event_stream() -> AsyncIterator[str]:
        final_state: dict | None = None
        # Track the last emitted value for each insight field so we only
        # emit when the value actually changes (handles revision loops
        # where a field gets updated across iterations).
        _last_insight_values: dict[str, str] = {}

        # Map of state field -> (attribute to extract, friendly label)
        _INSIGHT_FIELDS: list[tuple[str, str, str]] = [
            # (state_key, attribute_name, display_label)
            ("input_classification", "reasoning", "Input Classification Reasoning"),
            ("story_validation", "feedback", "Story Validation Feedback"),
            (
                "final_validation",
                "replan_instructions",
                "Final Validation Replan Instructions",
            ),
        ]

        try:
            async for mode, chunk in _graph.astream(
                initial_state, config, stream_mode=["tasks", "messages", "values"]
            ):
                # Check if the client disconnected.
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, aborting stream %s", thread_id)
                    return

                if mode == "tasks":
                    # task events have different shapes for start vs end
                    # start: has 'input' key     end: has 'result' key
                    node_name: str = chunk.get("name", "")
                    if node_name.startswith("__"):
                        continue  # skip __start__, __end__ internal nodes

                    is_start = "input" in chunk
                    status = "started" if is_start else "completed"

                    event_data = json.dumps(
                        {
                            "node": node_name,
                            "status": status,
                        }
                    )
                    yield f"event: progress\ndata: {event_data}\n\n"

                elif mode == "messages":
                    # messages mode yields (AIMessageChunk, metadata) tuples.
                    msg_chunk, metadata = chunk
                    node_name = metadata.get("langgraph_node", "")
                    if node_name.startswith("__"):
                        continue

                    # Extract thinking blocks from the message chunk content.
                    # With include_thoughts=True, thinking chunks arrive as
                    # content blocks: [{"type": "thinking", "thinking": "..."}]
                    content = msg_chunk.content
                    thinking_text = _extract_thinking(content)
                    if thinking_text:
                        event_data = json.dumps(
                            {
                                "node": node_name,
                                "text": thinking_text,
                            }
                        )
                        yield f"event: thinking\ndata: {event_data}\n\n"

                elif mode == "values":
                    # Keep the latest full state snapshot.
                    final_state = chunk

                    # Emit node_insight events for specific state fields
                    # that provide useful reasoning/feedback to the user.
                    for state_key, attr_name, label in _INSIGHT_FIELDS:
                        insight_id = f"{state_key}.{attr_name}"

                        model = chunk.get(state_key)
                        if model is None:
                            continue

                        value = (
                            getattr(model, attr_name, None)
                            if hasattr(model, attr_name)
                            else None
                        )
                        if not value:
                            continue

                        # Only emit if the value is new or changed (handles revision loops).
                        if _last_insight_values.get(insight_id) == value:
                            continue

                        _last_insight_values[insight_id] = value
                        event_data = json.dumps(
                            {
                                "field": insight_id,
                                "label": label,
                                "text": value,
                            }
                        )
                        yield f"event: node_insight\ndata: {event_data}\n\n"

        except Exception:
            logger.exception(
                "LangGraph streaming pipeline failed (thread=%s)", thread_id
            )
            error_data = json.dumps(
                {
                    "detail": "The AI analysis pipeline encountered an error. Please try again.",
                }
            )
            yield f"event: error\ndata: {error_data}\n\n"
            return

        if final_state is None:
            error_data = json.dumps({"detail": "Pipeline produced no output."})
            yield f"event: error\ndata: {error_data}\n\n"
            return

        # Validate completeness.
        required_keys = [
            "episode_planner",
            "episode_scripts",
            "emotional_arc",
            "retention_analysis",
            "cliffhanger_analysis",
            "optimization_report",
        ]
        if not all(final_state.get(k) for k in required_keys):
            error_data = json.dumps(
                {
                    "detail": "The analysis pipeline completed but produced incomplete results.",
                }
            )
            yield f"event: error\ndata: {error_data}\n\n"
            return

        # Build and emit the final response.
        response, run_id, now = _build_response(request, final_state)
        response_json = response.model_dump(mode="json")

        yield f"event: complete\ndata: {json.dumps(response_json)}\n\n"

        # Persist to DB (best-effort, after streaming the result to the client).
        _persist_run(db, run_id, request, response, now)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
