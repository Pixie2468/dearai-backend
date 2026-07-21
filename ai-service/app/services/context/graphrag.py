"""GraphRAG orchestration for Dear AI context retrieval.

Maintains a per-user cache of initialized GraphRAG instances to avoid
re-creating FalkorDB connections on every message.  Ingestion runs as a
fire-and-forget background task so it never blocks retrieval or streaming.
"""

import asyncio
import logging
import os
import time
import uuid

from graphrag_sdk import ConnectionConfig, GraphRAG

from app.schemas.graph_schema import create_graph_schema
from app.services.graph.generation import process_user_query
from app.services.graph.retrieval import get_graph_context
from app.utils.llm_setup import setup_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-user GraphRAG cache
# ---------------------------------------------------------------------------

_graph_cache: dict[str, GraphRAG] = {}
_graph_timestamps: dict[str, float] = {}  # last-access time per user
_user_locks: dict[str, asyncio.Lock] = {}
_global_lock = asyncio.Lock()

# Evict idle entries after 30 minutes
_TTL_SECONDS = 30 * 60


async def _get_user_lock(user_id: str) -> asyncio.Lock:
    """Return (or create) an asyncio.Lock for the given user."""
    async with _global_lock:
        if user_id not in _user_locks:
            _user_locks[user_id] = asyncio.Lock()
        return _user_locks[user_id]


async def get_graph_service(user_id: str) -> GraphRAG:
    """Return a cached, ready-to-use GraphRAG instance for *user_id*.

    On the first call for a user, the instance is created, entered, and
    cached.  Subsequent calls return the same instance without any
    connection overhead.
    """
    lock = await _get_user_lock(user_id)

    async with lock:
        if user_id in _graph_cache:
            _graph_timestamps[user_id] = time.monotonic()
            return _graph_cache[user_id]

        llm, embedder = setup_llm()

        connection = ConnectionConfig(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", 6379)),
            graph_name=f"graph_{user_id}",
            password=os.getenv("FALKORDB_PASSWORD", ""),
        )

        rag = GraphRAG(
            connection=connection,
            llm=llm,
            embedder=embedder,
            embedding_dimension=768,
            schema=create_graph_schema(),
        )

        await rag.__aenter__()

        _graph_cache[user_id] = rag
        _graph_timestamps[user_id] = time.monotonic()

        logger.info("Initialised GraphRAG for user %s", user_id)
        return rag


async def evict_idle_graphs() -> None:
    """Close and remove GraphRAG instances that have been idle for > TTL."""
    now = time.monotonic()
    stale_users = [
        uid
        for uid, ts in _graph_timestamps.items()
        if now - ts > _TTL_SECONDS
    ]

    for uid in stale_users:
        rag = _graph_cache.pop(uid, None)
        _graph_timestamps.pop(uid, None)
        if rag:
            try:
                await rag.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Error closing GraphRAG for user %s: %s", uid, exc)
            else:
                logger.info("Evicted idle GraphRAG for user %s", uid)


# ---------------------------------------------------------------------------
# Public pipeline helpers
# ---------------------------------------------------------------------------


async def retrieve_context(user_id: str, user_query: str) -> str:
    """Retrieve graph context for *user_query* using a cached GraphRAG.

    This is the fast path — only reads from the graph, does not write.
    """
    rag = await get_graph_service(user_id)
    return await get_graph_context(rag, user_query)


def schedule_ingestion(user_id: str, user_query: str) -> asyncio.Task:
    """Fire-and-forget: ingest *user_query* into the user's graph.

    Returns the background task so callers can optionally await it or
    attach error handlers.
    """
    return asyncio.create_task(_ingest_background(user_id, user_query))


async def _ingest_background(user_id: str, user_query: str) -> None:
    """Background coroutine that ingests a query into the graph."""
    try:
        rag = await get_graph_service(user_id)
        interaction_id = f"interaction_{uuid.uuid4().hex[:8]}"
        await process_user_query(rag, user_query, interaction_id)
        logger.debug("Background ingestion complete for user %s", user_id)
    except Exception as exc:
        logger.error("Background ingestion failed for user %s: %s", user_id, exc)
