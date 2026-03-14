"""
Redis caching wrapper around the Graph RAG pipeline.

This module adds a transparent cache layer **around** ``run_graph_rag``
without modifying its internal logic.  On a cache hit the stored response
is returned immediately; on a miss the pipeline runs normally and the
result is persisted for future requests.

Redis being unavailable is treated as a cache miss -- the system degrades
gracefully to live execution.
"""

import hashlib
import json
import logging
from collections.abc import AsyncGenerator

from app.core.config import settings
from app.core.redis import get_client
from app.services.context.graph_rag import run_graph_rag, run_graph_rag_stream
from app.services.llm.base import LLMMessage

logger = logging.getLogger(__name__)

_KEY_PREFIX = "rag_cache:"


def _build_cache_key(user_message: str) -> str:
    """Deterministic cache key from the user message."""
    digest = hashlib.sha256(user_message.strip().encode()).hexdigest()
    return f"{_KEY_PREFIX}{digest}"


async def get_cached_rag_response(
    user_message: str,
    conversation_history: list[LLMMessage] | None = None,
) -> str:
    """Return a Graph RAG response, using Redis cache when available.

    Flow:
        1. Compute cache key from *user_message*.
        2. If Redis is reachable **and** holds a cached value -> return it.
        3. Otherwise run ``run_graph_rag`` live.
        4. Store the fresh result in Redis (best-effort, non-blocking on failure).
    """
    cache_key = _build_cache_key(user_message)
    redis = get_client()

    # --- Try cache hit ---
    if redis is not None:
        try:
            cached = await redis.get(cache_key)
            if cached is not None:
                logger.debug("RAG cache hit for key %s", cache_key)
                return json.loads(cached)
        except Exception:
            logger.warning("Redis GET failed -- falling through to live RAG", exc_info=True)

    # --- Cache miss: execute pipeline ---
    result = await run_graph_rag(user_message, conversation_history)

    # --- Best-effort store ---
    if redis is not None:
        try:
            await redis.set(
                cache_key,
                json.dumps(result),
                ex=settings.rag_cache_ttl_seconds,
            )
            logger.debug("RAG result cached with TTL=%ds", settings.rag_cache_ttl_seconds)
        except Exception:
            logger.warning("Redis SET failed -- result not cached", exc_info=True)

    return result


async def get_cached_rag_response_stream(
    user_message: str,
    conversation_history: list[LLMMessage] | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming variant: check cache first, stream on miss.

    On cache hit, yields the entire cached string as a single chunk.
    On cache miss, streams from the pipeline and caches the assembled result.
    """
    cache_key = _build_cache_key(user_message)
    redis = get_client()

    # --- Try cache hit ---
    if redis is not None:
        try:
            cached = await redis.get(cache_key)
            if cached is not None:
                logger.debug("RAG cache hit (stream) for key %s", cache_key)
                yield json.loads(cached)
                return
        except Exception:
            logger.warning("Redis GET failed -- falling through to live RAG", exc_info=True)

    # --- Cache miss: stream from pipeline ---
    collected: list[str] = []
    async for chunk in run_graph_rag_stream(user_message, conversation_history):
        collected.append(chunk)
        yield chunk

    # --- Best-effort cache of assembled result ---
    full_response = "".join(collected)
    if redis is not None and full_response:
        try:
            await redis.set(
                cache_key,
                json.dumps(full_response),
                ex=settings.rag_cache_ttl_seconds,
            )
        except Exception:
            logger.warning("Redis SET failed -- result not cached", exc_info=True)
