"""
Async Redis client singleton.

Provides a single shared ``redis.asyncio.Redis`` instance that is
connected during application startup and closed on shutdown.
"""

import logging

from redis.asyncio import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton – initialised via ``connect()`` at startup.
_client: Redis | None = None


async def connect() -> None:
    """Open the async Redis connection pool."""
    global _client
    if _client is not None:
        return
    _client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
    )
    # Verify the connection is alive
    try:
        await _client.ping()
        logger.info("Redis connected (%s)", settings.redis_url)
    except Exception:
        logger.warning("Redis ping failed – caching will be unavailable", exc_info=True)
        await _client.aclose()
        _client = None


async def close() -> None:
    """Gracefully shut down the Redis connection pool."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Redis connection closed.")


def get_client() -> Redis | None:
    """Return the current Redis client (or *None* if unavailable)."""
    return _client
