"""
Async FalkorDB client singleton.

Provides a single shared FalkorDB graph handle that is
connected during application startup and closed on shutdown.
"""

import logging

from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: BlockingConnectionPool | None = None
_db: FalkorDB | None = None


async def connect() -> None:
    """Open the async FalkorDB connection pool."""
    global _pool, _db
    if _db is not None:
        return
    try:
        _pool = BlockingConnectionPool(
            host=settings.falkor_host,
            port=settings.falkor_port,
            decode_responses=True,
        )
        _db = FalkorDB(connection_pool=_pool)
        # Verify the connection is alive by listing graphs
        graph = _db.select_graph(settings.falkor_graph)
        # Simple connectivity check – run a no-op query
        await graph.query("RETURN 1")
        logger.info(
            "FalkorDB connected (%s:%s, graph=%s)",
            settings.falkor_host,
            settings.falkor_port,
            settings.falkor_graph,
        )
    except Exception:
        logger.warning(
            "FalkorDB connection failed – graph context will be unavailable",
            exc_info=True,
        )
        if _pool is not None:
            await _pool.aclose()
        _pool = None
        _db = None


async def close() -> None:
    """Gracefully shut down the FalkorDB connection pool."""
    global _pool, _db
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        _db = None
        logger.info("FalkorDB connection closed.")


def get_graph():
    """Return the FalkorDB graph handle (or None if unavailable)."""
    if _db is None:
        return None
    return _db.select_graph(settings.falkor_graph)
