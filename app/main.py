"""
Dear AI Backend – FastAPI application entry point.

Startup responsibilities:
  1. Configure structured logging (structlog)
  2. Connect to Redis (cache layer)
  3. Connect to FalkorDB (graph RAG), ensure schema & seed knowledge base
  4. Mount API routers

Shutdown responsibilities:
  1. Close FalkorDB connection pool
  2. Close Redis connection
  3. Dispose SQLAlchemy async engine
"""

import logging
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.users import router as users_router
from app.core import redis as redis_client
from app.core.config import settings
from app.core.database import engine

# Import models so that Base.metadata is fully populated (required for
# Alembic autogenerate and any create_all usage during development).
import app.models  # noqa: F401

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Structured logging configuration (structlog + stdlib)
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Set up structlog to wrap stdlib logging with structured JSON output.

    In development (``settings.debug``), human-readable console output is used.
    In production, JSON lines are emitted for log aggregation.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.debug:
        # Pretty console output for local development
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        # Machine-readable JSON for production / Cloud Logging
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# FalkorDB connection management
# ---------------------------------------------------------------------------

# Module-level reference to the FalkorDB connection pool so we can close it
# on shutdown.
_falkor_pool = None


async def _init_falkordb() -> None:
    """Connect to FalkorDB, ensure graph schema, and seed knowledge base.

    The async FalkorDB client uses a ``redis.asyncio.BlockingConnectionPool``
    under the hood, giving us proper connection pooling for concurrent Cypher
    queries across the Graph RAG pipeline.
    """
    global _falkor_pool

    try:
        from falkordb.asyncio import FalkorDB
        from redis.asyncio import BlockingConnectionPool

        pool = BlockingConnectionPool(
            host=settings.falkor_host,
            port=settings.falkor_port,
            max_connections=16,
            timeout=None,
            decode_responses=True,
        )
        db = FalkorDB(connection_pool=pool)
        graph = db.select_graph(settings.falkor_graph)

        # Inject the graph handle into the Graph RAG module
        from app.services.context.graph_rag import set_falkor_graph

        set_falkor_graph(graph)
        _falkor_pool = pool

        # Ensure indexes and seed the mental health knowledge base
        from app.services.context.graph_schema import (
            ensure_schema,
            seed_knowledge_base,
        )

        await ensure_schema(graph)
        await seed_knowledge_base(graph)

        logger.info(
            "falkordb_connected",
            host=settings.falkor_host,
            port=settings.falkor_port,
            graph=settings.falkor_graph,
        )
    except Exception:
        logger.warning(
            "falkordb_init_failed",
            host=settings.falkor_host,
            port=settings.falkor_port,
            exc_info=True,
        )


async def _close_falkordb() -> None:
    """Close the FalkorDB connection pool and clear the graph handle."""
    global _falkor_pool

    from app.services.context.graph_rag import set_falkor_graph

    set_falkor_graph(None)

    if _falkor_pool is not None:
        try:
            await _falkor_pool.aclose()
            logger.info("falkordb_disconnected")
        except Exception:
            logger.warning("falkordb_close_error", exc_info=True)
        _falkor_pool = None


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    _configure_logging()
    logger.info("startup_begin", app=settings.app_name, env=settings.environment)

    await redis_client.connect()
    await _init_falkordb()

    logger.info("startup_complete")
    yield
    # --- Shutdown ---
    logger.info("shutdown_begin")

    await _close_falkordb()
    await redis_client.close()
    await engine.dispose()

    logger.info("shutdown_complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description="Mental health companion chat API with voice support",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(users_router, prefix="/users", tags=["Users"])
app.include_router(
    conversations_router, prefix="/conversations", tags=["Conversations"]
)
app.include_router(chat_router, prefix="/chat", tags=["Chat"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
