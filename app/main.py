from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.core.config import settings
from app.core.database import engine
from app.core import redis as redis_client
from app.api.v1.conversations import router as conversations_router
from app.api.v1.users import router as users_router

# Import models so that Base.metadata is fully populated (required for
# Alembic autogenerate and any create_all usage during development).
import app.models  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await redis_client.connect()
    yield
    # Shutdown – dispose of the async engine's connection pool cleanly.
    await redis_client.close()
    await engine.dispose()


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
