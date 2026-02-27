"""
Async database engine, session factory, and base model.

Replaces the old app/repositories/database.py with a canonical location
in app/core/.  Uses SQLAlchemy 2.0 async ORM with the asyncpg driver.

Key components:
  - engine:             async engine bound to DATABASE_URL (postgresql+asyncpg)
  - async_session_maker: async session factory (expire_on_commit=False)
  - Base:               declarative base for all ORM models
  - get_db:             FastAPI dependency that yields an AsyncSession
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# ---------------------------------------------------------------------------
# Async engine – uses asyncpg driver for PostgreSQL
# echo=True in debug mode for SQL logging
# ---------------------------------------------------------------------------
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,  # verify connections before checkout (Cloud SQL friendly)
)

# ---------------------------------------------------------------------------
# Session factory – expire_on_commit=False so that attributes remain
# accessible after commit without an additional refresh
# ---------------------------------------------------------------------------
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Declarative base for all ORM models
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency – yields an AsyncSession with auto-commit / rollback
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            print(e)
            await session.rollback()
            raise
