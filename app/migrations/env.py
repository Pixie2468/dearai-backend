"""
Alembic environment configuration – async-compatible.

This env.py uses SQLAlchemy's async engine so that migrations can target
the same postgresql+asyncpg URL used at runtime.  It imports all ORM
models via ``app.models`` so that autogenerate can detect schema changes.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import settings

# ---------------------------------------------------------------------------
# Import ALL models so that Base.metadata is fully populated for autogenerate
# ---------------------------------------------------------------------------
from app.models import *  # noqa: F401, F403
from app.core.database import Base

# Alembic Config object – provides access to alembic.ini values
config = context.config

# Set the SQLAlchemy URL from the application settings
config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MetaData object for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL scripts without connecting to the database.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Helper that configures context and runs migrations synchronously."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode using an async engine.

    Creates an async engine from the alembic config, then runs migrations
    inside a synchronous callback via ``run_sync``.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations – delegates to async runner."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
