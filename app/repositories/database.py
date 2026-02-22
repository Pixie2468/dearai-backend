"""
DEPRECATED – database setup has moved to app/core/database.py

This shim re-exports all symbols for backward compatibility.
All new code should import from app.core.database directly.
"""

from app.core.database import Base, async_session_maker, engine, get_db

__all__ = ["Base", "async_session_maker", "engine", "get_db"]

