"""
Centralized ORM models package.

Importing from this module guarantees that every model is registered with
Base.metadata before Alembic or engine.create_all() runs.  All forward-
reference relationships (User ↔ Conversation, User ↔ RefreshToken) are
resolved at import time.
"""

from app.models.auth import RefreshToken
from app.models.conversation import (
    Conversation,
    ConversationType,
    Message,
    MessageRole,
    MessageType,
    Summary,
)
from app.models.user import User

__all__ = [
    "User",
    "RefreshToken",
    "Conversation",
    "ConversationType",
    "Message",
    "MessageRole",
    "MessageType",
    "Summary",
]
