"""
DEPRECATED – models have moved to app/models/conversation.py

This shim re-exports conversation models for backward compatibility.
All new code should import from app.models directly.
"""

from app.models.conversation import (
    Conversation,
    ConversationType,
    Message,
    MessageRole,
    MessageType,
    Summary,
)

__all__ = [
    "Conversation",
    "ConversationType",
    "Message",
    "MessageRole",
    "MessageType",
    "Summary",
]

