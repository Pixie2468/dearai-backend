"""
SQLAlchemy 2.0 ORM models – Conversation, Message, Summary.

Moved from app/services/conversations/models.py into the centralized
app/models/ package.  Includes pgvector-ready embedding column on Message
for future semantic search / RAG use-cases.

NOTE: The pgvector extension must be enabled manually in PostgreSQL:
    CREATE EXTENSION IF NOT EXISTS vector;
This is NOT auto-run in application code – create it via a migration or
manually on the database before using embedding columns.
"""

import uuid
from datetime import datetime
from enum import Enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ConversationType(str, Enum):
    friend = "friend"
    therapy = "therapy"


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class MessageType(str, Enum):
    text = "text"
    voice = "voice"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    type: Mapped[ConversationType] = mapped_column(
        SQLEnum(ConversationType), nullable=False, default=ConversationType.friend
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    summaries: Mapped[list["Summary"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conv_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(SQLEnum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[MessageType] = mapped_column(
        SQLEnum(MessageType), nullable=False, default=MessageType.text
    )
    audio_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    msg_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # -----------------------------------------------------------------------
    # pgvector embedding column (1536 dimensions – OpenAI text-embedding-3-small)
    # Requires: CREATE EXTENSION IF NOT EXISTS vector; in the database.
    # Nullable so existing rows are not affected until embeddings are computed.
    # -----------------------------------------------------------------------
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conv_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    last_message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="summaries")


# Forward-reference import resolved via app/models/__init__.py
from app.models.user import User  # noqa: E402, F401
