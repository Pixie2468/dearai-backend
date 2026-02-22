"""
SQLAlchemy 2.0 ORM model – RefreshToken.

Moved from app/services/auth/models.py into the centralized app/models/
package.  Stores hashed refresh tokens for session management, enabling
token revocation and refresh-token rotation.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class RefreshToken(Base):
    """
    Stores refresh tokens for session management.

    Each refresh token is tracked in the database with:
    - jti: Unique token identifier (JWT ID claim)
    - token_hash: SHA-256 hash of the token (never store plain tokens)
    - is_revoked: Soft-delete flag for token revocation

    This enables:
    - Token revocation (logout)
    - Refresh token rotation
    - Session tracking
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    jti: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationship back to User
    user: Mapped["User"] = relationship(back_populates="refresh_tokens")


# Forward-reference import resolved via app/models/__init__.py
from app.models.user import User  # noqa: E402, F401
