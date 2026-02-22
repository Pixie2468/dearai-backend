"""
Summary-based context provider.

Uses conversation summaries to provide context for the LLM.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Summary
from app.services.context.base import BaseContextProvider


class SummaryContextProvider(BaseContextProvider):
    """Provides context from conversation summaries."""

    async def get_context(
        self, db: AsyncSession, conversation_id: UUID, user_id: UUID
    ) -> str:
        """Get the most recent summary for the conversation."""
        result = await db.execute(
            select(Summary)
            .where(Summary.conv_id == conversation_id)
            .order_by(Summary.created_at.desc())
            .limit(1)
        )
        summary = result.scalar_one_or_none()

        if summary:
            return f"Previous conversation summary:\n{summary.content}"
        return ""


summary_context_provider = SummaryContextProvider()
