from abc import ABC, abstractmethod
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


class BaseContextProvider(ABC):
    """Abstract base class for context providers."""

    @abstractmethod
    async def get_context(
        self, db: AsyncSession, conversation_id: UUID, user_id: UUID
    ) -> str:
        """
        Retrieve relevant context for the conversation.

        Returns:
            A string containing context to be included in the LLM prompt.
        """
        pass
