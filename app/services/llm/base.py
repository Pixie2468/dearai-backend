from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self, messages: list[LLMMessage], system_prompt: str | None = None
    ) -> str:
        """Send messages to LLM and get response."""
        pass

    @abstractmethod
    def chat_stream(
        self, messages: list[LLMMessage], system_prompt: str | None = None
    ) -> AsyncIterator[str]:
        """Stream response from LLM."""
        ...
