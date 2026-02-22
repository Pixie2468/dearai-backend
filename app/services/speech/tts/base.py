from abc import ABC, abstractmethod


class BaseTTS(ABC):
    """Abstract base class for Text-to-Speech providers."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str = "alloy") -> bytes:
        """Convert text to speech audio."""
        pass
