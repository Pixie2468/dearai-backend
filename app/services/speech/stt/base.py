from abc import ABC, abstractmethod


class BaseSTT(ABC):
    """Abstract base class for Speech-to-Text providers."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str) -> str:
        """Transcribe audio to text."""
        pass
