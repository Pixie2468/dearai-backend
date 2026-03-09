from abc import ABC, abstractmethod


class BaseSTT(ABC):
    """Abstract base class for Speech-to-Text providers."""

    @abstractmethod
    async def transcribe(
        self, audio_data: bytes, language: str = "en"
    ) -> tuple[str, str]:
        """Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes.
            language: BCP-47 language hint (e.g. ``en-IN``). Use ``"unknown"``
                for automatic detection.

        Returns:
            A ``(transcript, detected_language)`` tuple. ``detected_language``
            is a BCP-47 code (e.g. ``"en-IN"``) or ``""`` when detection is
            unavailable.
        """
        pass
