"""Sarvam AI Speech-to-Text provider using Saaras v3."""

import logging

import httpx

from app.core.config import settings
from app.services.speech.stt.base import BaseSTT

logger = logging.getLogger(__name__)


class SarvamSTT(BaseSTT):
    """Sarvam AI Speech-to-Text using the Saaras v3 REST API.

    Endpoint: POST https://api.sarvam.ai/speech-to-text
    Supports 23 languages (22 Indian + English) with auto-detection.
    """

    def __init__(self):
        self.api_key = settings.sarvam_api_key
        self.base_url = "https://api.sarvam.ai"
        self.model = settings.sarvam_stt_model
        self.mode = settings.sarvam_stt_mode

    async def transcribe(
        self, audio_data: bytes, language: str = "unknown"
    ) -> tuple[str, str]:
        """Transcribe audio bytes using Sarvam Saaras v3.

        Args:
            audio_data: Raw audio bytes (WAV, MP3, WebM, OGG, FLAC, etc.).
            language: BCP-47 language code (e.g. ``en-IN``, ``hi-IN``).
                      Use ``"unknown"`` for automatic language detection.

        Returns:
            A ``(transcript, detected_language)`` tuple.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx response.
            ValueError: If the API key is not configured.
        """
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY is not configured. "
                "Set it in .env or as an environment variable."
            )

        # Map short codes to BCP-47 if needed
        lang_code = self._normalize_language(language)

        async with httpx.AsyncClient() as client:
            # Build multipart form data
            files = {"file": ("audio.wav", audio_data, "audio/wav")}
            data: dict[str, str] = {
                "model": self.model,
                "mode": self.mode,
            }
            if lang_code:
                data["language_code"] = lang_code

            response = await client.post(
                f"{self.base_url}/speech-to-text",
                headers={"api-subscription-key": self.api_key},
                files=files,
                data=data,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            transcript = result.get("transcript", "")
            detected_lang = result.get("language_code")
            confidence = result.get("language_probability")

            if detected_lang:
                logger.debug(
                    "Sarvam STT detected language=%s confidence=%.2f",
                    detected_lang,
                    confidence or 0.0,
                )

            return transcript.strip(), detected_lang or ""

    @staticmethod
    def _normalize_language(language: str) -> str:
        """Convert short language codes to BCP-47 format expected by Sarvam."""
        # Already in BCP-47 format
        if "-" in language:
            return language

        # Common short-code mappings
        mapping = {
            "en": "en-IN",
            "hi": "hi-IN",
            "bn": "bn-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "mr": "mr-IN",
            "gu": "gu-IN",
            "kn": "kn-IN",
            "ml": "ml-IN",
            "pa": "pa-IN",
            "od": "od-IN",
            "ur": "ur-IN",
            "as": "as-IN",
            "ne": "ne-IN",
            "unknown": "unknown",
        }
        return mapping.get(language.lower(), "unknown")
