"""Sarvam AI Text-to-Speech provider using Bulbul v3."""

import base64
import logging

import httpx

from app.core.config import settings
from app.services.speech.tts.base import BaseTTS

logger = logging.getLogger(__name__)


class SarvamTTS(BaseTTS):
    """Sarvam AI TTS using the Bulbul v3 REST API.

    Endpoint: POST https://api.sarvam.ai/text-to-speech
    Returns base64-encoded WAV audio that must be decoded.
    Supports 11 languages (10 Indian + English) with 30+ speaker voices.
    """

    def __init__(self):
        self.api_key = settings.sarvam_api_key
        self.base_url = "https://api.sarvam.ai"
        self.model = settings.sarvam_tts_model
        self.default_speaker = settings.sarvam_tts_speaker
        self.default_language = settings.sarvam_tts_language

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Convert text to speech via Sarvam Bulbul v3.

        Args:
            text: The text to synthesize (max 2500 characters for v3).
            voice: Optional speaker name (e.g. ``"shubh"``, ``"priya"``).
                   Falls back to ``settings.sarvam_tts_speaker``.

        Returns:
            Raw audio bytes (WAV format).

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx response.
            ValueError: If the API key is not configured or text is empty.
        """
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY is not configured. "
                "Set it in .env or as an environment variable."
            )

        if not text or not text.strip():
            raise ValueError("Text to synthesize cannot be empty.")

        speaker = voice or self.default_speaker

        # Bulbul v3 has a 2500 character limit; truncate gracefully
        if len(text) > 2500:
            logger.warning(
                "Text exceeds 2500 char limit (%d chars), truncating.", len(text)
            )
            text = text[:2497] + "..."

        payload = {
            "text": text,
            "target_language_code": self.default_language,
            "model": self.model,
            "speaker": speaker,
            "speech_sample_rate": "24000",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/text-to-speech",
                headers={
                    "api-subscription-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            # Response contains {"audios": ["<base64-wav-string>"], ...}
            audios = result.get("audios", [])
            if not audios:
                raise ValueError("Sarvam TTS returned no audio data.")

            # Decode the first base64 audio chunk
            audio_bytes = base64.b64decode(audios[0])
            return audio_bytes
