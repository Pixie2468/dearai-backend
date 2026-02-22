"""Hume AI Speech-to-Text provider."""

import base64

import httpx

from app.core.config import settings
from app.services.speech.stt.base import BaseSTT


class HumeSTT(BaseSTT):
    """Hume AI Speech-to-Text using the batch transcription API."""

    def __init__(self):
        self.api_key = settings.hume_api_key
        self.base_url = "https://api.hume.ai/v0"

    async def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """Transcribe audio bytes using Hume AI.

        Args:
            audio_data: Raw audio bytes (WAV, WebM, MP3, etc.).
            language: BCP-47 language code (currently unused by Hume;
                    included for interface compatibility).

        Returns:
            Transcribed text string.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/batch/jobs",
                headers={
                    "X-Hume-Api-Key": self.api_key,
                },
                files={"file": ("audio.webm", audio_data, "audio/webm")},
                data={"models": '{"language": {}}'},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            # Extract transcription text from the Hume batch response
            predictions = (
                result.get("results", {})
                .get("predictions", [{}])[0]
                .get("models", {})
                .get("language", {})
                .get("grouped_predictions", [{}])[0]
                .get("predictions", [])
            )

            text_parts = [p.get("text", "") for p in predictions]
            return " ".join(text_parts).strip()
