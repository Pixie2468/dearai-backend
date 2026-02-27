"""Hume AI Text-to-Speech provider using the Octave TTS API."""

import httpx

from app.core.config import settings
from app.services.speech.tts.base import BaseTTS


class HumeTTS(BaseTTS):
    """Hume AI TTS using the /v0/tts/file endpoint (returns raw audio bytes)."""

    def __init__(self):
        self.api_key = settings.hume_api_key
        self.base_url = "https://api.hume.ai/v0"
        self.default_voice = settings.hume_tts_voice

    async def synthesize(self, text, voice, lang=None) -> bytes:
        """Convert text to speech via Hume AI TTS API.

        Args:
            text: The text to synthesize.
            voice: Optional voice name from Hume's Voice Library.
                   Falls back to settings.hume_tts_voice.

        Returns:
            Raw audio bytes (MP3).
        """
        voice = voice or self.default_voice

        payload: dict = {
            "utterances": [
                {
                    "text": text,
                    "voice": {
                        "name": voice,
                        "provider": "HUME_AI",
                    },
                }
            ],
            "format": {
                "type": "mp3",
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/tts/file",
                headers={
                    "X-Hume-Api-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.content
