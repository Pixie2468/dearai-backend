"""Google Cloud Speech-to-Text service using ADC."""

import asyncio
import base64
import logging

import os
import google.auth
import google.auth.transport.requests
import httpx

logger = logging.getLogger(__name__)

GCP_STT_URL = "https://speech.googleapis.com/v1/speech:recognize"

_credentials = None
_project_id = None


def _get_auth_data() -> tuple[str, str | None]:
    """Get a valid access token and project ID using Application Default Credentials."""
    global _credentials, _project_id
    if _credentials is None:
        _credentials, _project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    if not _credentials.valid:
        _credentials.refresh(google.auth.transport.requests.Request())
        
    project_to_use = os.environ.get("VERTEX_PROJECT") or _project_id
    return _credentials.token, project_to_use


async def transcribe_audio(
    audio_bytes: bytes,
    *,
    encoding: str = "WEBM_OPUS",
    sample_rate_hertz: int = 48000,
    language_code: str = "en-US",
) -> str:
    """Transcribe audio bytes using Google Cloud STT.

    Returns the transcribed text, or an empty string if nothing was recognised.
    """
    access_token, project_id = await asyncio.to_thread(_get_auth_data)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    if project_id:
        headers["x-goog-user-project"] = project_id

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    body = {
        "config": {
            "encoding": encoding,
            "sampleRateHertz": sample_rate_hertz,
            "languageCode": language_code,
            "model": "latest_long",
            "enableAutomaticPunctuation": True,
        },
        "audio": {
            "content": audio_b64,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(GCP_STT_URL, headers=headers, json=body)

        if response.status_code != 200:
            logger.error(
                "Google STT failed (status=%s): %s",
                response.status_code,
                response.text,
            )
            raise RuntimeError(f"Google STT returned {response.status_code}")

        data = response.json()
        results = data.get("results", [])
        if not results:
            return ""

        transcript = " ".join(
            result["alternatives"][0]["transcript"]
            for result in results
            if result.get("alternatives")
        )
        return transcript.strip()
