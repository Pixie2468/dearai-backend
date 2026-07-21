"""Google Cloud Text-to-Speech service using ADC."""

import asyncio
import base64
import logging

import os
import google.auth
import google.auth.transport.requests
import httpx

logger = logging.getLogger(__name__)

GCP_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
DEFAULT_VOICE = "en-US-Journey-F"
DEFAULT_LANGUAGE = "en-US"

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


async def synthesize_speech(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    language_code: str = DEFAULT_LANGUAGE,
) -> bytes:
    """Synthesize speech from text using Google Cloud TTS.

    Returns the audio content as MP3 bytes.
    """
    access_token, project_id = await asyncio.to_thread(_get_auth_data)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    if project_id:
        headers["x-goog-user-project"] = project_id

    body = {
        "input": {"text": text},
        "voice": {
            "languageCode": language_code,
            "name": voice,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 1.0,
            "pitch": 0.0,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GCP_TTS_URL, headers=headers, json=body)

        if response.status_code != 200:
            logger.error(
                "Google TTS failed (status=%s): %s",
                response.status_code,
                response.text,
            )
            raise RuntimeError(f"Google TTS returned {response.status_code}")

        data = response.json()
        return base64.b64decode(data["audioContent"])
