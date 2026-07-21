"""Initialize a Google GenAI client with Vertex AI fallback logic."""

import os
import threading

from google import genai

_lock = threading.Lock()
_cached_client: genai.Client | None = None
_cached_model: str | None = None


def check_vertex() -> bool:
    """Return True when all Vertex AI env vars are present."""
    for key in ("VERTEX_MODEL_ID", "VERTEX_PROJECT", "VERTEX_LOCATION"):
        if key not in os.environ:
            return False

    return True


def get_client() -> tuple[genai.Client, str]:
    """Return a cached GenAI client and model name based on environment settings."""
    global _cached_client, _cached_model

    if _cached_client is not None and _cached_model is not None:
        return _cached_client, _cached_model

    with _lock:
        # Double-check after acquiring lock
        if _cached_client is not None and _cached_model is not None:
            return _cached_client, _cached_model

        if check_vertex():
            project = os.getenv("VERTEX_PROJECT", "dear-ai-485310")
            location = os.getenv("VERTEX_LOCATION", "us-central1")
            client = genai.Client(
                vertexai=True, project=project, location=location
            )
            model_id = os.getenv("VERTEX_MODEL_ID", "")

            # Numeric IDs are fine-tuned model endpoints, not publisher models.
            # Construct the full endpoint resource path so the SDK doesn't
            # mistakenly look under publishers/google/models/.
            if model_id.isdigit():
                model = f"projects/{project}/locations/{location}/endpoints/{model_id}"
            else:
                model = model_id
        else:
            api_key = os.getenv("GEMINI_API_KEY", "")
            client = genai.Client(api_key=api_key)
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

        _cached_client = client
        _cached_model = str(model)

    return _cached_client, _cached_model
