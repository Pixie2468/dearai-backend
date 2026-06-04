"""Initialize a Google GenAI client with Vertex AI fallback logic."""

import os

from google import genai


def check_vertex() -> bool:
    """Return True when all Vertex AI env vars are present."""
    for key in ("VERTEX_MODEL_ID", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        if key not in os.environ:
            return False

    return True


def get_client() -> tuple[genai.Client, str]:
    """Return a GenAI client and model name based on environment settings."""
    vertex = check_vertex()
    if vertex:
        client = genai.Client(vertexai=True)
        model = os.getenv("VERTEX_MODEL_ID")
    else:
        client = genai.Client()
        model = "gemini-2.5-flash"

    return client, model
