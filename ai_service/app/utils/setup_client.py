"""Initialize a Google GenAI client with Vertex AI fallback logic."""

import os

from google import genai


def check_vertex() -> bool:
    """Return True when all Vertex AI env vars are present."""
    for key in ("VERTEX_MODEL_ID", "VERTEX_PROJECT", "VERTEX_LOCATION"):
        if key not in os.environ:
            return False

    return True


def get_client() -> tuple[genai.Client, str]:
    """Return a GenAI client and model name based on environment settings."""
    vertex = check_vertex()
    if vertex:
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

    return client, str(model)
