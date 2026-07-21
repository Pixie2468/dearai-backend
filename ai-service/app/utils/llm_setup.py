"""Configure LiteLLM + embedder for GraphRAG usage."""

import os
import threading

from graphrag_sdk import LiteLLM, LiteLLMEmbedder

from app.utils.setup_client import check_vertex, get_client

_lock = threading.Lock()
_cached_llm: LiteLLM | None = None
_cached_embedder: LiteLLMEmbedder | None = None


def setup_llm() -> tuple[LiteLLM, LiteLLMEmbedder]:
    """Return a cached LLM and embedder based on Vertex or API key config."""
    global _cached_llm, _cached_embedder

    if _cached_llm is not None and _cached_embedder is not None:
        return _cached_llm, _cached_embedder

    with _lock:
        # Double-check after acquiring lock
        if _cached_llm is not None and _cached_embedder is not None:
            return _cached_llm, _cached_embedder

        _, model = get_client()
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "dear-ai-485310")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        # Check for both variable names to protect against Docker env omissions
        endpoint_id = os.getenv("VERTEX_ENDPOINT_ID") or os.getenv("VERTEX_MODEL_ID")

        if not endpoint_id:
            raise ValueError(
                "CRITICAL: Both VERTEX_ENDPOINT_ID and VERTEX_MODEL_ID are missing from the container environment!"
            )

        custom_api_base = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/{endpoint_id}"

        if check_vertex():
            llm = LiteLLM(
                model="vertex_ai/gemini-2.5-flash-lite",
                vertex_project=os.getenv("VERTEX_PROJECT"),
                vertex_location=os.getenv("VERTEX_LOCATION"),
                api_base=custom_api_base,
                temperature=0.0,
            )

            embed_model = f"vertex_ai/{os.getenv('EMBEDDING_MODEL', 'text-embedding-004')}"
        else:
            llm = LiteLLM(
                api_key=os.getenv("GEMINI_API_KEY"),
                model=str(model),
                temperature=0.0,
            )

            embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

        _cached_llm = llm
        _cached_embedder = LiteLLMEmbedder(model=embed_model)

    return _cached_llm, _cached_embedder
