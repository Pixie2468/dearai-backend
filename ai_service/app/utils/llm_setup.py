"""Configure LiteLLM + embedder for GraphRAG usage."""

import os

from graphrag_sdk import LiteLLM, LiteLLMEmbedder

from app.utils.setup_client import check_vertex, get_client


def setup_llm() -> tuple[LiteLLM, LiteLLMEmbedder]:
    """Create an LLM and embedder based on Vertex or API key config."""
    _, model = get_client()
    if check_vertex():
        llm = LiteLLM(
            model=f"vertex_ai/{model}",
            vertex_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION"),
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

    embedder = LiteLLMEmbedder(model=embed_model)

    return llm, embedder
