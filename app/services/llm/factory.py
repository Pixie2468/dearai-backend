from app.core.config import settings
from app.services.llm.base import BaseLLM


def get_llm(provider: str | None = None) -> BaseLLM:
    """Factory function to get LLM provider instance (default model)."""
    provider = provider or settings.llm_provider

    if provider == "vertex":
        from app.services.llm.vertex import VertexLLM

        # Use fine-tuned endpoint if configured, otherwise default model
        model = settings.llm_model if settings.llm_model else "gemini-2.5-flash"
        return VertexLLM(model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_layer1_llm() -> BaseLLM:
    """Return the fast model used for Layer 1 (instant acknowledgement).

    Uses ``settings.layer1_model`` (default: gemini-2.0-flash-lite).
    """
    from app.services.llm.vertex import VertexLLM

    return VertexLLM(model=settings.layer1_model)


def get_layer2_llm() -> BaseLLM:
    """Return the model used for Layer 2 (full Graph RAG response).

    Uses the fine-tuned Vertex endpoint from ``settings.llm_model``
    when available, otherwise falls back to the default model.
    """
    from app.services.llm.vertex import VertexLLM

    model = settings.llm_model if settings.llm_model else "gemini-2.5-flash"
    return VertexLLM(model=model)
