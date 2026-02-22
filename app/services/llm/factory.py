from app.core.config import settings
from app.services.llm.base import BaseLLM


def get_llm(provider: str | None = None) -> BaseLLM:
    """Factory function to get LLM provider instance."""
    provider = provider or settings.llm_provider

    if provider == "vertex":
        from app.services.llm.vertex import VertexLLM

        return VertexLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
