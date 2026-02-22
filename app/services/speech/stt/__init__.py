from app.core.config import settings
from app.services.speech.stt.base import BaseSTT


def get_stt(provider: str | None = None) -> BaseSTT:
    """Factory function to get STT provider instance."""
    provider = provider or settings.stt_provider

    if provider == "hume":
        from app.services.speech.stt.hume import HumeSTT

        return HumeSTT()
    else:
        raise ValueError(f"Unknown STT provider: {provider}")
