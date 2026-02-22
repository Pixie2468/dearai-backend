from app.core.config import settings
from app.services.speech.tts.base import BaseTTS


def get_tts(provider: str | None = None) -> BaseTTS:
    """Factory function to get TTS provider instance."""
    provider = provider or settings.tts_provider

    if provider == "hume":
        from app.services.speech.tts.hume import HumeTTS

        return HumeTTS()
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")
