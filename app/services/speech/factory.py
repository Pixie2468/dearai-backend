from app.core.config import settings
from app.services.speech.stt.base import BaseSTT
from app.services.speech.tts.base import BaseTTS


def get_tts(provider: str | None = None) -> BaseTTS:
    """Factory function to get TTS provider instance."""
    provider = provider or settings.tts_provider

    if provider == "sarvam":
        from app.services.speech.tts.sarvam import SarvamTTS

        return SarvamTTS()
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")


def get_stt(provider: str | None = None) -> BaseSTT:
    """Factory function to get STT provider instance."""
    provider = provider or settings.stt_provider

    if provider == "sarvam":
        from app.services.speech.stt.sarvam import SarvamSTT

        return SarvamSTT()
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")
