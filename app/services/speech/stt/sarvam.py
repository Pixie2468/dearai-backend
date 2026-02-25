import io

from sarvamai import SarvamAI

from app.core.config import settings
from app.services.speech.stt.base import BaseSTT


class SarvamSTT(BaseSTT):
    def __init__(self, model: str = "saaras:v3"):
        self.client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        self.model = model

    async def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.webm"

        response = self.client.speech_to_text.transcribe(model=self.model, file=audio_file)
        return response.transcript
