import base64

from sarvamai import SarvamAI

from app.core.config import settings
from app.services.speech.tts.base import BaseTTS


class SarvamTTS(BaseTTS):
    def __init__(self, model: str = "bulbul:v3"):
        self.client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        self.model = model

    async def synthesize(self, text: str, lang: str = "en-IN", voice: str = "shubh") -> bytes:
        response = self.client.text_to_speech.convert(
            model=self.model, speaker=voice, text=text, target_language_code=lang
        )
        response.audios[0]
        return base64.b64decode(response.audios[0])
