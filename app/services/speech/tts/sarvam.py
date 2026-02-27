import base64

from sarvamai import SarvamAI

# from sarvamai.play import play
from app.core.config import settings
from app.services.speech.tts.base import BaseTTS


class SarvamTTS(BaseTTS):
    def __init__(self, model: str = "bulbul:v3"):
        self.client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        self.model = model

    async def synthesize(self, text, voice="shubh", lang="hi-IN") -> bytes:
        response = self.client.text_to_speech.convert(
            model=self.model, text=text, speaker=voice, target_language_code=lang
        )
        # play(response)
        response.audios[0]
        return base64.b64decode(response.audios[0])
