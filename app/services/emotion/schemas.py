from pydantic import BaseModel


class EmotionScore(BaseModel):
    emotion: str
    score: float


class EmotionResult(BaseModel):
    emotions: list[EmotionScore]
    dominant_emotion: str
    confidence: float
