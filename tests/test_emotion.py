"""Tests for the Hume emotion detector."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.emotion.hume import HumeEmotionDetector


class TestHumeEmotionDetector:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.hume_api_key", "")
        self.detector = HumeEmotionDetector()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_api_key(self):
        result = await self.detector.detect_from_audio(b"audio")
        assert result is None

    def test_parse_response_empty_predictions(self):
        result = self.detector._parse_response({"predictions": []})
        assert result is None

    def test_parse_response_valid(self):
        data = {
            "predictions": [
                {"name": "joy", "score": 0.9},
                {"name": "sadness", "score": 0.1},
            ]
        }
        result = self.detector._parse_response(data)
        assert result is not None
        assert result.dominant_emotion == "joy"
        assert result.confidence == 0.9
        assert len(result.emotions) == 2

    def test_parse_response_invalid_data(self):
        result = self.detector._parse_response({})
        assert result is None

    @pytest.mark.asyncio
    async def test_detect_with_api_key_handles_error(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.hume_api_key", "test-key")
        detector = HumeEmotionDetector()
        # Will fail to reach the API, should return None
        result = await detector.detect_from_audio(b"audio")
        assert result is None
