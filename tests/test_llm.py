"""Tests for LLM base classes and factory."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm.base import BaseLLM, LLMMessage


class TestLLMMessage:
    def test_to_dict(self):
        msg = LLMMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}

    def test_attributes(self):
        msg = LLMMessage(role="assistant", content="hi")
        assert msg.role == "assistant"
        assert msg.content == "hi"


class TestLLMFactory:
    def test_get_llm_vertex(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.llm_provider", "vertex")
        # Patch vertexai.init to avoid Google Cloud credentials requirement
        with patch("app.services.llm.vertex.vertexai"):
            with patch("app.services.llm.vertex._initialized", False):
                from app.services.llm.factory import get_llm

                llm = get_llm("vertex")
                assert isinstance(llm, BaseLLM)

    def test_get_llm_unknown_raises(self):
        from app.services.llm.factory import get_llm

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm("nonexistent")

    def test_get_llm_default_provider(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.llm_provider", "vertex")
        with patch("app.services.llm.vertex.vertexai"):
            with patch("app.services.llm.vertex._initialized", False):
                from app.services.llm.factory import get_llm

                llm = get_llm()
                assert isinstance(llm, BaseLLM)


class TestVertexLLM:
    """Test VertexLLM chat methods with mocked Vertex AI."""

    @pytest.fixture(autouse=True)
    def _patch_vertex(self):
        with patch("app.services.llm.vertex.vertexai"):
            with patch("app.services.llm.vertex._initialized", False):
                yield

    def test_messages_to_contents(self):
        from app.services.llm.vertex import _messages_to_contents

        msgs = [
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="assistant", content="hello"),
        ]
        contents = _messages_to_contents(msgs)
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[1].role == "model"

    @pytest.mark.asyncio
    async def test_chat(self):
        from app.services.llm.vertex import VertexLLM

        with patch("app.services.llm.vertex.GenerativeModel") as MockModel:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "I'm here to help."
            mock_instance.generate_content_async = AsyncMock(return_value=mock_response)
            MockModel.return_value = mock_instance

            llm = VertexLLM()
            result = await llm.chat(
                [LLMMessage(role="user", content="hello")],
                system_prompt="Be helpful.",
            )
            assert result == "I'm here to help."
            MockModel.assert_called_once_with(
                "gemini-2.0-flash", system_instruction="Be helpful."
            )


class TestSpeechFactories:
    def test_get_stt_google(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.stt_provider", "google")
        with patch("app.services.speech.stt.google.speech"):
            from app.services.speech.stt import get_stt
            from app.services.speech.stt.base import BaseSTT

            stt = get_stt("google")
            assert isinstance(stt, BaseSTT)

    def test_get_stt_unknown(self):
        from app.services.speech.stt import get_stt

        with pytest.raises(ValueError, match="Unknown STT provider"):
            get_stt("nonexistent")

    def test_get_tts_hume(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.tts_provider", "hume")
        from app.services.speech.tts import get_tts
        from app.services.speech.tts.base import BaseTTS

        tts = get_tts("hume")
        assert isinstance(tts, BaseTTS)

    def test_get_tts_unknown(self):
        from app.services.speech.tts import get_tts

        with pytest.raises(ValueError, match="Unknown TTS provider"):
            get_tts("nonexistent")
