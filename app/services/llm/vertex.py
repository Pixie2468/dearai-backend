"""Vertex AI / Gemini LLM provider."""

import os
from typing import AsyncIterator

import vertexai
from vertexai.generative_models import Content, GenerativeModel, Part

from app.core.config import settings
from app.services.llm.base import BaseLLM, LLMMessage

_initialized = False


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        vertexai.init(
            project=settings.vertex_project or os.environ.get("VERTEX_PROJECT"),
            location=settings.vertex_location,
        )
        _initialized = True


def _messages_to_contents(messages: list[LLMMessage]) -> list[Content]:
    """Convert LLMMessage list to Vertex AI Content objects.

    Gemini uses "user" / "model" roles (not "assistant").
    """
    contents: list[Content] = []
    for msg in messages:
        role = "model" if msg.role == "assistant" else "user"
        contents.append(Content(role=role, parts=[Part.from_text(msg.content)]))
    return contents


class VertexLLM(BaseLLM):
    """Vertex AI Gemini LLM provider."""

    def __init__(self, model: str | None = None):
        _ensure_init()
        self.model_name = model or settings.llm_model
        self._model: GenerativeModel | None = None

    def _get_model(self, system_prompt: str | None = None) -> GenerativeModel:
        """Create a GenerativeModel, optionally with a system instruction."""
        kwargs: dict = {}
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        return GenerativeModel(self.model_name, **kwargs)

    async def chat(
        self, messages: list[LLMMessage], system_prompt: str | None = None
    ) -> str:
        model = self._get_model(system_prompt)
        contents = _messages_to_contents(messages)

        response = await model.generate_content_async(
            contents,
            generation_config={"max_output_tokens": 2048, "temperature": 0.7},
        )
        return response.text

    async def chat_stream(
        self, messages: list[LLMMessage], system_prompt: str | None = None
    ) -> AsyncIterator[str]:
        model = self._get_model(system_prompt)
        contents = _messages_to_contents(messages)

        response = await model.generate_content_async(
            contents,
            generation_config={"max_output_tokens": 2048, "temperature": 0.7},
            stream=True,
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text


# ---------------------------------------------------------------------------
# Standalone fast-acknowledgement helper (used by WebSocket layer)
# ---------------------------------------------------------------------------


async def generate_immediate_response(user_message: str) -> str:
    """Generate a fast, empathetic acknowledgement using gemini-2.0-flash-lite."""
    _ensure_init()

    model = GenerativeModel("gemini-2.0-flash-lite")

    prompt = (
        "You are a compassionate mental health companion. "
        "Give a very brief, empathetic acknowledgement (1-2 sentences) "
        "to the following message. Be warm and supportive. "
        "Do not give advice yet, just acknowledge.\n\n"
        f"User: {user_message}"
    )

    response = await model.generate_content_async(
        prompt,
        generation_config={"max_output_tokens": 80, "temperature": 0.7},
    )
    return response.text
