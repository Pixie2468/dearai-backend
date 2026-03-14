"""Vertex AI / Gemini LLM provider."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from google import genai
from google.genai import types

from app.core.config import settings
from app.services.llm.base import BaseLLM, LLMMessage

logger = logging.getLogger(__name__)

# Module-level shared client – avoids recreating the gRPC channel on every call.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=settings.vertex_project,
            location=settings.vertex_location,
        )
    return _client


class VertexLLM(BaseLLM):
    """Vertex AI Gemini LLM provider."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model

    @property
    def client(self) -> genai.Client:
        return _get_client()

    def _build_contents(self, messages: list[LLMMessage]) -> list[types.Content]:
        contents: list[types.Content] = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
        return contents

    async def chat(self, messages: list[LLMMessage], system_prompt: str | None = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or "",
        )
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=self._build_contents(messages),  # type: ignore[arg-type]
            config=config,
        )
        return response.text or ""

    async def chat_stream(
        self, messages: list[LLMMessage], system_prompt: str | None = None
    ) -> AsyncGenerator[str, Any]:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or "",
        )
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=self._build_contents(messages),  # type: ignore[arg-type]
            config=config,
        ):
            if chunk.text:
                yield chunk.text
