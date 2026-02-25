"""Vertex AI / Gemini LLM provider."""

import asyncio

# from collections.abc import AsyncGenerator
from google import genai
from google.genai import types

from app.core.config import settings
from app.services.llm.base import BaseLLM, LLMMessage


class VertexLLM(BaseLLM):
    """Vertex AI Gemini LLM provider."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(
            vertexai=True,
            project=settings.vertex_project,
            location=settings.vertex_location,
        )
        self.model = model

    def _build_contents(self, messages: list[LLMMessage]) -> list[types.Content]:
        contents = []
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
            contents=self._build_contents(messages),
            config=config,
        )
        return response.text or ""

    async def chat_stream(self, messages: list[LLMMessage], system_prompt: str | None = None):
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or "",
        )
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=self._build_contents(messages),
            config=config,
        ):
            if chunk.text:
                yield chunk.text


async def main():
    llm = VertexLLM()
    messages = [LLMMessage(role="user", content="Hello, how are you?")]
    response = await llm.chat(messages)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
