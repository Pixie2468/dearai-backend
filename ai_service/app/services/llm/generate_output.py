"""LLM streaming helpers."""

import asyncio
import logging
from typing import AsyncGenerator

from google.genai import types

from app.services.llm.prompt_manager import build_system_prompt
from app.utils.setup_client import get_client

logger = logging.getLogger(__name__)


async def stream_response(
    user_query: str, graph_context: str
) -> AsyncGenerator[str, None]:
    """Stream model output for a user query with optional context."""
    client, model = get_client()

    system_instruction = build_system_prompt(graph_context)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.6,
    )

    try:
        response_stream = await client.aio.models.generate_content_stream(
            model=str(model),
            contents=user_query,
            config=config,
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    except asyncio.CancelledError:
        logger.info("LLM generation cancelled")
        raise
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        yield "I'm having a little trouble connecting my thoughts right now. Could we try that again?"
