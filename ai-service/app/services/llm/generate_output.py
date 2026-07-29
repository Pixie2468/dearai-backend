"""LLM streaming helpers."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import List, Dict

from google.genai import types

from app.services.llm.prompt_manager import build_system_prompt
from app.utils.setup_client import get_client

logger = logging.getLogger(__name__)


async def stream_response(
    user_query: str,
    graph_context: str,
    history: List[Dict[str, str]] = None,
    emotion: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream model output for a user query with optional context, chat history, and emotion."""
    client, model = get_client()

    system_instruction = build_system_prompt(graph_context, emotion)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.6,
    )

    # Build contents from history
    contents = []
    if history:
        for msg in history:
            # Map roles: 'ai' -> 'model', 'user' -> 'user'
            role = "model" if msg.get("role") == "ai" else "user"
            contents.append(
                types.Content(
                    role=role, parts=[types.Part.from_text(text=msg.get("content", ""))]
                )
            )

    # Append the current user query
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
    )

    for attempt in range(3):
        try:
            response_stream = await client.aio.models.generate_content_stream(
                model=str(model),
                contents=contents,
                config=config,
            )

            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
            
            return  # Success, exit the retry loop

        except asyncio.CancelledError:
            logger.info("LLM generation cancelled")
            raise
        except Exception as exc:
            if "429" in str(exc) and attempt < 2:
                logger.warning("Hit 429 Quota limit on Vertex AI. Retrying in 2 seconds (attempt %d/3)...", attempt + 1)
                await asyncio.sleep(2)
                continue
            
            logger.error("LLM generation failed: %s", exc)
            yield f"I'm having a little trouble connecting my thoughts right now (Error: {str(exc)}). Could we try that again?"
            return
