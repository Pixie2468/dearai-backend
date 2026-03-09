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
# Standalone helpers used by the two-layer WebSocket handler
# ---------------------------------------------------------------------------

# Layer 1 system prompt – fast empathetic acknowledgement
_LAYER1_SYSTEM = (
    "You are a compassionate mental health companion. "
    "Give a very brief, empathetic acknowledgement (1-2 sentences) "
    "to the following message. Be warm and supportive. "
    "Do not give advice yet, just acknowledge the user's feelings."
)

# Layer 2 system prompt – continuation with graph context
_LAYER2_SYSTEM = (
    "You are a compassionate mental health companion. Your role is to:\n"
    "- Listen actively and empathetically\n"
    "- Provide emotional support without judgment\n"
    "- Help users explore their feelings\n"
    "- Encourage healthy coping strategies\n"
    "- Suggest professional help when appropriate\n\n"
    "Important guidelines:\n"
    "- Never provide medical diagnoses or prescribe medication\n"
    "- Always take crisis situations seriously\n"
    "- Use warm, understanding language\n"
    "- You are a supportive companion, not a replacement for "
    "professional mental health care.\n\n"
    "IMPORTANT: You are continuing a response that already started with "
    "an empathetic acknowledgement (provided below). Do NOT repeat that "
    "acknowledgement. Instead, smoothly CONTINUE from where it left off "
    "and add depth, context-aware guidance, or coping strategies."
)


async def generate_immediate_response(
    user_message: str,
    summary: str = "",
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Layer 1: Generate a fast, empathetic acknowledgement via gemini-2.0-flash-lite.

    Args:
        user_message: The current user message.
        summary: Previous conversation summary for context continuity.
        conversation_history: Recent messages as ``[{"role": ..., "content": ...}]``.

    Returns:
        A short (1-2 sentence) empathetic acknowledgement.
    """
    _ensure_init()

    model = GenerativeModel(
        "gemini-2.0-flash-lite",
        system_instruction=_LAYER1_SYSTEM,
    )

    # Build content with optional summary context
    parts: list[str] = []
    if summary:
        parts.append(f"Previous conversation summary:\n{summary}\n")
    if conversation_history:
        recent = conversation_history[-4:]  # last 4 messages for speed
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )
        parts.append(f"Recent conversation:\n{history_text}\n")
    parts.append(f"User: {user_message}")

    prompt = "\n\n".join(parts)

    response = await model.generate_content_async(
        prompt,
        generation_config={"max_output_tokens": 100, "temperature": 0.7},
    )
    return response.text


async def generate_rag_continuation(
    user_message: str,
    layer1_response: str,
    graph_context: list[str],
    summary: str = "",
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Layer 2: Generate a context-rich continuation from the Layer 1 acknowledgement.

    This function is called after both Layer 1 (fast acknowledgement) and
    the Graph RAG pipeline have completed. It produces a response that
    *continues seamlessly* from the Layer 1 text.

    Args:
        user_message: The current user message.
        layer1_response: The text already sent to the user by Layer 1.
        graph_context: List of context strings retrieved from FalkorDB.
        summary: Previous conversation summary.
        conversation_history: Recent messages as ``[{"role": ..., "content": ...}]``.

    Returns:
        The continuation text (to be appended after Layer 1).
    """
    _ensure_init()

    system_prompt = _LAYER2_SYSTEM

    if graph_context:
        context_block = "\n".join(f"- {c}" for c in graph_context)
        system_prompt += (
            f"\n\nRelevant knowledge graph context:\n{context_block}\n\n"
            "Use this context naturally. If coping strategies are relevant, "
            "suggest 1-2 gently. Do not list them mechanically."
        )

    model = GenerativeModel(
        settings.llm_model,
        system_instruction=system_prompt,
    )

    # Build the conversation contents
    contents: list[Content] = []

    # Inject summary context as a leading model message
    if summary:
        contents.append(
            Content(
                role="user",
                parts=[Part.from_text(f"[Context] {summary}")],
            )
        )
        contents.append(
            Content(
                role="model",
                parts=[Part.from_text("Understood, I'll keep this context in mind.")],
            )
        )

    # Add recent conversation history
    if conversation_history:
        for msg in conversation_history[-10:]:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                Content(role=role, parts=[Part.from_text(msg["content"])])
            )

    # Current user message
    contents.append(
        Content(role="user", parts=[Part.from_text(user_message)])
    )

    # Layer 1 response already delivered -- model should continue from here
    contents.append(
        Content(
            role="model",
            parts=[Part.from_text(layer1_response)],
        )
    )

    # Ask the model to continue
    contents.append(
        Content(
            role="user",
            parts=[
                Part.from_text(
                    "[System] Continue your response naturally. Add depth, "
                    "context-aware guidance, and relevant coping strategies "
                    "if appropriate. Do not repeat what you already said."
                )
            ],
        )
    )

    response = await model.generate_content_async(
        contents,
        generation_config={"max_output_tokens": 1024, "temperature": 0.7},
    )
    return response.text
