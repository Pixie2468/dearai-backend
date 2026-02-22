"""Text chat handler - processes text messages."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.chat.schemas import TextChatResponse
from app.services.context import summary_context_provider
from app.services.conversations.service import add_message, get_conversation_messages
from app.services.guardrails import input_guardrails, output_guardrails
from app.services.llm import LLMMessage, get_llm

SYSTEM_PROMPT = """You are a compassionate mental health companion. Your role is to:
- Listen actively and empathetically
- Provide emotional support without judgment
- Help users explore their feelings
- Encourage healthy coping strategies
- Suggest professional help when appropriate

Important guidelines:
- Never provide medical diagnoses or prescribe medication
- Always take crisis situations seriously
- Respect user boundaries and confidentiality
- Use warm, understanding language

Remember: You are a supportive companion, not a replacement for professional mental health care."""


async def handle_text_chat(
    db: AsyncSession,
    user_id: UUID,
    conversation_id: UUID,
    content: str,
) -> TextChatResponse:
    """
    Process a text chat message.

    Flow:
    1. Validate input with guardrails
    2. Check for crisis keywords
    3. Get conversation context
    4. Get conversation history
    5. Send to LLM
    6. Validate output with guardrails
    7. Save messages to database
    8. Return response
    """
    # 1. Validate input
    is_valid, error = await input_guardrails.validate(content)
    if not is_valid:
        raise ValueError(f"Input validation failed: {error}")

    # 2. Check for crisis
    is_crisis = await input_guardrails.check_crisis_keywords(content)

    # 3. Get context
    context = await summary_context_provider.get_context(db, conversation_id, user_id)

    # 4. Get conversation history
    messages = await get_conversation_messages(db, conversation_id, user_id, limit=20)
    llm_messages = [LLMMessage(role=msg.role, content=msg.content) for msg in messages]
    llm_messages.append(LLMMessage(role="user", content=content))

    # Build system prompt with context
    system_prompt = SYSTEM_PROMPT
    if context:
        system_prompt += f"\n\nContext:\n{context}"

    # 5. Send to LLM
    llm = get_llm()
    response_content = await llm.chat(llm_messages, system_prompt=system_prompt)

    # 6. Validate output
    is_valid, error = await output_guardrails.validate(response_content)
    if not is_valid:
        response_content = "I apologize, but I need to rephrase my response. Could you please share more about how you're feeling?"

    # Add crisis resources if needed
    response_content = await output_guardrails.add_crisis_resources(
        response_content, is_crisis
    )

    # 7. Save messages
    await add_message(db, conversation_id, user_id, "user", content, "text")
    assistant_msg = await add_message(
        db, conversation_id, user_id, "assistant", response_content, "text"
    )

    # 8. Return response
    return TextChatResponse(
        message_id=assistant_msg.id,
        content=response_content,
        is_crisis=is_crisis,
    )
