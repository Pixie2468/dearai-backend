"""Voice chat handler - processes voice messages."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.chat.handlers.text import SYSTEM_PROMPT
from app.services.chat.schemas import VoiceChatResponse
from app.services.context import summary_context_provider
from app.services.conversations.service import add_message, get_conversation_messages
from app.services.emotion import hume_detector
from app.services.guardrails import input_guardrails, output_guardrails
from app.services.llm import LLMMessage, get_llm
from app.services.speech import get_stt, get_tts


async def handle_voice_chat(
    db: AsyncSession,
    user_id: UUID,
    conversation_id: UUID,
    audio_data: bytes,
) -> tuple[VoiceChatResponse, bytes | None]:
    """
    Process a voice chat message.

    Flow:
    1. Transcribe audio (STT)
    2. Detect emotions from audio
    3. Validate input with guardrails
    4. Check for crisis keywords
    5. Get conversation context
    6. Get conversation history
    7. Send to LLM (with emotion context)
    8. Validate output with guardrails
    9. Convert response to speech (TTS)
    10. Save messages to database
    11. Return response with audio
    """
    # 1. Transcribe audio
    stt = get_stt()
    transcript = await stt.transcribe(audio_data)

    # 2. Detect emotions
    emotion_result = await hume_detector.detect_from_audio(audio_data)

    # 3. Validate input
    is_valid, error = await input_guardrails.validate(transcript)
    if not is_valid:
        raise ValueError(f"Input validation failed: {error}")

    # 4. Check for crisis
    is_crisis = await input_guardrails.check_crisis_keywords(transcript)

    # 5. Get context
    context = await summary_context_provider.get_context(db, conversation_id, user_id)

    # 6. Get conversation history
    messages = await get_conversation_messages(db, conversation_id, user_id, limit=20)
    llm_messages = [LLMMessage(role=msg.role, content=msg.content) for msg in messages]
    llm_messages.append(LLMMessage(role="user", content=transcript))

    # Build system prompt with context and emotion
    system_prompt = SYSTEM_PROMPT
    if context:
        system_prompt += f"\n\nContext:\n{context}"
    if emotion_result:
        system_prompt += f"\n\nUser's detected emotional state: {emotion_result.dominant_emotion} (confidence: {emotion_result.confidence:.2f})"

    # 7. Send to LLM
    llm = get_llm()
    response_content = await llm.chat(llm_messages, system_prompt=system_prompt)

    # 8. Validate output
    is_valid, error = await output_guardrails.validate(response_content)
    if not is_valid:
        response_content = "I apologize, but I need to rephrase my response. Could you please share more about how you're feeling?"

    # Add crisis resources if needed
    response_content = await output_guardrails.add_crisis_resources(
        response_content, is_crisis
    )

    # 9. Convert to speech
    tts = get_tts()
    audio_response = await tts.synthesize(response_content)

    # 10. Save messages
    metadata = None
    if emotion_result:
        metadata = {"emotion": emotion_result.model_dump()}

    await add_message(
        db, conversation_id, user_id, "user", transcript, "voice", metadata=metadata
    )
    assistant_msg = await add_message(
        db, conversation_id, user_id, "assistant", response_content, "voice"
    )

    # 11. Return response
    return (
        VoiceChatResponse(
            message_id=assistant_msg.id,
            content=response_content,
            emotion=emotion_result,
            is_crisis=is_crisis,
        ),
        audio_response,
    )
