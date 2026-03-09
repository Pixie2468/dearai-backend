"""
Automatic conversation summary generation.

Generates condensed summaries of conversations every N messages
(controlled by ``settings.summary_interval``). Summaries are used by:

- **Layer 1** (WebSocket): Provides conversation continuity context to
  ``gemini-2.0-flash-lite`` for fast empathetic acknowledgements.
- **SummaryContextProvider**: Injects summary into REST text/voice chat flow.

The summary captures: key topics, user's emotional state, important people
mentioned, coping strategies discussed, and any action items.
"""

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Message, Summary

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = (
    "You are a summarization assistant for a mental health companion chatbot.\n"
    "Given a conversation between a user and an AI companion, produce a concise "
    "summary (3-6 sentences) that captures:\n"
    "1. The user's main emotional state and concerns\n"
    "2. Key people or relationships mentioned\n"
    "3. Any mental health topics or conditions discussed\n"
    "4. Coping strategies or advice that was shared\n"
    "5. Any action items or commitments the user made\n\n"
    "The summary should be written in third person (e.g. 'The user expressed...').\n"
    "Be factual and concise. Do not add interpretation beyond what was said."
)


async def _count_messages_since_summary(
    db: AsyncSession,
    conversation_id: UUID,
) -> tuple[int, UUID | None]:
    """Count messages since the last summary was generated.

    Returns:
        (message_count, last_summary_message_id) -- the number of new messages
        and the message ID the last summary was anchored to (or None).
    """
    # Find the last summary for this conversation
    result = await db.execute(
        select(Summary.last_message_id)
        .where(Summary.conv_id == conversation_id)
        .order_by(Summary.created_at.desc())
        .limit(1)
    )
    last_summary_msg_id = result.scalar_one_or_none()

    # Count messages after that point
    query = select(func.count()).select_from(Message).where(
        Message.conv_id == conversation_id
    )
    if last_summary_msg_id is not None:
        # Get the created_at of the anchor message
        anchor_result = await db.execute(
            select(Message.created_at).where(Message.id == last_summary_msg_id)
        )
        anchor_ts = anchor_result.scalar_one_or_none()
        if anchor_ts is not None:
            query = query.where(Message.created_at > anchor_ts)

    count_result = await db.execute(query)
    count = count_result.scalar() or 0
    return count, last_summary_msg_id


async def should_generate_summary(
    db: AsyncSession,
    conversation_id: UUID,
) -> bool:
    """Check whether a new summary should be generated.

    Returns True when the number of messages since the last summary
    meets or exceeds ``settings.summary_interval``.
    """
    count, _ = await _count_messages_since_summary(db, conversation_id)
    return count >= settings.summary_interval


async def generate_summary(
    db: AsyncSession,
    conversation_id: UUID,
) -> Summary | None:
    """Generate and persist a new conversation summary.

    Fetches recent messages (since the last summary), sends them to the LLM,
    and stores the resulting summary in the database.

    Returns:
        The new Summary ORM instance, or None if generation was skipped/failed.
    """
    count, _ = await _count_messages_since_summary(db, conversation_id)

    if count < settings.summary_interval:
        logger.debug(
            "Skipping summary: only %d messages (need %d).",
            count,
            settings.summary_interval,
        )
        return None

    # Fetch the messages to summarize (get the last N*2 to include some overlap
    # with the previous summary for continuity, capped at 40)
    fetch_limit = min(count + 10, 40)
    result = await db.execute(
        select(Message)
        .where(Message.conv_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(fetch_limit)
    )
    messages = list(reversed(result.scalars().all()))

    if not messages:
        return None

    # Also fetch the previous summary for continuity
    prev_result = await db.execute(
        select(Summary.content)
        .where(Summary.conv_id == conversation_id)
        .order_by(Summary.created_at.desc())
        .limit(1)
    )
    prev_summary = prev_result.scalar_one_or_none()

    # Build the conversation text for the LLM
    conversation_text = ""
    if prev_summary:
        conversation_text += f"Previous summary:\n{prev_summary}\n\n"
    conversation_text += "New conversation messages:\n"
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Companion"
        conversation_text += f"{role_label}: {msg.content}\n"

    try:
        from app.services.llm import LLMMessage, get_llm

        llm = get_llm()
        llm_response = await llm.chat(
            [LLMMessage(role="user", content=conversation_text)],
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
        )
        summary_text = llm_response.strip()
    except Exception:
        logger.exception("Summary generation LLM call failed.")
        return None

    # Persist the summary
    last_message = messages[-1]
    summary = Summary(
        conv_id=conversation_id,
        content=summary_text,
        last_message_id=last_message.id,
    )
    db.add(summary)
    await db.flush()
    await db.refresh(summary)

    logger.info(
        "Generated summary for conversation %s (anchored to message %s).",
        conversation_id,
        last_message.id,
    )
    return summary


async def maybe_generate_summary(
    db: AsyncSession,
    conversation_id: UUID,
) -> Summary | None:
    """Convenience: check threshold and generate if needed.

    This is the primary entry point for callers who want a simple
    "generate if it's time" call. Safe to call after every message exchange.
    """
    if await should_generate_summary(db, conversation_id):
        return await generate_summary(db, conversation_id)
    return None
