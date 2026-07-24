"""Node A1: Story Expander.

Generates a detailed story description from the user's input, expanding it
into a semi-narrative form with characters, setting, and plot hooks
(300-600 words).

On revision passes (triggered by A2 validation failure), incorporates
feedback to produce an improved version.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import (
    STORY_EXPANDER_HUMAN,
    STORY_EXPANDER_REVISION_HUMAN,
    STORY_EXPANDER_SYSTEM,
)
from engine.state import EpisodeEngineState, ExpandedStory


def story_expander_node(state: EpisodeEngineState) -> dict:
    """Expand a brief story idea into a rich, detailed story description.

    On the first pass, generates a fresh expansion.
    On subsequent passes (after A2 rejection), uses accumulated feedback
    to produce an improved version.
    """
    model = get_model().with_structured_output(ExpandedStory)

    input_cls = state["input_classification"]
    classification = input_cls.classification
    task = input_cls.preprocessed_input

    feedback = state.get("story_validation_feedback", "")
    revision = state.get("story_revision_number", 1)

    # load optional inspirational context from the engine/context directory
    # the prompt constant now includes a ``{Context}`` placeholder that must be
    # filled before sending to the LLM. If the file doesn't exist we just pass an
    # empty string so the prompt remains valid.
    from pathlib import Path

    context_path = Path(__file__).parent.parent / "context" / "The_Yellow_Wallpaper.txt"
    try:
        context_text = context_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        context_text = ""

    system_prompt = STORY_EXPANDER_SYSTEM.format(Context=context_text)

    if feedback and revision > 1:
        human_content = STORY_EXPANDER_REVISION_HUMAN.format(
            task=task,
            classification=classification,
            feedback=feedback,
        )
    else:
        human_content = STORY_EXPANDER_HUMAN.format(
            task=task,
            classification=classification,
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    result: ExpandedStory = model.invoke(messages)

    return {"expanded_story": result}
