"""Node A3: Entry Planner.

Generates a structured entry planner for the full story, breaking it into
5-8 entrys with outlines, emotional arcs, emotional_peak ideas, and reflection
hooks per entry, tailored for 90-second vertical format.

On replan passes (triggered by A8 validation failure), incorporates targeted
feedback to produce an improved plan.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import (
    EPISODE_PLANNER_HUMAN,
    EPISODE_PLANNER_REPLAN_HUMAN,
    EPISODE_PLANNER_SYSTEM,
)
from engine.state import EntryEngineState, EntryPlanner


def entry_planner_node(state: EntryEngineState) -> dict:
    """Create a structured per-entry planner from the expanded story.

    On the first pass, generates a fresh plan.
    On subsequent passes (after A8 rejection), uses replan feedback
    to produce an improved version.
    """
    model = get_model().with_structured_output(EntryPlanner)

    task = state["task"]
    expanded_story = state["expanded_story"]
    story_text = expanded_story.model_dump_json(indent=2)

    feedback = state.get("final_validation_feedback", "")
    revision = state.get("pipeline_revision_number", 1)

    if feedback and revision > 1:
        human_content = EPISODE_PLANNER_REPLAN_HUMAN.format(
            task=task,
            expanded_story=story_text,
            feedback=feedback,
        )
    else:
        human_content = EPISODE_PLANNER_HUMAN.format(
            task=task,
            expanded_story=story_text,
        )

    messages = [
        SystemMessage(content=EPISODE_PLANNER_SYSTEM),
        HumanMessage(content=human_content),
    ]

    result: EntryPlanner = model.invoke(messages)

    return {"entry_planner": result}
