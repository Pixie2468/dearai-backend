"""Node A3: Episode Planner.

Generates a structured episode planner for the full story, breaking it into
5-8 episodes with outlines, emotional arcs, cliffhanger ideas, and retention
hooks per episode, tailored for 90-second vertical format.

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
from engine.state import EpisodeEngineState, EpisodePlanner


def episode_planner_node(state: EpisodeEngineState) -> dict:
    """Create a structured per-episode planner from the expanded story.

    On the first pass, generates a fresh plan.
    On subsequent passes (after A8 rejection), uses replan feedback
    to produce an improved version.
    """
    model = get_model().with_structured_output(EpisodePlanner)

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

    result: EpisodePlanner = model.invoke(messages)

    return {"episode_planner": result}
