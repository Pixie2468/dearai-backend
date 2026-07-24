"""Node A6: Cliffhanger Strength Scorer.

Scores the cliffhanger strength at the end of each episode (1-10),
evaluating suspense, unresolved elements, and hook effectiveness
based on the actual episode scripts.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import (
    CLIFFHANGER_STRENGTH_SCORER_HUMAN,
    CLIFFHANGER_STRENGTH_SCORER_SYSTEM,
)
from engine.state import CliffhangerAnalysis, EpisodeEngineState


def cliffhanger_strength_scorer_node(state: EpisodeEngineState) -> dict:
    """Score cliffhanger strength for each episode based on scripts."""
    model = get_model().with_structured_output(CliffhangerAnalysis)

    scripts = state["episode_scripts"]
    scripts_json = scripts.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=CLIFFHANGER_STRENGTH_SCORER_SYSTEM),
        HumanMessage(
            content=CLIFFHANGER_STRENGTH_SCORER_HUMAN.format(scripts_json=scripts_json)
        ),
    ]

    result: CliffhangerAnalysis = model.invoke(messages)

    return {"cliffhanger_analysis": result}
