"""Node A5: Emotional Arc Scorer.

Analyzes and scores the emotional arc for the full story and each episode
based on actual scripts, identifying emotion shifts, variance (1-10),
and flat zones where engagement might drop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import EMOTIONAL_ARC_SCORER_HUMAN, EMOTIONAL_ARC_SCORER_SYSTEM
from engine.state import EmotionalArc, EpisodeEngineState


def emotional_arc_scorer_node(state: EpisodeEngineState) -> dict:
    """Analyse the emotional progression of episode scripts."""
    model = get_model().with_structured_output(EmotionalArc)

    scripts = state["episode_scripts"]
    scripts_json = scripts.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=EMOTIONAL_ARC_SCORER_SYSTEM),
        HumanMessage(
            content=EMOTIONAL_ARC_SCORER_HUMAN.format(scripts_json=scripts_json)
        ),
    ]

    result: EmotionalArc = model.invoke(messages)

    return {"emotional_arc": result}
