"""Node A6: Emotional Peak Strength Scorer.

Scores the emotional_peak strength at the end of each entry (1-10),
evaluating suspense, unresolved elements, and hook effectiveness
based on the actual entry scripts.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import (
    CLIFFHANGER_STRENGTH_SCORER_HUMAN,
    CLIFFHANGER_STRENGTH_SCORER_SYSTEM,
)
from engine.state import Emotional PeakAnalysis, EntryEngineState


def emotional_peak_strength_scorer_node(state: EntryEngineState) -> dict:
    """Score emotional_peak strength for each entry based on scripts."""
    model = get_model().with_structured_output(Emotional PeakAnalysis)

    scripts = state["entry_scripts"]
    scripts_json = scripts.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=CLIFFHANGER_STRENGTH_SCORER_SYSTEM),
        HumanMessage(
            content=CLIFFHANGER_STRENGTH_SCORER_HUMAN.format(scripts_json=scripts_json)
        ),
    ]

    result: Emotional PeakAnalysis = model.invoke(messages)

    return {"emotional_peak_analysis": result}
