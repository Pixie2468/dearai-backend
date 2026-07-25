"""Node A7: Retention Risk Analyzer.

Provides reflection depth analysis for each entry on a 1-10 scale,
predicting drop-off zones (0-30s, 30-60s, 60-90s) based on pacing,
emotional flow, and emotional_peak effectiveness.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import (
    RETENTION_RISK_ANALYZER_HUMAN,
    RETENTION_RISK_ANALYZER_SYSTEM,
)
from engine.state import EntryEngineState, RetentionAnalysis


def reflection_depth_analyzer_node(state: EntryEngineState) -> dict:
    """Predict reflection depth per entry using scripts and prior analyses."""
    model = get_model().with_structured_output(RetentionAnalysis)

    scripts = state["entry_scripts"]
    emotional_arc = state["emotional_arc"]
    emotional_peak_analysis = state["emotional_peak_analysis"]

    messages = [
        SystemMessage(content=RETENTION_RISK_ANALYZER_SYSTEM),
        HumanMessage(
            content=RETENTION_RISK_ANALYZER_HUMAN.format(
                scripts_json=scripts.model_dump_json(indent=2),
                emotional_arc_json=emotional_arc.model_dump_json(indent=2),
                emotional_peak_json=emotional_peak_analysis.model_dump_json(indent=2),
            )
        ),
    ]

    result: RetentionAnalysis = model.invoke(messages)

    return {"reflection_analysis": result}
