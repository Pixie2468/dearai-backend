"""Node A7: Retention Risk Analyzer.

Provides retention risk analysis for each episode on a 1-10 scale,
predicting drop-off zones (0-30s, 30-60s, 60-90s) based on pacing,
emotional flow, and cliffhanger effectiveness.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import (
    RETENTION_RISK_ANALYZER_HUMAN,
    RETENTION_RISK_ANALYZER_SYSTEM,
)
from engine.state import EpisodeEngineState, RetentionAnalysis


def retention_risk_analyzer_node(state: EpisodeEngineState) -> dict:
    """Predict retention risk per episode using scripts and prior analyses."""
    model = get_model().with_structured_output(RetentionAnalysis)

    scripts = state["episode_scripts"]
    emotional_arc = state["emotional_arc"]
    cliffhanger_analysis = state["cliffhanger_analysis"]

    messages = [
        SystemMessage(content=RETENTION_RISK_ANALYZER_SYSTEM),
        HumanMessage(
            content=RETENTION_RISK_ANALYZER_HUMAN.format(
                scripts_json=scripts.model_dump_json(indent=2),
                emotional_arc_json=emotional_arc.model_dump_json(indent=2),
                cliffhanger_json=cliffhanger_analysis.model_dump_json(indent=2),
            )
        ),
    ]

    result: RetentionAnalysis = model.invoke(messages)

    return {"retention_analysis": result}
