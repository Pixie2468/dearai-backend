"""Node: Optimization Recommendation Engine.

Synthesises all analysis data (emotional arc, retention risk, cliffhanger
scores) and produces prioritized, actionable recommendations for the user.
This node is advisory-only — its output is presented to the user and does
NOT feed back into an automatic revision loop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import OPTIMIZER_HUMAN, OPTIMIZER_SYSTEM
from engine.state import EpisodeEngineState, OptimizationReport


def optimizer_node(state: EpisodeEngineState) -> dict:
    """Generate advisory recommendations based on all analysis results."""
    model = get_model().with_structured_output(OptimizationReport)

    episode_scripts = state["episode_scripts"]
    emotional_arc = state["emotional_arc"]
    retention_analysis = state["retention_analysis"]
    cliffhanger_analysis = state["cliffhanger_analysis"]

    messages = [
        SystemMessage(content=OPTIMIZER_SYSTEM),
        HumanMessage(
            content=OPTIMIZER_HUMAN.format(
                scripts_json=episode_scripts.model_dump_json(indent=2),
                emotional_arc_json=emotional_arc.model_dump_json(indent=2),
                retention_json=retention_analysis.model_dump_json(indent=2),
                cliffhanger_json=cliffhanger_analysis.model_dump_json(indent=2),
            )
        ),
    ]

    result: OptimizationReport = model.invoke(messages)

    return {"optimization_report": result}
