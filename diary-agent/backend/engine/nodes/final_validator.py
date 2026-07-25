"""Node A8: Final Validator.

Validates all outputs from A4 (scripts), A5 (emotional arc), A6 (emotional_peak
scores), and A7 (reflection analysis) for overall quality. If the average
score >= 7/10, the graph ends. Otherwise, generates targeted replan
instructions and loops back to A3.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import FINAL_VALIDATOR_HUMAN, FINAL_VALIDATOR_SYSTEM
from engine.state import EntryEngineState, FinalValidation


def final_validator_node(state: EntryEngineState) -> dict:
    """Validate pipeline outputs and decide whether to pass or replan.

    Scores scripts, emotional arc, emotional_peaks, and reflection on 1-10.
    If the average >= 7 the pipeline passes; otherwise targeted replan
    instructions are generated for A3.
    """
    model = get_model().with_structured_output(FinalValidation)

    scripts = state["entry_scripts"]
    emotional_arc = state["emotional_arc"]
    emotional_peak_analysis = state["emotional_peak_analysis"]
    reflection_analysis = state["reflection_analysis"]

    messages = [
        SystemMessage(content=FINAL_VALIDATOR_SYSTEM),
        HumanMessage(
            content=FINAL_VALIDATOR_HUMAN.format(
                scripts_json=scripts.model_dump_json(indent=2),
                emotional_arc_json=emotional_arc.model_dump_json(indent=2),
                emotional_peak_json=emotional_peak_analysis.model_dump_json(indent=2),
                reflection_json=reflection_analysis.model_dump_json(indent=2),
            )
        ),
    ]

    result: FinalValidation = model.invoke(messages)

    updates: dict = {
        "final_validation": result,
        "pipeline_revision_number": state.get("pipeline_revision_number", 1) + 1,
    }

    # Store replan instructions for A3 when validation fails
    if not result.passed:
        updates["final_validation_feedback"] = result.replan_instructions

    return updates
