"""Node A4: Entry Scripter.

Creates detailed entry scripts based on the planner, generating text
scripts for each entry while maintaining continuity, vertical-friendly
pacing, and word limits for 90 seconds (~225 words per entry).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import EPISODE_SCRIPTER_HUMAN, EPISODE_SCRIPTER_SYSTEM
from engine.state import EntryEngineState, EntryScripts


def entry_scripter_node(state: EntryEngineState) -> dict:
    """Generate full scripts for every entry from the entry planner."""
    model = get_model().with_structured_output(EntryScripts)

    planner = state["entry_planner"]
    planner_json = planner.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=EPISODE_SCRIPTER_SYSTEM),
        HumanMessage(
            content=EPISODE_SCRIPTER_HUMAN.format(planner_json=planner_json)
        ),
    ]

    result: EntryScripts = model.invoke(messages)

    return {"entry_scripts": result}
