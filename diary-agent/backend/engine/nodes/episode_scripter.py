"""Node A4: Episode Scripter.

Creates detailed episode scripts based on the planner, generating text
scripts for each episode while maintaining continuity, vertical-friendly
pacing, and word limits for 90 seconds (~225 words per episode).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import EPISODE_SCRIPTER_HUMAN, EPISODE_SCRIPTER_SYSTEM
from engine.state import EpisodeEngineState, EpisodeScripts


def episode_scripter_node(state: EpisodeEngineState) -> dict:
    """Generate full scripts for every episode from the episode planner."""
    model = get_model().with_structured_output(EpisodeScripts)

    planner = state["episode_planner"]
    planner_json = planner.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=EPISODE_SCRIPTER_SYSTEM),
        HumanMessage(
            content=EPISODE_SCRIPTER_HUMAN.format(planner_json=planner_json)
        ),
    ]

    result: EpisodeScripts = model.invoke(messages)

    return {"episode_scripts": result}
