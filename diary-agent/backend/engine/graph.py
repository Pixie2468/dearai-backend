"""LangGraph graph builder for the Episodic Intelligence Engine.

New topology (A0–A8 pipeline with two conditional loops):

    START ─> A0 (input_classifier)
              │
              ▼
         A1 (story_expander) ◄──────────────────┐
              │                                  │ (fail: score < 8)
              ▼                                  │
         A2 (story_validator) ──[should_retry_story]
              │ (pass: score >= 8)
              ▼
    ┌─> A3 (episode_planner) ◄──────────────────────────────┐
    │         │                                              │
    │         ▼                                              │
    │    A4 (episode_scripter)                               │
    │         │                                              │
    │         ├──> A5 (emotional_arc_scorer) ──┐             │
    │         └──> A6 (cliffhanger_scorer) ────┤             │
    │                                          ▼             │
    │                              A7 (retention_risk) ──────┤
    │                                          │             │
    │                                          ▼             │
    │                              A8 (final_validator) ─[should_replan]
    │                                          │ (pass)      │ (fail)
    │                                          ▼             │
    │                                   optimizer ──> END    │
    │                                                        │
    └────────────────────────────────────────────────────────┘

- A5 and A6 run in parallel (fan-out from A4, fan-in to A7).
- A1⇄A2 loop: retries story expansion up to max_story_revisions.
- A3→A8 loop: retries the full script pipeline up to max_pipeline_revisions.
- Optimizer is advisory-only (no feedback loop), runs after A8 passes.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from engine.nodes.cliffhanger_strength_scorer import cliffhanger_strength_scorer_node
from engine.nodes.emotional_arc_scorer import emotional_arc_scorer_node
from engine.nodes.episode_planner import episode_planner_node
from engine.nodes.episode_scripter import episode_scripter_node
from engine.nodes.final_validator import final_validator_node
from engine.nodes.input_classifier import input_classifier_node, story_validator_node
from engine.nodes.optimizer import optimizer_node
from engine.nodes.retention_risk_analyzer import retention_risk_analyzer_node
from engine.nodes.story_expander import story_expander_node
from engine.state import EpisodeEngineState


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def _should_retry_story(
    state: EpisodeEngineState,
) -> Literal["story_expander", "episode_planner"]:
    """After A2 (story_validator): loop back to A1 or proceed to A3.

    Passes when the story validation score >= 8 OR the maximum number of
    story revisions has been reached.
    """
    validation = state.get("story_validation")
    if validation is not None and validation.passed:
        return "episode_planner"

    revision = state.get("story_revision_number", 1)
    max_revisions = state.get("max_story_revisions", 3)
    if revision > max_revisions:
        # Exhausted retries — proceed with the best version we have
        return "episode_planner"

    return "story_expander"


def _should_replan(
    state: EpisodeEngineState,
) -> Literal["episode_planner", "optimizer"]:
    """After A8 (final_validator): loop back to A3 or proceed to optimizer.

    Passes when the final validation average >= 7 OR the maximum number of
    pipeline revisions has been reached.
    """
    validation = state.get("final_validation")
    if validation is not None and validation.passed:
        return "optimizer"

    revision = state.get("pipeline_revision_number", 1)
    max_revisions = state.get("max_pipeline_revisions", 3)
    if revision > max_revisions:
        # Exhausted retries — proceed with the best version we have
        return "optimizer"

    return "episode_planner"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(checkpointer: InMemorySaver | None = None) -> CompiledStateGraph:
    """Construct and compile the Episodic Intelligence Engine graph.

    Args:
        checkpointer: Optional checkpointer for state persistence.
            Defaults to an InMemorySaver if not provided.

    Returns:
        The compiled LangGraph graph, ready for .invoke() or .stream().
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    builder = StateGraph(EpisodeEngineState)

    # --- Register all nodes ---
    builder.add_node("input_classifier", input_classifier_node)       # A0
    builder.add_node("story_expander", story_expander_node)           # A1
    builder.add_node("story_validator", story_validator_node)         # A2
    builder.add_node("episode_planner", episode_planner_node)         # A3
    builder.add_node("episode_scripter", episode_scripter_node)       # A4
    builder.add_node("emotional_arc_scorer", emotional_arc_scorer_node)           # A5
    builder.add_node("cliffhanger_strength_scorer", cliffhanger_strength_scorer_node)  # A6
    builder.add_node("retention_risk_analyzer", retention_risk_analyzer_node)      # A7
    builder.add_node("final_validator", final_validator_node)         # A8
    builder.add_node("optimizer", optimizer_node)                     # Recommendations

    # --- Entry ---
    builder.add_edge(START, "input_classifier")

    # --- A0 -> A1 ---
    builder.add_edge("input_classifier", "story_expander")

    # --- A1 -> A2 ---
    builder.add_edge("story_expander", "story_validator")

    # --- A2 -> conditional: A1 (retry) or A3 (proceed) ---
    builder.add_conditional_edges(
        "story_validator",
        _should_retry_story,
        {
            "story_expander": "story_expander",
            "episode_planner": "episode_planner",
        },
    )

    # --- A3 -> A4 ---
    builder.add_edge("episode_planner", "episode_scripter")

    # --- A4 -> parallel fan-out: A5 + A6 ---
    builder.add_edge("episode_scripter", "emotional_arc_scorer")
    builder.add_edge("episode_scripter", "cliffhanger_strength_scorer")

    # --- Parallel fan-in: A5 + A6 -> A7 ---
    builder.add_edge("emotional_arc_scorer", "retention_risk_analyzer")
    builder.add_edge("cliffhanger_strength_scorer", "retention_risk_analyzer")

    # --- A7 -> A8 ---
    builder.add_edge("retention_risk_analyzer", "final_validator")

    # --- A8 -> conditional: A3 (replan) or optimizer (pass) ---
    builder.add_conditional_edges(
        "final_validator",
        _should_replan,
        {
            "episode_planner": "episode_planner",
            "optimizer": "optimizer",
        },
    )

    # --- Optimizer -> END ---
    builder.add_edge("optimizer", END)

    # --- Compile ---
    graph = builder.compile(checkpointer=checkpointer)

    return graph


if __name__ == "__main__":
    graph = build_graph()
    image_bytes = graph.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(image_bytes)

    print("Graph compiled successfully. Ready to invoke or stream.")
