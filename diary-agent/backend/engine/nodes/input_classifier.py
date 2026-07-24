"""Node A0: Input Classifier (LLM-based) + A2: Story Validator.

A0 — Uses an LLM to classify whether the user input is a concise one-liner
idea or a more detailed story outline.

A2 — Validates the quality of the expanded story from A1, checking coherence,
originality, engagement, and length. If the score >= 8, approve; otherwise
provide feedback for regeneration.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from engine.llm import get_model
from engine.prompts import (
    INPUT_CLASSIFIER_HUMAN,
    INPUT_CLASSIFIER_SYSTEM,
    STORY_VALIDATOR_HUMAN,
    STORY_VALIDATOR_SYSTEM,
)
from engine.state import EpisodeEngineState, InputClassification, StoryValidation


def input_classifier_node(state: EpisodeEngineState) -> dict:
    """Classify user input as a one-liner idea or a detailed story outline using an LLM."""
    model = get_model().with_structured_output(InputClassification)

    raw_input = state["task"].strip()

    messages = [
        SystemMessage(content=INPUT_CLASSIFIER_SYSTEM),
        HumanMessage(content=INPUT_CLASSIFIER_HUMAN.format(task=raw_input)),
    ]

    result: InputClassification = model.invoke(messages)

    return {"input_classification": result}


def story_validator_node(state: EpisodeEngineState) -> dict:
    """Validate the expanded story from A1 for quality and readiness.

    If score >= 8, the story passes. Otherwise, feedback is accumulated
    so A1 can retry with specific improvement instructions.
    """
    model = get_model().with_structured_output(StoryValidation)

    expanded_story = state["expanded_story"]
    story_text = expanded_story.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=STORY_VALIDATOR_SYSTEM),
        HumanMessage(content=STORY_VALIDATOR_HUMAN.format(expanded_story=story_text)),
    ]

    result: StoryValidation = model.invoke(messages)

    updates: dict = {
        "story_validation": result,
        "story_revision_number": state.get("story_revision_number", 1) + 1,
    }

    # Accumulate feedback for A1 retries when the story fails validation
    if not result.passed:
        prev_feedback = state.get("story_validation_feedback", "")
        separator = "\n---\n" if prev_feedback else ""
        updates["story_validation_feedback"] = (
            prev_feedback + separator + result.feedback
        )

    return updates
