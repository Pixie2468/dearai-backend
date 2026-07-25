"""State schema and Pydantic models for the Diary Intelligence Engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Pydantic models — used for structured LLM output
# ---------------------------------------------------------------------------


# --- Emotional Arc models ---


class EmotionBeat(BaseModel):
    """A single emotional beat within an entry."""

    timestamp_range: str = Field(
        description="Approximate time range within the 90s entry (e.g. '0-15s', '45-60s')"
    )
    emotion: str = Field(
        description="Primary emotion at this beat (e.g. curiosity, fear, hope, shock, sadness)"
    )
    intensity: int = Field(description="Emotion intensity on a 1-10 scale", ge=1, le=10)


class EntryEmotionProfile(BaseModel):
    """Emotional profile for a single entry."""

    entry_number: int
    emotion_beats: list[EmotionBeat] = Field(
        description="Sequence of emotional beats through the entry"
    )
    dominant_emotion: str = Field(
        description="The single most prominent emotion in this entry"
    )
    emotional_range: int = Field(
        description="How wide the emotional range is (1=flat, 10=extreme swings)",
        ge=1,
        le=10,
    )


class EmotionalArc(BaseModel):
    """Full emotional arc analysis across all entrys."""

    entrys: list[EntryEmotionProfile] = Field(
        description="Per-entry emotional profiles"
    )
    overall_progression: str = Field(
        description="Narrative description of how emotions progress across the full arc"
    )
    emotional_coherence_score: int = Field(
        description="How well emotions flow between entrys (1-10)", ge=1, le=10
    )
    tension_curve_description: str = Field(
        description="Description of the overall tension curve shape"
    )


# --- Retention Risk models ---


class RiskZone(BaseModel):
    """A specific zone within an entry where viewer drop-off is likely."""

    timestamp_range: str = Field(
        description="Time range where drop-off risk exists (e.g. '20-35s')"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity of the reflection depth"
    )
    reason: str = Field(description="Why viewers might drop off at this point")
    suggested_fix: str = Field(
        description="Quick suggestion to mitigate this risk zone"
    )


class EntryRetentionRisk(BaseModel):
    """Retention risk analysis for a single entry."""

    entry_number: int
    overall_reflection_score: int = Field(
        description="Predicted reflection score 0-100 (100=everyone stays)", ge=0, le=100
    )
    risk_zones: list[RiskZone] = Field(
        description="Specific time ranges with drop-off risk"
    )
    hook_strength: int = Field(
        description="How strong the opening hook is (1-10)", ge=1, le=10
    )
    pacing_score: int = Field(
        description="How well-paced the entry is (1-10)", ge=1, le=10
    )


class RetentionAnalysis(BaseModel):
    """Full reflection depth analysis across all entrys."""

    entrys: list[EntryRetentionRisk] = Field(
        description="Per-entry reflection analysis"
    )
    weakest_entry: int = Field(
        description="Entry number with the lowest reflection score"
    )
    strongest_entry: int = Field(
        description="Entry number with the highest reflection score"
    )
    overall_series_reflection_prediction: str = Field(
        description="Prediction of how many viewers who start ep 1 will finish the series"
    )


# --- Emotional Peak models ---


class Emotional PeakScore(BaseModel):
    """Emotional Peak quality assessment for a single entry."""

    entry_number: int
    score: int = Field(description="Emotional Peak strength score (1-10)", ge=1, le=10)
    emotional_peak_type: str = Field(
        description="Type of emotional_peak (e.g. Question, Danger, Revelation, Emotional, Decision, Twist)"
    )
    curiosity_gap: int = Field(
        description="How strong the curiosity gap is (1-10)", ge=1, le=10
    )
    stakes_level: int = Field(
        description="How high the stakes feel (1-10)", ge=1, le=10
    )
    emotional_charge: int = Field(
        description="Emotional impact of the emotional_peak (1-10)", ge=1, le=10
    )
    reasoning: str = Field(
        description="Explanation of why this emotional_peak works or doesn't"
    )


class Emotional PeakAnalysis(BaseModel):
    """Full emotional_peak analysis across all entrys."""

    scores: list[Emotional PeakScore] = Field(description="Per-entry emotional_peak scores")
    average_score: float = Field(
        description="Average emotional_peak score across all entrys"
    )
    weakest_emotional_peak: int = Field(
        description="Entry number with the weakest emotional_peak"
    )
    strongest_emotional_peak: int = Field(
        description="Entry number with the strongest emotional_peak"
    )


# --- Optimization models ---


# --- Input Classification (A0 — LLM-based) ---


class InputClassification(BaseModel):
    """LLM-based classification and initial assessment of the user's raw input."""

    classification: Literal["one-liner", "story"] = Field(
        description="Whether the input is a concise one-liner idea or a detailed story outline"
    )
    confidence: int = Field(
        description="Confidence in the classification (1-10)", ge=1, le=10
    )
    preprocessed_input: str = Field(
        description="The original input, lightly cleaned and normalised for downstream use"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )


# --- Story Validation (A2 — combined into A0 file) ---


class StoryValidation(BaseModel):
    """Quality validation of the expanded story from A1."""

    score: int = Field(
        description="Overall quality score for the story description (1-10)",
        ge=1,
        le=10,
    )
    passed: bool = Field(
        description="Whether the story meets the quality threshold (score >= 8)"
    )
    coherence: int = Field(
        description="How coherent and logical the story is (1-10)", ge=1, le=10
    )
    originality: int = Field(
        description="How original and non-clichéd the story is (1-10)", ge=1, le=10
    )
    engagement: int = Field(
        description="How engaging and compelling the story is (1-10)", ge=1, le=10
    )
    length_appropriate: bool = Field(
        description="Whether the story length is within the 300-600 word target"
    )
    feedback: str = Field(
        description="Specific feedback notes for improvement (populated when failed, empty when passed)"
    )


# --- Final Validation (A8) ---


class FinalValidation(BaseModel):
    """End-of-pipeline validation of all outputs from A4-A7."""

    passed: bool = Field(
        description="Whether all outputs meet quality thresholds (average score >= 7)"
    )
    average_score: float = Field(
        description="Weighted average quality score across all analyses (1-10)"
    )
    script_quality_score: int = Field(
        description="Quality score for the entry scripts (1-10)", ge=1, le=10
    )
    emotional_arc_score: int = Field(
        description="Quality score for the emotional arc analysis (1-10)", ge=1, le=10
    )
    emotional_peak_score: int = Field(
        description="Average emotional_peak strength across entrys (1-10)", ge=1, le=10
    )
    reflection_score: int = Field(
        description="Average reflection score across entrys (1-10)", ge=1, le=10
    )
    replan_instructions: str = Field(
        description="Targeted feedback for replanning if failed (e.g. 'Strengthen emotional_peaks in entrys 3-5'); empty when passed"
    )


# --- Expanded Story (A1) ---


class ExpandedStory(BaseModel):
    """A detailed story description expanded from user input."""

    title: str = Field(description="A compelling working title for the story")
    characters: list[str] = Field(
        description="Key characters with brief descriptors (e.g. 'Mira — a reclusive hacker')"
    )
    setting: str = Field(
        description="The world, time period, and atmosphere of the story"
    )
    plot_hooks: list[str] = Field(
        description="3-5 intriguing plot hooks that drive viewer curiosity"
    )
    expanded_description: str = Field(
        description="The full expanded story description (300-600 words) in semi-narrative form"
    )


# --- Entry Planner (A3) ---


class PlannedEntry(BaseModel):
    """A single entry entry in the entry planner."""

    entry_number: int = Field(description="Entry number (1-based)")
    title: str = Field(description="Short, punchy entry title")
    outline: str = Field(description="Concise outline of what happens in this entry")
    emotional_arc_notes: str = Field(
        description="Expected emotional trajectory within this entry"
    )
    emotional_peak_idea: str = Field(
        description="The planned emotional_peak or hook for the entry ending"
    )
    reflection_hooks: list[str] = Field(
        description="Specific moments designed to keep viewers watching"
    )
    estimated_word_count: int = Field(
        default=225,
        description="Target script word count for ~90 seconds of content",
    )


class EntryPlanner(BaseModel):
    """Full entry planner for the story."""

    total_entrys: int = Field(description="Total number of entrys (5-8)")
    overall_narrative_arc: str = Field(
        description="The overarching narrative arc type and description"
    )
    target_audience: str = Field(description="Intended audience for this content")
    entrys: list[PlannedEntry] = Field(
        description="The ordered list of planned entrys"
    )


# --- Entry Scripts (A4) ---


class EntryScript(BaseModel):
    """A single entry script."""

    entry_number: int = Field(description="Entry number (1-based)")
    title: str = Field(description="Entry title")
    script: str = Field(
        description="The full entry narrative voiceover script (~225 words for 90 seconds). Third-person narration, no direct dialogue."
    )
    word_count: int = Field(description="Actual word count of the script")
    scene_directions: list[str] = Field(
        description="Visual/camera directions for diary format format (close-ups, transitions, etc.)"
    )
    continuity_notes: str = Field(
        description="Notes on how this entry connects to the previous and next entrys"
    )


class EntryScripts(BaseModel):
    """Collection of all entry scripts."""

    scripts: list[EntryScript] = Field(
        description="The ordered list of entry scripts"
    )
    total_word_count: int = Field(description="Combined word count across all scripts")
    series_continuity_summary: str = Field(
        description="Brief summary of how entrys flow together narratively"
    )


# --- Optimization models ---


class Suggestion(BaseModel):
    """A single optimization suggestion."""

    entry_number: int = Field(
        description="Which entry this applies to (0 = series-wide)"
    )
    category: Literal[
        "hook", "pacing", "emotional_peak", "emotion", "structure", "dialogue"
    ] = Field(description="Category of improvement")
    current_issue: str = Field(description="What the current problem is")
    suggested_improvement: str = Field(description="Specific, actionable improvement")
    priority: Literal["critical", "high", "medium", "low"] = Field(
        description="Priority level for this fix"
    )
    expected_impact: str = Field(
        description="What impact this change would have on viewer engagement"
    )


class OptimizationReport(BaseModel):
    """Full set of optimization suggestions."""

    suggestions: list[Suggestion] = Field(description="All improvement suggestions")
    top_3_priorities: list[str] = Field(
        description="The 3 most impactful changes to make, in plain language"
    )
    overall_quality_score: int = Field(
        description="Current overall quality score for the series (1-100)", ge=1, le=100
    )
    predicted_quality_after_optimization: int = Field(
        description="Predicted quality score if all suggestions are applied (1-100)",
        ge=1,
        le=100,
    )


# ---------------------------------------------------------------------------
# LangGraph State — the TypedDict that flows through the graph
# ---------------------------------------------------------------------------


class EntryEngineState(TypedDict):
    """Central state object passed between all LangGraph nodes."""

    # Input
    task: str  # The user's raw chat history

    # A0 – Input Classifier (LLM-based)
    input_classification: InputClassification | None

    # A1 – Story Expander
    expanded_story: ExpandedStory | None

    # A2 – Story Validator (lives in input_classifier.py)
    story_validation: StoryValidation | None
    story_validation_feedback: str  # accumulated feedback for A1 retries

    # A3 – Entry Planner
    entry_planner: EntryPlanner | None

    # A4 – Entry Scripter
    entry_scripts: EntryScripts | None

    # A5-A7 analysis outputs
    emotional_arc: EmotionalArc | None
    reflection_analysis: RetentionAnalysis | None
    emotional_peak_analysis: Emotional PeakAnalysis | None

    # A8 – Final Validator
    final_validation: FinalValidation | None
    final_validation_feedback: str  # replan instructions for A3 retries

    # Optimizer (recommendation-only, no loop)
    optimization_report: OptimizationReport | None

    # Loop control
    story_revision_number: int  # A1↔A2 loop counter
    max_story_revisions: int
    pipeline_revision_number: int  # A3→A8 loop counter
    max_pipeline_revisions: int
