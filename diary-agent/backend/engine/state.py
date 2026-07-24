"""State schema and Pydantic models for the Episodic Intelligence Engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Pydantic models — used for structured LLM output
# ---------------------------------------------------------------------------


# --- Emotional Arc models ---


class EmotionBeat(BaseModel):
    """A single emotional beat within an episode."""

    timestamp_range: str = Field(
        description="Approximate time range within the 90s episode (e.g. '0-15s', '45-60s')"
    )
    emotion: str = Field(
        description="Primary emotion at this beat (e.g. curiosity, fear, hope, shock, sadness)"
    )
    intensity: int = Field(description="Emotion intensity on a 1-10 scale", ge=1, le=10)


class EpisodeEmotionProfile(BaseModel):
    """Emotional profile for a single episode."""

    episode_number: int
    emotion_beats: list[EmotionBeat] = Field(
        description="Sequence of emotional beats through the episode"
    )
    dominant_emotion: str = Field(
        description="The single most prominent emotion in this episode"
    )
    emotional_range: int = Field(
        description="How wide the emotional range is (1=flat, 10=extreme swings)",
        ge=1,
        le=10,
    )


class EmotionalArc(BaseModel):
    """Full emotional arc analysis across all episodes."""

    episodes: list[EpisodeEmotionProfile] = Field(
        description="Per-episode emotional profiles"
    )
    overall_progression: str = Field(
        description="Narrative description of how emotions progress across the full arc"
    )
    emotional_coherence_score: int = Field(
        description="How well emotions flow between episodes (1-10)", ge=1, le=10
    )
    tension_curve_description: str = Field(
        description="Description of the overall tension curve shape"
    )


# --- Retention Risk models ---


class RiskZone(BaseModel):
    """A specific zone within an episode where viewer drop-off is likely."""

    timestamp_range: str = Field(
        description="Time range where drop-off risk exists (e.g. '20-35s')"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity of the retention risk"
    )
    reason: str = Field(description="Why viewers might drop off at this point")
    suggested_fix: str = Field(
        description="Quick suggestion to mitigate this risk zone"
    )


class EpisodeRetentionRisk(BaseModel):
    """Retention risk analysis for a single episode."""

    episode_number: int
    overall_retention_score: int = Field(
        description="Predicted retention score 0-100 (100=everyone stays)", ge=0, le=100
    )
    risk_zones: list[RiskZone] = Field(
        description="Specific time ranges with drop-off risk"
    )
    hook_strength: int = Field(
        description="How strong the opening hook is (1-10)", ge=1, le=10
    )
    pacing_score: int = Field(
        description="How well-paced the episode is (1-10)", ge=1, le=10
    )


class RetentionAnalysis(BaseModel):
    """Full retention risk analysis across all episodes."""

    episodes: list[EpisodeRetentionRisk] = Field(
        description="Per-episode retention analysis"
    )
    weakest_episode: int = Field(
        description="Episode number with the lowest retention score"
    )
    strongest_episode: int = Field(
        description="Episode number with the highest retention score"
    )
    overall_series_retention_prediction: str = Field(
        description="Prediction of how many viewers who start ep 1 will finish the series"
    )


# --- Cliffhanger models ---


class CliffhangerScore(BaseModel):
    """Cliffhanger quality assessment for a single episode."""

    episode_number: int
    score: int = Field(description="Cliffhanger strength score (1-10)", ge=1, le=10)
    cliffhanger_type: str = Field(
        description="Type of cliffhanger (e.g. Question, Danger, Revelation, Emotional, Decision, Twist)"
    )
    curiosity_gap: int = Field(
        description="How strong the curiosity gap is (1-10)", ge=1, le=10
    )
    stakes_level: int = Field(
        description="How high the stakes feel (1-10)", ge=1, le=10
    )
    emotional_charge: int = Field(
        description="Emotional impact of the cliffhanger (1-10)", ge=1, le=10
    )
    reasoning: str = Field(
        description="Explanation of why this cliffhanger works or doesn't"
    )


class CliffhangerAnalysis(BaseModel):
    """Full cliffhanger analysis across all episodes."""

    scores: list[CliffhangerScore] = Field(description="Per-episode cliffhanger scores")
    average_score: float = Field(
        description="Average cliffhanger score across all episodes"
    )
    weakest_cliffhanger: int = Field(
        description="Episode number with the weakest cliffhanger"
    )
    strongest_cliffhanger: int = Field(
        description="Episode number with the strongest cliffhanger"
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
        description="Quality score for the episode scripts (1-10)", ge=1, le=10
    )
    emotional_arc_score: int = Field(
        description="Quality score for the emotional arc analysis (1-10)", ge=1, le=10
    )
    cliffhanger_score: int = Field(
        description="Average cliffhanger strength across episodes (1-10)", ge=1, le=10
    )
    retention_score: int = Field(
        description="Average retention score across episodes (1-10)", ge=1, le=10
    )
    replan_instructions: str = Field(
        description="Targeted feedback for replanning if failed (e.g. 'Strengthen cliffhangers in episodes 3-5'); empty when passed"
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


# --- Episode Planner (A3) ---


class PlannedEpisode(BaseModel):
    """A single episode entry in the episode planner."""

    episode_number: int = Field(description="Episode number (1-based)")
    title: str = Field(description="Short, punchy episode title")
    outline: str = Field(description="Concise outline of what happens in this episode")
    emotional_arc_notes: str = Field(
        description="Expected emotional trajectory within this episode"
    )
    cliffhanger_idea: str = Field(
        description="The planned cliffhanger or hook for the episode ending"
    )
    retention_hooks: list[str] = Field(
        description="Specific moments designed to keep viewers watching"
    )
    estimated_word_count: int = Field(
        default=225,
        description="Target script word count for ~90 seconds of content",
    )


class EpisodePlanner(BaseModel):
    """Full episode planner for the story."""

    total_episodes: int = Field(description="Total number of episodes (5-8)")
    overall_narrative_arc: str = Field(
        description="The overarching narrative arc type and description"
    )
    target_audience: str = Field(description="Intended audience for this content")
    episodes: list[PlannedEpisode] = Field(
        description="The ordered list of planned episodes"
    )


# --- Episode Scripts (A4) ---


class EpisodeScript(BaseModel):
    """A single episode script."""

    episode_number: int = Field(description="Episode number (1-based)")
    title: str = Field(description="Episode title")
    script: str = Field(
        description="The full episode narrative voiceover script (~225 words for 90 seconds). Third-person narration, no direct dialogue."
    )
    word_count: int = Field(description="Actual word count of the script")
    scene_directions: list[str] = Field(
        description="Visual/camera directions for vertical video format (close-ups, transitions, etc.)"
    )
    continuity_notes: str = Field(
        description="Notes on how this episode connects to the previous and next episodes"
    )


class EpisodeScripts(BaseModel):
    """Collection of all episode scripts."""

    scripts: list[EpisodeScript] = Field(
        description="The ordered list of episode scripts"
    )
    total_word_count: int = Field(description="Combined word count across all scripts")
    series_continuity_summary: str = Field(
        description="Brief summary of how episodes flow together narratively"
    )


# --- Optimization models ---


class Suggestion(BaseModel):
    """A single optimization suggestion."""

    episode_number: int = Field(
        description="Which episode this applies to (0 = series-wide)"
    )
    category: Literal[
        "hook", "pacing", "cliffhanger", "emotion", "structure", "dialogue"
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


class EpisodeEngineState(TypedDict):
    """Central state object passed between all LangGraph nodes."""

    # Input
    task: str  # The user's raw story idea

    # A0 – Input Classifier (LLM-based)
    input_classification: InputClassification | None

    # A1 – Story Expander
    expanded_story: ExpandedStory | None

    # A2 – Story Validator (lives in input_classifier.py)
    story_validation: StoryValidation | None
    story_validation_feedback: str  # accumulated feedback for A1 retries

    # A3 – Episode Planner
    episode_planner: EpisodePlanner | None

    # A4 – Episode Scripter
    episode_scripts: EpisodeScripts | None

    # A5-A7 analysis outputs
    emotional_arc: EmotionalArc | None
    retention_analysis: RetentionAnalysis | None
    cliffhanger_analysis: CliffhangerAnalysis | None

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
