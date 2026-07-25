"""Pydantic schemas for API request and response bodies."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from engine.state import (
    Emotional PeakAnalysis,
    EmotionalArc,
    EntryPlanner,
    EntryScripts,
    OptimizationReport,
    RetentionAnalysis,
)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Payload for POST /diary-intelligence/analyze."""

    chat_history: str = Field(
        ..., description="The raw chat history to analyze", min_length=1
    )
    genre: str = Field(default="", description="Genre hint (e.g. thriller, romance)")
    target_audience: str = Field(
        default="18-30 mobile-first viewers",
        description="Target audience description",
    )
    tone: str = Field(default="", description="Desired tone (e.g. tense, humorous)")
    entry_count_preference: int = Field(
        default=6,
        ge=5,
        le=8,
        description="Preferred number of entrys (5-8)",
    )
    max_revisions: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of optimization revision loops (1-5)",
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class AnalyzeResponse(BaseModel):
    """Structured response returned by the analysis endpoint."""

    run_id: uuid.UUID = Field(description="Unique identifier for this analysis run")
    chat_history: str
    revisions_completed: int
    entry_planner: EntryPlanner
    entry_scripts: EntryScripts
    emotional_arc: EmotionalArc
    reflection_analysis: RetentionAnalysis
    emotional_peak_analysis: Emotional PeakAnalysis
    optimization_report: OptimizationReport
    created_at: datetime

    model_config = {"from_attributes": True}
