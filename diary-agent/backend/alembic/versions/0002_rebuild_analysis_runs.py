"""rebuild analysis_runs table to match current model

Revision ID: 20260306_0002
Revises: 20260305_0001
Create Date: 2026-03-06

Drops the legacy analysis_runs table (with old columns like genre, tone,
output_json, etc.) and recreates it with the current schema: story_idea,
request_payload (JSONB), response_payload (JSONB), created_at.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260306_0002"
down_revision: str = "20260305_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("analysis_runs")
    op.create_table(
        "analysis_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("story_idea", sa.Text(), nullable=False),
        sa.Column("request_payload", postgresql.JSONB(), nullable=False),
        sa.Column("response_payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("analysis_runs")
    op.create_table(
        "analysis_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("story_idea", sa.String(2000), nullable=False),
        sa.Column("genre", sa.String(120), nullable=False),
        sa.Column("target_audience", sa.String(120), nullable=False),
        sa.Column("tone", sa.String(80), nullable=False),
        sa.Column("output_json", postgresql.JSONB(), nullable=False),
        sa.Column("score_summary_json", postgresql.JSONB(), nullable=False),
        sa.Column("node_timing_json", postgresql.JSONB(), nullable=False),
        sa.Column("overall_cliffhanger_strength", sa.Float(), nullable=False),
        sa.Column("overall_retention_risk", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
