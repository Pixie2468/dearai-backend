"""remove_username_from_users

Revision ID: b2252b83d45e
Revises: a1141a72c34d
Create Date: 2026-02-04

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2252b83d45e"
down_revision: str | None = "a1141a72c34d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Remove username column from users table."""
    op.drop_index("ix_users_username", table_name="users")
    op.drop_column("users", "username")


def downgrade() -> None:
    """Add username column back to users table."""
    import sqlalchemy as sa

    op.add_column(
        "users",
        sa.Column("username", sa.String(100), unique=True, nullable=False),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
