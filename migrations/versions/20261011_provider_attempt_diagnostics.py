"""Persist sanitized provider-attempt diagnostics.

Revision ID: 20261011_attempt_diagnostics
Revises: 20261010_model_alias_len
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261011_attempt_diagnostics"
down_revision: str | None = "20261010_model_alias_len"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable structured diagnostics to provider attempts."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.add_column(
        "provider_attempt_events",
        sa.Column("failure_details", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Remove provider-attempt diagnostics."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.drop_column("provider_attempt_events", "failure_details")
