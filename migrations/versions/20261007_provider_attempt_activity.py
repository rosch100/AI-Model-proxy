"""Record in-flight and completed provider cascade attempts.

Revision ID: 20261007_provider_attempts
Revises: 20261006_inference_activity
Create Date: 2026-10-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261007_provider_attempts"
down_revision: str | None = "20261006_inference_activity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create provider attempt outcome storage for dashboard hysteresis."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.create_table(
        "provider_attempt_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            sa.String(128),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column(
            "profile_id",
            sa.String(36),
            sa.ForeignKey("provider_profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("inbound_model", sa.String(256), nullable=False),
        sa.Column("routed_model", sa.String(256), nullable=False),
        sa.Column("outcome", sa.String(16), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter')",
            name="ck_provider_attempt_provider",
        ),
        sa.CheckConstraint(
            "outcome IN ('pending', 'success', 'failure', 'aborted')",
            name="ck_provider_attempt_outcome",
        ),
    )
    op.create_index(
        "ix_provider_attempt_tenant_profile_time",
        "provider_attempt_events",
        ["tenant_id", "profile_id", "occurred_at"],
    )
    op.create_index(
        "ix_provider_attempt_occurred_at",
        "provider_attempt_events",
        ["occurred_at"],
    )


def downgrade() -> None:
    """Drop provider attempt outcome storage."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.drop_index(
        "ix_provider_attempt_occurred_at",
        table_name="provider_attempt_events",
    )
    op.drop_index(
        "ix_provider_attempt_tenant_profile_time",
        table_name="provider_attempt_events",
    )
    op.drop_table("provider_attempt_events")
