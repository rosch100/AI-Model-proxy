"""Record per-request inference activity for the tenant overview.

Revision ID: 20261006_inference_activity
Revises: 20261005_provider_route
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261006_inference_activity"
down_revision: str | None = "20261005_provider_route"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the append-only inference activity table used by the dashboard."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.create_table(
        "inference_activity_events",
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
            sa.ForeignKey("provider_profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("inbound_model", sa.String(256), nullable=False),
        sa.Column("routed_model", sa.String(256), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_tokens", sa.Integer(), nullable=True),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter')",
            name="ck_inference_activity_provider",
        ),
        sa.CheckConstraint(
            "(input_tokens IS NULL) = (output_tokens IS NULL) "
            "AND (input_tokens IS NULL) = (total_tokens IS NULL)",
            name="ck_inference_activity_usage_complete",
        ),
    )
    op.create_index(
        "ix_inference_activity_tenant_occurred",
        "inference_activity_events",
        ["tenant_id", "occurred_at"],
    )
    op.create_index(
        "ix_inference_activity_tenant_provider_model",
        "inference_activity_events",
        ["tenant_id", "provider", "inbound_model", "occurred_at"],
    )


def downgrade() -> None:
    """Drop inference activity storage."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.drop_index(
        "ix_inference_activity_tenant_provider_model",
        table_name="inference_activity_events",
    )
    op.drop_index(
        "ix_inference_activity_tenant_occurred",
        table_name="inference_activity_events",
    )
    op.drop_table("inference_activity_events")
