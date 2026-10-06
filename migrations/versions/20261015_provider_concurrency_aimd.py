"""Persist AIMD provider concurrency bounds and active request leases.

Revision ID: 20261015_aimd
Revises: 20261014_batch_jobs
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261015_aimd"
down_revision: str | None = "20261014_batch_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create PostgreSQL-only persisted AIMD state and expiring leases."""
    _require_postgresql()
    op.create_table(
        "provider_concurrency_states",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            sa.String(128),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("scope_fingerprint", sa.String(64), nullable=False),
        sa.Column("concurrency_limit", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "tenant_id", "provider", "scope_fingerprint", name="uq_concurrency_scope"
        ),
        sa.CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_concurrency_provider",
        ),
        sa.CheckConstraint(
            "concurrency_limit BETWEEN 1 AND 64", name="ck_concurrency_limit"
        ),
    )
    op.create_table(
        "provider_concurrency_leases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "state_id",
            sa.Integer(),
            sa.ForeignKey("provider_concurrency_states.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("lease_token", sa.String(36), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("lease_token", name="uq_concurrency_lease_token"),
        sa.CheckConstraint(
            "status IN ('active', 'completed', 'released', 'expired')",
            name="ck_concurrency_lease_status",
        ),
    )
    op.create_index(
        "ix_concurrency_lease_state_expiry",
        "provider_concurrency_leases",
        ["state_id", "status", "expires_at"],
    )


def downgrade() -> None:
    """Drop persisted AIMD state and leases."""
    _require_postgresql()
    op.drop_index(
        "ix_concurrency_lease_state_expiry", table_name="provider_concurrency_leases"
    )
    op.drop_table("provider_concurrency_leases")
    op.drop_table("provider_concurrency_states")


def _require_postgresql() -> None:
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
