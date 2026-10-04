"""Persist quota circuit state and half-open probe leases.

Revision ID: 20261008_provider_circuit_breaker
Revises: 20261007_provider_attempts
Create Date: 2026-10-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261008_provider_circuit_breaker"
down_revision: str | None = "20261007_provider_attempts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create secret-free persistent state for provider quota breakers."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.create_table(
        "provider_circuit_states",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            sa.String(128),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("scope_type", sa.String(32), nullable=False),
        sa.Column("scope_fingerprint", sa.String(64), nullable=False),
        sa.Column("failure_category", sa.String(32), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("probe_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_token", sa.String(36), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "scope_fingerprint",
            name="uq_provider_circuit_scope",
        ),
        sa.CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter')",
            name="ck_provider_circuit_provider",
        ),
        sa.CheckConstraint(
            "scope_type IN ('profile', 'organization', 'project')",
            name="ck_provider_circuit_scope_type",
        ),
        sa.CheckConstraint(
            "failure_category = 'quota_exhausted'",
            name="ck_provider_circuit_failure_category",
        ),
        sa.CheckConstraint(
            "failure_count >= 1",
            name="ck_provider_circuit_failure_count",
        ),
        sa.CheckConstraint(
            "(lease_token IS NULL) = (lease_until IS NULL)",
            name="ck_provider_circuit_lease_pair",
        ),
    )
    op.create_index(
        "ix_provider_circuit_tenant_probe",
        "provider_circuit_states",
        ["tenant_id", "probe_at"],
    )


def downgrade() -> None:
    """Drop provider quota circuit state."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.drop_index(
        "ix_provider_circuit_tenant_probe", table_name="provider_circuit_states"
    )
    op.drop_table("provider_circuit_states")
