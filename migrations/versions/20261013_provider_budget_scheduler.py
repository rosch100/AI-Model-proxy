"""Persist provider-neutral request and token budget windows and leases.

Revision ID: 20261013_budget_sched
Revises: 20261012_catalog_pricing
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261013_budget_sched"
down_revision: str | None = "20261012_catalog_pricing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROVIDERS = "('azure', 'openai', 'openrouter', 'deepseek')"
_SCOPE_KINDS = "('provider', 'profile', 'model', 'organization', 'project')"


def upgrade() -> None:
    """Create PostgreSQL-only scheduler budget and lease tables."""
    _require_postgresql()
    op.create_table(
        "provider_budget_scope_states",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("scope_kind", sa.String(32), nullable=False),
        sa.Column("scope_fingerprint", sa.String(64), nullable=False),
        sa.Column("cooldown_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "provider",
            "scope_kind",
            "scope_fingerprint",
            name="uq_budget_scope_identity",
        ),
        sa.CheckConstraint(f"provider IN {_PROVIDERS}", name="ck_budget_scope_provider"),
        sa.CheckConstraint(
            f"scope_kind IN {_SCOPE_KINDS}", name="ck_budget_scope_kind"
        ),
    )
    op.create_table(
        "provider_budget_policies",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("scope_kind", sa.String(32), nullable=False),
        sa.Column("scope_fingerprint", sa.String(64), nullable=False),
        sa.Column("metric", sa.String(16), nullable=False),
        sa.Column("window_seconds", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "provider",
            "scope_kind",
            "scope_fingerprint",
            "metric",
            name="uq_budget_policy_identity",
        ),
        sa.CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_budget_policy_provider",
        ),
        sa.CheckConstraint(
            "scope_kind IN ('provider', 'profile', 'model', 'organization', 'project')",
            name="ck_budget_policy_scope",
        ),
        sa.CheckConstraint(
            "metric IN ('requests', 'tokens')", name="ck_budget_policy_metric"
        ),
        sa.CheckConstraint("window_seconds > 0", name="ck_budget_policy_window"),
    )
    op.create_table(
        "provider_budget_windows",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("scope_kind", sa.String(32), nullable=False),
        sa.Column("scope_fingerprint", sa.String(64), nullable=False),
        sa.Column("metric", sa.String(16), nullable=False),
        sa.Column("window_seconds", sa.Integer(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("limit_units", sa.Integer(), nullable=False),
        sa.Column("used_units", sa.Integer(), nullable=False),
        sa.Column("reserved_units", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "provider",
            "scope_kind",
            "scope_fingerprint",
            "metric",
            "window_seconds",
            "window_start",
            name="uq_budget_window_identity",
        ),
        sa.CheckConstraint(f"provider IN {_PROVIDERS}", name="ck_budget_window_provider"),
        sa.CheckConstraint(
            f"scope_kind IN {_SCOPE_KINDS}", name="ck_budget_window_scope"
        ),
        sa.CheckConstraint(
            "metric IN ('requests', 'tokens')", name="ck_budget_window_metric"
        ),
        sa.CheckConstraint("window_seconds > 0", name="ck_budget_window_seconds"),
        sa.CheckConstraint("limit_units > 0", name="ck_budget_window_limit"),
        sa.CheckConstraint(
            "used_units >= 0 AND reserved_units >= 0", name="ck_budget_window_units"
        ),
    )
    op.create_index(
        "ix_budget_window_expiry", "provider_budget_windows", ["window_start"]
    )
    op.create_table(
        "provider_budget_leases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(128),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("lease_token", sa.String(36), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("lease_token", name="uq_budget_lease_token"),
        sa.CheckConstraint(f"provider IN {_PROVIDERS}", name="ck_budget_lease_provider"),
        sa.CheckConstraint(
            "status IN ('active', 'completed', 'failed', 'released', 'expired')",
            name="ck_budget_lease_status",
        ),
    )
    op.create_index(
        "ix_budget_lease_tenant_status",
        "provider_budget_leases",
        ["tenant_id", "status"],
    )
    op.create_table(
        "provider_budget_lease_allocations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "lease_id",
            sa.String(36),
            sa.ForeignKey("provider_budget_leases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "window_id",
            sa.Integer(),
            sa.ForeignKey("provider_budget_windows.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("reserved_units", sa.Integer(), nullable=False),
        sa.UniqueConstraint("lease_id", "window_id", name="uq_budget_lease_window"),
        sa.CheckConstraint("reserved_units >= 0", name="ck_budget_allocation_units"),
    )


def downgrade() -> None:
    """Drop PostgreSQL scheduler tables in dependency order."""
    _require_postgresql()
    op.drop_table("provider_budget_lease_allocations")
    op.drop_index("ix_budget_lease_tenant_status", table_name="provider_budget_leases")
    op.drop_table("provider_budget_leases")
    op.drop_index("ix_budget_window_expiry", table_name="provider_budget_windows")
    op.drop_table("provider_budget_windows")
    op.drop_table("provider_budget_policies")
    op.drop_table("provider_budget_scope_states")


def _require_postgresql() -> None:
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
