"""Add tenant routing policy and permit equal-priority profile groups.

Revision ID: 20261016_capacity_route
Revises: 20261015_aimd
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261016_capacity_route"
down_revision: str | None = "20261015_aimd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Persist validated tenant routing choices and allow duplicate priorities."""
    _require_postgresql()
    op.execute("LOCK TABLE tenants, provider_profiles IN ACCESS EXCLUSIVE MODE")
    op.drop_constraint(
        "uq_profile_tenant_route_priority", "provider_profiles", type_="unique"
    )
    op.add_column(
        "provider_concurrency_states",
        sa.Column("minimum_limit", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "provider_concurrency_states",
        sa.Column("maximum_limit", sa.Integer(), nullable=False, server_default="64"),
    )
    op.create_check_constraint(
        "ck_concurrency_bounds",
        "provider_concurrency_states",
        "minimum_limit BETWEEN 1 AND 64 AND maximum_limit BETWEEN 1 AND 64 "
        "AND minimum_limit <= concurrency_limit AND concurrency_limit <= maximum_limit",
    )
    op.add_column(
        "tenants",
        sa.Column(
            "routing_strategy",
            sa.String(32),
            nullable=False,
            server_default="prioritized",
        ),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "routing_load_balancing_method",
            sa.String(32),
            nullable=False,
            server_default="weighted_least_loaded",
        ),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "routing_cost_policy",
            sa.String(32),
            nullable=False,
            server_default="ignore",
        ),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "routing_headroom_weight",
            sa.Float(),
            nullable=False,
            server_default="0.5",
        ),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "routing_max_retry_wait_seconds",
            sa.Integer(),
            nullable=False,
            server_default="30",
        ),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "routing_tie_breaker",
            sa.String(32),
            nullable=False,
            server_default="profile_id",
        ),
    )
    op.create_check_constraint(
        "ck_tenant_routing_strategy",
        "tenants",
        "routing_strategy IN ('prioritized', 'load_balanced')",
    )
    op.create_check_constraint(
        "ck_tenant_routing_balancer",
        "tenants",
        "routing_load_balancing_method = 'weighted_least_loaded'",
    )
    op.create_check_constraint(
        "ck_tenant_routing_cost_policy",
        "tenants",
        "routing_cost_policy IN ('ignore', 'prefer_lower_cost', 'cost_tiers')",
    )
    op.create_check_constraint(
        "ck_tenant_routing_headroom_weight",
        "tenants",
        "routing_headroom_weight BETWEEN 0 AND 1",
    )
    op.create_check_constraint(
        "ck_tenant_routing_retry_wait",
        "tenants",
        "routing_max_retry_wait_seconds BETWEEN 1 AND 300",
    )
    op.create_check_constraint(
        "ck_tenant_routing_tie_breaker",
        "tenants",
        "routing_tie_breaker = 'profile_id'",
    )


def downgrade() -> None:
    """Restore unique route positions only when no equal-priority group exists."""
    _require_postgresql()
    op.execute("LOCK TABLE tenants, provider_profiles IN ACCESS EXCLUSIVE MODE")
    op.execute(
        """DO $$
BEGIN
    IF EXISTS (
        SELECT tenant_id, route_priority
        FROM provider_profiles
        WHERE route_priority IS NOT NULL
        GROUP BY tenant_id, route_priority
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION 'Cannot downgrade capacity routing: duplicate route priorities exist';
    END IF;
END $$"""
    )
    op.drop_constraint(
        "ck_concurrency_bounds", "provider_concurrency_states", type_="check"
    )
    op.drop_column("provider_concurrency_states", "maximum_limit")
    op.drop_column("provider_concurrency_states", "minimum_limit")
    for constraint in (
        "ck_tenant_routing_tie_breaker",
        "ck_tenant_routing_retry_wait",
        "ck_tenant_routing_headroom_weight",
        "ck_tenant_routing_cost_policy",
        "ck_tenant_routing_balancer",
        "ck_tenant_routing_strategy",
    ):
        op.drop_constraint(constraint, "tenants", type_="check")
    for column in (
        "routing_tie_breaker",
        "routing_max_retry_wait_seconds",
        "routing_headroom_weight",
        "routing_cost_policy",
        "routing_load_balancing_method",
        "routing_strategy",
    ):
        op.drop_column("tenants", column)
    op.create_unique_constraint(
        "uq_profile_tenant_route_priority",
        "provider_profiles",
        ["tenant_id", "route_priority"],
    )


def _require_postgresql() -> None:
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
