"""Replace the single tenant profile pointer with an ordered provider route.

Revision ID: 20261005_provider_route
Revises: 20261004_clear_azure_billing
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261005_provider_route"
down_revision: str | None = "20261004_clear_azure_billing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Preserve existing nondeleted primary routes, leaving inactive rows NULL."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.execute("LOCK TABLE tenants, provider_profiles IN ACCESS EXCLUSIVE MODE")
    op.add_column(
        "provider_profiles", sa.Column("route_priority", sa.Integer(), nullable=True)
    )
    op.create_check_constraint(
        "ck_profile_route_priority_positive",
        "provider_profiles",
        "route_priority IS NULL OR route_priority > 0",
    )
    op.create_unique_constraint(
        "uq_profile_tenant_route_priority",
        "provider_profiles",
        ["tenant_id", "route_priority"],
    )
    op.execute(
        "UPDATE provider_profiles AS profile SET route_priority = 1 "
        "FROM tenants AS tenant "
        "WHERE tenant.id = profile.tenant_id AND tenant.active_profile_id = profile.id "
        "AND profile.deleted_at IS NULL"
    )
    op.drop_constraint(
        "fk_tenant_active_profile_same_tenant", "tenants", type_="foreignkey"
    )
    op.drop_column("tenants", "active_profile_id")


def downgrade() -> None:
    """Refuse lossy downgrade if a tenant has more than one routed profile."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.execute("LOCK TABLE tenants, provider_profiles IN ACCESS EXCLUSIVE MODE")
    # Also enforce this preflight in offline SQL, before any schema changes.
    op.execute("""DO $$
BEGIN
    IF EXISTS (
        SELECT tenant_id FROM provider_profiles
        WHERE route_priority IS NOT NULL
        GROUP BY tenant_id HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION 'Cannot downgrade provider routing: multiple active profiles would be lost';
    END IF;
END $$""")
    op.add_column(
        "tenants", sa.Column("active_profile_id", sa.String(36), nullable=True)
    )
    op.execute(
        "UPDATE tenants AS tenant SET active_profile_id = profile.id "
        "FROM provider_profiles AS profile "
        "WHERE profile.tenant_id = tenant.id AND profile.route_priority IS NOT NULL"
    )
    op.create_foreign_key(
        "fk_tenant_active_profile_same_tenant",
        "tenants",
        "provider_profiles",
        ["id", "active_profile_id"],
        ["tenant_id", "id"],
    )
    op.drop_constraint(
        "uq_profile_tenant_route_priority", "provider_profiles", type_="unique"
    )
    op.drop_constraint(
        "ck_profile_route_priority_positive", "provider_profiles", type_="check"
    )
    op.drop_column("provider_profiles", "route_priority")
