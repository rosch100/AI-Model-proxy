"""Remove legacy tenant-supplied Azure billing credentials.

Revision ID: 20261004_clear_azure_billing
Revises: 20261003_openrouter_workspace
Create Date: 2026-10-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261004_clear_azure_billing"
down_revision: str | None = "20261003_openrouter_workspace"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Permanently remove obsolete Azure billing credentials from profiles."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.execute(
        "UPDATE provider_profiles SET billing_secret_ciphertext = NULL "
        "WHERE provider = 'azure' AND billing_secret_ciphertext IS NOT NULL"
    )


def downgrade() -> None:
    """Do not recreate credentials that were intentionally erased."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
