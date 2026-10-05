"""Persist normalized optional provider catalog pricing.

Revision ID: 20261012_catalog_pricing
Revises: 20261011_attempt_diagnostics
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261012_catalog_pricing"
down_revision: str | None = "20261011_attempt_diagnostics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable per-million-token pricing metadata to catalog rows."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.add_column(
        "provider_catalog_entries",
        sa.Column("input_price_per_1m_tokens", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "provider_catalog_entries",
        sa.Column("output_price_per_1m_tokens", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "provider_catalog_entries",
        sa.Column("cache_price_per_1m_tokens", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "provider_catalog_entries",
        sa.Column("pricing_currency", sa.String(length=8), nullable=True),
    )
    op.add_column(
        "provider_catalog_entries",
        sa.Column("pricing_source", sa.String(length=256), nullable=True),
    )


def downgrade() -> None:
    """Remove optional provider catalog pricing metadata."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.drop_column("provider_catalog_entries", "pricing_source")
    op.drop_column("provider_catalog_entries", "pricing_currency")
    op.drop_column("provider_catalog_entries", "cache_price_per_1m_tokens")
    op.drop_column("provider_catalog_entries", "output_price_per_1m_tokens")
    op.drop_column("provider_catalog_entries", "input_price_per_1m_tokens")
