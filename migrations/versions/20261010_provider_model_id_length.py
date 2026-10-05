"""Allow account-qualified model IDs in provider activity records.

Revision ID: 20261010_model_alias_len
Revises: 20261009_deepseek_provider
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261010_model_alias_len"
down_revision: str | None = "20261009_deepseek_provider"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Expand inbound model columns for encoded profile aliases."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    for table in ("provider_attempt_events", "inference_activity_events"):
        op.alter_column(
            table,
            "inbound_model",
            existing_type=sa.String(length=256),
            type_=sa.String(length=2048),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Restore the original inbound model column size without truncation."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    connection = op.get_bind()
    for table in ("inference_activity_events", "provider_attempt_events"):
        contains_long_model_ids = connection.execute(
            sa.text(
                f"SELECT EXISTS (SELECT 1 FROM {table} "
                "WHERE char_length(inbound_model) > 256)"
            )
        ).scalar_one()
        if contains_long_model_ids:
            raise RuntimeError(
                f"Cannot downgrade while {table} contains model IDs longer than 256 characters"
            )
    for table in ("inference_activity_events", "provider_attempt_events"):
        op.alter_column(
            table,
            "inbound_model",
            existing_type=sa.String(length=2048),
            type_=sa.String(length=256),
            existing_nullable=False,
        )
