"""Allow DeepSeek inference profiles and activity without billing scopes.

Revision ID: 20261009_deepseek_provider
Revises: 20261008_provider_breaker
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261009_deepseek_provider"
down_revision: str | None = "20261008_provider_breaker"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Permit DeepSeek inference and circuit records, but not billing scopes."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")

    for table, constraint in (
        ("provider_profiles", "ck_profile_provider"),
        ("inference_activity_events", "ck_inference_activity_provider"),
        ("provider_attempt_events", "ck_provider_attempt_provider"),
        ("provider_circuit_states", "ck_provider_circuit_provider"),
    ):
        op.drop_constraint(constraint, table, type_="check")
        op.create_check_constraint(
            constraint,
            table,
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
        )


def downgrade() -> None:
    """Restore provider constraints only when no DeepSeek records would be stranded."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")

    has_deepseek_records = op.get_bind().execute(
        sa.text(
            "SELECT "
            "EXISTS (SELECT 1 FROM provider_profiles WHERE provider = 'deepseek') "
            "OR EXISTS (SELECT 1 FROM inference_activity_events "
            "WHERE provider = 'deepseek') "
            "OR EXISTS (SELECT 1 FROM provider_attempt_events "
            "WHERE provider = 'deepseek') "
            "OR EXISTS (SELECT 1 FROM provider_circuit_states "
            "WHERE provider = 'deepseek')"
        )
    ).scalar_one()
    if has_deepseek_records:
        raise RuntimeError("Cannot downgrade while DeepSeek records still exist")

    for table, constraint in (
        ("provider_circuit_states", "ck_provider_circuit_provider"),
        ("provider_attempt_events", "ck_provider_attempt_provider"),
        ("inference_activity_events", "ck_inference_activity_provider"),
        ("provider_profiles", "ck_profile_provider"),
    ):
        op.drop_constraint(constraint, table, type_="check")
        op.create_check_constraint(
            constraint,
            table,
            "provider IN ('azure', 'openai', 'openrouter')",
        )
