"""Allow DeepSeek inference profiles and activity without billing scopes.

Revision ID: 20261007_deepseek_provider
Revises: 20261006_inference_activity
Create Date: 2026-10-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261007_deepseek_provider"
down_revision: str | None = "20261006_inference_activity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Permit DeepSeek profiles and activity; keep scope constraints unchanged."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.drop_constraint("ck_profile_provider", "provider_profiles", type_="check")
    op.create_check_constraint(
        "ck_profile_provider",
        "provider_profiles",
        "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
    )
    op.drop_constraint(
        "ck_inference_activity_provider",
        "inference_activity_events",
        type_="check",
    )
    op.create_check_constraint(
        "ck_inference_activity_provider",
        "inference_activity_events",
        "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
    )


def downgrade() -> None:
    """Restore provider constraints after refusing to strand DeepSeek rows."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.drop_constraint(
        "ck_inference_activity_provider",
        "inference_activity_events",
        type_="check",
    )
    op.create_check_constraint(
        "ck_inference_activity_provider",
        "inference_activity_events",
        "provider IN ('azure', 'openai', 'openrouter')",
    )
    op.drop_constraint("ck_profile_provider", "provider_profiles", type_="check")
    op.create_check_constraint(
        "ck_profile_provider",
        "provider_profiles",
        "provider IN ('azure', 'openai', 'openrouter')",
    )
