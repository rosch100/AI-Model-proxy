"""Allow multiple named provider profiles and exclusive scope bindings.

Revision ID: 20261002_provider_accounts_usage
Revises: 20261002_passkeys
Create Date: 2026-10-02
"""

from collections import defaultdict
from collections.abc import Sequence
import unicodedata

import sqlalchemy as sa
from alembic import op

revision: str = "20261002_provider_accounts_usage"
down_revision: str | None = "20261002_passkeys"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROFILE_LABELS = {"azure": "Azure", "openai": "OpenAI", "openrouter": "OpenRouter"}


def upgrade() -> None:
    """Extend profile metadata and lift provider-wide uniqueness safely."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")

    collisions = op.get_bind().execute(
        sa.text(
            "SELECT node_id, array_agg(id ORDER BY id) AS binding_ids "
            "FROM provider_scope_bindings GROUP BY node_id HAVING count(*) > 1 "
            "ORDER BY node_id"
        )
    ).all()
    if collisions:
        detail = "; ".join(
            f"node_id={node_id} binding_ids={','.join(binding_ids)}"
            for node_id, binding_ids in collisions
        )
        raise RuntimeError(
            "Cannot enforce exclusive provider scope bindings; duplicate node "
            f"references detected: {detail}"
        )

    op.add_column(
        "provider_profiles", sa.Column("display_name", sa.String(128), nullable=True)
    )
    op.add_column(
        "provider_profiles", sa.Column("display_name_key", sa.String(384), nullable=True)
    )
    op.add_column(
        "provider_profiles",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "provider_profiles",
        sa.Column(
            "history_generation",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "provider_profiles",
        sa.Column(
            "catalog_generation",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )

    profiles = op.get_bind().execute(
        sa.text(
            "SELECT id, tenant_id, provider FROM provider_profiles "
            "ORDER BY tenant_id, provider, id"
        )
    ).all()
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for profile_id, tenant_id, provider in profiles:
        label = _PROFILE_LABELS.get(provider)
        if label is None:
            label = provider.title()
        counts[(tenant_id, provider)] += 1
        suffix = counts[(tenant_id, provider)]
        display_name = label if suffix == 1 else f"{label} {suffix}"
        op.get_bind().execute(
            sa.text(
            "UPDATE provider_profiles SET display_name = :display_name, "
            "display_name_key = :display_name_key, history_generation = 0, "
            "deleted_at = NULL WHERE id = :profile_id"
            ),
            {
                "display_name": display_name,
                "display_name_key": unicodedata.normalize(
                    "NFKC", display_name
                ).casefold(),
                "profile_id": profile_id,
            },
        )

    op.drop_constraint(
        "uq_profile_tenant_provider", "provider_profiles", type_="unique"
    )
    op.create_check_constraint(
        "ck_profile_history_generation",
        "provider_profiles",
        "history_generation >= 0",
    )
    op.create_index(
        "uq_profile_active_name",
        "provider_profiles",
        ["tenant_id", "provider", "display_name_key"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.drop_constraint(
        "uq_binding_tenant_purpose", "provider_scope_bindings", type_="unique"
    )
    op.create_unique_constraint(
        "uq_binding_node_id", "provider_scope_bindings", ["node_id"]
    )


def downgrade() -> None:
    """Restore provider-wide uniqueness when the stored data permits it."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")

    conflicting_profiles = op.get_bind().execute(
        sa.text(
            "SELECT tenant_id, provider, array_agg(id ORDER BY id) AS profile_ids "
            "FROM provider_profiles GROUP BY tenant_id, provider "
            "HAVING count(*) > 1 ORDER BY tenant_id, provider"
        )
    ).all()
    conflicting_bindings = op.get_bind().execute(
        sa.text(
            "SELECT tenant_id, provider, purpose, array_agg(id ORDER BY id) "
            "AS binding_ids FROM provider_scope_bindings "
            "GROUP BY tenant_id, provider, purpose "
            "HAVING count(*) > 1 ORDER BY tenant_id, provider, purpose"
        )
    ).all()
    if conflicting_profiles or conflicting_bindings:
        conflicts = []
        conflicts.extend(
            f"tenant_id={tenant_id} provider={provider} "
            f"profile_ids={','.join(profile_ids)}"
            for tenant_id, provider, profile_ids in conflicting_profiles
        )
        conflicts.extend(
            f"tenant_id={tenant_id} provider={provider} purpose={purpose} "
            f"binding_ids={','.join(binding_ids)}"
            for tenant_id, provider, purpose, binding_ids in conflicting_bindings
        )
        raise RuntimeError(
            "Cannot restore provider-wide uniqueness; conflicting data exists: "
            + "; ".join(conflicts)
        )

    op.drop_constraint(
        "uq_binding_node_id", "provider_scope_bindings", type_="unique"
    )
    op.create_unique_constraint(
        "uq_binding_tenant_purpose",
        "provider_scope_bindings",
        ["tenant_id", "provider", "purpose"],
    )
    op.drop_index("uq_profile_active_name", table_name="provider_profiles")
    op.drop_constraint(
        "ck_profile_history_generation", "provider_profiles", type_="check"
    )
    op.create_unique_constraint(
        "uq_profile_tenant_provider",
        "provider_profiles",
        ["tenant_id", "provider"],
    )
    op.drop_column("provider_profiles", "catalog_generation")
    op.drop_column("provider_profiles", "history_generation")
    op.drop_column("provider_profiles", "deleted_at")
    op.drop_column("provider_profiles", "display_name_key")
    op.drop_column("provider_profiles", "display_name")
