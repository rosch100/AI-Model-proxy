"""Create the initial tenant-admin persistence schema.

Revision ID: 20261001_initial
Revises:
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APPEND_ONLY_TABLES = (
    "provider_scope_nodes",
    "provider_scope_bindings",
    "cost_refresh_events",
    "cost_usage_records",
    "audit_events",
)


def upgrade() -> None:
    """Create the versioned schema and its database-enforced invariants."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")

    op.create_table(
        "tenants",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("api_key_hash", sa.String(64), nullable=False),
        sa.Column("custom_model_id", sa.String(128), nullable=False),
        sa.Column("active_profile_id", sa.String(36)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("api_key_hash", name="uq_tenants_api_key_hash"),
        sa.UniqueConstraint("custom_model_id", name="uq_tenants_custom_model_id"),
    )
    op.create_table(
        "login_rate_limits",
        sa.Column("subject_hash", sa.String(64), primary_key=True),
        sa.Column("failures", sa.Integer(), nullable=False),
        sa.Column("window_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("locked_until", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "admin_accounts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(128), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("username", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("tenant_id", "username", name="uq_admin_tenant_username"),
    )
    op.create_table(
        "provider_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(128), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("inference_secret_ciphertext", sa.String()),
        sa.Column("billing_secret_ciphertext", sa.String()),
        sa.Column("default_model", sa.String(256)),
        sa.Column("catalog_refreshed_at", sa.DateTime(timezone=True)),
        sa.Column("catalog_error", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("tenant_id", "provider", name="uq_profile_tenant_provider"),
        sa.UniqueConstraint("tenant_id", "provider", "id", name="uq_profile_tenant_provider_id"),
        sa.UniqueConstraint("tenant_id", "id", name="uq_profile_tenant_id"),
        sa.CheckConstraint("provider IN ('azure', 'openai', 'openrouter')", name="ck_profile_provider"),
    )
    op.create_table(
        "provider_catalog_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("profile_id", sa.String(36), sa.ForeignKey("provider_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_id", sa.String(256), nullable=False),
        sa.Column("deployment_id", sa.String(256)),
        sa.Column("source", sa.String(32), nullable=False),
        sa.UniqueConstraint("profile_id", "model_id", name="uq_catalog_profile_model"),
    )
    op.create_table(
        "provider_scope_nodes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(128), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("scope_type", sa.String(32), nullable=False),
        sa.Column("canonical_scope_id", sa.String(1024), nullable=False),
        sa.Column("parent_node_id", sa.String(36)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("provider", "scope_type", "canonical_scope_id", name="uq_scope_global_identity"),
        sa.UniqueConstraint("tenant_id", "provider", "id", name="uq_scope_tenant_provider_id"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "provider", "parent_node_id"],
            ["provider_scope_nodes.tenant_id", "provider_scope_nodes.provider", "provider_scope_nodes.id"],
            name="fk_scope_parent_same_tenant_provider",
        ),
        sa.CheckConstraint("provider IN ('azure', 'openai', 'openrouter')", name="ck_scope_provider"),
    )
    op.create_table(
        "provider_scope_bindings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(128), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("purpose", sa.String(16), nullable=False),
        sa.Column("node_id", sa.String(36), nullable=False),
        sa.Column("parent_binding_id", sa.String(36)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("profile_id", "provider", "purpose", name="uq_binding_profile_purpose"),
        sa.UniqueConstraint("tenant_id", "provider", "purpose", name="uq_binding_tenant_purpose"),
        sa.UniqueConstraint("tenant_id", "provider", "id", name="uq_binding_tenant_provider_id"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "provider", "profile_id"],
            ["provider_profiles.tenant_id", "provider_profiles.provider", "provider_profiles.id"],
            name="fk_binding_profile_same_tenant_provider",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "provider", "node_id"],
            ["provider_scope_nodes.tenant_id", "provider_scope_nodes.provider", "provider_scope_nodes.id"],
            name="fk_binding_node_same_tenant_provider",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "provider", "parent_binding_id"],
            ["provider_scope_bindings.tenant_id", "provider_scope_bindings.provider", "provider_scope_bindings.id"],
            name="fk_binding_parent_same_tenant_provider",
        ),
        sa.CheckConstraint("purpose IN ('billing', 'usage')", name="ck_binding_purpose"),
    )
    op.create_table(
        "admin_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("account_id", sa.Integer(), sa.ForeignKey("admin_accounts.id"), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(128), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("target", sa.String(256), nullable=False),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
    )
    op.create_table(
        "cost_refresh_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(128), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("binding_id", sa.String(36), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_api", sa.String(512), nullable=False),
        sa.Column("operation_key", sa.String(128), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("retry_at", sa.DateTime(timezone=True)),
        sa.Column("limitation", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("tenant_id", "operation_key", name="uq_refresh_tenant_operation"),
        sa.UniqueConstraint(
            "id",
            "tenant_id",
            "provider",
            "binding_id",
            name="uq_refresh_job_tenant_provider_binding",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "provider", "binding_id"],
            ["provider_scope_bindings.tenant_id", "provider_scope_bindings.provider", "provider_scope_bindings.id"],
            name="fk_refresh_binding_tenant_provider",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'retry_wait', 'success', 'unavailable', 'failed')",
            name="ck_refresh_status",
        ),
        sa.CheckConstraint(
            "(status = 'retry_wait' AND retry_at IS NOT NULL) OR "
            "(status != 'retry_wait' AND retry_at IS NULL)",
            name="ck_refresh_retry_at",
        ),
    )
    op.create_index("ix_refresh_job_tenant_created", "cost_refresh_jobs", ["tenant_id", "created_at"])
    op.create_table(
        "cost_refresh_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("cost_refresh_jobs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("previous_status", sa.String(16)),
        sa.Column("new_status", sa.String(16), nullable=False),
        sa.Column("reason", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "cost_usage_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("cost_refresh_jobs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("tenant_id", sa.String(128), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("binding_id", sa.String(36), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("metric", sa.String(128), nullable=False),
        sa.Column("value", sa.Numeric(28, 10), nullable=False),
        sa.Column("unit", sa.String(64), nullable=False),
        sa.Column("currency", sa.String(3)),
        sa.Column("bucket_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bucket_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("granularity", sa.String(64), nullable=False),
        sa.Column("dimensions", sa.JSON(), nullable=False),
        sa.Column("price_source", sa.String(512)),
        sa.Column("price_version", sa.String(128)),
        sa.Column("usage_source", sa.String(512)),
        sa.Column("usage_bucket_start", sa.DateTime(timezone=True)),
        sa.Column("formula_parameters", sa.JSON()),
        sa.Column("model_key", sa.String(256)),
        sa.Column("region_key", sa.String(128)),
        sa.Column("deployment_key", sa.String(256)),
        sa.ForeignKeyConstraint(
            ["job_id", "tenant_id", "provider", "binding_id"],
            [
                "cost_refresh_jobs.id",
                "cost_refresh_jobs.tenant_id",
                "cost_refresh_jobs.provider",
                "cost_refresh_jobs.binding_id",
            ],
            name="fk_cost_job_tenant_provider_binding",
        ),
        sa.CheckConstraint("kind IN ('actual', 'usage', 'estimate')", name="ck_cost_kind"),
        sa.CheckConstraint(
            "(kind IN ('actual', 'estimate') AND currency IS NOT NULL) OR "
            "(kind = 'usage' AND currency IS NULL)",
            name="ck_cost_currency_by_kind",
        ),
        sa.CheckConstraint(
            "kind != 'estimate' OR (price_source IS NOT NULL AND price_version IS NOT NULL "
            "AND usage_source IS NOT NULL AND usage_bucket_start IS NOT NULL "
            "AND formula_parameters IS NOT NULL AND model_key IS NOT NULL "
            "AND region_key IS NOT NULL AND deployment_key IS NOT NULL)",
            name="ck_estimate_provenance",
        ),
    )
    op.create_foreign_key(
        "fk_tenant_active_profile_same_tenant",
        "tenants",
        "provider_profiles",
        ["id", "active_profile_id"],
        ["tenant_id", "id"],
    )
    op.execute(
        """CREATE FUNCTION reject_append_only_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME;
        END;
        $$"""
    )
    for table_name in _APPEND_ONLY_TABLES:
        op.execute(
            f"CREATE TRIGGER trg_{table_name}_append_only "
            f"BEFORE UPDATE OR DELETE ON {table_name} "
            "FOR EACH ROW EXECUTE FUNCTION reject_append_only_mutation()"
        )
    op.execute(
        """CREATE FUNCTION require_successful_cost_job() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE job_status text;
        BEGIN
            SELECT status INTO job_status
            FROM cost_refresh_jobs
            WHERE id = NEW.job_id
            FOR UPDATE;
            IF job_status IS DISTINCT FROM 'success' THEN
                RAISE EXCEPTION 'cost usage records require a successful refresh job';
            END IF;
            RETURN NEW;
        END;
        $$"""
    )
    op.execute(
        """CREATE TRIGGER trg_cost_usage_success_only
        BEFORE INSERT ON cost_usage_records
        FOR EACH ROW EXECUTE FUNCTION require_successful_cost_job()"""
    )
    op.execute(
        """CREATE FUNCTION prevent_unsuccessful_job_with_cost_records() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.status <> 'success' AND EXISTS (
                SELECT 1 FROM cost_usage_records WHERE job_id = OLD.id
            ) THEN
                RAISE EXCEPTION 'cost refresh jobs with usage records must remain successful';
            END IF;
            RETURN NEW;
        END;
        $$"""
    )
    op.execute(
        """CREATE TRIGGER trg_cost_job_status_with_records
        BEFORE UPDATE OF status ON cost_refresh_jobs
        FOR EACH ROW EXECUTE FUNCTION prevent_unsuccessful_job_with_cost_records()"""
    )
    op.execute(
        """CREATE FUNCTION enforce_immutable_custom_model_id() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.custom_model_id IS DISTINCT FROM OLD.custom_model_id THEN
                RAISE EXCEPTION 'tenant custom_model_id is immutable';
            END IF;
            RETURN NEW;
        END;
        $$"""
    )
    op.execute(
        """CREATE TRIGGER trg_tenant_custom_model_immutable
        BEFORE UPDATE OF custom_model_id ON tenants
        FOR EACH ROW EXECUTE FUNCTION enforce_immutable_custom_model_id()"""
    )
    op.execute(
        """CREATE FUNCTION validate_provider_scope_binding() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE scope_kind text;
        DECLARE parent_scope_kind text;
        DECLARE parent_purpose text;
        DECLARE parent_node text;
        BEGIN
            SELECT scope_type, parent_node_id INTO scope_kind, parent_node
            FROM provider_scope_nodes
            WHERE id = NEW.node_id AND tenant_id = NEW.tenant_id AND provider = NEW.provider;
            IF NOT FOUND THEN
                RAISE EXCEPTION 'scope binding node does not match tenant/provider';
            END IF;
            IF NEW.provider = 'azure' AND NEW.purpose = 'billing' THEN
                IF scope_kind <> 'resource_group' OR NEW.parent_binding_id IS NOT NULL THEN
                    RAISE EXCEPTION 'Azure billing must bind a root resource_group node';
                END IF;
            ELSIF NEW.provider = 'azure' AND NEW.purpose = 'usage' THEN
                SELECT b.purpose, n.scope_type INTO parent_purpose, parent_scope_kind
                FROM provider_scope_bindings b
                JOIN provider_scope_nodes n ON n.id = b.node_id
                WHERE b.id = NEW.parent_binding_id
                  AND b.tenant_id = NEW.tenant_id AND b.provider = NEW.provider;
                IF scope_kind IS DISTINCT FROM 'cognitive_resource'
                   OR NEW.parent_binding_id IS NULL
                   OR parent_node IS NULL
                   OR parent_purpose IS DISTINCT FROM 'billing'
                   OR parent_scope_kind IS DISTINCT FROM 'resource_group'
                   OR parent_node IS DISTINCT FROM (
                       SELECT node_id FROM provider_scope_bindings
                       WHERE id = NEW.parent_binding_id
                   ) THEN
                    RAISE EXCEPTION 'Azure usage must be a child of its resource_group billing binding';
                END IF;
            ELSIF NEW.provider = 'openai' AND NEW.purpose = 'billing' THEN
                IF scope_kind <> 'project' OR parent_node IS NULL OR NEW.parent_binding_id IS NOT NULL THEN
                    RAISE EXCEPTION 'OpenAI billing must bind a project under an organization';
                END IF;
                SELECT scope_type INTO parent_scope_kind
                FROM provider_scope_nodes WHERE id = parent_node AND tenant_id = NEW.tenant_id;
                IF parent_scope_kind <> 'organization' THEN
                    RAISE EXCEPTION 'OpenAI project must be a child of an organization';
                END IF;
            ELSIF NEW.provider = 'openrouter' AND NEW.purpose = 'billing' THEN
                IF scope_kind <> 'account' OR parent_node IS NOT NULL OR NEW.parent_binding_id IS NOT NULL THEN
                    RAISE EXCEPTION 'OpenRouter billing must bind an exclusive account node';
                END IF;
            ELSE
                RAISE EXCEPTION 'unsupported provider billing/usage binding';
            END IF;
            RETURN NEW;
        END;
        $$"""
    )
    op.execute(
        """CREATE TRIGGER trg_validate_provider_scope_binding
        BEFORE INSERT ON provider_scope_bindings
        FOR EACH ROW EXECUTE FUNCTION validate_provider_scope_binding()"""
    )


def downgrade() -> None:
    """Drop the exact schema objects created by this revision."""
    op.execute("DROP TRIGGER IF EXISTS trg_validate_provider_scope_binding ON provider_scope_bindings")
    op.execute("DROP FUNCTION IF EXISTS validate_provider_scope_binding()")
    op.execute("DROP TRIGGER IF EXISTS trg_tenant_custom_model_immutable ON tenants")
    op.execute("DROP FUNCTION IF EXISTS enforce_immutable_custom_model_id()")
    op.execute("DROP TRIGGER IF EXISTS trg_cost_usage_success_only ON cost_usage_records")
    op.execute("DROP FUNCTION IF EXISTS require_successful_cost_job()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_cost_job_status_with_records ON cost_refresh_jobs"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS prevent_unsuccessful_job_with_cost_records()"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_cost_job_status_with_records ON cost_refresh_jobs"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS prevent_unsuccessful_job_with_cost_records()"
    )
    for table_name in reversed(_APPEND_ONLY_TABLES):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table_name}_append_only ON {table_name}")
    op.execute("DROP FUNCTION IF EXISTS reject_append_only_mutation()")
    op.drop_constraint(
        "fk_tenant_active_profile_same_tenant", "tenants", type_="foreignkey"
    )
    for table_name in (
        "cost_usage_records",
        "cost_refresh_events",
        "cost_refresh_jobs",
        "audit_events",
        "admin_sessions",
        "provider_scope_bindings",
        "provider_scope_nodes",
        "provider_catalog_entries",
        "provider_profiles",
        "admin_accounts",
        "login_rate_limits",
        "tenants",
    ):
        op.drop_table(table_name)
