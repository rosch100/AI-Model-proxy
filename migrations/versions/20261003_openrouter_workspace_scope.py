"""Allow OpenRouter billing bindings to use workspace scopes.

Revision ID: 20261003_openrouter_workspace
Revises: 20261002_passkeys
Create Date: 2026-10-02
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261003_openrouter_workspace"
down_revision: str | None = "20261002_passkeys"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


BINDING_VALIDATOR = """CREATE OR REPLACE FUNCTION validate_provider_scope_binding() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    scope_kind text;
    parent_scope_kind text;
    parent_purpose text;
    parent_node text;
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
        IF scope_kind <> '__OPENROUTER_SCOPE_TYPE__' OR parent_node IS NOT NULL OR NEW.parent_binding_id IS NOT NULL THEN
            RAISE EXCEPTION 'OpenRouter billing must bind an exclusive __OPENROUTER_SCOPE_TYPE__ node';
        END IF;
    ELSE
        RAISE EXCEPTION 'unsupported provider billing/usage binding';
    END IF;
    RETURN NEW;
END;
$$"""


def _replace_binding_validator(openrouter_scope_type: str) -> None:
    """Install the validator for the selected OpenRouter scope contract."""
    op.execute(
        BINDING_VALIDATOR.replace("__OPENROUTER_SCOPE_TYPE__", openrouter_scope_type)
    )


APPEND_ONLY_VALIDATOR = """CREATE OR REPLACE FUNCTION reject_append_only_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_TABLE_NAME = 'provider_scope_nodes'
       AND TG_OP = 'UPDATE'
       AND current_setting('app.openrouter_workspace_rebind', true) = 'on' THEN
        PERFORM 1 FROM provider_scope_bindings
        WHERE node_id = OLD.id AND tenant_id = OLD.tenant_id
          AND provider = 'openrouter' AND purpose = 'billing'
        FOR UPDATE;
        IF OLD.provider = 'openrouter'
           AND OLD.scope_type = 'account'
           AND NEW.scope_type = 'workspace'
           AND NEW.id = OLD.id
           AND NEW.tenant_id = OLD.tenant_id
           AND NEW.provider = OLD.provider
           AND NEW.parent_node_id IS NULL
           AND NEW.created_at = OLD.created_at
           AND NEW.canonical_scope_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
           AND EXISTS (
               SELECT 1 FROM provider_scope_bindings
               WHERE node_id = OLD.id AND tenant_id = OLD.tenant_id
                 AND provider = 'openrouter' AND purpose = 'billing'
                 AND parent_binding_id IS NULL
           )
           AND NOT EXISTS (
               SELECT 1 FROM provider_scope_bindings
               WHERE node_id = OLD.id AND tenant_id = OLD.tenant_id
                 AND provider = 'openrouter' AND purpose <> 'billing'
           )
           AND NOT EXISTS (
               SELECT 1 FROM provider_scope_bindings b
               JOIN cost_refresh_jobs j ON j.binding_id = b.id
               WHERE b.node_id = OLD.id AND b.tenant_id = OLD.tenant_id
                 AND b.provider = 'openrouter' AND b.purpose = 'billing'
                 AND j.status IN ('running', 'success')
           ) THEN
            RETURN NEW;
        END IF;
    END IF;
    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME;
END;
$$"""

STRICT_APPEND_ONLY_VALIDATOR = """CREATE OR REPLACE FUNCTION reject_append_only_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME;
END;
$$"""


def upgrade() -> None:
    """Allow workspace bindings and one constrained operator rebind."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    _replace_binding_validator("workspace")
    op.execute(APPEND_ONLY_VALIDATOR)


def downgrade() -> None:
    """Restore account-only bindings when no workspace bindings remain."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.execute(
        """DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM provider_scope_nodes
            WHERE provider = 'openrouter' AND scope_type = 'workspace'
        ) THEN
            RAISE EXCEPTION 'cannot downgrade while OpenRouter workspace bindings exist';
        END IF;
        END $$"""
    )
    _replace_binding_validator("account")
    op.execute(STRICT_APPEND_ONLY_VALIDATOR)
