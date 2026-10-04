"""Operator-only tenant bootstrap and legacy import commands."""

from __future__ import annotations

import os
import secrets
from uuid import UUID, uuid4

import click
from flask import Flask, current_app
from flask.cli import with_appcontext
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash

from app.tenants import hash_api_key, parse_tenants

from .persistence.database import Database
from .persistence.models import (
    AdminAccount,
    AuditEvent,
    CostRefreshJob,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from .persistence.repositories import import_tenants
from .providers.azure_scope import (
    canonical_cognitive_resource_id,
    canonical_resource_group_id,
)
from .providers.cost_jobs import fail_stale_running_jobs


@click.group("tenants")
def tenant_commands() -> None:
    """Create and import database-backed tenants."""


def _database() -> Database:
    database = current_app.extensions.get("database")
    if not isinstance(database, Database):
        raise click.ClickException(
            "Tenant operator commands require a configured database."
        )
    return database


@tenant_commands.command("create")
@click.argument("tenant_id")
@with_appcontext
def create_tenant(tenant_id: str) -> None:
    """Create a tenant and its initial administrator."""
    tenant_id = tenant_id.strip()
    if not tenant_id or len(tenant_id) > 128:
        raise click.ClickException("Tenant ID must contain 1 to 128 characters.")
    username = click.prompt("Admin username").strip()
    if not username:
        raise click.ClickException("Admin username cannot be empty.")
    password = click.prompt(
        "Initial password", hide_input=True, confirmation_prompt=True
    )
    api_key = secrets.token_urlsafe(32)
    tenant = Tenant(
        id=tenant_id,
        api_key_hash=hash_api_key(api_key),
        custom_model_id=f"cursor-{secrets.token_urlsafe(18)}",
    )
    admin = AdminAccount(
        tenant_id=tenant_id,
        username=username,
        password_hash=generate_password_hash(password),
    )
    try:
        with _database().sessions.begin() as session:
            if session.get(Tenant, tenant_id) is not None:
                raise ValueError(f"Tenant {tenant_id!r} already exists")
            session.add_all((tenant, admin))
    except (IntegrityError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Cursor API key: {api_key}")


@tenant_commands.command("import")
@with_appcontext
def import_legacy_tenants() -> None:
    """Atomically import tenant records from the legacy TENANTS environment value."""
    database = _database()
    raw_tenants = os.environ.get("TENANTS", "")
    try:
        tenants = parse_tenants(raw_tenants)
        with database.sessions.begin() as session:
            imported = import_tenants(tenants, session, database.secret_cipher)
    except IntegrityError as exc:
        raise click.ClickException(
            "Tenant import conflicts with existing database records; "
            "no tenants were imported."
        ) from exc
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Imported {imported} tenant(s).")


@tenant_commands.command("bind-billing-scope")
@with_appcontext
def bind_billing_scope() -> None:
    """Bind tenant-exclusive provider billing and usage scopes to one account."""
    tenant_id = click.prompt("Tenant ID")
    profile_id = click.prompt("Provider profile ID")
    database = _database()
    with database.sessions() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == tenant_id,
                ProviderProfile.id == profile_id,
                ProviderProfile.deleted_at.is_(None),
            )
        )
        if profile is None:
            raise click.ClickException(
                "Provider profile was not found for this tenant."
            )
        provider = profile.provider
        if provider not in {"azure", "openai", "openrouter"}:
            raise click.ClickException(
                f"Billing scopes are not supported for {provider!r}."
            )

    actor_id = f"uid:{os.getuid()}"
    scope_values = _prompt_scope_values(provider)
    if not click.confirm(
        "I confirm every entered provider scope is exclusively dedicated to this tenant"
    ):
        raise click.ClickException("Exclusive scope confirmation is required.")

    try:
        with database.sessions.begin() as session:
            tenant = session.get(Tenant, tenant_id)
            if tenant is None:
                raise ValueError(f"Tenant {tenant_id!r} does not exist")
            profile = session.scalar(
                select(ProviderProfile)
                .where(
                    ProviderProfile.tenant_id == tenant_id,
                    ProviderProfile.id == profile_id,
                    ProviderProfile.deleted_at.is_(None),
                )
                .with_for_update()
            )
            if profile is None:
                raise ValueError("Provider profile was not found for this tenant")
            provider = profile.provider
            rebound = provider == "openrouter" and _rebind_openrouter_workspace(
                session, tenant_id, profile.id, scope_values["billing"]
            )
            if not rebound:
                nodes, binding_values = _scope_records(
                    tenant_id, provider, profile.id, scope_values
                )
                for node in nodes:
                    existing_scope = session.scalar(
                        select(ProviderScopeNode).where(
                            ProviderScopeNode.provider == node.provider,
                            ProviderScopeNode.scope_type == node.scope_type,
                            ProviderScopeNode.canonical_scope_id
                            == node.canonical_scope_id,
                        )
                    )
                    if existing_scope is not None:
                        raise ValueError("Provider scope is already bound.")
                session.add_all(nodes)
                session.flush()
                bindings = []
                for purpose, node, parent_purpose in binding_values:
                    parent_binding_id = next(
                        (
                            binding.id
                            for binding in bindings
                            if binding.purpose == parent_purpose
                        ),
                        None,
                    )
                    binding = ProviderScopeBinding(
                        id=str(uuid4()),
                        tenant_id=tenant_id,
                        provider=provider,
                        profile_id=profile.id,
                        purpose=purpose,
                        node_id=node.id,
                        parent_binding_id=parent_binding_id,
                    )
                    bindings.append(binding)
                session.add_all(bindings)
                session.add(
                    AuditEvent(
                        tenant_id=tenant_id,
                        actor_id=actor_id,
                        target=f"{provider}:billing-scope",
                        action="billing_scope.bind",
                        outcome="success",
                        details={
                            "provider": provider,
                            "purposes": [b.purpose for b in bindings],
                        },
                    )
                )
    except IntegrityError as exc:
        raise click.ClickException(
            "Unable to bind provider scope: a database integrity constraint was "
            "violated; check whether the scope is already bound."
        ) from exc
    except ValueError as exc:
        raise click.ClickException(f"Unable to bind provider scope: {exc}") from exc
    action = "updated" if rebound else "bound"
    click.echo(
        f"{provider.capitalize()} billing scope {action} for tenant {tenant_id}."
    )


def _rebind_openrouter_workspace(
    session, tenant_id: str, profile_id: str, workspace_id: str
) -> bool:
    """Replace a legacy account binding with its explicitly confirmed workspace."""
    binding = session.scalar(
        select(ProviderScopeBinding)
        .where(
            ProviderScopeBinding.tenant_id == tenant_id,
            ProviderScopeBinding.provider == "openrouter",
            ProviderScopeBinding.profile_id == profile_id,
            ProviderScopeBinding.purpose == "billing",
        )
        .with_for_update()
    )
    if binding is None:
        return False

    node = session.get(ProviderScopeNode, binding.node_id)
    if node is None or node.scope_type != "account":
        raise ValueError("OpenRouter billing scope is already bound.")
    existing_workspace = session.scalar(
        select(ProviderScopeNode).where(
            ProviderScopeNode.provider == "openrouter",
            ProviderScopeNode.scope_type == "workspace",
            ProviderScopeNode.canonical_scope_id == workspace_id,
        )
    )
    if existing_workspace is not None and existing_workspace.id != node.id:
        raise ValueError("Provider scope is already bound.")
    if fail_stale_running_jobs(session, binding.id):
        raise ValueError(
            "OpenRouter billing scope cannot be rebound while a refresh is running."
        )
    has_successful_job = session.scalar(
        select(CostRefreshJob.id)
        .where(
            CostRefreshJob.binding_id == binding.id,
            CostRefreshJob.status == "success",
        )
        .limit(1)
    )
    if has_successful_job is not None:
        raise ValueError(
            "OpenRouter billing scope cannot be rebound after a successful refresh."
        )

    previous_scope_type = node.scope_type
    previous_scope_id = node.canonical_scope_id
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SET LOCAL app.openrouter_workspace_rebind = 'on'"))
    node.scope_type = "workspace"
    node.canonical_scope_id = workspace_id
    session.add(
        AuditEvent(
            tenant_id=tenant_id,
            actor_id="operator",
            target="openrouter:billing-scope",
            action="billing_scope.rebind",
            outcome="success",
            details={
                "provider": "openrouter",
                "purpose": "billing",
                "previous_scope_type": previous_scope_type,
                "previous_scope_id": previous_scope_id,
                "new_scope_type": "workspace",
                "new_scope_id": workspace_id,
            },
        )
    )
    return True


def _prompt_scope_values(provider: str) -> dict[str, str]:
    if provider == "azure":
        billing_scope = _canonical_azure_resource_group(
            click.prompt("Azure Resource Group ARM ID")
        )
        usage_scope = _canonical_azure_cognitive_resource(
            click.prompt("Azure Cognitive Services resource ARM ID")
        )
        usage_parent_prefix = (
            f"{billing_scope}/providers/microsoft.cognitiveservices/accounts/"
        )
        if not usage_scope.startswith(usage_parent_prefix):
            raise click.ClickException(
                "Azure usage scope must belong to the bound Resource Group."
            )
        return {"billing": billing_scope, "usage": usage_scope}
    if provider == "openai":
        return {
            "organization": _opaque_scope_id(
                click.prompt(
                    "OpenAI organization ID "
                    "(platform.openai.com/settings/organization/general, org-…)"
                )
            ),
            "billing": _opaque_scope_id(
                click.prompt(
                    "OpenAI project ID "
                    "(platform.openai.com/settings/organization/projects, proj_…)"
                )
            ),
        }
    return {
        "billing": _canonical_openrouter_workspace_id(
            click.prompt(
                "OpenRouter workspace UUID "
                "(openrouter.ai Settings → Workspaces, nicht der API-Key)"
            )
        )
    }


def _canonical_openrouter_workspace_id(workspace_id: str) -> str:
    try:
        return str(UUID(workspace_id.strip()))
    except ValueError as exc:
        raise click.ClickException("OpenRouter workspace ID must be a UUID.") from exc


def _canonical_azure_resource_group(scope_id: str) -> str:
    try:
        return canonical_resource_group_id(scope_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _canonical_azure_cognitive_resource(scope_id: str) -> str:
    try:
        return canonical_cognitive_resource_id(scope_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _opaque_scope_id(scope_id: str) -> str:
    normalized = scope_id.strip()
    if (
        not normalized
        or len(normalized) > 1024
        or any(character.isspace() or ord(character) < 32 for character in normalized)
    ):
        raise click.ClickException("Provider scope ID must be a valid opaque ID.")
    return normalized


def _scope_records(tenant_id, provider, profile_id, scope_values):
    if provider == "azure":
        billing_node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id=tenant_id,
            provider=provider,
            scope_type="resource_group",
            canonical_scope_id=scope_values["billing"],
        )
        usage_node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id=tenant_id,
            provider=provider,
            scope_type="cognitive_resource",
            canonical_scope_id=scope_values["usage"],
            parent_node_id=billing_node.id,
        )
        return [billing_node, usage_node], [
            ("billing", billing_node, None),
            ("usage", usage_node, "billing"),
        ]
    if provider == "openai":
        organization_node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id=tenant_id,
            provider=provider,
            scope_type="organization",
            canonical_scope_id=scope_values["organization"],
        )
        project_node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id=tenant_id,
            provider=provider,
            scope_type="project",
            canonical_scope_id=scope_values["billing"],
            parent_node_id=organization_node.id,
        )
        return [organization_node, project_node], [("billing", project_node, None)]
    workspace_node = ProviderScopeNode(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider=provider,
        scope_type="workspace",
        canonical_scope_id=scope_values["billing"],
    )
    return [workspace_node], [("billing", workspace_node, None)]


def register_tenant_commands(app: Flask) -> None:
    """Register tenant operator commands."""
    app.cli.add_command(tenant_commands)


def create_tenant_cli_app() -> Flask:
    """Create a minimal CLI app that bypasses web startup validation for imports."""
    app = Flask("tenant-operator-cli")
    app.config.update(
        DATABASE_URL=os.environ.get("DATABASE_URL"),
        PROVIDER_ENCRYPTION_KEY=os.environ.get("PROVIDER_ENCRYPTION_KEY"),
    )
    try:
        app.extensions["database"] = Database.from_config(app.config)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    register_tenant_commands(app)
    return app
