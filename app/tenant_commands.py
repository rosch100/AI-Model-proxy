"""Operator-only tenant bootstrap and legacy import commands."""

from __future__ import annotations

import hashlib
import os
import secrets
import unicodedata
from uuid import UUID, uuid4

import click
from flask import Flask, current_app
from flask.cli import with_appcontext
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash

from app.tenants import parse_tenants

from .persistence.database import Database
from .persistence.models import (
    AdminAccount,
    AuditEvent,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from .persistence.repositories import import_tenants


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
        api_key_hash=hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
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
            nodes, binding_values = _scope_records(
                tenant_id, provider, profile.id, scope_values
            )
            for node in nodes:
                existing_scope = session.scalar(
                    select(ProviderScopeNode).where(
                        ProviderScopeNode.provider == node.provider,
                        ProviderScopeNode.scope_type == node.scope_type,
                        ProviderScopeNode.canonical_scope_id == node.canonical_scope_id,
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
    click.echo(f"{provider.capitalize()} billing scope bound for tenant {tenant_id}.")


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
            "organization": _opaque_scope_id(click.prompt("OpenAI organization ID")),
            "billing": _opaque_scope_id(click.prompt("OpenAI project ID")),
        }
    return {"billing": _opaque_scope_id(click.prompt("OpenRouter account ID"))}


def _canonical_azure_resource_group(scope_id: str) -> str:
    parts = scope_id.strip("/").split("/")
    if (
        len(parts) != 4
        or parts[0].casefold() != "subscriptions"
        or parts[2].casefold() != "resourcegroups"
        or any(not part for part in parts)
    ):
        raise click.ClickException(
            "Azure billing scope must be a Resource Group ARM ID."
        )
    try:
        subscription_id = str(UUID(parts[1]))
    except ValueError as exc:
        raise click.ClickException(
            "Azure scope has an invalid subscription GUID."
        ) from exc
    if "?" in scope_id or "#" in scope_id or "://" in scope_id:
        raise click.ClickException(
            "Azure scope must not contain a URL, query, or fragment."
        )
    resource_group = parts[3]
    if not _is_valid_azure_resource_name(resource_group, max_length=90):
        raise click.ClickException("Azure Resource Group name is invalid.")
    return f"/subscriptions/{subscription_id}/resourcegroups/{resource_group.lower()}"


def _canonical_azure_cognitive_resource(scope_id: str) -> str:
    parts = scope_id.strip("/").split("/")
    if (
        len(parts) != 8
        or parts[0].casefold() != "subscriptions"
        or parts[2].casefold() != "resourcegroups"
        or parts[4].casefold() != "providers"
        or parts[5].casefold() != "microsoft.cognitiveservices"
        or parts[6].casefold() != "accounts"
        or any(not part for part in parts)
    ):
        raise click.ClickException(
            "Azure usage scope must be a Cognitive Services account ARM ID."
        )
    try:
        subscription_id = str(UUID(parts[1]))
    except ValueError as exc:
        raise click.ClickException(
            "Azure scope has an invalid subscription GUID."
        ) from exc
    if "?" in scope_id or "#" in scope_id or "://" in scope_id:
        raise click.ClickException(
            "Azure scope must not contain a URL, query, or fragment."
        )
    resource_group = parts[3]
    account_name = parts[7]
    if not _is_valid_azure_resource_name(resource_group, max_length=90):
        raise click.ClickException("Azure Resource Group name is invalid.")
    if not _is_valid_cognitive_account_name(account_name):
        raise click.ClickException("Azure Cognitive Services account name is invalid.")
    return (
        f"/subscriptions/{subscription_id}/resourcegroups/{resource_group.lower()}"
        f"/providers/microsoft.cognitiveservices/accounts/{account_name.casefold()}"
    )


def _is_valid_azure_resource_name(name: str, *, max_length: int) -> bool:
    return (
        1 <= len(name) <= max_length
        and name[-1] != "."
        and all(
            unicodedata.category(character) in {"Lu", "Ll", "Lt", "Lm", "Lo", "Nd"}
            or character in "_.()-"
            for character in name
        )
    )


def _is_valid_cognitive_account_name(name: str) -> bool:
    return (
        2 <= len(name) <= 64
        and name[0].isalnum()
        and name[-1].isalnum()
        and all(
            character.isascii() and (character.isalnum() or character == "-")
            for character in name
        )
    )


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
    account_node = ProviderScopeNode(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider=provider,
        scope_type="account",
        canonical_scope_id=scope_values["billing"],
    )
    return [account_node], [("billing", account_node, None)]


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
