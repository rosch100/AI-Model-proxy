"""Tenant-owned administrator mutations with audit records."""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.admin.passwords import hash_admin_password, verify_admin_password
from app.persistence.admin_auth import revoke_account_sessions
from app.persistence.models import (
    AdminAccount,
    AuditEvent,
    BatchJob,
    ProviderBudgetPolicy,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
    provider_profile_name_key,
)
from app.persistence.repositories import validate_routed_profile
from app.persistence.secrets import SecretCipher
from app.providers.azure_scope import canonical_cost_scopes
from app.providers.azure_url import validate_azure_base_url
from app.providers.catalog import selectable_catalog_models
from app.providers.scheduler_config import parse_scheduler_limits
from app.tenants import hash_api_key


def rotate_api_key(session: Session, tenant: Tenant, actor_id: str) -> str:
    """Replace the Cursor API key digest and return the plaintext key once."""
    api_key = secrets.token_urlsafe(32)
    tenant.api_key_hash = hash_api_key(api_key)
    session.add(
        AuditEvent(
            tenant_id=tenant.id,
            actor_id=actor_id,
            target="tenant:api-key",
            action="api_key.rotate",
            outcome="success",
            details={},
        )
    )
    return api_key


def change_admin_password(
    session: Session,
    account: AdminAccount,
    current_password: str,
    new_password: str,
    current_session_id: str,
) -> None:
    """Replace the password hash and revoke every other live session."""
    if not verify_admin_password(account.password_hash, current_password):
        raise ValueError("current password is incorrect")
    account.password_hash = hash_admin_password(new_password)
    revoke_account_sessions(session, account.id, except_session_id=current_session_id)
    session.add(
        AuditEvent(
            tenant_id=account.tenant_id,
            actor_id=account.username,
            target="admin:password",
            action="password.change",
            outcome="success",
            details={},
        )
    )


def upsert_provider_profile(
    session: Session,
    cipher: SecretCipher,
    tenant_id: str,
    provider: str,
    settings: dict[str, object],
    default_model: str,
    inference_secret: str | None,
    actor_id: str,
) -> ProviderProfile:
    """Save a legacy provider selection only when its profile is unambiguous."""
    _lock_tenant(session, tenant_id)
    profiles = list(
        session.scalars(
            select(ProviderProfile)
            .where(
                ProviderProfile.tenant_id == tenant_id,
                ProviderProfile.provider == provider,
                ProviderProfile.deleted_at.is_(None),
            )
            .execution_options(populate_existing=True)
        )
    )
    if len(profiles) > 1:
        raise ValueError(
            "Provider account selection is ambiguous; specify a profile ID"
        )
    if not profiles:
        profile = create_provider_profile(
            session,
            cipher,
            tenant_id,
            provider,
            provider,
            settings,
            default_model,
            inference_secret,
            actor_id,
        )
    else:
        profile = profiles[0]
        if profile.display_name is None:
            raise ValueError("Provider account name is required")
        profile = update_provider_profile(
            session,
            cipher,
            tenant_id,
            profile.id,
            profile.display_name,
            settings,
            default_model,
            inference_secret,
            actor_id,
        )
    session.add(
        AuditEvent(
            tenant_id=tenant_id,
            actor_id=actor_id,
            target=f"profile:{provider}",
            action="profile.save",
            outcome="success",
            details={"provider": provider},
        )
    )
    return profile


def _validate_provider_settings(
    provider: str, settings: dict[str, object]
) -> dict[str, object]:
    """Validate scheduler configuration before persisting profile settings."""
    if not isinstance(settings, dict):
        raise ValueError("Provider settings must be an object")
    normalized = dict(settings)
    if "scheduler_limits" in normalized:
        normalized["scheduler_limits"] = parse_scheduler_limits(
            normalized["scheduler_limits"]
        )
    estimate = normalized.get("token_reservation_estimate")
    if estimate is not None and (
        isinstance(estimate, bool) or not isinstance(estimate, int) or estimate <= 0
    ):
        raise ValueError("token_reservation_estimate must be a positive integer")
    if estimate is None and any(
        policy["metric"] == "tokens"
        for policy in normalized.get("scheduler_limits", [])
    ):
        raise ValueError("Token scheduler limits require token_reservation_estimate")
    if provider not in {"azure", "openai", "openrouter", "deepseek"}:
        raise ValueError("Unsupported provider")
    return normalized


def _validate_profile_name(display_name: str) -> str:
    """Normalize and validate the account name shown in admin pages."""
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError("Account name is required")
    normalized = display_name.strip()
    if len(normalized) > 128:
        raise ValueError("Account name must be at most 128 characters")
    return normalized


def _ensure_profile_name_available(
    session: Session,
    tenant_id: str,
    provider: str,
    display_name: str,
    *,
    exclude_profile_id: str | None = None,
) -> None:
    """Reject case-insensitive active account-name collisions."""
    query = select(ProviderProfile.id).where(
        ProviderProfile.tenant_id == tenant_id,
        ProviderProfile.provider == provider,
        ProviderProfile.deleted_at.is_(None),
        ProviderProfile.display_name_key == provider_profile_name_key(display_name),
    )
    if exclude_profile_id is not None:
        query = query.where(ProviderProfile.id != exclude_profile_id)
    if session.scalar(query) is not None:
        raise ValueError("An account with this name already exists")


def create_provider_profile(
    session: Session,
    cipher: SecretCipher,
    tenant_id: str,
    provider: str,
    display_name: str,
    settings: dict[str, object],
    default_model: str,
    inference_secret: str | None,
    actor_id: str,
) -> ProviderProfile:
    """Create a tenant-owned account outside the default route."""
    if provider not in {"azure", "openai", "openrouter", "deepseek"}:
        raise ValueError("Unsupported provider")
    if session.get(Tenant, tenant_id) is None:
        raise LookupError("Tenant was not found")
    name = _validate_profile_name(display_name)
    _ensure_profile_name_available(session, tenant_id, provider, name)
    normalized_settings = _validate_provider_settings(provider, settings)
    if provider == "azure":
        normalized_settings["base_url"] = validate_azure_base_url(
            normalized_settings.get("base_url")
        )
    if not isinstance(default_model, str):
        raise ValueError("Default model must be a string")
    normalized_default_model = default_model.strip() or None

    profile = ProviderProfile(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider=provider,
        display_name=name,
        settings=normalized_settings,
        default_model=normalized_default_model,
        inference_secret_ciphertext=(
            cipher.encrypt(inference_secret) if inference_secret else None
        ),
    )
    session.add(profile)
    session.flush()
    session.add(
        AuditEvent(
            tenant_id=tenant_id,
            actor_id=actor_id,
            target=f"profile:{profile.id}",
            action="profile.create",
            outcome="success",
            details={"provider": provider, "profile_id": profile.id},
        )
    )
    return profile


def update_provider_profile(
    session: Session,
    cipher: SecretCipher,
    tenant_id: str,
    profile_id: str,
    display_name: str,
    settings: dict[str, object],
    default_model: str,
    inference_secret: str | None,
    actor_id: str,
) -> ProviderProfile:
    """Update one account without changing its provider or blanking its secret."""
    _lock_tenant(session, tenant_id)
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.id == profile_id,
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.deleted_at.is_(None),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if profile is None:
        raise LookupError("Provider account was not found")
    name = _validate_profile_name(display_name)
    _ensure_profile_name_available(
        session,
        tenant_id,
        profile.provider,
        name,
        exclude_profile_id=profile.id,
    )
    if not isinstance(settings, dict):
        raise ValueError("Provider settings must be an object")
    merged_settings = dict(settings)
    for key in ("scheduler_limits", "token_reservation_estimate"):
        if key not in merged_settings and key in profile.settings:
            merged_settings[key] = profile.settings[key]
    normalized_settings = _validate_provider_settings(profile.provider, merged_settings)
    scheduler_limits = normalized_settings.get("scheduler_limits", [])
    if scheduler_limits:
        configured_scopes: dict[str, set[str]] = {
            "provider": {profile.provider},
            "profile": {profile.id},
            "organization": set(),
            "project": set(),
            "model": set(),
        }
        for kind in ("organization", "project"):
            scope_id = normalized_settings.get(kind)
            if isinstance(scope_id, str) and scope_id.strip():
                configured_scopes[kind].add(scope_id)
        catalog_models = session.scalars(
            select(ProviderCatalogEntry.model_id).where(
                ProviderCatalogEntry.profile_id == profile.id
            )
        )
        configured_scopes["model"].update(catalog_models)
        for policy_config in scheduler_limits:
            scope_kind = policy_config["scope_kind"]
            configured_ids = configured_scopes[scope_kind]
            explicit_scope_id = policy_config.get("scope_id")
            if explicit_scope_id is not None:
                configured_ids = {explicit_scope_id}
            for scope_id in configured_ids:
                fingerprint = cipher.provider_budget_fingerprint(
                    profile.provider,
                    scope_kind,
                    scope_id,
                    tenant_id=tenant_id,
                )
                registered_policy = session.scalar(
                    select(ProviderBudgetPolicy).where(
                        ProviderBudgetPolicy.provider == profile.provider,
                        ProviderBudgetPolicy.scope_kind == scope_kind,
                        ProviderBudgetPolicy.scope_fingerprint == fingerprint,
                        ProviderBudgetPolicy.metric == policy_config["metric"],
                    )
                )
                if (
                    registered_policy is not None
                    and registered_policy.window_seconds
                    != policy_config["window_seconds"]
                ):
                    raise ValueError(
                        "window_seconds for existing budget policies cannot be changed"
                    )
    azure_endpoint_changed = False
    if profile.provider == "azure":
        normalized_settings["base_url"] = validate_azure_base_url(
            normalized_settings.get("base_url")
        )
        previous_base_url = profile.settings.get("base_url")
        azure_endpoint_changed = normalized_settings["base_url"] != previous_base_url
        if not azure_endpoint_changed:
            existing_deployments = profile.settings.get("model_deployments")
            if existing_deployments is not None:
                normalized_settings["model_deployments"] = existing_deployments
    if not isinstance(default_model, str):
        raise ValueError("Default model must be a string")
    normalized_default_model = default_model.strip() or None
    if not normalized_default_model and profile.default_model:
        raise ValueError("Default model is required")

    if (
        profile.route_priority is not None
        and not azure_endpoint_changed
        and normalized_default_model is not None
    ):
        catalog = session.scalars(
            select(ProviderCatalogEntry).where(
                ProviderCatalogEntry.profile_id == profile.id
            )
        )
        selectable_models = {
            model_id
            for model_id, _deployment_id in selectable_catalog_models(
                profile.provider,
                [(entry.model_id, entry.deployment_id) for entry in catalog],
            )
        }
        if (
            normalized_default_model != profile.default_model
            and normalized_default_model not in selectable_models
        ):
            raise ValueError(
                f"The default model for {profile.display_name} is not in its available catalog"
            )
        candidate = ProviderProfile(
            provider=profile.provider,
            display_name=name,
            settings=normalized_settings,
            default_model=normalized_default_model,
            inference_secret_ciphertext=(
                cipher.encrypt(inference_secret)
                if inference_secret
                else profile.inference_secret_ciphertext
            ),
        )
        validate_routed_profile(
            candidate, [(entry.model_id, entry.deployment_id) for entry in catalog]
        )

    profile.display_name = name
    profile.settings = normalized_settings
    profile.default_model = None if azure_endpoint_changed else normalized_default_model
    if azure_endpoint_changed:
        profile.catalog_refreshed_at = None
        profile.catalog_error = None
        _remove_from_route(session, profile, actor_id, "azure_endpoint_changed")
        session.execute(
            delete(ProviderCatalogEntry).where(
                ProviderCatalogEntry.profile_id == profile.id
            )
        )
    if inference_secret:
        profile.inference_secret_ciphertext = cipher.encrypt(inference_secret)
    session.add(
        AuditEvent(
            tenant_id=tenant_id,
            actor_id=actor_id,
            target=f"profile:{profile.id}",
            action="profile.update",
            outcome="success",
            details={"provider": profile.provider, "profile_id": profile.id},
        )
    )
    return profile


def delete_provider_profile(
    session: Session, tenant_id: str, profile_id: str, actor_id: str
) -> None:
    """Soft-delete one tenant-owned profile while preserving audit history."""
    _lock_tenant(session, tenant_id)
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.id == profile_id,
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.deleted_at.is_(None),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if profile is None:
        raise LookupError("Provider account was not found")
    has_scope_bindings = session.scalar(
        select(ProviderScopeBinding.id)
        .where(ProviderScopeBinding.profile_id == profile.id)
        .limit(1)
    )
    if has_scope_bindings is not None:
        raise ValueError("Account cannot be deleted while scopes are bound")
    has_retained_batch = session.scalar(
        select(BatchJob.id)
        .where(BatchJob.tenant_id == tenant_id, BatchJob.profile_id == profile.id)
        .limit(1)
    )
    if has_retained_batch is not None:
        raise ValueError("Account cannot be deleted while batch jobs are retained")
    profile.inference_secret_ciphertext = None
    profile.billing_secret_ciphertext = None
    profile.deleted_at = datetime.now(timezone.utc)
    _remove_from_route(session, profile, actor_id, "profile_deleted")
    session.add(
        AuditEvent(
            tenant_id=tenant_id,
            actor_id=actor_id,
            target=f"profile:{profile.id}",
            action="profile.delete",
            outcome="success",
            details={"provider": profile.provider, "profile_id": profile.id},
        )
    )


def _lock_tenant(session: Session, tenant_id: str) -> Tenant:
    """Serialize every route mutation on the stable tenant row."""
    tenant = session.scalar(
        select(Tenant)
        .where(Tenant.id == tenant_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if tenant is None:
        raise LookupError("Tenant was not found")
    return tenant


def _routed_profiles(session: Session, tenant_id: str) -> list[ProviderProfile]:
    """Read the current route after acquiring the tenant lock."""
    return list(
        session.scalars(
            select(ProviderProfile)
            .where(
                ProviderProfile.tenant_id == tenant_id,
                ProviderProfile.route_priority.is_not(None),
            )
            .order_by(ProviderProfile.route_priority)
            .execution_options(populate_existing=True)
        )
    )


def _write_route(
    session: Session,
    previous: list[ProviderProfile],
    ordered: list[ProviderProfile],
) -> None:
    """Renumber atomically using a flushed NULL phase to avoid unique collisions."""
    for profile in previous:
        profile.route_priority = None
    session.flush()
    for priority, profile in enumerate(ordered, start=1):
        profile.route_priority = priority
    session.flush()


def _remove_from_route(
    session: Session,
    profile: ProviderProfile,
    actor_id: str,
    reason: str,
) -> None:
    """Remove one member and compact the route under an already-held tenant lock."""
    if profile.route_priority is None:
        return
    previous = _routed_profiles(session, profile.tenant_id)
    _write_route(
        session, previous, [item for item in previous if item.id != profile.id]
    )
    session.add(
        AuditEvent(
            tenant_id=profile.tenant_id,
            actor_id=actor_id,
            target=f"profile:{profile.id}",
            action="profile.deactivate",
            outcome="success",
            details={"profile_id": profile.id, "reason": reason},
        )
    )


def deactivate_provider_profile(
    session: Session, tenant: Tenant, profile_id: str, actor_id: str
) -> ProviderProfile:
    """Remove one tenant-owned account from routing without changing its bindings."""
    _lock_tenant(session, tenant.id)
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.id == profile_id,
            ProviderProfile.tenant_id == tenant.id,
            ProviderProfile.deleted_at.is_(None),
        )
        .execution_options(populate_existing=True)
    )
    if profile is None:
        raise LookupError("Provider account was not found")
    _remove_from_route(session, profile, actor_id, "manual")
    return profile


def reorder_provider_profile(
    session: Session,
    tenant: Tenant,
    profile_id: str,
    direction: str,
    actor_id: str,
) -> ProviderProfile:
    """Move a routed account one position up/down in the tenant-local route."""
    if direction not in {"up", "down"}:
        raise ValueError("Route direction must be up or down")
    _lock_tenant(session, tenant.id)
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.id == profile_id,
            ProviderProfile.tenant_id == tenant.id,
            ProviderProfile.deleted_at.is_(None),
        )
        .execution_options(populate_existing=True)
    )
    if profile is None:
        raise LookupError("Provider account was not found")
    if profile.route_priority is None:
        raise ValueError("Cannot reorder an inactive provider account")
    previous = _routed_profiles(session, tenant.id)
    ordered = list(previous)
    index = next(index for index, item in enumerate(ordered) if item.id == profile.id)
    destination = index + (-1 if direction == "up" else 1)
    if destination < 0 or destination >= len(ordered):
        return profile
    ordered[index], ordered[destination] = ordered[destination], ordered[index]
    _write_route(session, previous, ordered)
    session.add(
        AuditEvent(
            tenant_id=tenant.id,
            actor_id=actor_id,
            target=f"profile:{profile.id}",
            action="profile.reorder",
            outcome="success",
            details={
                "profile_id": profile.id,
                "direction": direction,
                "route_priority": profile.route_priority,
            },
        )
    )
    return profile


def activate_provider_profile(
    session: Session, tenant: Tenant, profile_id: str, actor_id: str
) -> ProviderProfile:
    """Append a validated account, preserving any existing route position."""
    tenant = _lock_tenant(session, tenant.id)
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.id == profile_id,
            ProviderProfile.tenant_id == tenant.id,
            ProviderProfile.deleted_at.is_(None),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if profile is None:
        raise LookupError("Provider account was not found")
    catalog = session.scalars(
        select(ProviderCatalogEntry).where(
            ProviderCatalogEntry.profile_id == profile.id
        )
    )
    validate_routed_profile(
        profile, [(entry.model_id, entry.deployment_id) for entry in catalog]
    )
    if profile.route_priority is None:
        previous = _routed_profiles(session, tenant.id)
        _write_route(session, previous, [*previous, profile])
    session.add(
        AuditEvent(
            tenant_id=tenant.id,
            actor_id=actor_id,
            target=f"profile:{profile.id}",
            action="profile.activate",
            outcome="success",
            details={
                "provider": profile.provider,
                "profile_id": profile.id,
                "route_priority": profile.route_priority,
            },
        )
    )
    return profile


def replace_catalog_entries(
    session: Session,
    profile: ProviderProfile,
    entries: list[tuple[str, str | None]],
    error: str | None,
    pricing: dict[str, dict[str, str]] | None = None,
) -> None:
    """Replace catalog rows, deactivating routes whose target disappeared."""
    _lock_tenant(session, profile.tenant_id)
    profile = session.scalar(
        select(ProviderProfile)
        .where(ProviderProfile.id == profile.id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if profile is None or profile.deleted_at is not None:
        raise LookupError("Provider account was not found")
    profile.catalog_error = error
    if error is not None:
        return

    existing = session.scalars(
        select(ProviderCatalogEntry).where(
            ProviderCatalogEntry.profile_id == profile.id
        )
    )
    for row in existing:
        session.delete(row)
    session.flush()
    for model_id, deployment_id in entries:
        rates = (
            pricing.get(model_id)
            if profile.provider == "openrouter" and pricing
            else None
        )
        has_complete_pricing = rates is not None and all(
            rates.get(key) is not None
            for key in (
                "input_per_1m_tokens",
                "output_per_1m_tokens",
                "cache_per_1m_tokens",
                "currency",
                "source",
            )
        )
        session.add(
            ProviderCatalogEntry(
                profile_id=profile.id,
                model_id=model_id,
                deployment_id=deployment_id,
                source="provider",
                input_price_per_1m_tokens=(
                    rates["input_per_1m_tokens"] if has_complete_pricing else None
                ),
                output_price_per_1m_tokens=(
                    rates["output_per_1m_tokens"] if has_complete_pricing else None
                ),
                cache_price_per_1m_tokens=(
                    rates["cache_per_1m_tokens"] if has_complete_pricing else None
                ),
                pricing_currency=rates["currency"] if has_complete_pricing else None,
                pricing_source=rates["source"] if has_complete_pricing else None,
            )
        )
    profile.catalog_refreshed_at = datetime.now(timezone.utc)
    selectable_models = {
        model_id for model_id, _ in selectable_catalog_models(profile.provider, entries)
    }
    if profile.default_model not in selectable_models:
        _remove_from_route(
            session, profile, "system:catalog", "default_model_unavailable"
        )


def bind_provider_cost_scope(
    session: Session,
    tenant_id: str,
    profile_id: str,
    provider: str,
    scope_values: dict[str, str],
    actor_id: str,
) -> ProviderScopeBinding:
    """Bind an exclusively confirmed OpenAI project or OpenRouter workspace."""
    if provider not in {"openai", "openrouter"}:
        raise ValueError("Provider billing scope is not configurable here.")
    tenant = _lock_tenant(session, tenant_id)
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.tenant_id == tenant.id,
            ProviderProfile.id == profile_id,
            ProviderProfile.provider == provider,
            ProviderProfile.deleted_at.is_(None),
        )
        .with_for_update()
    )
    if profile is None:
        raise LookupError("Provider account was not found for this tenant.")
    existing_binding = session.scalar(
        select(ProviderScopeBinding.id).where(
            ProviderScopeBinding.tenant_id == tenant.id,
            ProviderScopeBinding.profile_id == profile.id,
        )
    )
    if existing_binding is not None:
        raise ValueError("Billing scope is already bound to this account.")

    nodes: list[ProviderScopeNode]
    if provider == "openai":
        organization_id = scope_values.get("organization", "").strip()
        project_id = scope_values.get("project", "").strip()
        if re.fullmatch(r"org-[A-Za-z0-9_-]+", organization_id) is None:
            raise ValueError("OpenAI Organization-ID must start with 'org-'.")
        if re.fullmatch(r"proj_[A-Za-z0-9_-]+", project_id) is None:
            raise ValueError("OpenAI Project-ID must start with 'proj_'.")
        organization_node = session.scalar(
            select(ProviderScopeNode).where(
                ProviderScopeNode.provider == provider,
                ProviderScopeNode.scope_type == "organization",
                ProviderScopeNode.canonical_scope_id == organization_id,
            )
        )
        if organization_node is not None and organization_node.tenant_id != tenant.id:
            raise ValueError("Provider scope is already bound.")
        if organization_node is None:
            organization_node = ProviderScopeNode(
                id=str(uuid4()),
                tenant_id=tenant.id,
                provider=provider,
                scope_type="organization",
                canonical_scope_id=organization_id,
            )
            nodes = [organization_node]
        else:
            nodes = []
        billing_node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id=tenant.id,
            provider=provider,
            scope_type="project",
            canonical_scope_id=project_id,
            parent_node_id=organization_node.id,
        )
        nodes.append(billing_node)
    else:
        try:
            workspace_id = str(UUID(scope_values.get("workspace", "").strip()))
        except ValueError as exc:
            raise ValueError("OpenRouter Workspace-ID must be a UUID.") from exc
        billing_node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id=tenant.id,
            provider=provider,
            scope_type="workspace",
            canonical_scope_id=workspace_id,
        )
        nodes = [billing_node]

    for node in nodes:
        if (
            session.scalar(
                select(ProviderScopeNode.id).where(
                    ProviderScopeNode.provider == node.provider,
                    ProviderScopeNode.scope_type == node.scope_type,
                    ProviderScopeNode.canonical_scope_id == node.canonical_scope_id,
                )
            )
            is not None
        ):
            raise ValueError("Provider scope is already bound.")
    session.add_all(nodes)
    session.flush()
    binding = ProviderScopeBinding(
        id=str(uuid4()),
        tenant_id=tenant.id,
        provider=provider,
        profile_id=profile.id,
        purpose="billing",
        node_id=billing_node.id,
    )
    session.add_all(
        (
            binding,
            AuditEvent(
                tenant_id=tenant.id,
                actor_id=actor_id,
                target=f"{provider}:profile:{profile.id}:billing-scope",
                action="billing_scope.bind",
                outcome="success",
                details={"provider": provider, "purpose": "billing"},
            ),
        )
    )
    return binding


def bind_azure_cost_scopes(
    session: Session,
    tenant_id: str,
    profile_id: str,
    subscription_id: str,
    resource_group_arm_id: str,
    cognitive_resource_arm_id: str,
    actor_id: str,
) -> tuple[ProviderScopeBinding, ProviderScopeBinding]:
    """Bind one exclusively confirmed Azure resource group and child resource."""
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.id == profile_id,
            ProviderProfile.provider == "azure",
            ProviderProfile.deleted_at.is_(None),
        )
        .with_for_update()
    )
    if profile is None:
        raise LookupError("Azure provider profile was not found for this tenant.")
    if (
        session.scalar(
            select(ProviderScopeBinding.id).where(
                ProviderScopeBinding.tenant_id == tenant_id,
                ProviderScopeBinding.profile_id == profile_id,
            )
        )
        is not None
    ):
        raise ValueError("Azure scopes are already bound to this account.")

    billing_scope, usage_scope = canonical_cost_scopes(
        subscription_id,
        resource_group_arm_id,
        cognitive_resource_arm_id,
    )
    for scope_type, scope_id in (
        ("resource_group", billing_scope),
        ("cognitive_resource", usage_scope),
    ):
        if (
            session.scalar(
                select(ProviderScopeNode.id).where(
                    ProviderScopeNode.provider == "azure",
                    ProviderScopeNode.scope_type == scope_type,
                    ProviderScopeNode.canonical_scope_id == scope_id,
                )
            )
            is not None
        ):
            raise ValueError("Provider scope is already bound.")

    billing_node = ProviderScopeNode(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider="azure",
        scope_type="resource_group",
        canonical_scope_id=billing_scope,
    )
    usage_node = ProviderScopeNode(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider="azure",
        scope_type="cognitive_resource",
        canonical_scope_id=usage_scope,
        parent_node_id=billing_node.id,
    )
    session.add_all((billing_node, usage_node))
    session.flush()
    billing_binding = ProviderScopeBinding(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider="azure",
        profile_id=profile_id,
        purpose="billing",
        node_id=billing_node.id,
    )
    session.add(billing_binding)
    session.flush()
    usage_binding = ProviderScopeBinding(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider="azure",
        profile_id=profile_id,
        purpose="usage",
        node_id=usage_node.id,
        parent_binding_id=billing_binding.id,
    )
    session.add_all(
        (
            usage_binding,
            AuditEvent(
                tenant_id=tenant_id,
                actor_id=actor_id,
                target=f"azure:profile:{profile_id}:cost-scopes",
                action="billing_scope.bind",
                outcome="success",
                details={"provider": "azure", "purposes": ["billing", "usage"]},
            ),
        )
    )
    return billing_binding, usage_binding


def save_billing_secret(
    session: Session,
    cipher: SecretCipher,
    tenant_id: str,
    profile_id: str,
    secret: str,
    actor_id: str,
) -> ProviderProfile:
    """Store encrypted billing credentials on one tenant-owned profile."""
    profile = session.scalar(
        select(ProviderProfile).where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.id == profile_id,
            ProviderProfile.deleted_at.is_(None),
            ProviderProfile.provider.in_(("azure", "openai", "openrouter")),
        )
    )
    if profile is None:
        raise LookupError("Provider account was not found")
    if profile.provider == "azure":
        raise ValueError(
            "Azure billing credentials are not accepted; use operator host identity."
        )
    profile.billing_secret_ciphertext = cipher.encrypt(secret)
    session.add(
        AuditEvent(
            tenant_id=tenant_id,
            actor_id=actor_id,
            target=f"profile:{profile.id}:billing",
            action="billing_secret.save",
            outcome="success",
            details={"provider": profile.provider, "profile_id": profile.id},
        )
    )
    return profile


def parse_model_deployments_json(raw: str) -> dict[str, str]:
    """Parse a JSON object of Cursor model ids to Azure deployment names."""
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("model_deployments must be a non-empty JSON object")
    deployments: dict[str, str] = {}
    for model, deployment in parsed.items():
        if (
            not isinstance(model, str)
            or not isinstance(deployment, str)
            or not deployment.strip()
        ):
            raise ValueError("each model deployment must map a string to a string")
        deployments[model] = deployment.strip()
    return deployments
