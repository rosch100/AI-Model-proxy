"""Tenant-owned administrator mutations with audit records."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.admin.passwords import hash_admin_password, verify_admin_password
from app.persistence.admin_auth import revoke_account_sessions
from app.persistence.models import (
    AdminAccount,
    AuditEvent,
    ProviderCatalogEntry,
    ProviderProfile,
    Tenant,
    provider_profile_name_key,
)
from app.persistence.secrets import SecretCipher
from app.providers.azure_url import validate_azure_base_url
from app.providers.catalog import selectable_catalog_models
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
    """Create or update one provider profile without activating it."""
    profile = session.scalar(
        select(ProviderProfile).where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.provider == provider,
        )
    )
    if profile is None:
        profile = ProviderProfile(
            id=str(uuid4()),
            tenant_id=tenant_id,
            provider=provider,
            settings=settings,
            default_model=default_model,
        )
        session.add(profile)
        session.flush()
    else:
        profile.settings = settings
        profile.default_model = default_model
    if inference_secret:
        profile.inference_secret_ciphertext = cipher.encrypt(inference_secret)
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
    """Create an inactive, tenant-owned account with encrypted credentials."""
    if provider not in {"azure", "openai", "openrouter"}:
        raise ValueError("Unsupported provider")
    if session.get(Tenant, tenant_id) is None:
        raise LookupError("Tenant was not found")
    name = _validate_profile_name(display_name)
    _ensure_profile_name_available(session, tenant_id, provider, name)
    if not isinstance(settings, dict):
        raise ValueError("Provider settings must be an object")
    normalized_settings = dict(settings)
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
    tenant = session.scalar(
        select(Tenant).where(Tenant.id == tenant_id).with_for_update()
    )
    if tenant is None:
        raise LookupError("Tenant was not found")
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.id == profile_id,
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.deleted_at.is_(None),
        )
        .with_for_update()
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
    normalized_settings = dict(settings)
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

    tenant = session.get(Tenant, tenant_id)
    if (
        tenant.active_profile_id == profile.id
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
        if normalized_default_model not in selectable_models:
            raise ValueError(
                f"The default model for {profile.display_name} is not in its available catalog"
            )

    profile.display_name = name
    profile.settings = normalized_settings
    profile.default_model = None if azure_endpoint_changed else normalized_default_model
    if azure_endpoint_changed:
        profile.catalog_refreshed_at = None
        profile.catalog_error = None
        if tenant.active_profile_id == profile.id:
            tenant.active_profile_id = None
            session.add(
                AuditEvent(
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    target="tenant:active-profile",
                    action="profile.deactivate",
                    outcome="success",
                    details={
                        "profile_id": profile.id,
                        "reason": "azure_endpoint_changed",
                    },
                )
            )
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


def activate_provider_profile(
    session: Session, tenant: Tenant, profile_id: str, actor_id: str
) -> ProviderProfile:
    """Atomically activate one complete account identified by profile ID."""
    tenant = session.scalar(
        select(Tenant).where(Tenant.id == tenant.id).with_for_update()
    )
    if tenant is None:
        raise LookupError("Tenant was not found")
    profile = session.scalar(
        select(ProviderProfile)
        .where(
            ProviderProfile.id == profile_id,
            ProviderProfile.tenant_id == tenant.id,
            ProviderProfile.deleted_at.is_(None),
        )
        .with_for_update()
    )
    if profile is None:
        raise LookupError("Provider account was not found")
    if not profile.inference_secret_ciphertext or not profile.default_model:
        raise ValueError(f"The {profile.display_name} account is incomplete")
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
    if profile.default_model not in selectable_models:
        raise ValueError(
            f"The default model for {profile.display_name} is not in its available catalog"
        )
    tenant.active_profile_id = profile.id
    session.add(
        AuditEvent(
            tenant_id=tenant.id,
            actor_id=actor_id,
            target=f"profile:{profile.id}",
            action="profile.activate",
            outcome="success",
            details={"provider": profile.provider, "profile_id": profile.id},
        )
    )
    return profile


def replace_catalog_entries(
    session: Session,
    profile: ProviderProfile,
    entries: list[tuple[str, str | None]],
    error: str | None,
) -> None:
    """Replace catalog rows after a successful provider query."""
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
        session.add(
            ProviderCatalogEntry(
                profile_id=profile.id,
                model_id=model_id,
                deployment_id=deployment_id,
                source="provider",
            )
        )
    profile.catalog_refreshed_at = datetime.now(timezone.utc)


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
