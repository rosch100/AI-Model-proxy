"""Tenant-owned administrator mutations with audit records."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.admin.passwords import hash_admin_password, verify_admin_password
from app.persistence.admin_auth import revoke_account_sessions
from app.persistence.models import (
    AdminAccount,
    AuditEvent,
    ProviderCatalogEntry,
    ProviderProfile,
    Tenant,
)
from app.persistence.secrets import SecretCipher


def rotate_api_key(session: Session, tenant: Tenant, actor_id: str) -> str:
    """Replace the Cursor API key digest and return the plaintext key once."""
    api_key = secrets.token_urlsafe(32)
    tenant.api_key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
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


def activate_provider_profile(
    session: Session, tenant: Tenant, provider: str, actor_id: str
) -> ProviderProfile:
    """Atomically mark one saved profile as the active proxy provider."""
    profile = session.scalar(
        select(ProviderProfile).where(
            ProviderProfile.tenant_id == tenant.id,
            ProviderProfile.provider == provider,
        )
    )
    if profile is None:
        raise LookupError(f"No {provider} profile is configured")
    if not profile.inference_secret_ciphertext or not profile.default_model:
        raise ValueError(f"The {provider} profile is incomplete")
    tenant.active_profile_id = profile.id
    session.add(
        AuditEvent(
            tenant_id=tenant.id,
            actor_id=actor_id,
            target=f"profile:{provider}",
            action="profile.activate",
            outcome="success",
            details={"provider": provider, "profile_id": profile.id},
        )
    )
    return profile


def replace_catalog_entries(
    session: Session,
    profile: ProviderProfile,
    entries: list[tuple[str, str | None]],
    error: str | None,
) -> None:
    """Replace catalog rows for a profile after an out-of-band provider query."""
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
    profile.catalog_error = error


def save_billing_secret(
    session: Session,
    cipher: SecretCipher,
    tenant_id: str,
    provider: str,
    secret: str,
    actor_id: str,
) -> ProviderProfile:
    """Store an encrypted billing credential on an existing provider profile."""
    if provider not in {"openai", "openrouter"}:
        raise ValueError(
            "Billing secrets are supported only for OpenAI and OpenRouter; "
            "Azure uses operator host identity."
        )
    profile = session.scalar(
        select(ProviderProfile).where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.provider == provider,
        )
    )
    if profile is None:
        raise LookupError(f"No {provider} profile is configured")
    profile.billing_secret_ciphertext = cipher.encrypt(secret)
    session.add(
        AuditEvent(
            tenant_id=tenant_id,
            actor_id=actor_id,
            target=f"profile:{provider}:billing",
            action="billing_secret.save",
            outcome="success",
            details={"provider": provider},
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
