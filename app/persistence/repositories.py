"""Tenant and provider persistence operations."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.tenants import DatabaseTenantSnapshot, TenantConfig

from .models import ProviderProfile, Tenant
from .secrets import SecretCipher


class TenantRepository:
    """Resolve tenant identities and read tenant-owned configuration."""

    def __init__(self, session: Session) -> None:
        """Initialize tenant queries for an existing unit of work."""
        self._session = session

    def get_by_api_key(self, api_key: str) -> Tenant | None:
        """Resolve a tenant using the digest of a presented Cursor API key."""
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        tenant = self._session.scalar(
            select(Tenant).where(Tenant.api_key_hash == digest)
        )
        if tenant is None or not hmac.compare_digest(tenant.api_key_hash, digest):
            return None
        return tenant

    def get_proxy_snapshot_by_api_key(
        self, api_key: str, cipher: SecretCipher
    ) -> DatabaseTenantSnapshot | None:
        """Resolve a key and its active provider profile from one database read."""
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        row = self._session.execute(
            select(Tenant, ProviderProfile)
            .outerjoin(
                ProviderProfile,
                (ProviderProfile.tenant_id == Tenant.id)
                & (ProviderProfile.id == Tenant.active_profile_id),
            )
            .where(Tenant.api_key_hash == digest)
        ).one_or_none()
        if row is None:
            return None

        tenant, profile = row
        if not hmac.compare_digest(tenant.api_key_hash, digest):
            return None
        if profile is None:
            return DatabaseTenantSnapshot(
                id=tenant.id,
                api_key_hash=tenant.api_key_hash,
                custom_model_id=tenant.custom_model_id,
                provider=None,
                provider_settings={},
                inference_secret=None,
                default_model=None,
                profile_id=None,
                profile_name=None,
                history_generation=None,
                profile_deleted=False,
            )

        secret = (
            cipher.decrypt(profile.inference_secret_ciphertext)
            if profile.inference_secret_ciphertext is not None
            else None
        )
        return DatabaseTenantSnapshot(
            id=tenant.id,
            api_key_hash=tenant.api_key_hash,
            custom_model_id=tenant.custom_model_id,
            provider=profile.provider,
            provider_settings=dict(profile.settings),
            inference_secret=secret,
            default_model=profile.default_model,
            profile_id=profile.id,
            profile_name=profile.display_name,
            history_generation=profile.history_generation,
            profile_deleted=profile.deleted_at is not None,
        )

    def get_admin_snapshot(
        self, tenant_id: str
    ) -> tuple[Tenant, ProviderProfile | None]:
        """Return a tenant and its active provider profile in one query."""
        row = self._session.execute(
            select(Tenant, ProviderProfile)
            .outerjoin(
                ProviderProfile,
                (ProviderProfile.tenant_id == Tenant.id)
                & (ProviderProfile.id == Tenant.active_profile_id),
            )
            .where(Tenant.id == tenant_id)
        ).one_or_none()
        if row is None:
            raise LookupError(f"Tenant {tenant_id!r} does not exist")
        return row[0], row[1]


def import_tenants(
    tenants: Sequence[TenantConfig], session: Session, cipher: SecretCipher
) -> int:
    """Atomically import static tenants without overwriting differing records."""
    imported = 0
    for tenant_config in tenants:
        if (
            tenant_config.azure_default_model
            not in tenant_config.azure_model_deployments
        ):
            raise ValueError(
                f"TENANTS[{tenant_config.id!r}].azure_default_model must name "
                "a configured Azure model deployment."
            )
        existing = session.get(Tenant, tenant_config.id)
        if existing is not None:
            if existing.api_key_hash != tenant_config.api_key_hash:
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has a different API key hash"
                )
            profiles = list(
                session.scalars(
                    select(ProviderProfile).where(
                        ProviderProfile.tenant_id == tenant_config.id,
                        ProviderProfile.provider == "azure",
                    )
                )
            )
            profile = next(
                (
                    candidate
                    for candidate in profiles
                    if candidate.id == existing.active_profile_id
                    and candidate.deleted_at is None
                ),
                None,
            )
            if profile is None:
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has ambiguous active "
                    "Azure profile resolution for environment import"
                )
            expected_settings = {
                "base_url": tenant_config.azure_base_url,
                "model_deployments": dict(tenant_config.azure_model_deployments),
            }
            if (
                profile is None
                or profile.settings != expected_settings
                or profile.default_model != tenant_config.azure_default_model
            ):
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has different provider data"
                )
            if profile.inference_secret_ciphertext is None:
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has no provider secret"
                )
            stored_secret = cipher.decrypt(profile.inference_secret_ciphertext)
            if not hmac.compare_digest(stored_secret, tenant_config.azure_api_key):
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has a different provider secret"
                )
            continue

        custom_model_id = f"cursor-{uuid4().hex}"
        profile_id = str(uuid4())
        tenant = Tenant(
            id=tenant_config.id,
            api_key_hash=tenant_config.api_key_hash,
            custom_model_id=custom_model_id,
        )
        session.add(tenant)
        session.flush()

        profile = ProviderProfile(
            id=profile_id,
            tenant_id=tenant_config.id,
            provider="azure",
            display_name="Azure",
            settings={
                "base_url": tenant_config.azure_base_url,
                "model_deployments": dict(tenant_config.azure_model_deployments),
            },
            default_model=tenant_config.azure_default_model,
            inference_secret_ciphertext=cipher.encrypt(tenant_config.azure_api_key),
        )
        session.add(profile)
        session.flush()
        tenant.active_profile_id = profile_id
        imported += 1
    session.flush()
    return imported
