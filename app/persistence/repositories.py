"""Tenant and provider persistence operations."""

from __future__ import annotations

import hmac
from collections.abc import Sequence
from copy import deepcopy
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import select_fallback_public_model
from app.providers.azure_url import validate_azure_base_url
from app.providers.catalog import selectable_catalog_models
from app.tenants import (
    DatabaseTenantRoutingSnapshot,
    DatabaseTenantSnapshot,
    TenantConfig,
    hash_api_key,
)

from .models import ProviderCatalogEntry, ProviderProfile, Tenant
from .secrets import SecretCipher


def validate_routed_profile(
    profile: ProviderProfile, entries: Sequence[tuple[str, str | None]]
) -> None:
    """Reject incomplete settings or a target absent from the selectable catalog."""
    if profile.deleted_at is not None:
        raise ValueError("Provider account has been removed")
    if (
        not profile.inference_secret_ciphertext
        or not isinstance(profile.default_model, str)
        or not profile.default_model.strip()
    ):
        raise ValueError(f"The {profile.display_name} account is incomplete")
    if profile.provider not in {"azure", "openai", "openrouter", "deepseek"}:
        raise ValueError("Unsupported provider")
    selectable = dict(selectable_catalog_models(profile.provider, entries))
    if profile.default_model not in selectable:
        raise ValueError(
            f"The default model for {profile.display_name} is not in its available catalog"
        )
    if not isinstance(profile.settings, dict):
        raise ValueError("Provider settings must be an object")
    if profile.provider == "azure":
        validate_azure_base_url(profile.settings.get("base_url"))
        deployments = profile.settings.get("model_deployments")
        if (
            not isinstance(deployments, dict)
            or any(
                not isinstance(model, str)
                or not isinstance(deployment, str)
                or not deployment.strip()
                for model, deployment in deployments.items()
            )
            or deployments.get(profile.default_model)
            != selectable[profile.default_model]
        ):
            raise ValueError(
                "Azure model deployments do not match the available catalog"
            )
    else:
        for key in ("organization", "project"):
            value = profile.settings.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Provider setting {key} must be a string")


class TenantRepository:
    """Resolve tenant identities and read tenant-owned configuration."""

    def __init__(self, session: Session) -> None:
        """Initialize tenant queries for an existing unit of work."""
        self._session = session

    def get_by_api_key(self, api_key: str) -> Tenant | None:
        """Resolve a tenant using the digest of a presented Cursor API key."""
        digest = hash_api_key(api_key)
        tenant = self._session.scalar(
            select(Tenant).where(Tenant.api_key_hash == digest)
        )
        if tenant is None or not hmac.compare_digest(tenant.api_key_hash, digest):
            return None
        return tenant

    def get_proxy_snapshot_by_api_key(
        self, api_key: str, cipher: SecretCipher
    ) -> DatabaseTenantRoutingSnapshot | None:
        """Read identity, ordered route, and validation catalogs in one statement."""
        digest = hash_api_key(api_key)
        rows = self._session.execute(
            select(Tenant, ProviderProfile, ProviderCatalogEntry)
            .outerjoin(
                ProviderProfile,
                (ProviderProfile.tenant_id == Tenant.id)
                & ProviderProfile.route_priority.is_not(None)
                & ProviderProfile.deleted_at.is_(None),
            )
            .outerjoin(
                ProviderCatalogEntry,
                ProviderCatalogEntry.profile_id == ProviderProfile.id,
            )
            .where(Tenant.api_key_hash == digest)
            .order_by(ProviderProfile.route_priority, ProviderCatalogEntry.id)
            .execution_options(populate_existing=True)
        ).all()
        if not rows:
            return None
        tenant = rows[0][0]
        if not hmac.compare_digest(tenant.api_key_hash, digest):
            return None

        catalogs: dict[str, list[tuple[str, str | None]]] = {}
        profiles: dict[str, ProviderProfile] = {}
        for _tenant, profile, entry in rows:
            if profile is None:
                continue
            profiles[profile.id] = profile
            catalog = catalogs.setdefault(profile.id, [])
            if entry is not None:
                catalog.append((entry.model_id, entry.deployment_id))
        snapshots = []
        for profile in profiles.values():
            try:
                validate_routed_profile(profile, catalogs[profile.id])
                secret = cipher.decrypt(profile.inference_secret_ciphertext)
                if not secret.strip():
                    continue
            except (ValueError, InvalidTag):
                # Invalid route members cannot be used; never invent another model.
                continue
            snapshots.append(
                DatabaseTenantSnapshot(
                    id=tenant.id,
                    api_key_hash=tenant.api_key_hash,
                    custom_model_id=tenant.custom_model_id,
                    provider=profile.provider,
                    provider_settings=deepcopy(profile.settings),
                    inference_secret=secret,
                    default_model=profile.default_model,
                    profile_id=profile.id,
                    profile_name=profile.display_name,
                    history_generation=profile.history_generation,
                    profile_deleted=False,
                    catalog_model_ids=tuple(
                        model
                        for model, _ in selectable_catalog_models(
                            profile.provider, catalogs[profile.id]
                        )
                    ),
                )
            )
        return DatabaseTenantRoutingSnapshot(
            id=tenant.id,
            api_key_hash=tenant.api_key_hash,
            custom_model_id=tenant.custom_model_id,
            profiles=tuple(snapshots),
        )

    def get_admin_snapshot(
        self, tenant_id: str
    ) -> tuple[Tenant, ProviderProfile | None]:
        """Return a tenant and its first routed, nondeleted profile in one query."""
        row = self._session.execute(
            select(Tenant, ProviderProfile)
            .outerjoin(
                ProviderProfile,
                (ProviderProfile.tenant_id == Tenant.id)
                & ProviderProfile.deleted_at.is_(None)
                & ProviderProfile.route_priority.is_not(None),
            )
            .where(Tenant.id == tenant_id)
            .order_by(ProviderProfile.route_priority)
            .limit(1)
            .execution_options(populate_existing=True)
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
        default_model = tenant_config.azure_default_model
        if default_model is None:
            default_model = select_fallback_public_model(
                tenant_config.azure_model_deployments
            )
            if default_model is None:
                raise ValueError(
                    f"TENANTS[{tenant_config.id!r}].azure_default_model is omitted "
                    "and no preferred fallback model deployment is configured."
                )
        elif default_model not in tenant_config.azure_model_deployments:
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
            azure_profiles = session.scalars(
                select(ProviderProfile)
                .where(
                    ProviderProfile.tenant_id == tenant_config.id,
                    ProviderProfile.provider == "azure",
                    ProviderProfile.deleted_at.is_(None),
                    ProviderProfile.route_priority.is_not(None),
                )
                .order_by(ProviderProfile.route_priority)
            )
            azure_profiles = list(azure_profiles)
            if not azure_profiles:
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has ambiguous active "
                    "Azure profile resolution for environment import"
                )
            expected_settings = {
                "base_url": tenant_config.azure_base_url,
                "model_deployments": dict(tenant_config.azure_model_deployments),
            }
            settings_matches = [
                profile
                for profile in azure_profiles
                if profile.settings == expected_settings
                and profile.default_model == default_model
            ]
            if not settings_matches:
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has different provider data"
                )
            profiles_with_secrets = [
                profile
                for profile in settings_matches
                if profile.inference_secret_ciphertext is not None
            ]
            if not profiles_with_secrets:
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has no provider secret"
                )
            matching_profiles = [
                profile
                for profile in profiles_with_secrets
                if hmac.compare_digest(
                    cipher.decrypt(profile.inference_secret_ciphertext),
                    tenant_config.azure_api_key,
                )
            ]
            if not matching_profiles:
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has a different provider secret"
                )
            if len(matching_profiles) > 1:
                raise ValueError(
                    f"Existing tenant {tenant_config.id!r} has ambiguous active "
                    "Azure profile resolution for environment import"
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
            default_model=default_model,
            route_priority=1,
            inference_secret_ciphertext=cipher.encrypt(tenant_config.azure_api_key),
        )
        session.add(profile)
        session.flush()
        session.add_all(
            ProviderCatalogEntry(
                profile_id=profile_id,
                model_id=model_id,
                deployment_id=deployment_id,
                source="environment",
            )
            for model_id, deployment_id in tenant_config.azure_model_deployments.items()
        )
        imported += 1
    session.flush()
    return imported
