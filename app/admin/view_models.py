"""View models for tenant administrator pages."""

from __future__ import annotations

from dataclasses import dataclass

from app.persistence.models import (
    CostRefreshJob,
    CostUsageRecord,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)


@dataclass(frozen=True)
class ProviderStatus:
    """Dashboard status for one configured or missing provider."""

    provider: str
    label: str
    state: str
    is_active: bool


@dataclass(frozen=True)
class DashboardView:
    """Aggregated tenant status shown on the admin home page."""

    tenant_id: str
    custom_model_id: str
    providers: tuple[ProviderStatus, ...]


@dataclass(frozen=True)
class ConnectionView:
    """Provider accounts grouped for the connection settings page."""

    tenant_id: str
    profiles_by_provider: dict[str, tuple[ProviderProfile, ...]]
    profiles_with_bindings: frozenset[str]
    catalogs: dict[str, tuple[ProviderCatalogEntry, ...]]
    selectable_models: dict[str, tuple[tuple[str, str | None], ...]]
    selectable_model_ids: dict[str, tuple[str, ...]]
    active_profile_id: str | None
    active_profile_name: str | None
    active_provider: str | None


@dataclass(frozen=True)
class CostsView:
    """Bound scopes, latest refresh, and recorded cost buckets."""

    tenant_id: str
    profiles: tuple[ProviderProfile, ...]
    bindings: tuple[
        tuple[ProviderProfile, ProviderScopeBinding, ProviderScopeNode], ...
    ]
    binding_profile_names: dict[str, str]
    jobs: tuple[CostRefreshJob, ...]
    records: tuple[CostUsageRecord, ...]
    billing_key_masks: dict[str, str | None]
    azure_costs_ready_profile_ids: frozenset[str]


def provider_state(profile: ProviderProfile | None) -> str:
    """Return the German dashboard label for one provider profile."""
    if profile is None:
        return "Nicht konfiguriert"
    if not profile.inference_secret_ciphertext or not profile.default_model:
        return "Prüfung erforderlich"
    if profile.catalog_error:
        return "Prüfung erforderlich"
    return "Gespeichert"


def dashboard_view(
    tenant: Tenant, profiles: tuple[ProviderProfile, ...]
) -> DashboardView:
    """Build dashboard status rows without collapsing account identities."""
    profiles_by_provider: dict[str, ProviderProfile] = {}
    active_provider = None
    for profile in profiles:
        profiles_by_provider.setdefault(profile.provider, profile)
        if profile.id == tenant.active_profile_id:
            profiles_by_provider[profile.provider] = profile
            active_provider = profile.provider
    labels = {"azure": "Azure", "openai": "OpenAI", "openrouter": "OpenRouter"}
    statuses = tuple(
        ProviderStatus(
            provider=name,
            label=labels[name],
            state=provider_state(profiles_by_provider.get(name)),
            is_active=active_provider == name,
        )
        for name in ("azure", "openai", "openrouter")
    )
    return DashboardView(
        tenant_id=tenant.id,
        custom_model_id=tenant.custom_model_id,
        providers=statuses,
    )
