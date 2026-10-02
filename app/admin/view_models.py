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
    """Editable connection state for every supported provider."""

    tenant_id: str
    azure: ProviderProfile | None
    openai: ProviderProfile | None
    openrouter: ProviderProfile | None
    catalog: tuple[ProviderCatalogEntry, ...]
    active_provider: str | None


@dataclass(frozen=True)
class CostsView:
    """Bound scopes, latest refresh, and recorded cost buckets."""

    tenant_id: str
    bindings: tuple[tuple[ProviderScopeBinding, ProviderScopeNode], ...]
    jobs: tuple[CostRefreshJob, ...]
    records: tuple[CostUsageRecord, ...]


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
    tenant: Tenant, profiles: dict[str, ProviderProfile]
) -> DashboardView:
    """Build dashboard status rows from the persisted tenant and profiles."""
    active = None
    for profile in profiles.values():
        if tenant.active_profile_id == profile.id:
            active = profile.provider
            break
    labels = {"azure": "Azure", "openai": "OpenAI", "openrouter": "OpenRouter"}
    statuses = tuple(
        ProviderStatus(
            provider=name,
            label=labels[name],
            state=provider_state(profiles.get(name)),
            is_active=active == name,
        )
        for name in ("azure", "openai", "openrouter")
    )
    return DashboardView(
        tenant_id=tenant.id,
        custom_model_id=tenant.custom_model_id,
        providers=statuses,
    )
