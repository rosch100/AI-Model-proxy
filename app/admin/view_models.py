"""View models for tenant administrator pages."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """Aggregated tenant status and cost preview shown on the home page."""

    tenant_id: str
    custom_model_id: str
    providers: tuple[ProviderStatus, ...]
    cost_record_counts: dict[str, int] = field(default_factory=dict)
    cost_records: tuple[CostUsageRecord, ...] = ()
    cost_record_profile_names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionView:
    """Provider accounts grouped for the connection settings page."""

    tenant_id: str
    profiles_by_provider: dict[str, tuple[ProviderProfile, ...]]
    profiles_with_bindings: frozenset[str]
    catalogs: dict[str, tuple[ProviderCatalogEntry, ...]]
    selectable_models: dict[str, tuple[tuple[str, str | None], ...]]
    selectable_model_ids: dict[str, tuple[str, ...]]
    routed_profiles: tuple[ProviderProfile, ...]

    @property
    def active_profile_id(self) -> str | None:
        """Expose the primary ID derived from the ordered route."""
        return self.routed_profiles[0].id if self.routed_profiles else None

    @property
    def active_profile_name(self) -> str | None:
        """Expose the primary account name, without a second routing source."""
        return self.routed_profiles[0].display_name if self.routed_profiles else None

    @property
    def active_provider(self) -> str | None:
        """Expose the primary provider derived from the ordered route."""
        return self.routed_profiles[0].provider if self.routed_profiles else None


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
    azure_scope_bound_profile_ids: frozenset[str]


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
    active_providers: set[str] = set()
    visible_profiles = sorted(
        (profile for profile in profiles if profile.deleted_at is None),
        key=lambda profile: (
            profile.route_priority is None,
            profile.route_priority or 0,
            profile.id,
        ),
    )
    for profile in visible_profiles:
        profiles_by_provider.setdefault(profile.provider, profile)
        if profile.route_priority is not None:
            active_providers.add(profile.provider)
    labels = {"azure": "Azure", "openai": "OpenAI", "openrouter": "OpenRouter"}
    statuses = tuple(
        ProviderStatus(
            provider=name,
            label=labels[name],
            state=provider_state(profiles_by_provider.get(name)),
            is_active=name in active_providers,
        )
        for name in ("azure", "openai", "openrouter")
    )
    return DashboardView(
        tenant_id=tenant.id,
        custom_model_id=tenant.custom_model_id,
        providers=statuses,
    )
