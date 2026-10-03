"""View models for tenant administrator pages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

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
class CostMetricSummary:
    """A currency- or metric-specific cost/usage total."""

    metric: str
    label: str
    value: Decimal
    formatted_value: str
    currency: str | None


@dataclass(frozen=True)
class DashboardAccountCost:
    """Cost and refresh status for one provider profile and billing scope."""

    profile_id: str
    profile_name: str
    provider_label: str
    has_billing_scope: bool
    has_billing_key: bool
    actual_costs: tuple[CostMetricSummary, ...]
    estimates: tuple[CostMetricSummary, ...]
    usage: tuple[CostMetricSummary, ...]
    period_start: datetime | None
    period_end: datetime | None
    last_successful_at: datetime | None
    last_job_status: str | None
    last_job_message: str | None


@dataclass(frozen=True)
class DashboardView:
    """Aggregated tenant status and latest cost snapshot for the home page."""

    tenant_id: str
    custom_model_id: str
    providers: tuple[ProviderStatus, ...]
    billing_scope_count: int = 0
    accounts_with_cost_data: int = 0
    accounts_with_successful_refresh: int = 0
    cost_accounts: tuple[DashboardAccountCost, ...] = ()


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


_PROVIDER_LABELS = {"azure": "Azure", "openai": "OpenAI", "openrouter": "OpenRouter"}
_COST_METRIC_LABELS = {"cost": "Providerkosten", "byok_inference_cost": "BYOK-Kosten"}
_USAGE_METRIC_LABELS = {
    "input_tokens": "Eingabe-Tokens",
    "output_tokens": "Ausgabe-Tokens",
    "prompt_tokens": "Prompt-Tokens",
    "completion_tokens": "Completion-Tokens",
    "reasoning_tokens": "Reasoning-Tokens",
}
_REFRESH_STATUS_LABELS = {
    "running": "Aktualisierung läuft",
    "retry_wait": "Wiederholung geplant",
    "success": "Erfolgreich",
    "unavailable": "Nicht verfügbar",
    "failed": "Fehlgeschlagen",
}


def _format_amount(value: Decimal, *, tokens: bool) -> str:
    if tokens:
        return format(value, ",.0f").replace(",", ".")
    integer, separator, fraction = format(value, ",f").partition(".")
    fraction = fraction.rstrip("0") if separator else ""
    if len(fraction) < 2:
        fraction = fraction.ljust(2, "0")
    formatted = integer.replace(",", ".")
    return f"{formatted},{fraction}"


def _summarize_records(
    records: Iterable[CostUsageRecord], kind: str
) -> tuple[CostMetricSummary, ...]:
    totals: dict[tuple[str, str | None], Decimal] = {}
    for record in records:
        if record.kind != kind:
            continue
        key = (record.metric, record.currency)
        totals[key] = totals.get(key, Decimal(0)) + record.value
    summaries = []
    for (metric, currency), value in sorted(totals.items()):
        is_usage = kind == "usage"
        labels = _USAGE_METRIC_LABELS if is_usage else _COST_METRIC_LABELS
        label = labels.get(metric, metric.replace("_", " ").capitalize())
        summaries.append(
            CostMetricSummary(
                metric=metric,
                label=label,
                value=value,
                formatted_value=_format_amount(value, tokens=is_usage),
                currency=currency,
            )
        )
    return tuple(summaries)


def provider_state(profile: ProviderProfile | None) -> str:
    """Return the German dashboard label for one provider profile."""
    if profile is None:
        return "Nicht konfiguriert"
    if not profile.inference_secret_ciphertext or not profile.default_model:
        return "Prüfung erforderlich"
    if profile.catalog_error:
        return "Prüfung erforderlich"
    return "Gespeichert"


def _latest_jobs_by_binding(
    jobs: tuple[CostRefreshJob, ...], *, successful_only: bool = False
) -> dict[str, CostRefreshJob]:
    latest: dict[str, CostRefreshJob] = {}
    for job in sorted(jobs, key=lambda item: (item.created_at, item.id), reverse=True):
        if successful_only and job.status != "success":
            continue
        latest.setdefault(job.binding_id, job)
    return latest


def dashboard_view(
    tenant: Tenant,
    profiles: tuple[ProviderProfile, ...],
    bindings: tuple[
        tuple[ProviderProfile, ProviderScopeBinding, ProviderScopeNode], ...
    ] = (),
    jobs: tuple[CostRefreshJob, ...] = (),
    records: tuple[CostUsageRecord, ...] = (),
) -> DashboardView:
    """Build provider status and latest-snapshot cost summaries for a tenant."""
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
    statuses = tuple(
        ProviderStatus(
            provider=name,
            label=_PROVIDER_LABELS[name],
            state=provider_state(profiles_by_provider.get(name)),
            is_active=name in active_providers,
        )
        for name in ("azure", "openai", "openrouter")
    )

    billing_bindings = tuple(
        (profile, binding, node)
        for profile, binding, node in bindings
        if binding.purpose == "billing"
    )
    binding_ids = {binding.id for _profile, binding, _node in billing_bindings}
    latest_attempts = _latest_jobs_by_binding(jobs)
    latest_successes = _latest_jobs_by_binding(jobs, successful_only=True)
    latest_attempts = {
        binding_id: job
        for binding_id, job in latest_attempts.items()
        if binding_id in binding_ids
    }
    latest_successes = {
        binding_id: job
        for binding_id, job in latest_successes.items()
        if binding_id in binding_ids
    }
    successful_job_ids = {job.id for job in latest_successes.values()}
    active_records = tuple(
        record for record in records if record.job_id in successful_job_ids
    )
    records_by_binding: dict[str, list[CostUsageRecord]] = {}
    for record in active_records:
        records_by_binding.setdefault(record.binding_id, []).append(record)

    bindings_by_profile = {
        profile.id: binding for profile, binding, _node in billing_bindings
    }
    cost_accounts = []
    for profile in visible_profiles:
        binding = bindings_by_profile.get(profile.id)
        latest_success = latest_successes.get(binding.id) if binding else None
        account_records = records_by_binding.get(binding.id, []) if binding else []
        latest_attempt = latest_attempts.get(binding.id) if binding else None
        cost_accounts.append(
            DashboardAccountCost(
                profile_id=profile.id,
                profile_name=profile.display_name or _PROVIDER_LABELS[profile.provider],
                provider_label=_PROVIDER_LABELS[profile.provider],
                has_billing_scope=binding is not None,
                has_billing_key=(
                    profile.provider == "azure"
                    or bool(profile.billing_secret_ciphertext)
                ),
                actual_costs=_summarize_records(account_records, "actual"),
                estimates=_summarize_records(account_records, "estimate"),
                usage=_summarize_records(account_records, "usage"),
                period_start=latest_success.period_start if latest_success else None,
                period_end=latest_success.period_end if latest_success else None,
                last_successful_at=(
                    latest_success.completed_at if latest_success else None
                ),
                last_job_status=(
                    _REFRESH_STATUS_LABELS.get(
                        latest_attempt.status, latest_attempt.status
                    )
                    if latest_attempt
                    else None
                ),
                last_job_message=latest_attempt.limitation if latest_attempt else None,
            )
        )

    accounts_with_cost_data = sum(
        bool(account.actual_costs or account.estimates or account.usage)
        for account in cost_accounts
    )
    accounts_with_successful_refresh = sum(
        account.has_billing_scope and account.last_successful_at is not None
        for account in cost_accounts
    )
    return DashboardView(
        tenant_id=tenant.id,
        custom_model_id=tenant.custom_model_id,
        providers=statuses,
        billing_scope_count=len(billing_bindings),
        accounts_with_cost_data=accounts_with_cost_data,
        accounts_with_successful_refresh=accounts_with_successful_refresh,
        cost_accounts=tuple(cost_accounts),
    )
