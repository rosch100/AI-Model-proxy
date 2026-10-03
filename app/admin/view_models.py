"""View models for tenant administrator pages."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.persistence.inference_activity import (
    ACTIVITY_LOOKBACK_HOURS,
    ACTIVITY_WINDOW_MINUTES,
)
from app.persistence.models import (
    CostRefreshJob,
    CostUsageRecord,
    InferenceActivityEvent,
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
    recency: str | None = None
    last_request_label: str | None = None


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
class ModelActivityRow:
    """One model series inside a provider on the live activity board."""

    model: str
    inbound_model: str
    routed_model: str | None
    tokens_per_minute: float | None
    tokens_per_minute_label: str | None
    requests_in_window: int
    last_request_at: datetime
    last_request_label: str
    bar_percent: int
    recency: str


@dataclass(frozen=True)
class ProviderActivityGroup:
    """Live activity for one provider."""

    provider: str
    label: str
    rows: tuple[ModelActivityRow, ...]


@dataclass(frozen=True)
class ActivityBoard:
    """Current proxy throughput for the tenant overview."""

    window_minutes: int
    generated_at: datetime
    total_tokens_per_minute: float | None
    total_tokens_per_minute_label: str | None
    total_requests_in_window: int
    last_request_at: datetime | None
    last_request_label: str | None
    providers: tuple[ProviderActivityGroup, ...]


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
    activity: ActivityBoard | None = None


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
    activity_events: Sequence[InferenceActivityEvent] = (),
    *,
    now: datetime | None = None,
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
    activity = activity_board(
        activity_events,
        custom_model_id=tenant.custom_model_id,
        now=now,
    )
    latest_activity = {
        group.provider: max(group.rows, key=lambda row: row.last_request_at)
        for group in activity.providers
        if group.rows
    }
    statuses = tuple(
        ProviderStatus(
            provider=status.provider,
            label=status.label,
            state=status.state,
            is_active=status.is_active,
            recency=(
                latest_activity[status.provider].recency
                if status.provider in latest_activity
                else None
            ),
            last_request_label=(
                latest_activity[status.provider].last_request_label
                if status.provider in latest_activity
                else None
            ),
        )
        for status in statuses
    )
    return DashboardView(
        tenant_id=tenant.id,
        custom_model_id=tenant.custom_model_id,
        providers=statuses,
        billing_scope_count=len(billing_bindings),
        accounts_with_cost_data=accounts_with_cost_data,
        accounts_with_successful_refresh=accounts_with_successful_refresh,
        cost_accounts=tuple(cost_accounts),
        activity=activity,
    )


def _aware_utc(moment: datetime) -> datetime:
    """Normalize SQLite-naive and PostgreSQL-aware timestamps to UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def format_activity_rate(value: float) -> str:
    """Format tokens or requests per minute with German grouping."""
    if value >= 100:
        return format(round(value), ",.0f").replace(",", ".")
    formatted = format(value, ".1f")
    integer, _separator, fraction = formatted.partition(".")
    return f"{integer},{fraction}"


def format_relative_time(moment: datetime, now: datetime) -> str:
    """Return a compact German relative timestamp for live activity."""
    if moment.tzinfo is None or now.tzinfo is None:
        raise ValueError("Activity timestamps must be timezone-aware")
    elapsed = (now - moment).total_seconds()
    if elapsed < 10:
        return "gerade eben"
    if elapsed < 60:
        return f"vor {int(elapsed)} s"
    if elapsed < 3600:
        return f"vor {int(elapsed // 60)} min"
    if moment.astimezone(timezone.utc).date() == now.astimezone(timezone.utc).date():
        return moment.astimezone(timezone.utc).strftime("%H:%M UTC")
    return moment.astimezone(timezone.utc).strftime("%d.%m. %H:%M UTC")


@dataclass
class _ActivityAccumulator:
    inbound_model: str
    routed_model: str | None
    last_request_at: datetime
    window_tokens: int = 0
    window_requests: int = 0
    window_with_usage: int = 0


def _display_model(
    inbound_model: str, routed_model: str | None, custom_model_id: str
) -> str:
    if inbound_model == custom_model_id and routed_model:
        return routed_model
    return inbound_model


def _recency_class(
    last_request_at: datetime, window_start: datetime, now: datetime
) -> str:
    if last_request_at >= now - timedelta(minutes=2):
        return "live"
    if last_request_at >= window_start:
        return "recent"
    return "idle"


def activity_board(
    events: Sequence[InferenceActivityEvent],
    *,
    custom_model_id: str,
    now: datetime | None = None,
    window_minutes: int = ACTIVITY_WINDOW_MINUTES,
    lookback_hours: int = ACTIVITY_LOOKBACK_HOURS,
) -> ActivityBoard:
    """Aggregate recent inferences into provider/model throughput bars."""
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        raise ValueError("Activity board clock must be timezone-aware")
    window_start = clock - timedelta(minutes=window_minutes)
    lookback_start = clock - timedelta(hours=lookback_hours)
    buckets: dict[tuple[str, str], _ActivityAccumulator] = {}
    for event in events:
        occurred_at = _aware_utc(event.occurred_at)
        if occurred_at < lookback_start:
            continue
        display = _display_model(
            event.inbound_model, event.routed_model, custom_model_id
        )
        key = (event.provider, display)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _ActivityAccumulator(
                inbound_model=event.inbound_model,
                routed_model=event.routed_model,
                last_request_at=occurred_at,
            )
            buckets[key] = bucket
        if occurred_at > bucket.last_request_at:
            bucket.last_request_at = occurred_at
            bucket.inbound_model = event.inbound_model
            bucket.routed_model = event.routed_model
        if occurred_at < window_start:
            continue
        bucket.window_requests += 1
        if event.total_tokens is None:
            continue
        bucket.window_tokens += event.total_tokens
        bucket.window_with_usage += 1

    prepared: list[tuple[str, str, _ActivityAccumulator, float | None]] = []
    scale = 0.0
    for (provider, model), bucket in buckets.items():
        tokens_per_minute = (
            bucket.window_tokens / window_minutes if bucket.window_with_usage else None
        )
        weight = (
            tokens_per_minute
            if tokens_per_minute is not None
            else float(bucket.window_requests)
        )
        scale = max(scale, weight)
        prepared.append((provider, model, bucket, tokens_per_minute))

    grouped: dict[str, list[ModelActivityRow]] = {
        name: [] for name in ("azure", "openai", "openrouter")
    }
    for provider, model, bucket, tokens_per_minute in prepared:
        weight = (
            tokens_per_minute
            if tokens_per_minute is not None
            else float(bucket.window_requests)
        )
        grouped.setdefault(provider, []).append(
            ModelActivityRow(
                model=model,
                inbound_model=bucket.inbound_model,
                routed_model=bucket.routed_model,
                tokens_per_minute=tokens_per_minute,
                tokens_per_minute_label=(
                    None
                    if tokens_per_minute is None
                    else format_activity_rate(tokens_per_minute)
                ),
                requests_in_window=bucket.window_requests,
                last_request_at=bucket.last_request_at,
                last_request_label=format_relative_time(bucket.last_request_at, clock),
                bar_percent=0 if scale <= 0 else round(100 * weight / scale),
                recency=_recency_class(bucket.last_request_at, window_start, clock),
            )
        )

    providers = []
    total_tokens = 0.0
    total_with_usage = 0
    total_requests = 0
    last_request_at = None
    for provider, rows in grouped.items():
        ordered = tuple(
            sorted(
                rows,
                key=lambda row: (
                    -(row.tokens_per_minute or 0.0),
                    -row.requests_in_window,
                    -row.last_request_at.timestamp(),
                    row.model,
                ),
            )
        )
        if not ordered:
            continue
        providers.append(
            ProviderActivityGroup(
                provider=provider,
                label=_PROVIDER_LABELS[provider],
                rows=ordered,
            )
        )
        for row in ordered:
            total_requests += row.requests_in_window
            if row.tokens_per_minute is not None:
                total_tokens += row.tokens_per_minute * window_minutes
                total_with_usage += 1
            if last_request_at is None or row.last_request_at > last_request_at:
                last_request_at = row.last_request_at

    total_tokens_per_minute = (
        total_tokens / window_minutes if total_with_usage else None
    )
    return ActivityBoard(
        window_minutes=window_minutes,
        generated_at=clock,
        total_tokens_per_minute=total_tokens_per_minute,
        total_tokens_per_minute_label=(
            None
            if total_tokens_per_minute is None
            else format_activity_rate(total_tokens_per_minute)
        ),
        total_requests_in_window=total_requests,
        last_request_at=last_request_at,
        last_request_label=(
            format_relative_time(last_request_at, clock)
            if last_request_at is not None
            else None
        ),
        providers=tuple(providers),
    )
