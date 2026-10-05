"""View models for tenant administrator pages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal

from app.persistence.inference_activity import (
    ACTIVITY_LOOKBACK_HOURS,
    ACTIVITY_WINDOW_MINUTES,
)
from app.persistence.models import (
    CostRefreshJob,
    CostUsageRecord,
    InferenceActivityEvent,
    ProviderAttemptEvent,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from app.persistence.provider_circuit_breaker import CircuitSnapshot


@dataclass(frozen=True)
class ProviderStatus:
    """Dashboard status for one provider profile or a missing provider."""

    provider: str
    label: str
    state: str
    is_active: bool
    profile_id: str | None = None
    profile_name: str | None = None
    route_priority: int | None = None
    recency: str | None = None
    last_request_label: str | None = None
    last_request_at: datetime | None = None
    outcome: str | None = None
    last_failure_status_code: int | None = None
    has_current_activity: bool = False
    quota_probe_at: datetime | None = None
    is_disabled: bool = False
    is_overloaded: bool = False
    has_recent_failure: bool = False


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
    tokens_per_request: float | None
    tokens_per_request_label: str | None
    requests_in_window: int
    requests_with_usage_in_window: int
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
class ActivityRequestRow:
    """One recent inference request for the tenant activity list."""

    provider: str
    provider_label: str
    requested_model: str
    routed_model: str | None
    occurred_at: datetime
    time_label: str
    total_tokens_label: str | None
    input_tokens_label: str | None
    output_tokens_label: str | None
    kind: Literal["request"] = "request"


@dataclass(frozen=True)
class FailedProviderAttemptRow:
    """One failed upstream attempt included in the activity list."""

    provider_label: str
    profile_name: str | None
    requested_model: str
    routed_model: str
    occurred_at: datetime
    time_label: str
    status_code: int | None
    outcome_label: str
    failure_details_label: str | None
    kind: Literal["failure"] = "failure"


@dataclass(frozen=True)
class ActivityWindowSummary:
    """Request and reported-token totals for one rolling time window."""

    minutes: int
    label: str
    total_requests: int
    total_requests_with_usage: int
    total_tokens_label: str | None


@dataclass(frozen=True)
class ActivityBoard:
    """Recent proxy requests and aggregate status for the overview."""

    window_minutes: int
    generated_at: datetime
    requests: tuple[ActivityRequestRow, ...]
    window_summaries: tuple[ActivityWindowSummary, ...]
    total_tokens_per_request: float | None
    total_tokens_per_request_label: str | None
    total_requests_in_window: int
    total_requests_with_usage_in_window: int
    last_request_at: datetime | None
    last_request_label: str | None
    providers: tuple[ProviderActivityGroup, ...]
    request_period_hours: int
    failed_attempts: tuple[FailedProviderAttemptRow, ...] = ()
    entries: tuple[ActivityRequestRow | FailedProviderAttemptRow, ...] = ()


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


_PROVIDER_LABELS = {
    "azure": "Azure",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "deepseek": "DeepSeek",
}
_ACTIVITY_REQUEST_LIST_LIMIT = 50
_COST_METRIC_LABELS = {
    "cost": "Anbieterkosten",
    "byok_inference_cost": "Kosten über den eigenen API-Schlüssel",
}
_USAGE_METRIC_LABELS = {
    "input_tokens": "Eingabe-Token",
    "output_tokens": "Ausgabe-Token",
    "prompt_tokens": "Eingabe-Token",
    "completion_tokens": "Ausgabe-Token",
    "reasoning_tokens": "Token für Schlussfolgerungen",
}
_REFRESH_STATUS_LABELS = {
    "running": "Abruf läuft",
    "retry_wait": "Neuer Versuch ausstehend",
    "success": "Abruf erfolgreich",
    "unavailable": "Nicht verfügbar",
    "failed": "Abruf fehlgeschlagen",
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
    """Return a clear setup status for one provider profile."""
    if profile is None:
        return "Nicht eingerichtet"
    if not profile.inference_secret_ciphertext or not profile.default_model:
        return "Einrichtung prüfen"
    if profile.catalog_error:
        return "Einrichtung prüfen"
    return "Eingerichtet"


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
    request_events: Sequence[InferenceActivityEvent] | None = None,
    request_period_hours: int = ACTIVITY_LOOKBACK_HOURS,
    provider_attempts: Sequence[ProviderAttemptEvent] = (),
    circuit_scopes_by_profile: Mapping[str, Sequence[CircuitSnapshot]] | None = None,
    include_activity_board: bool = True,
    failed_attempts: Sequence[tuple[ProviderAttemptEvent, str | None]] = (),
) -> DashboardView:
    """Build provider status and latest-snapshot cost summaries for a tenant."""
    visible_profiles = sorted(
        (profile for profile in profiles if profile.deleted_at is None),
        key=lambda profile: (
            profile.provider,
            profile.route_priority is None,
            profile.route_priority or 0,
            profile.id,
        ),
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
                    _aware_utc(latest_success.completed_at)
                    if latest_success and latest_success.completed_at
                    else None
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
    activity = (
        activity_board(
            activity_events,
            custom_model_id=tenant.custom_model_id,
            now=now,
            request_events=request_events,
            request_period_hours=request_period_hours,
            failed_attempts=failed_attempts,
        )
        if include_activity_board
        else None
    )
    latest_activity_by_profile = {
        event.profile_id: event
        for event in sorted(
            activity_events, key=lambda event: _aware_utc(event.occurred_at)
        )
        if event.profile_id is not None
    }
    attempts_by_profile: dict[str, list[ProviderAttemptEvent]] = {}
    for attempt in provider_attempts:
        attempts_by_profile.setdefault(attempt.profile_id, []).append(attempt)
    now_utc = _aware_utc(
        activity.generated_at
        if activity is not None
        else now or datetime.now(timezone.utc)
    )
    attempt_statuses = {
        profile_id: _provider_activity_status(
            attempts,
            now=now_utc,
            fallback_activity=latest_activity_by_profile.get(profile_id),
        )
        for profile_id, attempts in attempts_by_profile.items()
    }
    circuit_scopes_by_profile = circuit_scopes_by_profile or {}
    profile_statuses = tuple(
        _provider_status(
            profile,
            activity=latest_activity_by_profile.get(profile.id),
            attempt_status=attempt_statuses.get(profile.id),
            now=now_utc,
            activity_window_minutes=(
                activity.window_minutes
                if activity is not None
                else ACTIVITY_WINDOW_MINUTES
            ),
            circuit_snapshots=circuit_scopes_by_profile.get(profile.id, ()),
        )
        for profile in visible_profiles
    )
    configured_providers = {profile.provider for profile in visible_profiles}
    missing_provider_statuses = tuple(
        ProviderStatus(
            provider=provider,
            label=_PROVIDER_LABELS[provider],
            state="Nicht eingerichtet",
            is_active=False,
        )
        for provider in _PROVIDER_LABELS
        if provider not in configured_providers
    )
    statuses = profile_statuses + missing_provider_statuses
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


def _provider_status(
    profile: ProviderProfile,
    *,
    activity: InferenceActivityEvent | None,
    attempt_status: dict[str, object] | None,
    now: datetime,
    activity_window_minutes: int,
    circuit_snapshots: Sequence[CircuitSnapshot],
) -> ProviderStatus:
    """Build a profile card from inference recency and its hysteresis state."""
    outcome = None if attempt_status is None else attempt_status["outcome"]
    has_current_activity = (
        False
        if attempt_status is None
        else bool(attempt_status["has_current_activity"])
    )
    activity_at = _aware_utc(activity.occurred_at) if activity else None
    attempt_at = None if attempt_status is None else attempt_status["last_attempt_at"]
    last_request_at = max(
        (moment for moment in (activity_at, attempt_at) if moment is not None),
        default=None,
    )
    has_current_activity = has_current_activity or bool(
        activity_at and activity_at >= now - timedelta(seconds=20)
    )
    has_recent_status = bool(
        (attempt_status is not None and attempt_status["last_attempt_at"] is not None)
        or (
            activity_at
            and activity_at >= now - timedelta(minutes=activity_window_minutes)
        )
    )
    if outcome is None and has_recent_status:
        outcome = "success"
    if (
        attempt_status is not None
        and attempt_status["has_pending_attempt"]
        and outcome != "failure"
    ):
        outcome = "pending"
    quota_probe_at = max(
        (_aware_utc(snapshot.probe_at) for snapshot in circuit_snapshots),
        default=None,
    )
    quota_lease_until = max(
        (
            _aware_utc(snapshot.lease_until)
            for snapshot in circuit_snapshots
            if snapshot.lease_until is not None
        ),
        default=None,
    )
    if quota_probe_at is not None and quota_probe_at > now:
        quota_status_at = quota_probe_at
    elif quota_lease_until is not None and quota_lease_until > now:
        quota_status_at = quota_lease_until
    else:
        quota_status_at = None
    latest_attempt_is_rate_limited = (
        attempt_status is not None
        and attempt_status["last_attempt_outcome"] == "failure"
        and attempt_status["last_attempt_status_code"] == 429
        and attempt_status["last_attempt_at"] >= now - timedelta(seconds=20)
    )
    has_recent_failure = (
        attempt_status is not None
        and attempt_status["last_attempt_outcome"] == "failure"
        and attempt_status["last_attempt_at"] >= now - timedelta(seconds=20)
    )
    is_overloaded = latest_attempt_is_rate_limited or quota_status_at is not None
    state = provider_state(profile)
    return ProviderStatus(
        provider=profile.provider,
        label=_PROVIDER_LABELS[profile.provider],
        state=state,
        is_active=profile.route_priority is not None,
        is_disabled=profile.route_priority is None and state == "Eingerichtet",
        is_overloaded=is_overloaded,
        has_recent_failure=has_recent_failure,
        profile_id=profile.id,
        profile_name=profile.display_name,
        route_priority=profile.route_priority,
        outcome=outcome,
        last_failure_status_code=(
            None
            if attempt_status is None
            else attempt_status["last_failure_status_code"]
        ),
        has_current_activity=has_current_activity,
        quota_probe_at=quota_status_at,
        recency=(
            "live"
            if has_current_activity
            else (
                _recency_class(
                    last_request_at,
                    now - timedelta(minutes=activity_window_minutes),
                    now,
                )
                if last_request_at is not None
                else None
            )
        ),
        last_request_label=(
            format_relative_time(last_request_at, now) if last_request_at else None
        ),
        last_request_at=last_request_at,
    )


def _provider_activity_status(
    attempts: Sequence[ProviderAttemptEvent],
    *,
    now: datetime,
    fallback_activity: InferenceActivityEvent | None,
) -> dict[str, object]:
    """Apply two-failure activation and 20-second quiet recovery hysteresis."""
    ordered = sorted(
        attempts,
        key=lambda attempt: _aware_utc(attempt.completed_at or attempt.occurred_at),
    )
    last_attempt = ordered[-1] if ordered else None
    latest_at = (
        _aware_utc(last_attempt.completed_at or last_attempt.occurred_at)
        if last_attempt is not None
        else None
    )
    cutoff = now - timedelta(seconds=20)
    has_current_activity = any(
        attempt.outcome == "pending"
        or _aware_utc(attempt.completed_at or attempt.occurred_at) >= cutoff
        for attempt in ordered
    )
    previous_failure_at: datetime | None = None
    error_until: datetime | None = None
    for attempt in ordered:
        attempt_at = _aware_utc(attempt.completed_at or attempt.occurred_at)
        if attempt.outcome == "success":
            previous_failure_at = None
        elif attempt.outcome == "failure":
            consecutive_failure = (
                previous_failure_at is not None
                and attempt_at - previous_failure_at <= timedelta(seconds=20)
            )
            if consecutive_failure or (
                error_until is not None and attempt_at < error_until
            ):
                error_until = attempt_at + timedelta(seconds=20)
            previous_failure_at = attempt_at
    if error_until is not None and now < error_until:
        outcome = "failure"
    elif last_attempt is not None or (
        fallback_activity is not None
        and _aware_utc(fallback_activity.occurred_at)
        >= now - timedelta(minutes=ACTIVITY_WINDOW_MINUTES)
    ):
        outcome = "success"
    else:
        outcome = None
    latest_failure = next(
        (attempt for attempt in reversed(ordered) if attempt.outcome == "failure"),
        None,
    )
    return {
        "outcome": outcome,
        "has_current_activity": has_current_activity,
        "has_pending_attempt": any(attempt.outcome == "pending" for attempt in ordered),
        "last_failure_status_code": (
            None if latest_failure is None else latest_failure.status_code
        ),
        "last_attempt_status_code": (
            None if last_attempt is None else last_attempt.status_code
        ),
        "last_attempt_outcome": None if last_attempt is None else last_attempt.outcome,
        "last_attempt_at": latest_at,
    }


def _aware_utc(moment: datetime) -> datetime:
    """Normalize SQLite-naive and PostgreSQL-aware timestamps to UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def format_token_count(value: int | float) -> str:
    """Format a token count with German digit grouping."""
    if isinstance(value, int) or value.is_integer():
        return format(int(value), ",").replace(",", ".")
    integer, fraction = format(value, ",.1f").split(".")
    return f"{integer.replace(',', '.')},{fraction}"


def format_relative_time(moment: datetime, now: datetime) -> str:
    """Return a compact German relative timestamp for live activity."""
    if moment.tzinfo is None or now.tzinfo is None:
        raise ValueError("Activity timestamps must be timezone-aware")
    elapsed = (now - moment).total_seconds()
    if elapsed < 10:
        return "gerade eben"
    if elapsed < 60:
        seconds = int(elapsed)
        unit = "Sekunde" if seconds == 1 else "Sekunden"
        return f"vor {seconds} {unit}"
    if elapsed < 3600:
        minutes = int(elapsed // 60)
        unit = "Minute" if minutes == 1 else "Minuten"
        return f"vor {minutes} {unit}"
    if elapsed < 86400:
        hours = int(elapsed // 3600)
        unit = "Stunde" if hours == 1 else "Stunden"
        return f"vor {hours} {unit}"
    days = int(elapsed // 86400)
    return f"vor {days} Tag" if days == 1 else f"vor {days} Tagen"


@dataclass
class _ActivityAccumulator:
    inbound_model: str
    routed_model: str | None
    last_request_at: datetime
    window_tokens: int = 0
    window_requests: int = 0
    window_with_usage: int = 0


@dataclass
class _ActivityWindowAccumulator:
    requests: int = 0
    requests_with_usage: int = 0
    total_tokens: int = 0


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


def _activity_request_row(event: InferenceActivityEvent) -> ActivityRequestRow:
    """Prepare one inference event for the chronological request list."""
    occurred_at = _aware_utc(event.occurred_at)
    return ActivityRequestRow(
        provider=event.provider,
        provider_label=_PROVIDER_LABELS[event.provider],
        requested_model=event.inbound_model,
        routed_model=event.routed_model,
        occurred_at=occurred_at,
        time_label=occurred_at.isoformat(),
        total_tokens_label=(
            None
            if event.total_tokens is None
            else format_token_count(event.total_tokens)
        ),
        input_tokens_label=(
            None
            if event.input_tokens is None
            else format_token_count(event.input_tokens)
        ),
        output_tokens_label=(
            None
            if event.output_tokens is None
            else format_token_count(event.output_tokens)
        ),
    )


def _failed_provider_attempt_row(
    attempt: ProviderAttemptEvent, profile_name: str | None
) -> FailedProviderAttemptRow:
    """Prepare a failed upstream attempt without treating it as a user request."""
    occurred_at = _aware_utc(attempt.completed_at or attempt.occurred_at)
    return FailedProviderAttemptRow(
        provider_label=_PROVIDER_LABELS[attempt.provider],
        profile_name=profile_name,
        requested_model=attempt.inbound_model,
        routed_model=attempt.routed_model,
        occurred_at=occurred_at,
        time_label=occurred_at.isoformat(),
        status_code=attempt.status_code,
        outcome_label=_provider_attempt_outcome_label(attempt.status_code),
        failure_details_label=_failure_details_label(attempt.failure_details),
    )


def _failure_details_label(details: Mapping[str, object] | None) -> str | None:
    """Format persisted allowlisted failure diagnostics for the activity list."""
    if not details:
        return None
    labels = [
        str(value)
        for key in ("error_code", "exception_type", "provider_error_code")
        if isinstance((value := details.get(key)), str) and value
    ]
    return " · ".join(labels) or None


def _provider_attempt_outcome_label(status_code: int | None) -> str:
    """Render a concise, accurate label without guessing provider error details."""
    if status_code == 429:
        return "Rate-Limit oder Überlastung · HTTP 429"
    if status_code is None:
        return "Provider-Versuch fehlgeschlagen · kein HTTP-Status aufgezeichnet"
    return f"Provider-Versuch fehlgeschlagen · HTTP {status_code}"


def activity_board(
    events: Sequence[InferenceActivityEvent],
    *,
    custom_model_id: str,
    now: datetime | None = None,
    window_minutes: int = ACTIVITY_WINDOW_MINUTES,
    request_events: Sequence[InferenceActivityEvent] | None = None,
    request_period_hours: int = ACTIVITY_LOOKBACK_HOURS,
    lookback_hours: int = ACTIVITY_LOOKBACK_HOURS,
    failed_attempts: Sequence[tuple[ProviderAttemptEvent, str | None]] = (),
) -> ActivityBoard:
    """Prepare recent request rows and provider recency for the overview."""
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        raise ValueError("Activity board clock must be timezone-aware")
    window_start = clock - timedelta(minutes=window_minutes)
    hour_start = clock - timedelta(hours=1)
    lookback_start = clock - timedelta(hours=lookback_hours)
    request_start = clock - timedelta(hours=request_period_hours)
    list_events = request_events if request_events is not None else events
    request_rows = sorted(
        (
            _activity_request_row(event)
            for event in list_events
            if _aware_utc(event.occurred_at) >= request_start
        ),
        key=lambda row: row.occurred_at,
        reverse=True,
    )[:_ACTIVITY_REQUEST_LIST_LIMIT]
    summary_windows = (
        (
            ACTIVITY_WINDOW_MINUTES,
            f"Letzte {ACTIVITY_WINDOW_MINUTES} Minuten",
            clock - timedelta(minutes=ACTIVITY_WINDOW_MINUTES),
        ),
        (60, "Letzte Stunde", hour_start),
    )
    summary_accumulators = {
        minutes: _ActivityWindowAccumulator()
        for minutes, _label, _start in summary_windows
    }
    buckets: dict[tuple[str, str], _ActivityAccumulator] = {}
    total_tokens = 0
    total_requests = 0
    total_requests_with_usage = 0
    for event in events:
        occurred_at = _aware_utc(event.occurred_at)
        if occurred_at <= clock and occurred_at >= hour_start:
            for minutes, _label, summary_start in summary_windows:
                if occurred_at >= summary_start:
                    summary = summary_accumulators[minutes]
                    summary.requests += 1
                    if event.total_tokens is not None:
                        summary.requests_with_usage += 1
                        summary.total_tokens += event.total_tokens
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
        total_requests += 1
        if event.total_tokens is None:
            continue
        bucket.window_tokens += event.total_tokens
        bucket.window_with_usage += 1
        total_tokens += event.total_tokens
        total_requests_with_usage += 1

    window_summaries = tuple(
        ActivityWindowSummary(
            minutes=minutes,
            label=label,
            total_requests=summary_accumulators[minutes].requests,
            total_requests_with_usage=summary_accumulators[minutes].requests_with_usage,
            total_tokens_label=(
                format_token_count(summary_accumulators[minutes].total_tokens)
                if summary_accumulators[minutes].requests_with_usage
                else None
            ),
        )
        for minutes, label, _summary_start in summary_windows
    )
    prepared: list[tuple[str, str, _ActivityAccumulator, float | None]] = []
    scale = 0.0
    for (provider, model), bucket in buckets.items():
        tokens_per_request = (
            bucket.window_tokens / bucket.window_with_usage
            if bucket.window_with_usage
            else None
        )
        weight = tokens_per_request or 0.0
        scale = max(scale, weight)
        prepared.append((provider, model, bucket, tokens_per_request))

    grouped: dict[str, list[ModelActivityRow]] = {
        name: [] for name in ("azure", "openai", "openrouter", "deepseek")
    }
    for provider, model, bucket, tokens_per_request in prepared:
        weight = tokens_per_request or 0.0
        grouped.setdefault(provider, []).append(
            ModelActivityRow(
                model=model,
                inbound_model=bucket.inbound_model,
                routed_model=bucket.routed_model,
                tokens_per_request=tokens_per_request,
                tokens_per_request_label=(
                    None
                    if tokens_per_request is None
                    else format_token_count(tokens_per_request)
                ),
                requests_in_window=bucket.window_requests,
                requests_with_usage_in_window=bucket.window_with_usage,
                last_request_at=bucket.last_request_at,
                last_request_label=format_relative_time(bucket.last_request_at, clock),
                bar_percent=0 if scale <= 0 else round(100 * weight / scale),
                recency=_recency_class(bucket.last_request_at, window_start, clock),
            )
        )

    providers = []
    last_request_at = None
    for provider, rows in grouped.items():
        ordered = tuple(
            sorted(
                rows,
                key=lambda row: (
                    -(row.tokens_per_request or 0.0),
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
            if last_request_at is None or row.last_request_at > last_request_at:
                last_request_at = row.last_request_at

    prepared_failures = tuple(
        _failed_provider_attempt_row(attempt, profile_name)
        for attempt, profile_name in sorted(
            (
                item
                for item in failed_attempts
                if request_start
                <= _aware_utc(item[0].completed_at or item[0].occurred_at)
                <= clock
            ),
            key=lambda item: _aware_utc(item[0].completed_at or item[0].occurred_at),
            reverse=True,
        )[:_ACTIVITY_REQUEST_LIST_LIMIT]
    )
    entries = tuple(
        sorted(
            (*request_rows, *prepared_failures),
            key=lambda row: row.occurred_at,
            reverse=True,
        )[:_ACTIVITY_REQUEST_LIST_LIMIT]
    )
    total_tokens_per_request = (
        total_tokens / total_requests_with_usage if total_requests_with_usage else None
    )
    return ActivityBoard(
        window_minutes=window_minutes,
        generated_at=clock,
        requests=tuple(request_rows),
        window_summaries=tuple(window_summaries),
        total_tokens_per_request=total_tokens_per_request,
        total_tokens_per_request_label=(
            None
            if total_tokens_per_request is None
            else format_token_count(total_tokens_per_request)
        ),
        total_requests_in_window=total_requests,
        total_requests_with_usage_in_window=total_requests_with_usage,
        last_request_at=last_request_at,
        last_request_label=(
            format_relative_time(last_request_at, clock)
            if last_request_at is not None
            else None
        ),
        providers=tuple(providers),
        request_period_hours=request_period_hours,
        failed_attempts=prepared_failures,
        entries=entries,
    )
