"""Sequential tenant inference routing with explicit logical model identity."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from random import SystemRandom
from threading import Event, Lock, Thread
from time import monotonic

from flask import Request, Response, current_app, g, jsonify
from gevent import sleep as cooperative_sleep

from app.azure.adapter import AzureAdapter
from app.exceptions import ServiceConfigurationError
from app.persistence.database import Database
from app.persistence.inference_activity import (
    complete_provider_attempt,
    start_provider_attempt,
)
from app.persistence.provider_circuit_breaker import (
    CircuitPermit,
    ProviderCircuitBreakerStore,
    ProviderCircuitStoreError,
)
from app.persistence.provider_scheduler import (
    BudgetLease,
    BudgetMetric,
    BudgetPolicy,
    BudgetScope,
    BudgetScopeKind,
    ProviderBudgetScheduler,
    ProviderConcurrencyCandidate,
    ProviderConcurrencyController,
    ProviderConcurrencyLease,
    ProviderSchedulerPolicyConflict,
    ProviderSchedulerStoreError,
    ReservationRequest,
)
from app.providers.circuit_breaker import (
    ProviderCircuitAttempt,
    retry_after_header,
)
from app.providers.failover_upstream import UpstreamError
from app.providers.model_ids import account_model_id, qualified_model_id
from app.providers.openai_compat import forward_openai_compatible
from app.tenants import DatabaseTenantRoutingSnapshot, DatabaseTenantSnapshot

MAX_TRANSIENT_RETRY_WAIT_SECONDS = 300.0
DEFAULT_TRANSIENT_RETRY_DELAY_SECONDS = 1.0
RETRY_COOLDOWN_GRACE_SECONDS = 0.05


@dataclass
class ProviderConcurrencyAttempt:
    """Renew and settle one adaptive provider concurrency permit."""

    controller: ProviderConcurrencyController
    lease: ProviderConcurrencyLease | None
    settled: bool = False
    _stopping: Event = field(default_factory=Event, init=False, repr=False)
    lease_lost: Event = field(default_factory=Event, init=False, repr=False)
    _settlement_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)
    _last_renewal: float = field(
        default_factory=lambda: monotonic(), init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Start renewal only for a permit issued by durable storage."""
        if self.lease is not None:
            self._thread = Thread(
                target=self._heartbeat,
                name=f"provider-concurrency-{self.lease.id}",
                daemon=True,
            )
            self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stopping.wait(self.controller.heartbeat_interval_seconds):
            try:
                if not self.controller.renew(self.lease):
                    self.lease_lost.set()
                    return
                self._last_renewal = monotonic()
            except ProviderSchedulerStoreError:
                if (
                    monotonic() - self._last_renewal
                    >= self.controller.lease_ttl_seconds
                ):
                    self.lease_lost.set()
                    return

    def complete(self) -> None:
        """Release capacity and additively increase the bound after success."""
        self._settle("success")

    def failed(self, error: UpstreamError) -> None:
        """Release capacity and multiplicatively decrease on transient failures."""
        outcome = (
            "transient_failure"
            if error.classification.category == "transient"
            else "terminal_failure"
        )
        self._settle(outcome)

    def release(self) -> None:
        """Release capacity without changing AIMD after an aborted/unstarted call."""
        self._settle("released")

    def _settle(self, outcome: str) -> None:
        with self._settlement_lock:
            if self.settled:
                return
            self.settled = True
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)
        if self.lease is not None:
            self.controller.settle(self.lease, outcome=outcome)


@dataclass
class ProviderBudgetAttempt:
    """Settle one durable reservation exactly once at the stream terminal."""

    scheduler: ProviderBudgetScheduler
    lease: BudgetLease | None
    failure_handler: Callable[[UpstreamError, float | None], datetime | None]
    concurrency_attempt: ProviderConcurrencyAttempt | None = None
    settled: bool = False
    cooldown_recorded: bool = False
    _completion_pending: bool = field(default=False, init=False, repr=False)
    _pending_actual_tokens: int | None = field(default=None, init=False, repr=False)

    def failed(
        self, error: UpstreamError | None, retry_delay: float | None = None
    ) -> datetime | None:
        """Charge failed work, adjust AIMD, and persist available retry guidance."""
        if (
            retry_delay is None
            and error is not None
            and error.classification.category == "transient"
        ):
            retry_delay = DEFAULT_TRANSIENT_RETRY_DELAY_SECONDS
        if not self.settled:
            try:
                if self.lease is not None:
                    settle = (
                        self.scheduler.failed
                        if error is None or error.upstream_started
                        else self.scheduler.release
                    )
                    settle(self.lease)
            finally:
                self.settled = True
                if self.concurrency_attempt is not None:
                    if error is None or not error.upstream_started:
                        self.concurrency_attempt.release()
                    else:
                        self.concurrency_attempt.failed(error)
        if self.cooldown_recorded or error is None or not error.upstream_started:
            return None
        retry_at = self.failure_handler(error, retry_delay)
        self.cooldown_recorded = True
        return retry_at

    def complete(self, actual_tokens: int | None) -> None:
        """Charge request/known usage and increase AIMD after successful completion."""
        if self.settled:
            return
        self._completion_pending = True
        self._pending_actual_tokens = actual_tokens
        self._complete_pending()

    def _complete_pending(self) -> None:
        if self.lease is not None:
            self.scheduler.complete(
                self.lease, actual_tokens=self._pending_actual_tokens
            )
        self._completion_pending = False
        self._pending_actual_tokens = None
        self.settled = True
        if self.concurrency_attempt is not None:
            self.concurrency_attempt.complete()

    def release(self) -> None:
        """Release reservations on error, retrying any pending successful charge."""
        if self.settled:
            return
        if self._completion_pending:
            self._complete_pending()
            return
        try:
            if self.lease is not None:
                self.scheduler.release(self.lease)
        finally:
            self.settled = True
            if self.concurrency_attempt is not None:
                self.concurrency_attempt.release()


ProviderForwarder = Callable[
    [
        Request,
        DatabaseTenantSnapshot,
        str,
        int | None,
        ProviderCircuitAttempt | None,
        ProviderBudgetAttempt | None,
    ],
    Response,
]


@dataclass
class RouteAttemptState:
    """Accumulate request-level provider availability results."""

    blocked_until: list[datetime | None]
    lease_contention_seen: bool = False
    transient_failure_seen: bool = False
    transient_retry_after: float | None = None
    last_retryable_error: UpstreamError | None = None
    circuit_state_unavailable: bool = False
    scheduler_state_unavailable: bool = False
    attempts_started: int = 0
    budget_blocked_until: list[datetime] = field(default_factory=list)
    concurrency_blocked_until: list[datetime] = field(default_factory=list)
    retry_count: int = 0
    candidate_configuration_errors: list[tuple[str, str]] = field(default_factory=list)

    def record_error(self, error: UpstreamError) -> None:
        """Retain transient retry guidance and the latest retryable failure."""
        self.last_retryable_error = error
        if error.classification.category != "transient":
            return
        self.transient_failure_seen = True
        retry_after = error.classification.retry_after_seconds
        if retry_after is not None:
            self.transient_retry_after = max(
                self.transient_retry_after or 0, retry_after
            )

    def record_blocked(self, retry_at: datetime | None, *, lease_blocked: bool) -> None:
        """Record a circuit block and whether an active probe lease may expire."""
        self.blocked_until.append(retry_at)
        self.lease_contention_seen |= lease_blocked


@dataclass(frozen=True)
class SchedulerCandidate:
    """Validated explicit policies and durable cooldown scopes for one candidate."""

    policies: tuple[BudgetPolicy, ...]
    cooldown_scopes: tuple[BudgetScope, ...]
    token_estimate: int | None


@dataclass(frozen=True)
class RouteReservationResult:
    """Reservations acquired before forwarding or reason a candidate was skipped."""

    concurrency_attempt: ProviderConcurrencyAttempt | None = None
    budget_attempt: ProviderBudgetAttempt | None = None
    skip_candidate: bool = False
    response: Response | None = None


@dataclass(frozen=True)
class RouteCandidate:
    """Validated settings needed to compare and attempt one route target."""

    profile: DatabaseTenantSnapshot
    target_model: str
    scheduler: SchedulerCandidate
    budget_headroom: float = 1.0


def _route_candidate_groups(
    route_targets: tuple[tuple[DatabaseTenantSnapshot, str], ...],
    snapshot: DatabaseTenantRoutingSnapshot,
) -> tuple[tuple[tuple[DatabaseTenantSnapshot, str], ...], ...]:
    """Group candidates by priority and applicable cost policy, stably."""
    priority_groups: dict[int, list[tuple[DatabaseTenantSnapshot, str]]] = {}
    for profile, model in route_targets:
        priority = (
            profile.route_priority if profile.route_priority is not None else 2**31
        )
        priority_groups.setdefault(priority, []).append((profile, model))

    groups = []
    for priority in sorted(priority_groups):
        candidates = sorted(
            priority_groups[priority], key=lambda target: target[0].profile_id or ""
        )
        policy = snapshot.routing_settings.cost_policy
        if policy == "ignore":
            groups.append(tuple(candidates))
            continue
        if policy == "cost_tiers":
            by_tier: dict[int, list[tuple[DatabaseTenantSnapshot, str]]] = {}
            for candidate in candidates:
                by_tier.setdefault(candidate[0].routing_settings.cost_tier, []).append(
                    candidate
                )
            groups.extend(tuple(by_tier[tier]) for tier in sorted(by_tier))
            continue

        comparable = _comparable_cost_groups(candidates)
        groups.extend(comparable)
    return tuple(groups)


def _comparable_cost_groups(
    candidates: list[tuple[DatabaseTenantSnapshot, str]],
) -> tuple[tuple[tuple[DatabaseTenantSnapshot, str], ...], ...]:
    """Sort complete same-model, same-currency prices; never infer unknown cost."""
    if len({model for _profile, model in candidates}) > 1:
        return (tuple(candidates),)

    pricing_fields = (
        "input_per_1m_tokens",
        "output_per_1m_tokens",
        "cache_per_1m_tokens",
        "currency",
    )
    fully_priced = [
        (profile, model, profile.catalog_pricing[model])
        for profile, model in candidates
        if model in profile.catalog_pricing
        and all(key in profile.catalog_pricing[model] for key in pricing_fields)
    ]
    currencies = {price["currency"] for _profile, _model, price in fully_priced}
    if len(currencies) != 1 or len(fully_priced) != len(candidates):
        return (tuple(candidates),)

    def total_price(
        item: tuple[DatabaseTenantSnapshot, str, Mapping[str, str]],
    ) -> Decimal:
        values = [Decimal(item[2][key]) for key in pricing_fields[:-1]]
        if any(not value.is_finite() or value < 0 for value in values):
            raise InvalidOperation
        return sum(values)

    try:
        ranked = sorted(
            fully_priced,
            key=lambda item: (total_price(item), item[0].profile_id or ""),
        )
        by_cost: dict[Decimal, list[tuple[DatabaseTenantSnapshot, str]]] = {}
        for profile, model, price in ranked:
            by_cost.setdefault(total_price((profile, model, price)), []).append(
                (profile, model)
            )
    except (InvalidOperation, ValueError):
        return (tuple(candidates),)
    return tuple(tuple(by_cost[cost]) for cost in sorted(by_cost))


def _forward_azure(
    req: Request,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    attempt_id: int | None,
    circuit_attempt: ProviderCircuitAttempt | None,
    budget_attempt: ProviderBudgetAttempt | None,
) -> Response:
    """Make one Azure attempt for the centrally selected provider model."""
    return AzureAdapter().forward_attempt(
        req,
        profile,
        target_model=target_model,
        attempt_id=attempt_id,
        circuit_attempt=circuit_attempt,
        budget_attempt=budget_attempt,
    )


def _forward_openai_compatible(
    req: Request,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    attempt_id: int | None,
    circuit_attempt: ProviderCircuitAttempt | None,
    budget_attempt: ProviderBudgetAttempt | None,
) -> Response:
    """Make one OpenAI-compatible attempt for the centrally selected model."""
    return forward_openai_compatible(
        req,
        profile,
        target_model=target_model,
        attempt_id=attempt_id,
        circuit_attempt=circuit_attempt,
        budget_attempt=budget_attempt,
    )


_PROVIDER_FORWARDERS: dict[str, ProviderForwarder] = {
    "azure": _forward_azure,
    "openai": _forward_openai_compatible,
    "openrouter": _forward_openai_compatible,
    "deepseek": _forward_openai_compatible,
}


def routed_profiles(
    snapshot: DatabaseTenantRoutingSnapshot, *, azure_only: bool = False
) -> tuple[DatabaseTenantSnapshot, ...]:
    """Return enabled profiles in configured order for tenant-default routing."""
    return _eligible_profiles(snapshot.profiles, azure_only=azure_only)


def catalog_profiles(
    snapshot: DatabaseTenantRoutingSnapshot, *, azure_only: bool = False
) -> tuple[DatabaseTenantSnapshot, ...]:
    """Return ready profiles available for explicit catalog-model selection."""
    return _eligible_profiles(snapshot.available_profiles, azure_only=azure_only)


def _eligible_profiles(
    profiles: tuple[DatabaseTenantSnapshot, ...], *, azure_only: bool
) -> tuple[DatabaseTenantSnapshot, ...]:
    """Apply provider switches while preserving the database snapshot order."""
    return tuple(
        profile
        for profile in profiles
        if (not azure_only or profile.provider == "azure")
        and (
            profile.provider != "azure" or current_app.config.get("ENABLE_AZURE", False)
        )
    )


def _route_targets(
    profiles: tuple[DatabaseTenantSnapshot, ...],
    inbound_model: object,
    custom_model_id: str,
) -> tuple[tuple[DatabaseTenantSnapshot, str], ...]:
    """Resolve each profile target while preserving the configured route priority."""
    if not isinstance(inbound_model, str) or not inbound_model:
        raise ServiceConfigurationError(
            "Request model must match the tenant's Cursor model ID."
        )
    if inbound_model == custom_model_id:
        route_targets = []
        for profile in profiles:
            if profile.default_model is None:
                raise ServiceConfigurationError(
                    "The active provider profile has no default model configured."
                )
            route_targets.append((profile, profile.default_model))
        return tuple(route_targets)

    requested_model = inbound_model.casefold()
    native_targets = [
        (profile, model)
        for profile in profiles
        for model in profile.catalog_model_ids
        if model.casefold() == requested_model
    ]
    if native_targets:
        return tuple(native_targets)

    qualified_targets = [
        (profile, model)
        for profile in profiles
        if profile.provider is not None and profile.profile_name is not None
        for model in profile.catalog_model_ids
        if qualified_model_id(profile.provider, profile.profile_name, model).casefold()
        == requested_model
    ]
    if len(qualified_targets) == 1:
        return (qualified_targets[0],)
    if len(qualified_targets) > 1:
        raise ServiceConfigurationError(
            "Provider-qualified model ID is ambiguous; check provider account names."
        )

    account_targets = [
        (profile, model)
        for profile in profiles
        if profile.profile_name is not None
        for model in profile.catalog_model_ids
        if account_model_id(profile.profile_name, model).casefold() == requested_model
    ]
    if len(account_targets) == 1:
        return (account_targets[0],)
    if len(account_targets) > 1:
        raise ServiceConfigurationError(
            "Account-qualified model ID is ambiguous; include its provider name."
        )

    raise ServiceConfigurationError(
        "Request model must match the tenant's Cursor model ID."
    )


def _forward_profile(
    req: Request,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    attempt_id: int | None,
    circuit_attempt: ProviderCircuitAttempt | None,
    budget_attempt: ProviderBudgetAttempt | None,
) -> Response:
    """Dispatch one profile to its registered protocol adapter."""
    if profile.provider is None:
        raise ServiceConfigurationError("The provider profile has no provider.")
    forwarder = _PROVIDER_FORWARDERS.get(profile.provider)
    if forwarder is None:
        raise ServiceConfigurationError(f"Unsupported provider {profile.provider!r}.")
    return forwarder(
        req, profile, target_model, attempt_id, circuit_attempt, budget_attempt
    )


def _scheduler_candidate(
    snapshot: DatabaseTenantRoutingSnapshot,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    payload: Mapping[str, object],
) -> SchedulerCandidate:
    """Validate profile scheduler tuples and derive this candidate's scopes."""
    if profile.provider is None or profile.profile_id is None:
        raise ServiceConfigurationError("Provider scheduling identity is unavailable.")
    settings = profile.provider_settings
    raw_limits = settings.get("scheduler_limits")
    if "scheduler_limits" in settings and not isinstance(raw_limits, list):
        raise ServiceConfigurationError(
            "scheduler_limits must be a list of policy tuples."
        )

    policies: list[BudgetPolicy] = []
    for index, entry in enumerate(raw_limits or []):
        if (
            not isinstance(entry, dict)
            or not {"scope_kind", "metric", "limit", "window_seconds"}.issubset(entry)
            or not set(entry).issubset(
                {"scope_kind", "scope_id", "metric", "limit", "window_seconds"}
            )
        ):
            raise ServiceConfigurationError(
                f"scheduler_limits[{index}] must contain exactly "
                "scope_kind, scope_id, metric, limit, and window_seconds."
            )
        try:
            kind = BudgetScopeKind(entry["scope_kind"])
            metric = BudgetMetric(entry["metric"])
            configured_scope_id = entry.get("scope_id")
            if (
                kind is BudgetScopeKind.MODEL
                and isinstance(configured_scope_id, str)
                and configured_scope_id != target_model
            ):
                continue
            scope_id = _scheduler_scope_id(
                profile.provider, kind, configured_scope_id, profile, target_model
            )
            scope = _scheduler_scope(
                profile.provider, kind, scope_id, profile, target_model
            )
            policy = BudgetPolicy(
                scope, metric, entry["limit"], entry["window_seconds"]
            )
            if any(
                existing.scope == policy.scope and existing.metric is policy.metric
                for existing in policies
            ):
                raise ValueError("duplicate scope_kind/scope_id/metric policy")
            policies.append(policy)
        except (TypeError, ValueError) as exc:
            raise ServiceConfigurationError(
                f"scheduler_limits[{index}] is invalid: {exc}"
            ) from exc

    scopes = [
        _scheduler_scope(
            profile.provider,
            BudgetScopeKind.PROFILE,
            profile.profile_id,
            profile,
            target_model,
        )
    ]
    for policy in policies:
        if policy.scope not in scopes:
            scopes.append(policy.scope)
    for key, kind in (
        ("organization", BudgetScopeKind.ORGANIZATION),
        ("project", BudgetScopeKind.PROJECT),
    ):
        configured_id = settings.get(key)
        if isinstance(configured_id, str) and configured_id.strip():
            scope = _scheduler_scope(
                profile.provider, kind, configured_id, profile, target_model
            )
            if scope not in scopes:
                scopes.append(scope)

    token_policies = [
        policy for policy in policies if policy.metric is BudgetMetric.TOKENS
    ]
    estimate = _token_reservation(payload, settings) if token_policies else None
    if token_policies and estimate is None:
        raise ServiceConfigurationError(
            "Token scheduler limits require token_reservation_estimate because "
            "output caps do not estimate prompt usage."
        )
    return SchedulerCandidate(tuple(policies), tuple(scopes), estimate)


def _scheduler_scope_id(
    provider: str,
    kind: BudgetScopeKind,
    configured_scope_id: object,
    profile: DatabaseTenantSnapshot,
    target_model: str,
) -> str:
    """Resolve an omitted policy identity and reject any explicit mismatch."""
    expected = {
        BudgetScopeKind.PROVIDER: provider,
        BudgetScopeKind.PROFILE: profile.profile_id,
        BudgetScopeKind.MODEL: target_model,
        BudgetScopeKind.ORGANIZATION: profile.provider_settings.get("organization"),
        BudgetScopeKind.PROJECT: profile.provider_settings.get("project"),
    }[kind]
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError(f"{kind.value} scope identity is not configured")
    if configured_scope_id is not None and configured_scope_id != expected:
        raise ValueError(
            f"scope_id for {kind.value} must match the persisted provider identity"
        )
    return expected


def _scheduler_scope(
    provider: str,
    kind: BudgetScopeKind,
    scope_id: str,
    profile: DatabaseTenantSnapshot,
    target_model: str,
) -> BudgetScope:
    """Bind configured policy IDs to the actual routed provider identity."""
    expected = {
        BudgetScopeKind.PROVIDER: provider,
        BudgetScopeKind.PROFILE: profile.profile_id,
        BudgetScopeKind.MODEL: target_model,
        BudgetScopeKind.ORGANIZATION: profile.provider_settings.get("organization"),
        BudgetScopeKind.PROJECT: profile.provider_settings.get("project"),
    }[kind]
    if scope_id != expected:
        raise ServiceConfigurationError(
            f"scheduler_limits scope_id for {kind.value} must match the persisted provider identity."
        )
    return BudgetScope(provider, kind, scope_id)


def _token_reservation(
    payload: Mapping[str, object], settings: Mapping[str, object]
) -> int | None:
    """Reserve from a visible output cap or an explicit conservative estimate."""
    visible_cap = None
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        value = payload.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ServiceConfigurationError(
                    f"{key} must be a nonnegative integer for token scheduling."
                )
            visible_cap = value
            break
    estimate = settings.get("token_reservation_estimate")
    if estimate is not None and (
        isinstance(estimate, bool) or not isinstance(estimate, int) or estimate <= 0
    ):
        raise ServiceConfigurationError(
            "token_reservation_estimate must be a positive integer."
        )
    if estimate is None:
        return None
    return max(visible_cap or 0, estimate)


def _scheduler_failure_handler(
    scheduler: ProviderBudgetScheduler,
    snapshot: DatabaseTenantRoutingSnapshot,
    candidate: SchedulerCandidate,
) -> Callable[[UpstreamError, float | None], datetime | None]:
    """Persist transient cooldowns to every provider-local and shared scope."""

    def persist(error: UpstreamError, retry_delay: float | None) -> datetime | None:
        if error.classification.category != "transient":
            return None
        delay = error.classification.retry_after_seconds
        if delay is None:
            delay = retry_delay
        if delay is None:
            return None
        retry_times = [
            scheduler.apply_cooldown(scope, delay, tenant_id=snapshot.id)
            for scope in candidate.cooldown_scopes
        ]
        return max(retry_times) if retry_times else None

    return persist


def _reservation_request(
    snapshot: DatabaseTenantRoutingSnapshot, candidate: SchedulerCandidate
) -> ReservationRequest:
    """Bind one validated candidate's budgets to its authenticated tenant."""
    return ReservationRequest(
        tenant_id=snapshot.id,
        policies=candidate.policies,
        estimated_tokens=candidate.token_estimate,
        cooldown_scopes=candidate.cooldown_scopes,
    )


def _budget_attempt(
    scheduler: ProviderBudgetScheduler,
    snapshot: DatabaseTenantRoutingSnapshot,
    candidate: SchedulerCandidate,
    concurrency_attempt: ProviderConcurrencyAttempt | None,
) -> ProviderBudgetAttempt:
    """Check durable cooldowns and reserve explicit budgets before upstream I/O."""
    request = _reservation_request(snapshot, candidate)
    decision = scheduler.reserve(request)
    if not decision.allowed:
        raise BudgetUnavailableError(decision.retry_at)
    if decision.lease is None and candidate.policies:
        raise ProviderSchedulerStoreError(
            "Provider scheduler allowed a budgeted request without issuing a lease"
        )
    return ProviderBudgetAttempt(
        scheduler,
        decision.lease,
        _scheduler_failure_handler(scheduler, snapshot, candidate),
        concurrency_attempt,
    )


class BudgetUnavailableError(Exception):
    """One provider profile is currently blocked by its durable scheduler."""

    def __init__(self, retry_at: datetime | None) -> None:
        """Store the scheduler-provided retry time, when known."""
        self.retry_at = retry_at


def _provider_failure_details(error: UpstreamError) -> dict[str, object]:
    """Return only the classified and allowlisted fields for attempt storage."""
    details: dict[str, object] = {"error_code": error.code}
    for key in (
        "provider_error_code",
        "provider_error_type",
        "provider_error_param",
        "provider_limit_source",
        "provider_request_id",
    ):
        value = getattr(error, key)
        if value is not None:
            details[key] = value
    provider_diagnostics = getattr(error, "provider_diagnostics", None)
    if isinstance(provider_diagnostics, dict) and provider_diagnostics:
        details["provider_diagnostics"] = provider_diagnostics
    return details


def _service_unavailable(code: str, message: str, retry_after: str) -> Response:
    response = jsonify({"error": {"code": code, "message": message}})
    response.status_code = 503
    response.headers["Retry-After"] = retry_after
    return response


def _quota_unavailable(retry_at: datetime) -> Response:
    return _service_unavailable(
        "provider_quota_unavailable",
        "All configured providers are temporarily unavailable.",
        retry_after_header(retry_at),
    )


def _resolve_request_targets(
    snapshot: DatabaseTenantRoutingSnapshot,
    inbound_model: object,
    route_profiles: tuple[DatabaseTenantSnapshot, ...],
    catalog_candidates: tuple[DatabaseTenantSnapshot, ...],
    *,
    azure_only: bool,
) -> tuple[tuple[DatabaseTenantSnapshot, str], ...]:
    """Resolve the default cascade or an explicit catalog model."""
    if inbound_model == snapshot.custom_model_id:
        if not route_profiles:
            message = "No active provider profile is configured for this tenant."
            if azure_only:
                message = (
                    "The active tenant provider is not available on the Azure route."
                )
            raise ServiceConfigurationError(message)
        candidates = route_profiles
    else:
        candidates = route_profiles if azure_only else catalog_candidates
    return _route_targets(candidates, inbound_model, snapshot.custom_model_id)


def _acquire_route_permit(
    snapshot: DatabaseTenantRoutingSnapshot,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    is_retry: bool,
    breaker_store: ProviderCircuitBreakerStore | None,
    state: RouteAttemptState,
    retry_targets: list,
    deadline: float,
) -> CircuitPermit | None:
    """Acquire breaker leases and queue one bounded retry for lease contention."""
    if breaker_store is None:
        return CircuitPermit(True)
    scopes = breaker_store.scopes_for_profile(
        snapshot.id,
        profile.provider,
        profile.profile_id,
        profile.provider_settings,
    )
    try:
        permit = breaker_store.acquire(scopes)
    except ProviderCircuitStoreError:
        state.circuit_state_unavailable = True
        current_app.logger.exception(
            "Provider circuit state unavailable; skipping profile=%s",
            profile.profile_id,
        )
        return None
    if not permit.allowed:
        state.record_blocked(permit.retry_at, lease_blocked=permit.lease_blocked)
        current_app.logger.info(
            "Provider route candidate skipped: request_id=%s tenant=%s "
            "provider=%s profile=%s model=%s fallback_reason=%s retry_at=%s",
            getattr(g, "proxy_request_id", "unavailable"),
            snapshot.id,
            profile.provider,
            profile.profile_id,
            target_model,
            ("probe_lease_contention" if permit.lease_blocked else "circuit_cooldown"),
            (permit.retry_at.isoformat() if permit.retry_at is not None else "unknown"),
        )
        if permit.lease_blocked:
            if not is_retry and permit.retry_at is not None:
                lease_remaining = max(
                    0.0,
                    (permit.retry_at - datetime.now(timezone.utc)).total_seconds(),
                )
                retry_at = monotonic() + lease_remaining
                if retry_at <= deadline:
                    retry_targets.append((profile, target_model, True, retry_at))
        return None
    return permit


def _transient_retry_delay(
    error: UpstreamError,
    is_retry: bool,
    retry_count: int,
    max_retry_wait_seconds: int = int(MAX_TRANSIENT_RETRY_WAIT_SECONDS),
) -> float | None:
    """Calculate a same-candidate delay bounded by validated routing policy."""
    retry_after = error.classification.retry_after_seconds
    if (
        error.classification.category != "transient"
        or is_retry
        or error.status not in {402, 429}
    ):
        return None
    if retry_after is not None:
        if retry_after > max_retry_wait_seconds:
            return None
        delay = retry_after
    else:
        max_exponent = math.ceil(
            math.log2(max_retry_wait_seconds / DEFAULT_TRANSIENT_RETRY_DELAY_SECONDS)
        )
        ceiling = min(
            max_retry_wait_seconds,
            DEFAULT_TRANSIENT_RETRY_DELAY_SECONDS
            * (2 ** min(retry_count, max_exponent)),
        )
        delay = SystemRandom().uniform(0, ceiling)
    return max(delay, 0.1)


def _queue_transient_retry(
    profile: DatabaseTenantSnapshot,
    target_model: str,
    retry_delay: float,
    retry_at: datetime | None,
    retry_targets: list,
    deadline: float,
) -> None:
    """Queue the retry no earlier than both provider guidance and saved cooldown."""
    remaining = (
        max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        if retry_at is not None
        else retry_delay
    )
    eligible_at = monotonic() + remaining + RETRY_COOLDOWN_GRACE_SECONDS
    if eligible_at <= deadline:
        retry_targets.append((profile, target_model, True, eligible_at))


def _log_provider_failure(
    snapshot: DatabaseTenantRoutingSnapshot,
    profile: DatabaseTenantSnapshot,
    error: UpstreamError,
) -> None:
    """Log classified, allowlisted provider diagnostics without raw error text."""
    current_app.logger.warning(
        "Provider attempt failed: request_id=%s tenant=%s profile=%s "
        "provider=%s status=%s retryable=%s error_code=%s "
        "error_category=%s provider_error_code=%s provider_error_type=%s "
        "provider_error_param=%s provider_limit_source=%s "
        "provider_request_id=%s provider_diagnostics=%s routing_strategy=%s "
        "route_priority=%s cost_tier=%s fallback_reason=%s provider_429=%s",
        getattr(g, "proxy_request_id", "unavailable"),
        snapshot.id,
        profile.profile_id,
        profile.provider,
        error.status,
        error.retryable,
        error.code,
        error.classification.category,
        error.provider_error_code or "unknown",
        error.provider_error_type or "unknown",
        error.provider_error_param or "unknown",
        error.provider_limit_source or "unknown",
        error.provider_request_id or "unknown",
        json.dumps(error.provider_diagnostics, sort_keys=True, ensure_ascii=False),
        snapshot.routing_settings.strategy,
        profile.route_priority if profile.route_priority is not None else "unset",
        profile.routing_settings.cost_tier,
        (
            "provider_transient_failure"
            if error.classification.category == "transient"
            else "provider_failure"
        ),
        error.status == 429,
    )


def _record_provider_failure(
    snapshot: DatabaseTenantRoutingSnapshot,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    attempt_id: int | None,
    circuit_attempt: ProviderCircuitAttempt | None,
    error: UpstreamError,
    state: RouteAttemptState,
    is_retry: bool,
    retry_targets: list,
    budget_attempt: ProviderBudgetAttempt | None,
    concurrency_attempt: ProviderConcurrencyAttempt | None,
    deadline: float,
) -> Response | None:
    """Persist one failure, apply cooldowns, and retain cascade semantics."""
    if circuit_attempt is not None:
        try:
            circuit_attempt.failed(error.classification)
        except ProviderCircuitStoreError:
            state.circuit_state_unavailable = True
            current_app.logger.exception(
                "Could not resolve provider circuit state for profile=%s",
                profile.profile_id,
            )
            complete_provider_attempt(
                attempt_id,
                outcome="failure",
                status_code=error.status,
                failure_details=_provider_failure_details(error),
            )
            try:
                if budget_attempt is not None:
                    budget_attempt.failed(error)
                elif concurrency_attempt is not None:
                    concurrency_attempt.failed(error)
            except ProviderSchedulerStoreError:
                state.scheduler_state_unavailable = True
                current_app.logger.exception(
                    "Could not settle provider budget after circuit-store failure "
                    "for profile=%s",
                    profile.profile_id,
                )
            return None
    complete_provider_attempt(
        attempt_id,
        outcome="failure",
        status_code=error.status,
        failure_details=_provider_failure_details(error),
    )
    state.record_error(error)
    retry_delay = _transient_retry_delay(
        error,
        is_retry,
        state.retry_count,
        snapshot.routing_settings.max_retry_wait_seconds,
    )
    retry_at = None
    try:
        if budget_attempt is not None:
            retry_at = budget_attempt.failed(error, retry_delay)
        elif concurrency_attempt is not None:
            concurrency_attempt.failed(error)
    except ProviderSchedulerStoreError:
        state.scheduler_state_unavailable = True
        current_app.logger.exception(
            "Could not persist provider cooldown for profile=%s", profile.profile_id
        )
        _log_provider_failure(snapshot, profile, error)
        return _service_unavailable(
            "provider_scheduler_unavailable",
            "Provider availability could not be verified; retry shortly.",
            "30",
        )
    _log_provider_failure(snapshot, profile, error)
    if retry_delay is not None:
        _queue_transient_retry(
            profile,
            target_model,
            retry_delay,
            retry_at,
            retry_targets,
            deadline,
        )
        state.retry_count += 1
    return error.response() if not error.retryable else None


def _release_circuit_attempt(
    circuit_attempt: ProviderCircuitAttempt | None, *, reason: str
) -> None:
    """Release a probe permit, logging store failures without hiding the main error."""
    if circuit_attempt is None:
        return
    try:
        circuit_attempt.release()
    except ProviderCircuitStoreError:
        current_app.logger.exception(
            "Could not release circuit permit after %s", reason
        )


def _queue_scheduler_retry(
    retry_at: datetime | None,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    is_retry: bool,
    retry_targets: list,
    deadline: float,
) -> None:
    """Queue one capacity-denied candidate if its eligibility fits the deadline."""
    if retry_at is None or is_retry:
        return
    delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    eligible_at = monotonic() + delay
    if eligible_at <= deadline:
        retry_targets.append((profile, target_model, True, eligible_at))


def _acquire_route_reservations(
    snapshot: DatabaseTenantRoutingSnapshot,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    is_retry: bool,
    candidate: SchedulerCandidate,
    scheduler: ProviderBudgetScheduler | None,
    concurrency_controller: ProviderConcurrencyController | None,
    circuit_attempt: ProviderCircuitAttempt | None,
    state: RouteAttemptState,
    retry_targets: list,
    deadline: float,
    pre_acquired_concurrency: ProviderConcurrencyAttempt | None = None,
) -> RouteReservationResult:
    """Acquire concurrency and budget leases before any provider network request."""
    concurrency_attempt = pre_acquired_concurrency
    if concurrency_controller is not None and concurrency_attempt is None:
        try:
            profile_bounds = profile.routing_settings.profile_concurrency
            model_bounds = profile.routing_settings.concurrency_for_model(target_model)
            decision = concurrency_controller.acquire(
                snapshot.id,
                profile.provider,
                profile.profile_id,
                target_model=target_model,
                profile_initial_limit=profile_bounds.initial,
                profile_minimum_limit=profile_bounds.minimum,
                profile_maximum_limit=profile_bounds.maximum,
                model_initial_limit=model_bounds.initial,
                model_minimum_limit=model_bounds.minimum,
                model_maximum_limit=model_bounds.maximum,
            )
            if not decision.allowed:
                current_app.logger.info(
                    "Provider route candidate skipped: request_id=%s tenant=%s "
                    "provider=%s profile=%s model=%s fallback_reason=concurrency_limit "
                    "retry_at=%s",
                    getattr(g, "proxy_request_id", "unavailable"),
                    snapshot.id,
                    profile.provider,
                    profile.profile_id,
                    target_model,
                    (
                        decision.retry_at.isoformat()
                        if decision.retry_at is not None
                        else "unknown"
                    ),
                )
                if decision.retry_at is not None:
                    state.concurrency_blocked_until.append(decision.retry_at)
                _queue_scheduler_retry(
                    decision.retry_at,
                    profile,
                    target_model,
                    is_retry,
                    retry_targets,
                    deadline,
                )
                _release_circuit_attempt(circuit_attempt, reason="AIMD capacity denial")
                return RouteReservationResult(skip_candidate=True)
            if decision.lease is None:
                raise ProviderSchedulerStoreError(
                    "AIMD concurrency allowed a request without issuing a lease"
                )
            concurrency_attempt = ProviderConcurrencyAttempt(
                concurrency_controller, decision.lease
            )
        except ProviderSchedulerStoreError:
            state.scheduler_state_unavailable = True
            _release_circuit_attempt(circuit_attempt, reason="AIMD failure")
            current_app.logger.exception(
                "Provider AIMD controller unavailable for profile=%s",
                profile.profile_id,
            )
            return RouteReservationResult(
                response=_service_unavailable(
                    "provider_scheduler_unavailable",
                    "Provider concurrency state could not be verified; retry shortly.",
                    "30",
                )
            )

    budget_attempt = None
    if scheduler is None:
        return RouteReservationResult(concurrency_attempt=concurrency_attempt)
    try:
        budget_attempt = _budget_attempt(
            scheduler, snapshot, candidate, concurrency_attempt
        )
    except BudgetUnavailableError as blocked:
        current_app.logger.info(
            "Provider route candidate skipped: request_id=%s tenant=%s "
            "provider=%s profile=%s model=%s fallback_reason=budget_exhausted "
            "retry_at=%s",
            getattr(g, "proxy_request_id", "unavailable"),
            snapshot.id,
            profile.provider,
            profile.profile_id,
            target_model,
            blocked.retry_at.isoformat() if blocked.retry_at is not None else "unknown",
        )
        if blocked.retry_at is not None:
            state.budget_blocked_until.append(blocked.retry_at)
        _queue_scheduler_retry(
            blocked.retry_at,
            profile,
            target_model,
            is_retry,
            retry_targets,
            deadline,
        )
        if concurrency_attempt is not None:
            concurrency_attempt.release()
        _release_circuit_attempt(circuit_attempt, reason="budget denial")
        return RouteReservationResult(skip_candidate=True)
    except (ProviderSchedulerStoreError, ProviderSchedulerPolicyConflict):
        state.scheduler_state_unavailable = True
        if concurrency_attempt is not None:
            concurrency_attempt.release()
        _release_circuit_attempt(circuit_attempt, reason="scheduler failure")
        current_app.logger.exception(
            "Provider scheduler unavailable for profile=%s", profile.profile_id
        )
        return RouteReservationResult(
            response=_service_unavailable(
                "provider_scheduler_unavailable",
                "Provider budget state could not be verified; retry shortly.",
                "30",
            )
        )
    return RouteReservationResult(
        concurrency_attempt=concurrency_attempt, budget_attempt=budget_attempt
    )


def _guard_concurrency_lease(
    response: Response, attempt: ProviderConcurrencyAttempt | None
) -> None:
    """Stop emitting provider stream data after its durable lease is lost."""
    if attempt is None or attempt.lease is None:
        return
    stream = response.response

    def guarded_stream():
        try:
            for chunk in stream:
                if attempt.lease_lost.is_set():
                    yield (
                        b'data: {"error":{"code":"provider_concurrency_lease_lost",'
                        b'"message":"Provider concurrency lease expired; stream stopped."}}\n\n'
                    )
                    return
                yield chunk
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    response.response = guarded_stream()


def _forward_route_attempt(
    req: Request,
    snapshot: DatabaseTenantRoutingSnapshot,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    attempt_id: int,
    circuit_attempt: ProviderCircuitAttempt | None,
    reservations: RouteReservationResult,
    state: RouteAttemptState,
    is_retry: bool,
    retry_targets: list,
    deadline: float,
) -> Response | None:
    """Forward one started attempt and settle all leases on every terminal path."""
    concurrency_attempt = reservations.concurrency_attempt
    budget_attempt = reservations.budget_attempt
    try:
        response = _forward_profile(
            req, profile, target_model, attempt_id, circuit_attempt, budget_attempt
        )
        _guard_concurrency_lease(response, concurrency_attempt)
        if circuit_attempt is not None:
            response.call_on_close(circuit_attempt.release)
        if budget_attempt is not None:
            response.call_on_close(budget_attempt.release)
        elif concurrency_attempt is not None:
            response.call_on_close(concurrency_attempt.release)
        return response
    except ServiceConfigurationError:
        if budget_attempt is not None:
            budget_attempt.release()
        elif concurrency_attempt is not None:
            concurrency_attempt.release()
        _release_circuit_attempt(circuit_attempt, reason="configuration failure")
        complete_provider_attempt(attempt_id, outcome="failure", status_code=400)
        raise
    except UpstreamError as error:
        return _record_provider_failure(
            snapshot,
            profile,
            target_model,
            attempt_id,
            circuit_attempt,
            error,
            state,
            is_retry,
            retry_targets,
            budget_attempt,
            concurrency_attempt,
            deadline,
        )
    except Exception:  # noqa: BLE001 - release resources before re-raising
        if budget_attempt is not None:
            try:
                budget_attempt.release()
            except ProviderSchedulerStoreError:
                current_app.logger.exception(
                    "Could not release provider budget after unexpected failure "
                    "for profile=%s",
                    profile.profile_id,
                )
        elif concurrency_attempt is not None:
            try:
                concurrency_attempt.release()
            except ProviderSchedulerStoreError:
                current_app.logger.exception(
                    "Could not release AIMD permit after unexpected failure "
                    "for profile=%s",
                    profile.profile_id,
                )
        _release_circuit_attempt(circuit_attempt, reason="unexpected failure")
        complete_provider_attempt(attempt_id, outcome="failure", status_code=None)
        raise


def _attempt_route_target(
    req: Request,
    snapshot: DatabaseTenantRoutingSnapshot,
    inbound_model: object,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    is_retry: bool,
    breaker_store: ProviderCircuitBreakerStore | None,
    scheduler: ProviderBudgetScheduler | None,
    concurrency_controller: ProviderConcurrencyController | None,
    state: RouteAttemptState,
    retry_targets: list,
    deadline: float,
    prevalidated_candidate: SchedulerCandidate | None = None,
    pre_acquired_concurrency: ProviderConcurrencyAttempt | None = None,
) -> Response | None:
    """Run one candidate after validation, availability, and capacity checks."""
    try:
        candidate = prevalidated_candidate or _scheduler_candidate(
            snapshot, profile, target_model, req.get_json(silent=True) or {}
        )
    except ServiceConfigurationError as exc:
        state.candidate_configuration_errors.append(
            (profile.profile_id or "unknown", str(exc))
        )
        current_app.logger.warning(
            "Skipping provider candidate with invalid scheduler configuration: "
            "tenant=%s profile=%s reason=%s",
            snapshot.id,
            profile.profile_id or "unknown",
            str(exc),
        )
        return None

    permit = _acquire_route_permit(
        snapshot,
        profile,
        target_model,
        is_retry,
        breaker_store,
        state,
        retry_targets,
        deadline,
    )
    if permit is None:
        if pre_acquired_concurrency is not None:
            pre_acquired_concurrency.release()
        return None
    circuit_attempt = (
        ProviderCircuitAttempt(
            breaker_store,
            permit,
            snapshot.id,
            profile.provider,
            profile.profile_id,
            profile.provider_settings,
        )
        if breaker_store is not None
        else None
    )
    reservations = _acquire_route_reservations(
        snapshot,
        profile,
        target_model,
        is_retry,
        candidate,
        scheduler,
        concurrency_controller,
        circuit_attempt,
        state,
        retry_targets,
        deadline,
        pre_acquired_concurrency=pre_acquired_concurrency,
    )
    if reservations.response is not None:
        return reservations.response
    if reservations.skip_candidate:
        return None

    attempt_id = start_provider_attempt(
        tenant_id=snapshot.id,
        provider=profile.provider,
        profile_id=profile.profile_id,
        inbound_model=inbound_model,
        routed_model=target_model,
    )
    state.attempts_started += 1
    return _forward_route_attempt(
        req,
        snapshot,
        profile,
        target_model,
        attempt_id,
        circuit_attempt,
        reservations,
        state,
        is_retry,
        retry_targets,
        deadline,
    )


def _final_route_failure(state: RouteAttemptState) -> Response:
    """Choose the response after every available route and retry was exhausted."""
    if state.scheduler_state_unavailable:
        return _service_unavailable(
            "provider_scheduler_unavailable",
            "Provider budget state could not be verified; retry shortly.",
            "30",
        )
    if state.circuit_state_unavailable:
        return _service_unavailable(
            "provider_circuit_unavailable",
            "Provider availability could not be verified; retry the request shortly.",
            "30",
        )
    if state.transient_failure_seen:
        retry_after_seconds = (
            max(1, math.ceil(state.transient_retry_after))
            if state.transient_retry_after is not None
            else 30
        )
        retry_times = [
            retry_at for retry_at in state.blocked_until if retry_at is not None
        ]
        retry_times.extend(state.budget_blocked_until)
        retry_times.extend(state.concurrency_blocked_until)
        cooldown_seconds = (
            int(retry_after_header(retry_at)) for retry_at in retry_times
        )
        retry_after = str(max(retry_after_seconds, max(cooldown_seconds, default=0)))
        return _service_unavailable(
            "provider_temporarily_unavailable",
            "Providers are temporarily rate limited or unavailable.",
            retry_after,
        )
    if state.last_retryable_error is not None:
        return state.last_retryable_error.response()
    if state.budget_blocked_until:
        retry_at = min(state.budget_blocked_until)
        return _service_unavailable(
            "provider_temporarily_unavailable",
            "Provider budgets are temporarily exhausted; retry shortly.",
            retry_after_header(retry_at),
        )
    if state.concurrency_blocked_until:
        retry_at = min(state.concurrency_blocked_until)
        return _service_unavailable(
            "provider_temporarily_unavailable",
            "Provider concurrency limits are temporarily exhausted; retry shortly.",
            retry_after_header(retry_at),
        )
    if state.blocked_until:
        retry_at = min(value for value in state.blocked_until if value is not None)
        if state.lease_contention_seen:
            return _service_unavailable(
                "provider_temporarily_unavailable",
                "Providers are handling concurrent recovery probes; retry shortly.",
                retry_after_header(retry_at),
            )
        return _quota_unavailable(retry_at)
    if state.attempts_started:
        raise AssertionError(
            "A started provider route must return a response or failure"
        )
    if state.candidate_configuration_errors:
        raise ServiceConfigurationError(
            "No provider candidate has a valid scheduler configuration."
        )
    raise ServiceConfigurationError(
        "No provider circuit state could be verified; no upstream request was sent."
    )


def _retry_route_target(
    target: tuple[DatabaseTenantSnapshot, str, bool, float],
    deadline: float,
) -> bool:
    """Wait cooperatively for an eligible retry unless its request budget expired."""
    retry_at = target[3]
    if retry_at > deadline or monotonic() > deadline:
        return False
    remaining = retry_at - monotonic()
    if remaining > 0:
        cooperative_sleep(remaining)
    return True


def forward_tenant_route(
    req: Request, snapshot: DatabaseTenantRoutingSnapshot, *, azure_only: bool = False
) -> Response:
    """Try priority/cost tiers, then eligible bounded transient retries."""
    route_profiles = routed_profiles(snapshot, azure_only=azure_only)
    catalog_candidates = catalog_profiles(snapshot, azure_only=azure_only)
    payload = req.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ServiceConfigurationError(
            "Request model must match the tenant's Cursor model ID."
        )
    inbound_model = payload.get("model")
    route_targets = _resolve_request_targets(
        snapshot,
        inbound_model,
        route_profiles,
        catalog_candidates,
        azure_only=azure_only,
    )
    candidate_groups = _route_candidate_groups(route_targets, snapshot)
    database = current_app.extensions.get("database")
    if not isinstance(database, Database):
        return _service_unavailable(
            "provider_scheduler_unavailable",
            "Provider budget state could not be verified; retry shortly.",
            "30",
        )
    breaker_store = ProviderCircuitBreakerStore(
        database.sessions, database.secret_cipher
    )
    scheduler = ProviderBudgetScheduler(database.sessions, database.secret_cipher)
    concurrency_controller = ProviderConcurrencyController(
        database.sessions, database.secret_cipher
    )
    state = RouteAttemptState(blocked_until=[])
    retry_targets = []
    deadline = monotonic() + min(
        MAX_TRANSIENT_RETRY_WAIT_SECONDS,
        snapshot.routing_settings.max_retry_wait_seconds,
    )
    for group_index, route_group in enumerate(candidate_groups):
        remaining = list(route_group)
        while remaining:
            validated = []
            for profile, target_model in remaining:
                try:
                    candidate = _scheduler_candidate(
                        snapshot, profile, target_model, payload
                    )
                except ServiceConfigurationError as exc:
                    state.candidate_configuration_errors.append(
                        (profile.profile_id or "unknown", str(exc))
                    )
                    current_app.logger.warning(
                        "Skipping provider candidate with invalid scheduler configuration: "
                        "tenant=%s profile=%s reason=%s",
                        snapshot.id,
                        profile.profile_id or "unknown",
                        str(exc),
                    )
                    continue
                budget_headroom = 1.0
                if scheduler is not None and candidate.policies:
                    try:
                        budget_headroom = scheduler.headroom(
                            _reservation_request(snapshot, candidate)
                        )
                    except ProviderSchedulerStoreError:
                        state.scheduler_state_unavailable = True
                        current_app.logger.exception(
                            "Provider budget headroom unavailable for profile=%s",
                            profile.profile_id,
                        )
                        return _service_unavailable(
                            "provider_scheduler_unavailable",
                            "Provider budget state could not be verified; retry shortly.",
                            "30",
                        )
                validated.append(
                    RouteCandidate(profile, target_model, candidate, budget_headroom)
                )
            remaining.clear()
            if not validated:
                break

            selected = validated[0]
            pre_acquired = None
            capacity_score = None
            if (
                snapshot.routing_settings.strategy == "load_balanced"
                and len(validated) > 1
            ):
                if concurrency_controller is None:
                    selected_group = sorted(
                        validated, key=lambda item: item.profile.profile_id or ""
                    )
                    selected = selected_group[0]
                    remaining.extend(
                        (item.profile, item.target_model) for item in selected_group[1:]
                    )
                else:
                    concurrency_candidates = tuple(
                        ProviderConcurrencyCandidate(
                            provider=item.profile.provider,
                            profile_id=item.profile.profile_id,
                            target_model=item.target_model,
                            profile_initial_limit=item.profile.routing_settings.profile_concurrency.initial,
                            profile_minimum_limit=item.profile.routing_settings.profile_concurrency.minimum,
                            profile_maximum_limit=item.profile.routing_settings.profile_concurrency.maximum,
                            model_initial_limit=(
                                item.profile.routing_settings.concurrency_for_model(
                                    item.target_model
                                ).initial
                            ),
                            model_minimum_limit=(
                                item.profile.routing_settings.concurrency_for_model(
                                    item.target_model
                                ).minimum
                            ),
                            model_maximum_limit=(
                                item.profile.routing_settings.concurrency_for_model(
                                    item.target_model
                                ).maximum
                            ),
                            weight=item.profile.routing_settings.load_balancing_weight,
                            budget_headroom=item.budget_headroom,
                            headroom_weight=snapshot.routing_settings.headroom_weight,
                        )
                        for item in validated
                    )
                    try:
                        decision = concurrency_controller.acquire_best(
                            snapshot.id, concurrency_candidates
                        )
                    except ProviderSchedulerStoreError:
                        state.scheduler_state_unavailable = True
                        current_app.logger.exception(
                            "Provider AIMD controller unavailable for tenant=%s",
                            snapshot.id,
                        )
                        return _service_unavailable(
                            "provider_scheduler_unavailable",
                            "Provider concurrency state could not be verified; retry shortly.",
                            "30",
                        )
                    if not decision.allowed:
                        if decision.retry_at is not None:
                            state.concurrency_blocked_until.append(decision.retry_at)
                        for item in validated:
                            _queue_scheduler_retry(
                                decision.retry_at,
                                item.profile,
                                item.target_model,
                                False,
                                retry_targets,
                                deadline,
                            )
                        break
                    selected = next(
                        item
                        for item in validated
                        if item.profile.profile_id == decision.profile_id
                    )
                    if decision.lease is None:
                        raise ProviderSchedulerStoreError(
                            "AIMD group selection allowed a request without issuing a lease"
                        )
                    pre_acquired = ProviderConcurrencyAttempt(
                        concurrency_controller, decision.lease
                    )
                    capacity_score = decision.capacity_score
                    remaining.extend(
                        (item.profile, item.target_model)
                        for item in validated
                        if item is not selected
                    )
            else:
                remaining.extend(
                    (item.profile, item.target_model) for item in validated[1:]
                )

            profile = selected.profile
            target_model = selected.target_model
            permit = _acquire_route_permit(
                snapshot,
                profile,
                target_model,
                False,
                breaker_store,
                state,
                retry_targets,
                deadline,
            )
            if permit is None:
                if pre_acquired is not None:
                    pre_acquired.release()
                continue
            circuit_attempt = (
                ProviderCircuitAttempt(
                    breaker_store,
                    permit,
                    snapshot.id,
                    profile.provider,
                    profile.profile_id,
                    profile.provider_settings,
                )
                if breaker_store is not None
                else None
            )
            reservations = _acquire_route_reservations(
                snapshot,
                profile,
                target_model,
                False,
                selected.scheduler,
                scheduler,
                concurrency_controller,
                circuit_attempt,
                state,
                retry_targets,
                deadline,
                pre_acquired_concurrency=pre_acquired,
            )
            if reservations.response is not None:
                return reservations.response
            if reservations.skip_candidate:
                remaining = [
                    target for target in remaining if target != (profile, target_model)
                ]
                continue
            current_app.logger.info(
                "Provider route candidate selected: request_id=%s tenant=%s "
                "provider=%s profile=%s model=%s routing_strategy=%s "
                "route_priority=%s cost_policy=%s cost_tier=%s weight=%s "
                "budget_headroom=%s capacity_score=%s fallback=%s",
                getattr(g, "proxy_request_id", "unavailable"),
                snapshot.id,
                profile.provider,
                profile.profile_id,
                target_model,
                snapshot.routing_settings.strategy,
                (
                    profile.route_priority
                    if profile.route_priority is not None
                    else "unset"
                ),
                snapshot.routing_settings.cost_policy,
                profile.routing_settings.cost_tier,
                profile.routing_settings.load_balancing_weight,
                selected.budget_headroom,
                capacity_score if capacity_score is not None else "unmeasured",
                group_index > 0 or state.attempts_started > 0,
            )
            attempt_id = start_provider_attempt(
                tenant_id=snapshot.id,
                provider=profile.provider,
                profile_id=profile.profile_id,
                inbound_model=inbound_model,
                routed_model=target_model,
            )
            state.attempts_started += 1
            response = _forward_route_attempt(
                req,
                snapshot,
                profile,
                target_model,
                attempt_id,
                circuit_attempt,
                reservations,
                state,
                False,
                retry_targets,
                deadline,
            )
            if response is not None:
                return response

    for retry in sorted(retry_targets, key=lambda target: target[3]):
        if not _retry_route_target(retry, deadline):
            continue
        profile, target_model, is_retry, _retry_at = retry
        response = _attempt_route_target(
            req,
            snapshot,
            inbound_model,
            profile,
            target_model,
            is_retry,
            breaker_store,
            scheduler,
            concurrency_controller,
            state,
            [],
            deadline,
        )
        if response is not None:
            return response
    return _final_route_failure(state)
