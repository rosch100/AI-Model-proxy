"""PostgreSQL-backed, provider-neutral request and token budget reservations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.common.retry_after import RETRY_AFTER_MAX_SECONDS
from app.persistence.models import (
    ProviderBudgetLease,
    ProviderBudgetLeaseAllocation,
    ProviderBudgetPolicy,
    ProviderBudgetScopeState,
    ProviderBudgetWindow,
)
from app.persistence.models import (
    ProviderConcurrencyLease as ProviderConcurrencyLeaseRecord,
)
from app.persistence.models import ProviderConcurrencyState
from app.persistence.secrets import SecretCipher

_SUPPORTED_PROVIDERS = frozenset({"azure", "openai", "openrouter", "deepseek"})
AIMD_MIN_CONCURRENCY = 1
AIMD_MAX_CONCURRENCY = 64
AIMD_INITIAL_CONCURRENCY = 4
AIMD_LEASE_TTL = timedelta(seconds=90)
MAX_POSTGRES_INTEGER = 2**31 - 1


def _capacity_score(
    profile_active: int,
    profile_limit: int,
    model_active: int,
    model_limit: int,
    weight: float,
    *,
    budget_headroom: float = 1.0,
    headroom_weight: float = 0.0,
) -> float:
    """Score the profile/model bottleneck and configured budget headroom."""
    counts = (profile_active, profile_limit, model_active, model_limit)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise ValueError("AIMD capacity counts must be integers")
    if profile_limit <= 0 or model_limit <= 0:
        raise ValueError("AIMD capacity limits must be positive")
    if not 0 <= profile_active <= profile_limit or not 0 <= model_active <= model_limit:
        raise ValueError("AIMD active permits must fit within their capacity limits")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("Load-balancing weight must be finite and positive")
    if (
        isinstance(budget_headroom, bool)
        or not isinstance(budget_headroom, (int, float))
        or not math.isfinite(budget_headroom)
        or not 0 <= budget_headroom <= 1
        or isinstance(headroom_weight, bool)
        or not isinstance(headroom_weight, (int, float))
        or not math.isfinite(headroom_weight)
        or not 0 <= headroom_weight <= 1
    ):
        raise ValueError("Budget headroom inputs must be finite values between 0 and 1")
    utilization = max(profile_active / profile_limit, model_active / model_limit)
    budget_pressure = (1.0 - budget_headroom) * headroom_weight
    return max(utilization, budget_pressure) / weight


def _validate_aimd_bounds(
    initial: int, minimum: int, maximum: int
) -> tuple[int, int, int]:
    """Validate one configured scope's initial and adaptive concurrency bounds."""
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (initial, minimum, maximum)
    ):
        raise ValueError("AIMD limits must be integers")
    if (
        not AIMD_MIN_CONCURRENCY
        <= minimum
        <= initial
        <= maximum
        <= AIMD_MAX_CONCURRENCY
    ):
        raise ValueError(
            "AIMD limits must satisfy 1 <= minimum <= initial <= maximum <= 64"
        )
    return initial, minimum, maximum


def _adjust_aimd_limit(
    current: int,
    *,
    outcome: str,
    minimum: int = AIMD_MIN_CONCURRENCY,
    maximum: int = AIMD_MAX_CONCURRENCY,
) -> int:
    """Apply additive increase or multiplicative decrease to one bounded limit."""
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not AIMD_MIN_CONCURRENCY <= minimum <= maximum <= AIMD_MAX_CONCURRENCY
        or isinstance(current, bool)
        or not isinstance(current, int)
        or not minimum <= current <= maximum
    ):
        raise ValueError("AIMD concurrency limit is outside its supported range")
    if outcome == "success":
        return min(maximum, current + 1)
    if outcome == "transient_failure":
        return max(minimum, current // 2)
    if outcome in {"terminal_failure", "released"}:
        return current
    raise ValueError("Unsupported AIMD outcome")


class BudgetScopeKind(StrEnum):
    """Identity level against which a configured provider budget applies."""

    PROVIDER = "provider"
    PROFILE = "profile"
    MODEL = "model"
    ORGANIZATION = "organization"
    PROJECT = "project"


class BudgetMetric(StrEnum):
    """Supported request and token accounting dimensions."""

    REQUESTS = "requests"
    TOKENS = "tokens"


@dataclass(frozen=True)
class SharedBudgetIdentity:
    """Trusted canonical provider identity explicitly approved for cross-tenant use.

    Callers must supply a stable provider-issued or otherwise verified identity;
    display labels and unvalidated local names must never be passed here.
    """

    canonical_id: str

    def __post_init__(self) -> None:
        """Reject absent or whitespace-only canonical provider identities."""
        if not isinstance(self.canonical_id, str) or not self.canonical_id.strip():
            raise ValueError(
                "Shared provider budget identity must be canonical and non-empty"
            )


@dataclass(frozen=True)
class BudgetScope:
    """Typed provider scope; the raw identity is fingerprinted before persistence."""

    provider: str
    kind: BudgetScopeKind
    scope_id: str
    shared_identity: SharedBudgetIdentity | None = None

    def __post_init__(self) -> None:
        """Validate tenant-local keys and explicit trusted cross-tenant identities."""
        if self.provider not in _SUPPORTED_PROVIDERS:
            raise ValueError("Unsupported provider budget provider")
        if not isinstance(self.kind, BudgetScopeKind):
            raise ValueError("Invalid provider budget scope kind")
        if not isinstance(self.scope_id, str) or not self.scope_id.strip():
            raise ValueError("Provider budget scope identity must be non-empty")
        if self.shared_identity is not None and (
            self.kind not in {BudgetScopeKind.ORGANIZATION, BudgetScopeKind.PROJECT}
            or not isinstance(self.shared_identity, SharedBudgetIdentity)
        ):
            raise ValueError(
                "Only organization or project scopes accept a trusted shared identity"
            )


@dataclass(frozen=True)
class BudgetPolicy:
    """Explicit fixed-window capacity; unknown provider limits remain unconfigured."""

    scope: BudgetScope
    metric: BudgetMetric
    limit_units: int
    window_seconds: int

    def __post_init__(self) -> None:
        """Enforce explicit, typed, positive budget dimensions."""
        if not isinstance(self.scope, BudgetScope):
            raise ValueError("Budget policy requires a typed scope")
        if not isinstance(self.metric, BudgetMetric):
            raise ValueError("Invalid provider budget metric")
        if (
            isinstance(self.limit_units, bool)
            or not isinstance(self.limit_units, int)
            or self.limit_units <= 0
            or self.limit_units > MAX_POSTGRES_INTEGER
        ):
            raise ValueError(
                "Budget limit must fit a positive 32-bit PostgreSQL INTEGER"
            )
        if (
            isinstance(self.window_seconds, bool)
            or not isinstance(self.window_seconds, int)
            or self.window_seconds <= 0
            or self.window_seconds > MAX_POSTGRES_INTEGER
        ):
            raise ValueError(
                "Budget window must fit a positive 32-bit PostgreSQL INTEGER"
            )


@dataclass(frozen=True)
class ReservationRequest:
    """One provider attempt with all hierarchical budgets applied together."""

    tenant_id: str
    policies: tuple[BudgetPolicy, ...]
    estimated_tokens: int | None = None
    cooldown_scopes: tuple[BudgetScope, ...] = ()

    def __post_init__(self) -> None:
        """Validate tenant, policies, and any token estimate before persistence."""
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("Provider budget tenant identity must be non-empty")
        if not self.policies and not self.cooldown_scopes:
            raise ValueError(
                "At least one provider policy or cooldown scope is required"
            )
        if any(not isinstance(policy, BudgetPolicy) for policy in self.policies):
            raise ValueError("Provider budget policies must be typed")
        if any(not isinstance(scope, BudgetScope) for scope in self.cooldown_scopes):
            raise ValueError("Provider cooldown scopes must be typed")
        providers = {policy.scope.provider for policy in self.policies}
        providers.update(scope.provider for scope in self.cooldown_scopes)
        if len(providers) != 1:
            raise ValueError(
                "All policies in one reservation must use the same provider"
            )
        keys = [
            (
                policy.scope.provider,
                policy.scope.kind.value,
                (
                    policy.scope.shared_identity.canonical_id
                    if policy.scope.shared_identity is not None
                    else (self.tenant_id, policy.scope.scope_id)
                ),
                policy.metric,
            )
            for policy in self.policies
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate policy for scope and metric; window_seconds cannot vary"
            )
        requires_tokens = any(
            policy.metric is BudgetMetric.TOKENS for policy in self.policies
        )
        if requires_tokens and (
            isinstance(self.estimated_tokens, bool)
            or not isinstance(self.estimated_tokens, int)
            or self.estimated_tokens < 0
        ):
            raise ValueError("estimated_tokens must be a nonnegative integer")
        if (
            not requires_tokens
            and self.estimated_tokens is not None
            and (
                isinstance(self.estimated_tokens, bool)
                or not isinstance(self.estimated_tokens, int)
                or self.estimated_tokens < 0
            )
        ):
            raise ValueError("estimated_tokens must be a nonnegative integer")

    @property
    def scopes(self) -> tuple[BudgetScope, ...]:
        """Return each tenant-local or explicitly shared scope exactly once."""
        scopes_by_identity = {}
        for scope in self.cooldown_scopes:
            identity = (
                scope.provider,
                scope.kind.value,
                (
                    scope.shared_identity.canonical_id
                    if scope.shared_identity is not None
                    else (self.tenant_id, scope.scope_id)
                ),
            )
            scopes_by_identity.setdefault(identity, scope)
        for policy in self.policies:
            scope = policy.scope
            identity = (
                scope.provider,
                scope.kind.value,
                (
                    scope.shared_identity.canonical_id
                    if scope.shared_identity is not None
                    else (self.tenant_id, scope.scope_id)
                ),
            )
            scopes_by_identity.setdefault(identity, scope)
        return tuple(scopes_by_identity.values())

    @property
    def provider(self) -> str:
        """Return the provider represented by policies or cooldown scopes."""
        if self.policies:
            return self.policies[0].scope.provider
        return self.cooldown_scopes[0].provider


@dataclass(frozen=True)
class BudgetLease:
    """Opaque lease capability bound to a persisted tenant and lease token."""

    id: str
    tenant_id: str
    provider: str
    token: str


@dataclass(frozen=True)
class ProviderConcurrencyPermit:
    """Opaque token for one scope's persisted AIMD permit."""

    id: str
    state_id: int
    token: str


@dataclass(frozen=True)
class ProviderConcurrencyLease:
    """Opaque tokens for all durable permits held by one provider request."""

    id: str
    state_id: int
    tenant_id: str
    provider: str
    token: str
    additional_permits: tuple[ProviderConcurrencyPermit, ...] = ()

    @property
    def permits(self) -> tuple[ProviderConcurrencyPermit, ...]:
        """Return all scope permits in deterministic state-lock order."""
        return (
            ProviderConcurrencyPermit(self.id, self.state_id, self.token),
            *self.additional_permits,
        )


@dataclass(frozen=True)
class ProviderConcurrencyCandidate:
    """One routable candidate's profile/model bounds and scheduling weight."""

    provider: str
    profile_id: str
    target_model: str | None
    profile_initial_limit: int
    profile_minimum_limit: int
    profile_maximum_limit: int
    model_initial_limit: int
    model_minimum_limit: int
    model_maximum_limit: int
    weight: float = 1.0
    budget_headroom: float = 1.0
    headroom_weight: float = 0.0


@dataclass(frozen=True)
class ProviderConcurrencyDecision:
    """Persistent AIMD capacity decision and retry hint."""

    allowed: bool
    lease: ProviderConcurrencyLease | None = None
    retry_at: datetime | None = None
    capacity_score: float | None = None
    profile_id: str | None = None


@dataclass(frozen=True)
class ReservationDecision:
    """Atomic reservation outcome, including a bounded retry time if denied."""

    allowed: bool
    lease: BudgetLease | None = None
    retry_at: datetime | None = None


@dataclass(frozen=True)
class _WindowClaim:
    policy: BudgetPolicy
    fingerprint: str
    window_start: datetime
    units: int


class ProviderBudgetScheduler:
    """Coordinate explicit budget windows and cooldowns across PostgreSQL workers."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        cipher: SecretCipher | None = None,
    ) -> None:
        """Bind the PostgreSQL session factory and scope fingerprint cipher."""
        self._sessions = sessions
        self._cipher = cipher

    def headroom(
        self, request: ReservationRequest, *, now: datetime | None = None
    ) -> float:
        """Estimate the minimum remaining fraction across configured budget windows."""
        if not isinstance(request, ReservationRequest):
            raise ValueError("A typed provider budget request is required")
        if not request.policies:
            return 1.0
        moment = _utc(now or datetime.now(timezone.utc))
        try:
            with self._sessions() as session:
                _require_postgresql(session)
                fingerprints = self._fingerprints(request.tenant_id, request.scopes)
                fractions = []
                for policy in request.policies:
                    window_start = _window_start(moment, policy.window_seconds)
                    window = session.scalar(
                        select(ProviderBudgetWindow).where(
                            ProviderBudgetWindow.provider == policy.scope.provider,
                            ProviderBudgetWindow.scope_kind == policy.scope.kind.value,
                            ProviderBudgetWindow.scope_fingerprint
                            == fingerprints[policy.scope],
                            ProviderBudgetWindow.metric == policy.metric.value,
                            ProviderBudgetWindow.window_seconds
                            == policy.window_seconds,
                            ProviderBudgetWindow.window_start == window_start,
                        )
                    )
                    if window is None:
                        fractions.append(1.0)
                        continue
                    units = (
                        1
                        if policy.metric is BudgetMetric.REQUESTS
                        else int(request.estimated_tokens or 0)
                    )
                    remaining = max(
                        0,
                        window.limit_units
                        - window.used_units
                        - window.reserved_units
                        - units,
                    )
                    fractions.append(remaining / window.limit_units)
                return min(fractions, default=1.0)
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not read provider budget headroom from PostgreSQL"
            ) from exc

    def reserve(
        self, request: ReservationRequest, *, now: datetime | None = None
    ) -> ReservationDecision:
        """Reserve every request/token window atomically or reserve none."""
        moment = _utc(now or datetime.now(timezone.utc))
        try:
            with self._sessions.begin() as session:
                _require_postgresql(session)
                policy_scopes = request.scopes
                fingerprints = self._fingerprints(request.tenant_id, policy_scopes)
                ordered_scopes = tuple(
                    sorted(
                        request.scopes,
                        key=lambda scope: _scope_sort_key(scope, fingerprints[scope]),
                    )
                )
                self._lock_scope_states(session, ordered_scopes, fingerprints, moment)
                scope_states = self._read_scope_states(
                    session, request.scopes, fingerprints
                )
                blocked_until = [
                    _utc(state.cooldown_until)
                    for state in scope_states
                    if state.cooldown_until is not None
                    and _utc(state.cooldown_until) > moment
                ]
                if blocked_until:
                    return ReservationDecision(False, retry_at=max(blocked_until))

                claims = tuple(
                    sorted(
                        (
                            _WindowClaim(
                                policy=policy,
                                fingerprint=fingerprints[policy.scope],
                                window_start=_window_start(
                                    moment, policy.window_seconds
                                ),
                                units=(
                                    1
                                    if policy.metric is BudgetMetric.REQUESTS
                                    else int(request.estimated_tokens or 0)
                                ),
                            )
                            for policy in request.policies
                        ),
                        key=lambda claim: _window_lock_key_from_values(
                            request.provider,
                            claim.policy.scope.kind.value,
                            claim.fingerprint,
                            claim.policy.metric.value,
                            claim.policy.window_seconds,
                            claim.window_start,
                            0,
                        ),
                    )
                )
                if not claims:
                    return ReservationDecision(True)
                for claim in claims:
                    self._register_policy(session, claim)
                window_rows = self._lock_windows(session, request.provider, claims)
                denials = []
                for claim, row in zip(claims, window_rows, strict=True):
                    if (
                        row.used_units + row.reserved_units + claim.units
                        > row.limit_units
                    ):
                        denials.append(
                            claim.window_start
                            + timedelta(seconds=claim.policy.window_seconds)
                        )
                if denials:
                    return ReservationDecision(False, retry_at=max(denials))

                lease_id = str(uuid4())
                token = str(uuid4())
                session.add(
                    ProviderBudgetLease(
                        id=lease_id,
                        tenant_id=request.tenant_id,
                        provider=request.provider,
                        lease_token=token,
                        status="active",
                        created_at=moment,
                    )
                )
                for claim, row in zip(claims, window_rows, strict=True):
                    row.reserved_units += claim.units
                    session.add(
                        ProviderBudgetLeaseAllocation(
                            lease_id=lease_id,
                            window_id=row.id,
                            reserved_units=claim.units,
                        )
                    )
                session.flush()
                return ReservationDecision(
                    True,
                    BudgetLease(lease_id, request.tenant_id, request.provider, token),
                )
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not reserve provider budget in PostgreSQL"
            ) from exc

    def release(self, lease: BudgetLease, *, now: datetime | None = None) -> bool:
        """Release an uncharged reservation when work did not complete."""
        return self._settle(lease, actual_tokens=None, status="released", now=now)

    def failed(self, lease: BudgetLease, *, now: datetime | None = None) -> bool:
        """Charge a started failed request while releasing its token estimate."""
        return self._settle(lease, actual_tokens=None, status="failed", now=now)

    def complete(
        self,
        lease: BudgetLease,
        *,
        actual_tokens: int | None,
        now: datetime | None = None,
    ) -> bool:
        """Charge a completed request and actual usage, or retain its estimate."""
        if actual_tokens is not None and (
            isinstance(actual_tokens, bool)
            or not isinstance(actual_tokens, int)
            or actual_tokens < 0
        ):
            raise ValueError("actual_tokens must be a nonnegative integer")
        return self._settle(
            lease, actual_tokens=actual_tokens, status="completed", now=now
        )

    def check_cooldowns(
        self,
        tenant_id: str,
        scopes: tuple[BudgetScope, ...],
        *,
        now: datetime | None = None,
    ) -> datetime | None:
        """Return the latest active durable cooldown for any candidate scope."""
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Provider budget tenant identity must be non-empty")
        if not scopes or any(not isinstance(scope, BudgetScope) for scope in scopes):
            raise ValueError("At least one typed provider budget scope is required")
        providers = {scope.provider for scope in scopes}
        if len(providers) != 1:
            raise ValueError("Cooldown scopes must use the same provider")
        moment = _utc(now or datetime.now(timezone.utc))
        try:
            with self._sessions() as session:
                _require_postgresql(session)
                fingerprints = self._fingerprints(tenant_id, scopes)
                blocked_until = []
                for scope in set(scopes):
                    state = session.scalar(
                        select(ProviderBudgetScopeState).where(
                            *_scope_predicate(scope, fingerprints[scope])
                        )
                    )
                    if state is not None and state.cooldown_until is not None:
                        cooldown_until = _utc(state.cooldown_until)
                        if cooldown_until > moment:
                            blocked_until.append(cooldown_until)
                return max(blocked_until) if blocked_until else None
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not read provider cooldowns from PostgreSQL"
            ) from exc

    def apply_cooldown(
        self,
        scope: BudgetScope,
        retry_after_seconds: int | float,
        *,
        tenant_id: str,
        now: datetime | None = None,
    ) -> datetime:
        """Persist the maximum existing or provider-requested scope cooldown."""
        if (
            isinstance(retry_after_seconds, bool)
            or not isinstance(retry_after_seconds, (int, float))
            or not math.isfinite(retry_after_seconds)
            or retry_after_seconds < 0
        ):
            raise ValueError("retry_after_seconds must be a finite nonnegative number")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Provider budget tenant identity must be non-empty")
        moment = _utc(now or datetime.now(timezone.utc))
        retry_at = moment + timedelta(
            seconds=min(retry_after_seconds, RETRY_AFTER_MAX_SECONDS)
        )
        try:
            with self._sessions.begin() as session:
                _require_postgresql(session)
                fingerprint = self._fingerprint(tenant_id, scope)
                self._lock_scope_states(session, (scope,), {scope: fingerprint}, moment)
                state = self._read_scope_states(
                    session, (scope,), {scope: fingerprint}
                )[0]
                if state.cooldown_until is not None:
                    retry_at = max(retry_at, _utc(state.cooldown_until))
                state.cooldown_until = retry_at
                state.updated_at = moment
                session.flush()
                return retry_at
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not persist provider budget cooldown in PostgreSQL"
            ) from exc

    def _settle(
        self,
        lease: BudgetLease,
        *,
        actual_tokens: int | None,
        status: str,
        now: datetime | None,
    ) -> bool:
        if not isinstance(lease, BudgetLease):
            raise ValueError("A typed provider budget lease is required")
        try:
            with self._sessions.begin() as session:
                _require_postgresql(session)
                record = session.scalar(
                    select(ProviderBudgetLease)
                    .where(
                        ProviderBudgetLease.id == lease.id,
                        ProviderBudgetLease.tenant_id == lease.tenant_id,
                        ProviderBudgetLease.provider == lease.provider,
                        ProviderBudgetLease.lease_token == lease.token,
                        ProviderBudgetLease.status == "active",
                    )
                    .with_for_update()
                )
                if record is None:
                    return False
                allocations = session.scalars(
                    select(ProviderBudgetLeaseAllocation).where(
                        ProviderBudgetLeaseAllocation.lease_id == lease.id
                    )
                ).all()
                window_ids = tuple(allocation.window_id for allocation in allocations)
                windows = session.scalars(
                    select(ProviderBudgetWindow).where(
                        ProviderBudgetWindow.id.in_(window_ids)
                    )
                ).all()
                lock_keys = sorted(_window_lock_key(window) for window in windows)
                locked_windows = {}
                for lock_key in lock_keys:
                    window = session.scalar(
                        select(ProviderBudgetWindow)
                        .where(ProviderBudgetWindow.id == lock_key[-1])
                        .with_for_update()
                    )
                    if window is not None:
                        locked_windows[window.id] = window
                moment = _utc(now or datetime.now(timezone.utc))
                for allocation in allocations:
                    window = locked_windows.get(allocation.window_id)
                    if window is None:
                        # Cleanup already reclaimed this expired allocation.
                        continue
                    # Fixed windows are the lease boundary: late stream completions
                    # may settle the lease, but never charge a newer/expired bucket.
                    reserved_units = allocation.reserved_units
                    window.reserved_units -= reserved_units
                    allocation.reserved_units = 0
                    window_end = _utc(window.window_start) + timedelta(
                        seconds=window.window_seconds
                    )
                    if window_end > moment:
                        if window.metric == BudgetMetric.REQUESTS.value and status in {
                            "completed",
                            "failed",
                        }:
                            window.used_units += 1
                        elif (
                            window.metric == BudgetMetric.TOKENS.value
                            and actual_tokens is not None
                        ):
                            window.used_units += actual_tokens
                        elif (
                            window.metric == BudgetMetric.TOKENS.value
                            and status == "completed"
                        ):
                            window.used_units += reserved_units
                record.status = status
                record.completed_at = moment
                session.flush()
                return True
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not settle provider budget lease in PostgreSQL"
            ) from exc

    @staticmethod
    def _register_policy(session: Session, claim: _WindowClaim) -> None:
        identity = {
            "provider": claim.policy.scope.provider,
            "scope_kind": claim.policy.scope.kind.value,
            "scope_fingerprint": claim.fingerprint,
            "metric": claim.policy.metric.value,
        }
        session.execute(
            postgresql_insert(ProviderBudgetPolicy)
            .values(**identity, window_seconds=claim.policy.window_seconds)
            .on_conflict_do_nothing(constraint="uq_budget_policy_identity")
        )
        policy = session.scalar(
            select(ProviderBudgetPolicy)
            .where(
                ProviderBudgetPolicy.provider == claim.policy.scope.provider,
                ProviderBudgetPolicy.scope_kind == claim.policy.scope.kind.value,
                ProviderBudgetPolicy.scope_fingerprint == claim.fingerprint,
                ProviderBudgetPolicy.metric == claim.policy.metric.value,
            )
            .with_for_update()
        )
        if policy is None or policy.window_seconds != claim.policy.window_seconds:
            raise ProviderSchedulerPolicyConflict(
                "Provider budget window policy conflicts for this scope and metric"
            )

    def cleanup(
        self,
        *,
        now: datetime | None = None,
        batch_size: int = 100,
        terminal_retention: timedelta = timedelta(days=30),
    ) -> int:
        """Atomically remove at most batch_size rows from each retention category.

        Active leases are preserved for live streams. Their allocations stop affecting
        capacity at the fixed-window boundary, and stale completion cannot charge it.
        """
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if terminal_retention.total_seconds() < 0:
            raise ValueError("terminal_retention must not be negative")
        moment = _utc(now or datetime.now(timezone.utc))
        cutoff = moment - terminal_retention
        try:
            with self._sessions.begin() as session:
                _require_postgresql(session)
                terminal_lease_ids = tuple(
                    session.scalars(
                        select(ProviderBudgetLease.id)
                        .where(
                            ProviderBudgetLease.status != "active",
                            ProviderBudgetLease.completed_at <= cutoff,
                        )
                        .order_by(
                            ProviderBudgetLease.completed_at,
                            ProviderBudgetLease.id,
                        )
                        .limit(batch_size)
                    ).all()
                )
                expired_allocation_ids = tuple(
                    session.scalars(
                        select(ProviderBudgetLeaseAllocation.id)
                        .join(
                            ProviderBudgetWindow,
                            ProviderBudgetWindow.id
                            == ProviderBudgetLeaseAllocation.window_id,
                        )
                        .where(
                            ProviderBudgetWindow.window_start
                            + func.make_interval(
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                ProviderBudgetWindow.window_seconds,
                            )
                            <= moment
                        )
                        .order_by(
                            ProviderBudgetLeaseAllocation.lease_id.collate("C"),
                            ProviderBudgetWindow.provider.collate("C"),
                            ProviderBudgetWindow.scope_kind.collate("C"),
                            ProviderBudgetWindow.scope_fingerprint.collate("C"),
                            ProviderBudgetWindow.metric.collate("C"),
                            ProviderBudgetWindow.window_seconds,
                            ProviderBudgetWindow.window_start,
                            ProviderBudgetWindow.id,
                            ProviderBudgetLeaseAllocation.id,
                        )
                        .limit(batch_size)
                    ).all()
                )
                allocation_candidates = (
                    session.scalars(
                        select(ProviderBudgetLeaseAllocation).where(
                            ProviderBudgetLeaseAllocation.id.in_(expired_allocation_ids)
                        )
                    ).all()
                    if expired_allocation_ids
                    else []
                )
                lease_ids = tuple(
                    sorted({item.lease_id for item in allocation_candidates})
                )
                locked_leases = {}
                for lease_id in lease_ids:
                    lease = session.scalar(
                        select(ProviderBudgetLease)
                        .where(ProviderBudgetLease.id == lease_id)
                        .with_for_update(skip_locked=True)
                    )
                    if lease is not None:
                        locked_leases[lease.id] = lease
                candidate_window_ids = tuple(
                    sorted({item.window_id for item in allocation_candidates})
                )
                candidate_windows = (
                    session.scalars(
                        select(ProviderBudgetWindow).where(
                            ProviderBudgetWindow.id.in_(candidate_window_ids)
                        )
                    ).all()
                    if candidate_window_ids
                    else []
                )
                locked_windows = {}
                for key in sorted(
                    _window_lock_key(window) for window in candidate_windows
                ):
                    window = session.scalar(
                        select(ProviderBudgetWindow)
                        .where(ProviderBudgetWindow.id == key[-1])
                        .with_for_update(skip_locked=True)
                    )
                    if window is not None:
                        locked_windows[window.id] = window
                deletable_allocation_ids = tuple(
                    item.id
                    for item in allocation_candidates
                    if item.lease_id in locked_leases
                    and item.window_id in locked_windows
                )
                if deletable_allocation_ids:
                    session.execute(
                        delete(ProviderBudgetLeaseAllocation).where(
                            ProviderBudgetLeaseAllocation.id.in_(
                                deletable_allocation_ids
                            )
                        )
                    )
                session.flush()
                reclaimed_window_ids = tuple(
                    window_id
                    for window_id in locked_windows
                    if session.scalar(
                        select(ProviderBudgetLeaseAllocation.id)
                        .where(ProviderBudgetLeaseAllocation.window_id == window_id)
                        .limit(1)
                    )
                    is None
                )
                if reclaimed_window_ids:
                    session.execute(
                        delete(ProviderBudgetWindow).where(
                            ProviderBudgetWindow.id.in_(reclaimed_window_ids)
                        )
                    )
                for lease in locked_leases.values():
                    has_allocations = session.scalar(
                        select(ProviderBudgetLeaseAllocation.id)
                        .where(ProviderBudgetLeaseAllocation.lease_id == lease.id)
                        .limit(1)
                    )
                    if lease.status == "active" and has_allocations is None:
                        lease.status = "expired"
                        lease.completed_at = moment
                old_lease_ids = tuple(
                    session.scalars(
                        select(ProviderBudgetLease.id)
                        .where(ProviderBudgetLease.id.in_(terminal_lease_ids))
                        .order_by(ProviderBudgetLease.id)
                        .with_for_update(skip_locked=True)
                    ).all()
                )
                if old_lease_ids:
                    session.execute(
                        delete(ProviderBudgetLease).where(
                            ProviderBudgetLease.id.in_(old_lease_ids)
                        )
                    )
                session.flush()
                remaining_window_batch = max(0, batch_size - len(reclaimed_window_ids))
                orphan_windows = session.scalars(
                    select(ProviderBudgetWindow)
                    .where(
                        ProviderBudgetWindow.window_start
                        + func.make_interval(
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            ProviderBudgetWindow.window_seconds,
                        )
                        <= moment,
                        ~select(ProviderBudgetLeaseAllocation.id)
                        .where(
                            ProviderBudgetLeaseAllocation.window_id
                            == ProviderBudgetWindow.id
                        )
                        .exists(),
                    )
                    .order_by(
                        ProviderBudgetWindow.provider.collate("C"),
                        ProviderBudgetWindow.scope_kind.collate("C"),
                        ProviderBudgetWindow.scope_fingerprint.collate("C"),
                        ProviderBudgetWindow.metric.collate("C"),
                        ProviderBudgetWindow.window_seconds,
                        ProviderBudgetWindow.window_start,
                        ProviderBudgetWindow.id,
                    )
                    .limit(remaining_window_batch)
                ).all()
                orphan_ids = set()
                for key in sorted(
                    _window_lock_key(window) for window in orphan_windows
                ):
                    if len(orphan_ids) >= remaining_window_batch:
                        break
                    window = session.scalar(
                        select(ProviderBudgetWindow)
                        .where(ProviderBudgetWindow.id == key[-1])
                        .with_for_update(skip_locked=True)
                    )
                    if window is not None:
                        orphan_ids.add(window.id)
                if orphan_ids:
                    session.execute(
                        delete(ProviderBudgetWindow).where(
                            ProviderBudgetWindow.id.in_(orphan_ids)
                        )
                    )
                inactive_cooldown = or_(
                    ProviderBudgetScopeState.cooldown_until.is_(None),
                    ProviderBudgetScopeState.cooldown_until <= moment,
                )
                has_budget_policy = (
                    select(ProviderBudgetPolicy.id)
                    .where(
                        ProviderBudgetPolicy.provider
                        == ProviderBudgetScopeState.provider,
                        ProviderBudgetPolicy.scope_kind
                        == ProviderBudgetScopeState.scope_kind,
                        ProviderBudgetPolicy.scope_fingerprint
                        == ProviderBudgetScopeState.scope_fingerprint,
                    )
                    .exists()
                )
                scope_candidates = tuple(
                    session.scalars(
                        select(ProviderBudgetScopeState.id)
                        .where(inactive_cooldown, ~has_budget_policy)
                        .order_by(
                            ProviderBudgetScopeState.provider.collate("C"),
                            ProviderBudgetScopeState.scope_kind.collate("C"),
                            ProviderBudgetScopeState.scope_fingerprint.collate("C"),
                        )
                        .limit(batch_size)
                    ).all()
                )
                expired_scope_ids = tuple(
                    session.scalars(
                        select(ProviderBudgetScopeState.id)
                        .where(
                            ProviderBudgetScopeState.id.in_(scope_candidates),
                            inactive_cooldown,
                            ~has_budget_policy,
                        )
                        .order_by(ProviderBudgetScopeState.id)
                        .with_for_update(skip_locked=True)
                    ).all()
                )
                if expired_scope_ids:
                    session.execute(
                        delete(ProviderBudgetScopeState).where(
                            ProviderBudgetScopeState.id.in_(expired_scope_ids)
                        )
                    )
                return (
                    len(deletable_allocation_ids)
                    + len(old_lease_ids)
                    + len(reclaimed_window_ids)
                    + len(orphan_ids)
                    + len(expired_scope_ids)
                )
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not clean up provider budget state in PostgreSQL"
            ) from exc

    def _fingerprints(
        self, tenant_id: str, scopes: tuple[BudgetScope, ...]
    ) -> dict[BudgetScope, str]:
        return {scope: self._fingerprint(tenant_id, scope) for scope in scopes}

    def _fingerprint(self, tenant_id: str, scope: BudgetScope) -> str:
        if self._cipher is None:
            raise ProviderSchedulerStoreError(
                "Provider budget scope fingerprinting requires the configured secret cipher"
            )
        shared_identity = scope.shared_identity
        tenant_scope = shared_identity is None
        return self._cipher.provider_budget_fingerprint(
            scope.provider,
            scope.kind.value,
            shared_identity.canonical_id if shared_identity else scope.scope_id,
            tenant_id=tenant_id if tenant_scope else None,
        )

    @staticmethod
    def _lock_scope_states(
        session: Session,
        scopes: tuple[BudgetScope, ...],
        fingerprints: dict[BudgetScope, str],
        moment: datetime,
    ) -> None:
        for scope in sorted(
            scopes, key=lambda item: _scope_sort_key(item, fingerprints[item])
        ):
            identity = _scope_identity(scope, fingerprints[scope])
            session.execute(
                postgresql_insert(ProviderBudgetScopeState)
                .values(**identity, cooldown_until=None, updated_at=moment)
                .on_conflict_do_nothing(constraint="uq_budget_scope_identity")
            )
            session.scalar(
                select(ProviderBudgetScopeState)
                .where(*_scope_predicate(scope, fingerprints[scope]))
                .with_for_update()
            )

    @staticmethod
    def _read_scope_states(
        session: Session,
        scopes: tuple[BudgetScope, ...],
        fingerprints: dict[BudgetScope, str],
    ) -> list[ProviderBudgetScopeState]:
        states = []
        for scope in sorted(
            scopes, key=lambda item: _scope_sort_key(item, fingerprints[item])
        ):
            state = session.scalar(
                select(ProviderBudgetScopeState).where(
                    *_scope_predicate(scope, fingerprints[scope])
                )
            )
            if state is None:
                raise ProviderSchedulerStoreError(
                    "Could not load provider budget scope"
                )
            states.append(state)
        return states

    @staticmethod
    def _lock_windows(
        session: Session,
        provider: str,
        claims: tuple[_WindowClaim, ...],
    ) -> list[ProviderBudgetWindow]:
        claim_keys = tuple(_window_claim_lock_key(provider, claim) for claim in claims)
        if claim_keys != tuple(sorted(claim_keys)):
            raise AssertionError("Provider budget claims must use canonical lock order")
        rows = []
        for claim in claims:
            session.execute(
                postgresql_insert(ProviderBudgetWindow)
                .values(
                    provider=provider,
                    scope_kind=claim.policy.scope.kind.value,
                    scope_fingerprint=claim.fingerprint,
                    metric=claim.policy.metric.value,
                    window_seconds=claim.policy.window_seconds,
                    window_start=claim.window_start,
                    limit_units=claim.policy.limit_units,
                    used_units=0,
                    reserved_units=0,
                )
                .on_conflict_do_nothing(constraint="uq_budget_window_identity")
            )
            row = session.scalar(
                select(ProviderBudgetWindow)
                .where(
                    ProviderBudgetWindow.provider == provider,
                    ProviderBudgetWindow.scope_kind == claim.policy.scope.kind.value,
                    ProviderBudgetWindow.scope_fingerprint == claim.fingerprint,
                    ProviderBudgetWindow.metric == claim.policy.metric.value,
                    ProviderBudgetWindow.window_seconds == claim.policy.window_seconds,
                    ProviderBudgetWindow.window_start == claim.window_start,
                )
                .with_for_update()
            )
            if row is None:
                raise ProviderSchedulerStoreError(
                    "Could not create provider budget window"
                )
            if row.limit_units != claim.policy.limit_units:
                row.limit_units = claim.policy.limit_units
            rows.append(row)
        return rows


class ProviderConcurrencyController:
    """Persist per-profile AIMD capacity and renewable cross-worker leases."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        cipher: SecretCipher,
        *,
        initial_limit: int = AIMD_INITIAL_CONCURRENCY,
        lease_ttl: timedelta = AIMD_LEASE_TTL,
        minimum_limit: int = AIMD_MIN_CONCURRENCY,
        maximum_limit: int = AIMD_MAX_CONCURRENCY,
    ) -> None:
        """Bind PostgreSQL storage and bounded AIMD lease settings."""
        if not isinstance(cipher, SecretCipher):
            raise ValueError("AIMD concurrency requires the configured secret cipher")
        if (
            isinstance(initial_limit, bool)
            or not isinstance(initial_limit, int)
            or isinstance(minimum_limit, bool)
            or not isinstance(minimum_limit, int)
            or isinstance(maximum_limit, bool)
            or not isinstance(maximum_limit, int)
            or not AIMD_MIN_CONCURRENCY
            <= minimum_limit
            <= initial_limit
            <= maximum_limit
            <= AIMD_MAX_CONCURRENCY
        ):
            raise ValueError(
                "AIMD limits must satisfy 1 <= minimum <= initial <= maximum <= 64"
            )
        if lease_ttl.total_seconds() <= 0:
            raise ValueError("lease_ttl must be positive")
        self._sessions = sessions
        self._cipher = cipher
        self._initial_limit = initial_limit
        self._minimum_limit = minimum_limit
        self._maximum_limit = maximum_limit
        self._lease_ttl = lease_ttl

    @property
    def heartbeat_interval_seconds(self) -> float:
        """Return a renewal cadence that leaves two thirds of the lease as margin."""
        return max(0.1, self._lease_ttl.total_seconds() / 3)

    @property
    def lease_ttl_seconds(self) -> float:
        """Return the finite lease lifetime used to bound renewal outages."""
        return self._lease_ttl.total_seconds()

    def acquire(
        self,
        tenant_id: str,
        provider: str,
        profile_id: str,
        *,
        target_model: str | None = None,
        profile_initial_limit: int | None = None,
        profile_minimum_limit: int | None = None,
        profile_maximum_limit: int | None = None,
        model_initial_limit: int | None = None,
        model_minimum_limit: int | None = None,
        model_maximum_limit: int | None = None,
        now: datetime | None = None,
    ) -> ProviderConcurrencyDecision:
        """Acquire profile and optional model capacity for one candidate."""
        candidate = ProviderConcurrencyCandidate(
            provider=provider,
            profile_id=profile_id,
            target_model=target_model,
            profile_initial_limit=(
                profile_initial_limit
                if profile_initial_limit is not None
                else self._initial_limit
            ),
            profile_minimum_limit=(
                profile_minimum_limit
                if profile_minimum_limit is not None
                else self._minimum_limit
            ),
            profile_maximum_limit=(
                profile_maximum_limit
                if profile_maximum_limit is not None
                else self._maximum_limit
            ),
            model_initial_limit=(
                model_initial_limit
                if model_initial_limit is not None
                else self._initial_limit
            ),
            model_minimum_limit=(
                model_minimum_limit
                if model_minimum_limit is not None
                else self._minimum_limit
            ),
            model_maximum_limit=(
                model_maximum_limit
                if model_maximum_limit is not None
                else self._maximum_limit
            ),
        )
        if target_model is None and any(
            value is not None
            for value in (model_initial_limit, model_minimum_limit, model_maximum_limit)
        ):
            raise ValueError("Model AIMD bounds require a target_model")
        decision = self.acquire_best(tenant_id, (candidate,), now=now)
        if target_model is None and decision.lease is not None:
            return decision
        return decision

    def acquire_best(
        self,
        tenant_id: str,
        candidates: tuple[ProviderConcurrencyCandidate, ...],
        *,
        now: datetime | None = None,
    ) -> ProviderConcurrencyDecision:
        """Lock a candidate group, score current utilization, and lease one winner."""
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("AIMD tenant identity must be non-empty")
        if not candidates or any(
            not isinstance(candidate, ProviderConcurrencyCandidate)
            for candidate in candidates
        ):
            raise ValueError("At least one typed AIMD candidate is required")
        identities = [
            (item.provider, item.profile_id, item.target_model) for item in candidates
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("AIMD candidate group contains duplicate identities")

        candidate_scopes = {}
        candidate_scope_roles = {}
        bounds_by_scope = {}
        for candidate in candidates:
            if (
                not isinstance(candidate.profile_id, str)
                or not candidate.profile_id.strip()
            ):
                raise ValueError("AIMD profile identity must be non-empty")
            profile_scope = BudgetScope(
                candidate.provider, BudgetScopeKind.PROFILE, candidate.profile_id
            )
            model_scope = None
            if candidate.target_model is not None:
                if not candidate.target_model.strip():
                    raise ValueError("AIMD target model must be non-empty")
                model_scope = BudgetScope(
                    candidate.provider,
                    BudgetScopeKind.MODEL,
                    json.dumps(
                        [candidate.profile_id, candidate.target_model],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            profile_bounds = _validate_aimd_bounds(
                candidate.profile_initial_limit,
                candidate.profile_minimum_limit,
                candidate.profile_maximum_limit,
            )
            model_bounds = _validate_aimd_bounds(
                candidate.model_initial_limit,
                candidate.model_minimum_limit,
                candidate.model_maximum_limit,
            )
            candidate_key = (
                candidate.provider,
                candidate.profile_id,
                candidate.target_model,
            )
            candidate_scopes[candidate_key] = tuple(
                sorted(
                    (
                        scope
                        for scope in (profile_scope, model_scope)
                        if scope is not None
                    ),
                    key=lambda scope: (
                        scope.provider,
                        scope.kind.value,
                        scope.scope_id,
                    ),
                )
            )
            candidate_scope_roles[candidate_key] = (profile_scope, model_scope)
            scoped_bounds = ((profile_scope, profile_bounds),)
            if model_scope is not None:
                scoped_bounds += ((model_scope, model_bounds),)
            for scope, bounds in scoped_bounds:
                existing = bounds_by_scope.setdefault(scope, bounds)
                if existing != bounds:
                    raise ValueError("AIMD bounds conflict for a shared scope")

        ordered_scopes = sorted(
            bounds_by_scope,
            key=lambda scope: (scope.provider, scope.kind.value, scope.scope_id),
        )
        moment = _utc(now or datetime.now(timezone.utc))
        try:
            with self._sessions.begin() as session:
                _require_postgresql(session)
                states_by_scope = {}
                for scope in ordered_scopes:
                    initial, minimum, maximum = bounds_by_scope[scope]
                    fingerprint = self._fingerprint(tenant_id, scope)
                    session.execute(
                        postgresql_insert(ProviderConcurrencyState)
                        .values(
                            tenant_id=tenant_id,
                            provider=scope.provider,
                            scope_fingerprint=fingerprint,
                            concurrency_limit=initial,
                            minimum_limit=minimum,
                            maximum_limit=maximum,
                            updated_at=moment,
                        )
                        .on_conflict_do_nothing(constraint="uq_concurrency_scope")
                    )
                    state = session.scalar(
                        select(ProviderConcurrencyState)
                        .where(
                            ProviderConcurrencyState.tenant_id == tenant_id,
                            ProviderConcurrencyState.provider == scope.provider,
                            ProviderConcurrencyState.scope_fingerprint == fingerprint,
                        )
                        .with_for_update()
                    )
                    if state is None:
                        raise ProviderSchedulerStoreError(
                            "Could not create AIMD concurrency state"
                        )
                    state.minimum_limit = minimum
                    state.maximum_limit = maximum
                    state.concurrency_limit = min(
                        maximum, max(minimum, state.concurrency_limit)
                    )
                    session.execute(
                        update(ProviderConcurrencyLeaseRecord)
                        .where(
                            ProviderConcurrencyLeaseRecord.state_id == state.id,
                            ProviderConcurrencyLeaseRecord.status == "active",
                            ProviderConcurrencyLeaseRecord.expires_at <= moment,
                        )
                        .values(status="expired", completed_at=moment)
                    )
                    states_by_scope[scope] = state

                active_counts = {}
                active_expiries = {}
                for scope in ordered_scopes:
                    state = states_by_scope[scope]
                    active_rows = session.scalars(
                        select(ProviderConcurrencyLeaseRecord)
                        .where(
                            ProviderConcurrencyLeaseRecord.state_id == state.id,
                            ProviderConcurrencyLeaseRecord.status == "active",
                            ProviderConcurrencyLeaseRecord.expires_at > moment,
                        )
                        .order_by(ProviderConcurrencyLeaseRecord.expires_at)
                    ).all()
                    active_counts[scope] = len(active_rows)
                    active_expiries[scope] = tuple(
                        row.expires_at for row in active_rows
                    )

                eligible = []
                blocked_expiries = []
                for candidate in candidates:
                    profile_scope, model_scope = candidate_scope_roles[
                        (
                            candidate.provider,
                            candidate.profile_id,
                            candidate.target_model,
                        )
                    ]
                    profile_state = states_by_scope[profile_scope]
                    profile_active = active_counts[profile_scope]
                    model_state = (
                        states_by_scope[model_scope]
                        if model_scope is not None
                        else None
                    )
                    model_active = (
                        active_counts[model_scope] if model_scope is not None else 0
                    )
                    if profile_active >= profile_state.concurrency_limit or (
                        model_state is not None
                        and model_active >= model_state.concurrency_limit
                    ):
                        profile_expiries = active_expiries[profile_scope]
                        model_expiries = (
                            active_expiries[model_scope]
                            if model_scope is not None
                            else ()
                        )
                        blocked_expiries.extend(
                            expiry for expiry in (*profile_expiries, *model_expiries)
                        )
                        continue
                    score = _capacity_score(
                        profile_active,
                        profile_state.concurrency_limit,
                        model_active,
                        model_state.concurrency_limit if model_state is not None else 1,
                        candidate.weight,
                        budget_headroom=candidate.budget_headroom,
                        headroom_weight=candidate.headroom_weight,
                    )
                    eligible.append((score, candidate.profile_id, candidate))

                if not eligible:
                    retry_at = min(blocked_expiries) if blocked_expiries else None
                    return ProviderConcurrencyDecision(False, retry_at=retry_at)

                score, _profile_id, winner = min(
                    eligible,
                    key=lambda item: (
                        item[0],
                        item[1],
                        item[2].target_model or "",
                    ),
                )
                permits = []
                winner_scopes = candidate_scopes[
                    (winner.provider, winner.profile_id, winner.target_model)
                ]
                for scope in winner_scopes:
                    state = states_by_scope[scope]
                    permit_id = str(uuid4())
                    token = str(uuid4())
                    session.add(
                        ProviderConcurrencyLeaseRecord(
                            id=permit_id,
                            state_id=state.id,
                            lease_token=token,
                            status="active",
                            created_at=moment,
                            expires_at=moment + self._lease_ttl,
                        )
                    )
                    permits.append(
                        ProviderConcurrencyPermit(permit_id, state.id, token)
                    )
                session.flush()
                primary, *additional = permits
                return ProviderConcurrencyDecision(
                    True,
                    ProviderConcurrencyLease(
                        primary.id,
                        primary.state_id,
                        tenant_id,
                        winner.provider,
                        primary.token,
                        tuple(additional),
                    ),
                    capacity_score=score,
                    profile_id=winner.profile_id,
                )
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not acquire AIMD provider concurrency lease"
            ) from exc

    def renew(
        self, lease: ProviderConcurrencyLease, *, now: datetime | None = None
    ) -> bool:
        """Renew every scope permit together or reject the whole lease."""
        moment = _utc(now or datetime.now(timezone.utc))
        try:
            with self._sessions.begin() as session:
                _require_postgresql(session)
                states = self._lock_lease_states(session, lease)
                if len(states) != len(lease.permits):
                    return False
                records = self._lock_lease_records(session, lease)
                if len(records) != len(lease.permits) or any(
                    record.status != "active" or record.expires_at <= moment
                    for record in records
                ):
                    return False
                for record in records:
                    record.expires_at = moment + self._lease_ttl
                return True
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not renew AIMD provider concurrency lease"
            ) from exc

    def settle(
        self,
        lease: ProviderConcurrencyLease,
        *,
        outcome: str,
        now: datetime | None = None,
    ) -> bool:
        """Settle all scope permits and update each independent AIMD bound."""
        if outcome not in {
            "success",
            "transient_failure",
            "terminal_failure",
            "released",
        }:
            raise ValueError("Unsupported AIMD lease outcome")
        moment = _utc(now or datetime.now(timezone.utc))
        try:
            with self._sessions.begin() as session:
                _require_postgresql(session)
                states = self._lock_lease_states(session, lease)
                if len(states) != len(lease.permits):
                    return False
                records = self._lock_lease_records(session, lease)
                if len(records) != len(lease.permits):
                    for record in records:
                        if record.status == "active":
                            record.status = "released"
                            record.completed_at = moment
                    return False
                if any(record.status != "active" for record in records):
                    for record in records:
                        if record.status == "active":
                            record.status = "released"
                            record.completed_at = moment
                    return False
                if any(record.expires_at <= moment for record in records):
                    for record in records:
                        if record.expires_at <= moment:
                            record.status = "expired"
                            record.completed_at = moment
                    return False
                for state, record in zip(states, records):
                    record.status = "released" if outcome == "released" else "completed"
                    record.completed_at = moment
                    state.concurrency_limit = _adjust_aimd_limit(
                        state.concurrency_limit,
                        outcome=outcome,
                        minimum=state.minimum_limit,
                        maximum=state.maximum_limit,
                    )
                    state.updated_at = moment
                return True
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not settle AIMD provider concurrency lease"
            ) from exc

    def _lock_lease_states(
        self, session: Session, lease: ProviderConcurrencyLease
    ) -> list[ProviderConcurrencyState]:
        """Lock all lease scope states in ascending primary-key order."""
        permits = lease.permits
        states = []
        for permit in permits:
            state = session.scalar(
                select(ProviderConcurrencyState)
                .where(
                    ProviderConcurrencyState.id == permit.state_id,
                    ProviderConcurrencyState.tenant_id == lease.tenant_id,
                    ProviderConcurrencyState.provider == lease.provider,
                )
                .with_for_update()
            )
            if state is not None:
                states.append(state)
        return states

    def _lock_lease_records(
        self,
        session: Session,
        lease: ProviderConcurrencyLease,
    ) -> list[ProviderConcurrencyLeaseRecord]:
        """Lock every permit row in the same deterministic state order."""
        permits = lease.permits
        records = []
        for permit in permits:
            statement = select(ProviderConcurrencyLeaseRecord).where(
                ProviderConcurrencyLeaseRecord.id == permit.id,
                ProviderConcurrencyLeaseRecord.state_id == permit.state_id,
                ProviderConcurrencyLeaseRecord.lease_token == permit.token,
                ProviderConcurrencyLeaseRecord.status == "active",
            )
            record = session.scalar(statement.with_for_update())
            if record is not None:
                records.append(record)
        return records

    def cleanup(
        self,
        *,
        now: datetime | None = None,
        batch_size: int = 100,
        terminal_retention: timedelta = timedelta(days=30),
    ) -> int:
        """Delete a bounded number of old terminal lease records, preserving limits."""
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if terminal_retention.total_seconds() < 0:
            raise ValueError("terminal_retention must not be negative")
        moment = _utc(now or datetime.now(timezone.utc))
        cutoff = moment - terminal_retention
        try:
            with self._sessions.begin() as session:
                _require_postgresql(session)
                expired_ids = tuple(
                    session.scalars(
                        select(ProviderConcurrencyLeaseRecord.id)
                        .where(
                            ProviderConcurrencyLeaseRecord.status == "active",
                            ProviderConcurrencyLeaseRecord.expires_at <= moment,
                        )
                        .order_by(
                            ProviderConcurrencyLeaseRecord.expires_at,
                            ProviderConcurrencyLeaseRecord.id,
                        )
                        .with_for_update(skip_locked=True)
                        .limit(batch_size)
                    ).all()
                )
                if expired_ids:
                    session.execute(
                        update(ProviderConcurrencyLeaseRecord)
                        .where(ProviderConcurrencyLeaseRecord.id.in_(expired_ids))
                        .values(status="expired", completed_at=moment)
                    )
                remaining_batch = batch_size - len(expired_ids)
                ids = (
                    tuple(
                        session.scalars(
                            select(ProviderConcurrencyLeaseRecord.id)
                            .where(
                                ProviderConcurrencyLeaseRecord.status != "active",
                                ProviderConcurrencyLeaseRecord.completed_at <= cutoff,
                            )
                            .order_by(
                                ProviderConcurrencyLeaseRecord.completed_at,
                                ProviderConcurrencyLeaseRecord.id,
                            )
                            .with_for_update(skip_locked=True)
                            .limit(remaining_batch)
                        ).all()
                    )
                    if remaining_batch
                    else ()
                )
                if ids:
                    session.execute(
                        delete(ProviderConcurrencyLeaseRecord).where(
                            ProviderConcurrencyLeaseRecord.id.in_(ids)
                        )
                    )
                return len(expired_ids) + len(ids)
        except SQLAlchemyError as exc:
            raise ProviderSchedulerStoreError(
                "Could not clean up AIMD concurrency leases"
            ) from exc

    def _fingerprint(self, tenant_id: str, scope: BudgetScope) -> str:
        return self._cipher.provider_budget_fingerprint(
            scope.provider,
            "aimd_profile",
            scope.scope_id,
            tenant_id=tenant_id,
        )


def _window_claim_lock_key(
    provider: str,
    claim: _WindowClaim,
) -> tuple[str, str, str, str, int, datetime, int]:
    return _window_lock_key_from_values(
        provider,
        claim.policy.scope.kind.value,
        claim.fingerprint,
        claim.policy.metric.value,
        claim.policy.window_seconds,
        claim.window_start,
        0,
    )


def _window_lock_key(
    window: ProviderBudgetWindow,
) -> tuple[str, str, str, str, int, datetime, int]:
    return _window_lock_key_from_values(
        window.provider,
        window.scope_kind,
        window.scope_fingerprint,
        window.metric,
        window.window_seconds,
        _utc(window.window_start),
        window.id,
    )


def _window_lock_key_from_values(
    provider: str,
    scope_kind: str,
    fingerprint: str,
    metric: str,
    window_seconds: int,
    window_start: datetime,
    row_id: int,
) -> tuple[str, str, str, str, int, datetime, int]:
    return (
        provider,
        scope_kind,
        fingerprint,
        metric,
        window_seconds,
        window_start,
        row_id,
    )


def _scope_sort_key(scope: BudgetScope, fingerprint: str) -> tuple[str, str, str]:
    return scope.provider, scope.kind.value, fingerprint


def _scope_identity(scope: BudgetScope, fingerprint: str) -> dict[str, str]:
    return {
        "provider": scope.provider,
        "scope_kind": scope.kind.value,
        "scope_fingerprint": fingerprint,
    }


def _scope_predicate(scope: BudgetScope, fingerprint: str) -> tuple[object, ...]:
    return (
        ProviderBudgetScopeState.provider == scope.provider,
        ProviderBudgetScopeState.scope_kind == scope.kind.value,
        ProviderBudgetScopeState.scope_fingerprint == fingerprint,
    )


def _window_start(moment: datetime, window_seconds: int) -> datetime:
    timestamp = int(moment.timestamp())
    return datetime.fromtimestamp(timestamp - timestamp % window_seconds, timezone.utc)


def _utc(moment: datetime) -> datetime:
    return (
        moment.replace(tzinfo=timezone.utc)
        if moment.tzinfo is None
        else moment.astimezone(timezone.utc)
    )


def _require_postgresql(session: Session) -> None:
    if session.get_bind().dialect.name != "postgresql":
        raise ProviderSchedulerStoreError(
            "Provider scheduler persistence requires PostgreSQL; no local fallback is available"
        )


class ProviderSchedulerStoreError(RuntimeError):
    """A scheduler operation could not be safely completed in PostgreSQL."""


class ProviderSchedulerPolicyConflict(RuntimeError):
    """Concurrent workers supplied different limits for the same active window."""
