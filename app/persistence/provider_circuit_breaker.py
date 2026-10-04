"""Persistent, worker-shared quota circuit state and half-open leases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.common.retry_after import RETRY_AFTER_MAX_SECONDS
from app.persistence.models import ProviderCircuitState
from app.persistence.secrets import SecretCipher

_INITIAL_COOLDOWN = timedelta(hours=1)
_MAX_COOLDOWN = timedelta(hours=24)
_MAX_BACKOFF_STAGE = 6
PROBE_LEASE_SECONDS = 30
_PROBE_LEASE = timedelta(seconds=PROBE_LEASE_SECONDS)
_SUPPORTED_PROVIDERS = frozenset({"azure", "openai", "openrouter", "deepseek"})
_SUPPORTED_SCOPES = frozenset({"profile", "organization", "project"})


@dataclass(frozen=True)
class BreakerScope:
    """One tenant-owned breaker identity without a persisted raw scope ID."""

    tenant_id: str
    provider: str
    scope_type: str
    fingerprint: str

    def __post_init__(self) -> None:
        """Reject invalid scope identifiers before any store operation."""
        if self.provider not in _SUPPORTED_PROVIDERS:
            raise ValueError("Unsupported provider circuit provider")
        if self.scope_type not in _SUPPORTED_SCOPES:
            raise ValueError("Unsupported provider circuit scope type")
        if not self.tenant_id or len(self.fingerprint) != 64:
            raise ValueError("Invalid provider circuit scope identity")


@dataclass(frozen=True)
class ProbeLease:
    """Lease token bound to a single persisted breaker scope."""

    scope: BreakerScope
    token: str
    lease_until: datetime


@dataclass(frozen=True)
class CircuitPermit:
    """Result of atomically checking/claiming all route scopes."""

    allowed: bool
    leases: tuple[ProbeLease, ...] = ()
    retry_at: datetime | None = None


@dataclass(frozen=True)
class CircuitSnapshot:
    """Secret-free state for one configured scope."""

    scope: BreakerScope
    probe_at: datetime
    failure_count: int
    lease_until: datetime | None
    failure_category: str = "quota_exhausted"


class ProviderCircuitBreakerStore:
    """Own durable breaker mutations with short, explicit transactions."""

    def __init__(self, sessions: sessionmaker[Session], cipher: SecretCipher) -> None:
        """Bind the transaction factory and tenant-scope fingerprint service."""
        self._sessions = sessions
        self._cipher = cipher

    def scope(
        self, tenant_id: str, provider: str, scope_type: str, scope_id: str
    ) -> BreakerScope:
        """Build the stable HMAC identity for a profile/org/project scope."""
        return BreakerScope(
            tenant_id,
            provider,
            scope_type,
            self._cipher.scope_fingerprint(tenant_id, provider, scope_type, scope_id),
        )

    def open_quota(
        self,
        scope: BreakerScope,
        *,
        now: datetime | None = None,
        retry_after_seconds: float | None = None,
    ) -> CircuitSnapshot:
        """Open or extend a quota breaker using capped exponential cooldown."""
        moment = _utc(now or datetime.now(timezone.utc))
        retry_after = max(0.0, retry_after_seconds or 0.0)
        try:
            with self._sessions.begin() as session:
                state, created = self._locked_state(session, scope, moment)
                failure_count = 1 if created else state.failure_count + 1
                stage = min(failure_count, _MAX_BACKOFF_STAGE)
                cooldown = min(_INITIAL_COOLDOWN * (2 ** (stage - 1)), _MAX_COOLDOWN)
                probe_at = max(
                    moment + cooldown,
                    moment
                    + timedelta(seconds=min(retry_after, RETRY_AFTER_MAX_SECONDS)),
                )
                state.failure_category = "quota_exhausted"
                state.failure_count = failure_count
                state.probe_at = probe_at
                state.lease_token = None
                state.lease_until = None
                state.updated_at = moment
                session.flush()
                return CircuitSnapshot(scope, probe_at, failure_count, None)
        except SQLAlchemyError as exc:
            raise ProviderCircuitStoreError(
                "Could not persist quota circuit state"
            ) from exc

    def acquire(
        self,
        scopes: tuple[BreakerScope, ...],
        *,
        now: datetime | None = None,
    ) -> CircuitPermit:
        """Check all scopes and exclusively lease due probes in one transaction."""
        moment = _utc(now or datetime.now(timezone.utc))
        unique_scopes = tuple(
            sorted(
                set(scopes),
                key=lambda item: (item.provider, item.scope_type, item.fingerprint),
            )
        )
        if not unique_scopes:
            return CircuitPermit(True)
        try:
            with self._sessions.begin() as session:
                states = session.scalars(
                    select(ProviderCircuitState)
                    .where(
                        ProviderCircuitState.tenant_id == unique_scopes[0].tenant_id,
                        ProviderCircuitState.provider.in_(
                            {scope.provider for scope in unique_scopes}
                        ),
                        ProviderCircuitState.scope_fingerprint.in_(
                            {scope.fingerprint for scope in unique_scopes}
                        ),
                    )
                    .order_by(ProviderCircuitState.id)
                    .with_for_update()
                ).all()
                scope_by_identity = {_identity(scope): scope for scope in unique_scopes}
                states = [
                    state
                    for state in states
                    if _identity_from_state(state) in scope_by_identity
                ]
                blocked_until = []
                for state in states:
                    lease_until = _utc(state.lease_until) if state.lease_until else None
                    if _utc(state.probe_at) > moment:
                        blocked_until.append(_utc(state.probe_at))
                    elif lease_until is not None and lease_until > moment:
                        blocked_until.append(lease_until)
                if blocked_until:
                    return CircuitPermit(False, retry_at=max(blocked_until))

                leases = []
                lease_until = moment + _PROBE_LEASE
                for state in states:
                    token = str(uuid4())
                    state.lease_token = token
                    state.lease_until = lease_until
                    state.updated_at = moment
                    leases.append(
                        ProbeLease(
                            scope_by_identity[_identity_from_state(state)],
                            token,
                            lease_until,
                        )
                    )
                return CircuitPermit(True, tuple(leases))
        except SQLAlchemyError as exc:
            raise ProviderCircuitStoreError(
                "Could not acquire provider probe lease"
            ) from exc

    def resolve_probe(
        self,
        permit: CircuitPermit,
        *,
        category: str,
        retry_after_seconds: float | None = None,
        now: datetime | None = None,
    ) -> None:
        """Close success/non-quota probes or renew quota probes token-safely."""
        if category not in {
            "success",
            "quota_exhausted",
            "transient",
            "terminal",
            "unknown",
        }:
            raise ValueError("Invalid provider probe result category")
        moment = _utc(now or datetime.now(timezone.utc))
        try:
            with self._sessions.begin() as session:
                for lease in permit.leases:
                    predicate = _lease_predicate(lease)
                    if category == "quota_exhausted":
                        state = session.scalar(
                            select(ProviderCircuitState)
                            .where(*predicate)
                            .with_for_update()
                        )
                        if state is None:
                            continue
                        failure_count = (
                            state.failure_count + 1
                            if state.failure_category == "quota_exhausted"
                            else 1
                        )
                        stage = min(failure_count, _MAX_BACKOFF_STAGE)
                        cooldown = min(
                            _INITIAL_COOLDOWN * (2 ** (stage - 1)), _MAX_COOLDOWN
                        )
                        retry_after = max(0.0, retry_after_seconds or 0.0)
                        state.failure_count = failure_count
                        state.failure_category = "quota_exhausted"
                        state.probe_at = max(
                            moment + cooldown,
                            moment
                            + timedelta(
                                seconds=min(retry_after, RETRY_AFTER_MAX_SECONDS)
                            ),
                        )
                        state.lease_token = None
                        state.lease_until = None
                        state.updated_at = moment
                    else:
                        session.execute(delete(ProviderCircuitState).where(*predicate))
        except SQLAlchemyError as exc:
            raise ProviderCircuitStoreError(
                "Could not resolve provider probe lease"
            ) from exc

    def release(self, permit: CircuitPermit, *, now: datetime | None = None) -> None:
        """Release unfinished probe leases without changing the cooldown state."""
        moment = _utc(now or datetime.now(timezone.utc))
        try:
            with self._sessions.begin() as session:
                for lease in permit.leases:
                    session.execute(
                        update(ProviderCircuitState)
                        .where(*_lease_predicate(lease))
                        .values(lease_token=None, lease_until=None, updated_at=moment)
                    )
        except SQLAlchemyError as exc:
            raise ProviderCircuitStoreError(
                "Could not release provider probe lease"
            ) from exc

    def snapshots(
        self, tenant_id: str, scopes: tuple[BreakerScope, ...]
    ) -> tuple[CircuitSnapshot, ...]:
        """Read active state only for the requested tenant-owned route scopes."""
        if not scopes:
            return ()
        try:
            with self._sessions() as session:
                states = session.scalars(
                    select(ProviderCircuitState).where(
                        ProviderCircuitState.tenant_id == tenant_id,
                        ProviderCircuitState.scope_fingerprint.in_(
                            {scope.fingerprint for scope in scopes}
                        ),
                    )
                ).all()
                scope_by_fingerprint = {scope.fingerprint: scope for scope in scopes}
                return tuple(
                    CircuitSnapshot(
                        scope_by_fingerprint[state.scope_fingerprint],
                        _utc(state.probe_at),
                        state.failure_count,
                        _utc(state.lease_until) if state.lease_until else None,
                        state.failure_category,
                    )
                    for state in states
                    if state.scope_fingerprint in scope_by_fingerprint
                )
        except SQLAlchemyError as exc:
            raise ProviderCircuitStoreError(
                "Could not read provider circuit state"
            ) from exc

    def scopes_for_profile(
        self,
        tenant_id: str,
        provider: str,
        profile_id: str,
        settings: Mapping[str, Any],
    ) -> tuple[BreakerScope, ...]:
        """Resolve profile and explicitly configured shared scopes consistently."""
        scopes = [self.scope(tenant_id, provider, "profile", profile_id)]
        if provider == "openai":
            for scope_type in ("organization", "project"):
                scope_id = settings.get(scope_type)
                if isinstance(scope_id, str) and scope_id.strip():
                    scopes.append(
                        self.scope(tenant_id, provider, scope_type, scope_id.strip())
                    )
        return tuple(scopes)

    def _locked_state(
        self, session: Session, scope: BreakerScope, now: datetime
    ) -> tuple[ProviderCircuitState, bool]:
        state = session.scalar(
            select(ProviderCircuitState)
            .where(
                ProviderCircuitState.tenant_id == scope.tenant_id,
                ProviderCircuitState.provider == scope.provider,
                ProviderCircuitState.scope_fingerprint == scope.fingerprint,
            )
            .with_for_update()
        )
        if state is not None:
            return state, False
        dialect = session.bind.dialect.name
        insert = (
            postgresql_insert(ProviderCircuitState)
            if dialect == "postgresql"
            else sqlite_insert(ProviderCircuitState) if dialect == "sqlite" else None
        )
        if insert is None:
            raise ProviderCircuitStoreError("Unsupported provider circuit database")
        inserted = session.execute(
            insert.values(
                tenant_id=scope.tenant_id,
                provider=scope.provider,
                scope_type=scope.scope_type,
                scope_fingerprint=scope.fingerprint,
                failure_category="quota_exhausted",
                failure_count=1,
                probe_at=now,
                updated_at=now,
            ).on_conflict_do_nothing(
                index_elements=[
                    "tenant_id",
                    "provider",
                    "scope_fingerprint",
                ]
            )
        )
        state = session.scalar(
            select(ProviderCircuitState)
            .where(
                ProviderCircuitState.tenant_id == scope.tenant_id,
                ProviderCircuitState.provider == scope.provider,
                ProviderCircuitState.scope_fingerprint == scope.fingerprint,
            )
            .with_for_update()
        )
        if state is None:
            raise ProviderCircuitStoreError("Could not create provider circuit state")
        return state, inserted.rowcount == 1


class ProviderCircuitStoreError(RuntimeError):
    """The persistent breaker state could not be safely read or mutated."""


def _identity(scope: BreakerScope) -> tuple[str, str, str]:
    return scope.provider, scope.scope_type, scope.fingerprint


def _identity_from_state(state: ProviderCircuitState) -> tuple[str, str, str]:
    return state.provider, state.scope_type, state.scope_fingerprint


def _lease_predicate(lease: ProbeLease):
    return (
        ProviderCircuitState.tenant_id == lease.scope.tenant_id,
        ProviderCircuitState.provider == lease.scope.provider,
        ProviderCircuitState.scope_type == lease.scope.scope_type,
        ProviderCircuitState.scope_fingerprint == lease.scope.fingerprint,
        ProviderCircuitState.lease_token == lease.token,
    )


def _utc(moment: datetime) -> datetime:
    """Normalize SQLite-naive and PostgreSQL-aware timestamps as UTC."""
    return (
        moment.replace(tzinfo=timezone.utc)
        if moment.tzinfo is None
        else moment.astimezone(timezone.utc)
    )
