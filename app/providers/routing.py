"""Sequential tenant inference routing with explicit logical model identity."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
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
from app.providers.circuit_breaker import (
    ProviderCircuitAttempt,
    retry_after_header,
)
from app.providers.failover_upstream import UpstreamError
from app.providers.model_ids import account_model_id, qualified_model_id
from app.providers.openai_compat import forward_openai_compatible
from app.tenants import DatabaseTenantRoutingSnapshot, DatabaseTenantSnapshot

MAX_TRANSIENT_RETRY_WAIT_SECONDS = 30.0
DEFAULT_TRANSIENT_RETRY_DELAY_SECONDS = 1.0

ProviderForwarder = Callable[
    [Request, DatabaseTenantSnapshot, str, int | None, ProviderCircuitAttempt | None],
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
    attempts_started: int = 0

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


def _forward_azure(
    req: Request,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    attempt_id: int | None,
    circuit_attempt: ProviderCircuitAttempt | None,
) -> Response:
    """Make one Azure attempt for the centrally selected provider model."""
    return AzureAdapter().forward_attempt(
        req,
        profile,
        target_model=target_model,
        attempt_id=attempt_id,
        circuit_attempt=circuit_attempt,
    )


def _forward_openai_compatible(
    req: Request,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    attempt_id: int | None,
    circuit_attempt: ProviderCircuitAttempt | None,
) -> Response:
    """Make one OpenAI-compatible attempt for the centrally selected model."""
    return forward_openai_compatible(
        req,
        profile,
        target_model=target_model,
        attempt_id=attempt_id,
        circuit_attempt=circuit_attempt,
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
    """Apply provider switches without borrowing global credentials."""
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
) -> Response:
    """Dispatch one profile to its registered protocol adapter."""
    if profile.provider is None:
        raise ServiceConfigurationError("The provider profile has no provider.")
    forwarder = _PROVIDER_FORWARDERS.get(profile.provider)
    if forwarder is None:
        raise ServiceConfigurationError(f"Unsupported provider {profile.provider!r}.")
    return forwarder(req, profile, target_model, attempt_id, circuit_attempt)


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


def _schedule_transient_retry(
    error: UpstreamError,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    is_retry: bool,
    retry_targets: list,
) -> None:
    """Queue one safe retry for a transient 402/429 within the request budget."""
    retry_after = error.classification.retry_after_seconds
    if (
        error.classification.category != "transient"
        or is_retry
        or error.status not in {402, 429}
        or retry_after is not None
        and retry_after > MAX_TRANSIENT_RETRY_WAIT_SECONDS
    ):
        return
    delay = (
        retry_after
        if retry_after is not None
        else DEFAULT_TRANSIENT_RETRY_DELAY_SECONDS
    )
    retry_targets.append((profile, target_model, True, monotonic() + max(delay, 0.1)))


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
        "provider_request_id=%s provider_diagnostics=%s",
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
) -> Response | None:
    """Persist one failed attempt and return only terminal provider errors."""
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
            return None
    complete_provider_attempt(
        attempt_id,
        outcome="failure",
        status_code=error.status,
        failure_details=_provider_failure_details(error),
    )
    state.record_error(error)
    _schedule_transient_retry(error, profile, target_model, is_retry, retry_targets)
    _log_provider_failure(snapshot, profile, error)
    return error.response() if not error.retryable else None


def _attempt_route_target(
    req: Request,
    snapshot: DatabaseTenantRoutingSnapshot,
    inbound_model: object,
    profile: DatabaseTenantSnapshot,
    target_model: str,
    is_retry: bool,
    breaker_store: ProviderCircuitBreakerStore | None,
    state: RouteAttemptState,
    retry_targets: list,
    deadline: float,
) -> Response | None:
    """Run one upstream attempt, persisting failures and releasing its lease."""
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
        return None
    state.attempts_started += 1
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
    attempt_id = start_provider_attempt(
        tenant_id=snapshot.id,
        provider=profile.provider,
        profile_id=profile.profile_id,
        inbound_model=inbound_model,
        routed_model=target_model,
    )
    try:
        response = _forward_profile(
            req, profile, target_model, attempt_id, circuit_attempt
        )
        if circuit_attempt is not None:
            response.call_on_close(circuit_attempt.release)
        return response
    except ServiceConfigurationError:
        if circuit_attempt is not None:
            circuit_attempt.release()
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
        )
    except Exception:  # noqa: BLE001 - release resources before re-raising
        if circuit_attempt is not None:
            try:
                circuit_attempt.release()
            except ProviderCircuitStoreError:
                current_app.logger.exception(
                    "Could not release provider circuit lease after unexpected failure "
                    "for profile=%s",
                    profile.profile_id,
                )
        complete_provider_attempt(attempt_id, outcome="failure", status_code=None)
        raise
    return None


def _final_route_failure(state: RouteAttemptState) -> Response:
    """Choose the response after every available route and retry was exhausted."""
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
        cooldown_seconds = (
            int(retry_after_header(retry_at))
            for retry_at in state.blocked_until
            if retry_at is not None
        )
        retry_after = str(max(retry_after_seconds, max(cooldown_seconds, default=0)))
        return _service_unavailable(
            "provider_temporarily_unavailable",
            "Providers are temporarily rate limited or unavailable.",
            retry_after,
        )
    if state.last_retryable_error is not None:
        return state.last_retryable_error.response()
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
    """Try each configured candidate, then eligible bounded transient retries."""
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
    attempts = deque((profile, model, False, 0.0) for profile, model in route_targets)
    database = current_app.extensions.get("database")
    breaker_store = (
        ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
        if isinstance(database, Database)
        else None
    )
    state = RouteAttemptState(blocked_until=[])
    retry_targets = []
    deadline = monotonic() + MAX_TRANSIENT_RETRY_WAIT_SECONDS
    while attempts or retry_targets:
        if not attempts:
            attempts.extend(sorted(retry_targets, key=lambda target: target[3]))
            retry_targets.clear()
        profile, target_model, is_retry, _ = attempts.popleft()
        if is_retry and not _retry_route_target(
            (profile, target_model, is_retry, _), deadline
        ):
            continue
        response = _attempt_route_target(
            req,
            snapshot,
            inbound_model,
            profile,
            target_model,
            is_retry,
            breaker_store,
            state,
            retry_targets,
            deadline,
        )
        if response is not None:
            return response
    return _final_route_failure(state)
