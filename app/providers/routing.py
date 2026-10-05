"""Sequential tenant inference routing with explicit logical model identity."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime

from flask import Request, Response, current_app, g, jsonify

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

ProviderForwarder = Callable[
    [Request, DatabaseTenantSnapshot, str, int | None, ProviderCircuitAttempt | None],
    Response,
]


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
    """Apply operator provider switches without ever borrowing global credentials."""
    profiles = tuple(
        profile
        for profile in snapshot.profiles
        if (not azure_only or profile.provider == "azure")
        and (
            profile.provider != "azure" or current_app.config.get("ENABLE_AZURE", False)
        )
    )
    return profiles


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


def forward_tenant_route(
    req: Request, snapshot: DatabaseTenantRoutingSnapshot, *, azure_only: bool = False
) -> Response:
    """Try each configured candidate at most once, before committing headers."""
    profiles = routed_profiles(snapshot, azure_only=azure_only)
    if not profiles:
        message = "No active provider profile is configured for this tenant."
        if azure_only:
            message = "The active tenant provider is not available on the Azure route."
        raise ServiceConfigurationError(message)
    payload = req.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ServiceConfigurationError(
            "Request model must match the tenant's Cursor model ID."
        )
    inbound_model = payload.get("model")
    route_targets = _route_targets(profiles, inbound_model, snapshot.custom_model_id)
    database = current_app.extensions.get("database")
    breaker_store = (
        ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
        if isinstance(database, Database)
        else None
    )
    blocked_until = []
    circuit_state_unavailable = False
    attempts_started = 0
    last_retryable_error: UpstreamError | None = None
    transient_failure_seen = False
    transient_retry_after: float | None = None
    for profile, target_model in route_targets:
        permit = CircuitPermit(True)
        if breaker_store is not None:
            scopes = breaker_store.scopes_for_profile(
                snapshot.id,
                profile.provider,
                profile.profile_id,
                profile.provider_settings,
            )
            try:
                permit = breaker_store.acquire(scopes)
            except ProviderCircuitStoreError:
                circuit_state_unavailable = True
                current_app.logger.exception(
                    "Provider circuit state unavailable; skipping profile=%s",
                    profile.profile_id,
                )
                continue
            if not permit.allowed:
                blocked_until.append(permit.retry_at)
                continue
        attempts_started += 1
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
        except UpstreamError as exc:
            circuit_update_failed = False
            if circuit_attempt is not None:
                try:
                    circuit_attempt.failed(exc.classification)
                except ProviderCircuitStoreError:
                    circuit_update_failed = True
                    circuit_state_unavailable = True
                    current_app.logger.exception(
                        "Could not resolve provider circuit state for profile=%s",
                        profile.profile_id,
                    )
            complete_provider_attempt(
                attempt_id,
                outcome="failure",
                status_code=exc.status,
                failure_details=_provider_failure_details(exc),
            )
            last_retryable_error = exc
            if exc.classification.category == "transient":
                transient_failure_seen = True
                retry_after = exc.classification.retry_after_seconds
                if retry_after is not None:
                    transient_retry_after = max(transient_retry_after or 0, retry_after)
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
                exc.status,
                exc.retryable,
                exc.code,
                exc.classification.category,
                exc.provider_error_code or "unknown",
                exc.provider_error_type or "unknown",
                exc.provider_error_param or "unknown",
                exc.provider_limit_source or "unknown",
                exc.provider_request_id or "unknown",
                json.dumps(
                    exc.provider_diagnostics, sort_keys=True, ensure_ascii=False
                ),
            )
            if circuit_update_failed:
                continue
            if not exc.retryable:
                return exc.response()
        except Exception:  # noqa: BLE001 - release resources before re-raising
            if circuit_attempt is not None:
                try:
                    circuit_attempt.release()
                except ProviderCircuitStoreError:
                    current_app.logger.exception(
                        "Could not release provider circuit lease after unexpected failure for profile=%s",
                        profile.profile_id,
                    )
            complete_provider_attempt(attempt_id, outcome="failure", status_code=None)
            raise
    if circuit_state_unavailable:
        return _service_unavailable(
            "provider_circuit_unavailable",
            "Provider availability could not be verified; retry the request shortly.",
            "30",
        )
    if transient_failure_seen:
        retry_after_seconds = (
            max(1, math.ceil(transient_retry_after))
            if transient_retry_after is not None
            else 30
        )
        cooldown_seconds = (
            int(retry_after_header(retry_at))
            for retry_at in blocked_until
            if retry_at is not None
        )
        retry_after = str(max(retry_after_seconds, max(cooldown_seconds, default=0)))
        return _service_unavailable(
            "provider_quota_unavailable",
            "All configured providers are temporarily unavailable.",
            retry_after,
        )
    if last_retryable_error is not None:
        return last_retryable_error.response()
    if blocked_until:
        retry_at = min(value for value in blocked_until if value is not None)
        return _quota_unavailable(retry_at)
    if attempts_started:
        raise AssertionError(
            "A started provider route must return a response or failure"
        )
    raise ServiceConfigurationError(
        "No provider circuit state could be verified; no upstream request was sent."
    )
