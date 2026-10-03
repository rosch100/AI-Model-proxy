"""Sequential tenant inference routing with explicit logical model identity."""

from __future__ import annotations

from flask import Request, Response, current_app

from app.azure.adapter import AzureAdapter
from app.exceptions import ServiceConfigurationError
from app.providers.failover_upstream import UpstreamError
from app.providers.openai_compat import forward_openai_compatible
from app.tenants import DatabaseTenantRoutingSnapshot, DatabaseTenantSnapshot


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
    return profiles[:1] if azure_only else profiles


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
    if (
        not isinstance(payload, dict)
        or payload.get("model") != snapshot.custom_model_id
    ):
        raise ServiceConfigurationError(
            "Request model must match the tenant's Cursor model ID."
        )
    if azure_only:
        return AzureAdapter().forward(req, profiles[0])
    for profile in profiles:
        try:
            if profile.provider == "azure":
                return AzureAdapter().forward_attempt(req, profile)
            if profile.provider in {"openai", "openrouter"}:
                return forward_openai_compatible(req, profile, routed=True)
            raise ServiceConfigurationError(
                f"Unsupported provider {profile.provider!r}."
            )
        except UpstreamError as exc:
            current_app.logger.warning(
                "Provider attempt failed: tenant=%s profile=%s provider=%s status=%s retryable=%s",
                snapshot.id,
                profile.profile_id,
                profile.provider,
                exc.status,
                exc.retryable,
            )
            if not exc.retryable or profile is profiles[-1]:
                return exc.response()
    raise AssertionError("A nonempty route must return a response or failure")
