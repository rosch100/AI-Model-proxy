"""OpenAI-compatible streaming forwarder for tenant Chat Completions providers."""

from __future__ import annotations

import json

import requests
from flask import Request, Response, stream_with_context

from app.exceptions import ServiceConfigurationError
from app.providers.circuit_breaker import ProviderCircuitAttempt
from app.providers.failover_upstream import (
    ROUTED_READ_TIMEOUT_SECONDS,
    chat_stream,
    prepare_upstream,
    transport_failure,
)
from app.tenants import DatabaseTenantSnapshot


def openai_compatible_base_url(provider: str) -> str:
    """Return the public Chat Completions origin for a supported provider."""
    if provider == "openai":
        return "https://api.openai.com/v1"
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "deepseek":
        return "https://api.deepseek.com"
    raise ServiceConfigurationError(
        f"Unsupported OpenAI-compatible provider {provider!r}"
    )


def forward_openai_compatible(
    req: Request,
    snapshot: DatabaseTenantSnapshot,
    *,
    target_model: str,
    attempt_id: int | None = None,
    circuit_attempt: ProviderCircuitAttempt | None = None,
) -> Response:
    """Forward a Cursor request to an OpenAI-compatible Chat Completions API."""
    if snapshot.profile_id is None:
        raise ServiceConfigurationError(
            "No active provider profile is configured for this tenant."
        )
    if snapshot.profile_deleted:
        raise ServiceConfigurationError("The active provider profile has been removed.")
    if snapshot.inference_secret is None or snapshot.default_model is None:
        raise ServiceConfigurationError(
            "The active provider profile is missing credentials or a default model."
        )
    payload = req.get_json(silent=True, force=False) or {}
    if not isinstance(payload, dict):
        payload = {}
    inbound_model = payload.get("model")
    payload = {**payload, "model": target_model}
    tools = payload.get("tools")
    if (
        snapshot.provider == "openai"
        and payload.get("model") == "gpt-6-luna"
        and isinstance(tools, list)
        and any(
            isinstance(tool, dict) and tool.get("type") == "function" for tool in tools
        )
    ):
        payload["reasoning_effort"] = "none"
    payload["stream"] = True
    stream_options = payload.get("stream_options")
    if not isinstance(stream_options, dict):
        stream_options = {}
    payload["stream_options"] = {**stream_options, "include_usage": True}
    origin = openai_compatible_base_url(snapshot.provider or "")
    headers = {
        "Authorization": f"Bearer {snapshot.inference_secret}",
        "Content-Type": "application/json",
    }
    if snapshot.provider == "openai":
        organization = snapshot.provider_settings.get("organization")
        if isinstance(organization, str) and organization:
            headers["OpenAI-Organization"] = organization
        project = snapshot.provider_settings.get("project")
        if isinstance(project, str) and project:
            headers["OpenAI-Project"] = project
    try:
        upstream = requests.post(
            f"{origin}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            stream=True,
            timeout=(10.0, ROUTED_READ_TIMEOUT_SECONDS),
        )
    except requests.RequestException as exc:
        raise transport_failure(exc) from exc
    prepared = prepare_upstream(
        upstream,
        provider=snapshot.provider,
        settings=snapshot.provider_settings,
    )
    if circuit_attempt is not None:
        circuit_attempt.preflight_succeeded()
    response = Response(
        stream_with_context(
            chat_stream(
                prepared,
                snapshot.custom_model_id,
                activity_tenant_id=snapshot.id,
                activity_provider=snapshot.provider,
                activity_profile_id=snapshot.profile_id,
                inbound_model=inbound_model,
                routed_model=target_model,
                attempt_id=attempt_id,
                provider=snapshot.provider,
                settings=snapshot.provider_settings,
                circuit_attempt=circuit_attempt,
            )
        ),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    response.call_on_close(prepared.close)
    return response
