"""OpenAI-compatible streaming forwarder for OpenAI and OpenRouter."""

from __future__ import annotations

import json
from typing import Any

import requests
from flask import Request, Response, stream_with_context

from app.exceptions import ServiceConfigurationError
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
    raise ServiceConfigurationError(
        f"Unsupported OpenAI-compatible provider {provider!r}"
    )


def forward_openai_compatible(
    req: Request, snapshot: DatabaseTenantSnapshot, *, routed: bool = False
) -> Response:
    """Forward a Cursor request to OpenAI or OpenRouter Chat Completions."""
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
    if inbound_model == snapshot.custom_model_id or not inbound_model:
        payload = {**payload, "model": snapshot.default_model}
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
    origin = openai_compatible_base_url(snapshot.provider or "")
    headers = {
        "Authorization": f"Bearer {snapshot.inference_secret}",
        "Content-Type": "application/json",
    }
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
            timeout=(10.0, ROUTED_READ_TIMEOUT_SECONDS) if routed else 600,
        )
    except requests.RequestException as exc:
        if routed:
            raise transport_failure(exc) from exc
        raise
    if routed:
        prepared = prepare_upstream(upstream)
        response = Response(
            stream_with_context(chat_stream(prepared, snapshot.custom_model_id)),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        response.call_on_close(prepared.close)
        return response

    def generate() -> Any:
        try:
            for chunk in upstream.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return Response(
        stream_with_context(generate()),
        status=upstream.status_code,
        content_type=upstream.headers.get("Content-Type", "text/event-stream"),
    )
