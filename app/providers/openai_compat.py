"""OpenAI-compatible streaming forwarder for OpenAI and OpenRouter."""

from __future__ import annotations

import json
from typing import Any

import requests
from flask import Request, Response, stream_with_context

from app.exceptions import ServiceConfigurationError
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
    req: Request, snapshot: DatabaseTenantSnapshot
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
    upstream = requests.post(
        f"{origin}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        stream=True,
        timeout=600,
    )

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
