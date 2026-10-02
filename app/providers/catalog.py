"""Catalog refresh against Azure, OpenAI, and OpenRouter HTTP APIs."""

from __future__ import annotations

import requests


class CatalogRefreshError(Exception):
    """Raised when a provider catalog cannot be retrieved."""


# Azure OpenAI data-plane listing of deployments is only served on the
# legacy deployments API version. Newer api-version values return HTTP 404.
AZURE_DEPLOYMENTS_API_VERSION = "2022-12-01"


def refresh_provider_catalog(
    provider: str, settings: dict[str, object], inference_secret: str
) -> list[tuple[str, str | None]]:
    """Return provider model ids and optional Azure deployment names."""
    if provider == "azure":
        return _refresh_azure(settings, inference_secret)
    if provider == "openai":
        return _refresh_openai(inference_secret)
    if provider == "openrouter":
        return _refresh_openrouter(inference_secret)
    raise CatalogRefreshError(f"Unsupported provider {provider!r}")


def _refresh_azure(
    settings: dict[str, object], api_key: str
) -> list[tuple[str, str | None]]:
    base_url = settings.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise CatalogRefreshError("Azure base URL is missing.")
    url = (
        f"{base_url.rstrip('/')}/openai/deployments"
        f"?api-version={AZURE_DEPLOYMENTS_API_VERSION}"
    )
    response = requests.get(url, headers={"api-key": api_key}, timeout=30)
    if response.status_code >= 400:
        raise CatalogRefreshError(f"Azure catalog HTTP {response.status_code}")
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise CatalogRefreshError("Azure catalog payload is invalid.")
    entries: list[tuple[str, str | None]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        deployment_id = item.get("id")
        model = item.get("model")
        if isinstance(deployment_id, str):
            model_id = model if isinstance(model, str) else deployment_id
            entries.append((model_id, deployment_id))
    return entries


def _refresh_openai(api_key: str) -> list[tuple[str, str | None]]:
    response = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise CatalogRefreshError(f"OpenAI catalog HTTP {response.status_code}")
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise CatalogRefreshError("OpenAI catalog payload is invalid.")
    return [
        (item["id"], None)
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def _refresh_openrouter(api_key: str) -> list[tuple[str, str | None]]:
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise CatalogRefreshError(f"OpenRouter catalog HTTP {response.status_code}")
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise CatalogRefreshError("OpenRouter catalog payload is invalid.")
    return [
        (item["id"], None)
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
