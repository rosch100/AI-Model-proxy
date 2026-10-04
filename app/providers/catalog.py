"""Catalog refresh against supported inference-provider HTTP APIs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import requests

from app.models import SUPPORTED_MODELS


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
    if provider == "deepseek":
        return _refresh_deepseek(inference_secret)
    raise CatalogRefreshError(f"Unsupported provider {provider!r}")


def selectable_catalog_models(
    provider: str, entries: Sequence[tuple[str, str | None]]
) -> list[tuple[str, str | None]]:
    """Return catalog rows that the proxy can expose for query routing."""
    if provider == "azure":
        supported = set(SUPPORTED_MODELS)
        selected: list[tuple[str, str | None]] = []
        seen: set[str] = set()
        for model_id, deployment_id in entries:
            cursor_id = None
            for candidate in (model_id, deployment_id):
                if isinstance(candidate, str) and candidate in supported:
                    cursor_id = candidate
                    break
            if cursor_id is None or cursor_id in seen:
                continue
            seen.add(cursor_id)
            selected.append((cursor_id, deployment_id or cursor_id))
        return selected
    return [(model_id, deployment_id) for model_id, deployment_id in entries]


def azure_deployments_from_catalog(
    entries: Iterable[tuple[str, str | None]],
) -> dict[str, str]:
    """Build Cursor-model → Azure-deployment map from selectable catalog rows."""
    return {
        model_id: (deployment_id or model_id)
        for model_id, deployment_id in selectable_catalog_models("azure", list(entries))
    }


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
    try:
        response = requests.get(url, headers={"api-key": api_key}, timeout=30)
    except requests.RequestException as exc:
        raise CatalogRefreshError("Azure catalog request failed.") from exc
    if response.status_code >= 400:
        raise CatalogRefreshError(f"Azure catalog HTTP {response.status_code}")
    try:
        payload = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError) as exc:
        raise CatalogRefreshError("Azure catalog payload is invalid.") from exc
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
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise CatalogRefreshError("OpenAI catalog request failed.") from exc
    if response.status_code >= 400:
        raise CatalogRefreshError(f"OpenAI catalog HTTP {response.status_code}")
    try:
        payload = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError) as exc:
        raise CatalogRefreshError("OpenAI catalog payload is invalid.") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise CatalogRefreshError("OpenAI catalog payload is invalid.")
    return [
        (item["id"], None)
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def _refresh_deepseek(api_key: str) -> list[tuple[str, str | None]]:
    """Load DeepSeek models from its OpenAI-compatible model-list endpoint."""
    try:
        response = requests.get(
            "https://api.deepseek.com/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise CatalogRefreshError("DeepSeek catalog request failed.") from exc
    if response.status_code >= 400:
        raise CatalogRefreshError(f"DeepSeek catalog HTTP {response.status_code}")
    try:
        payload = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError) as exc:
        raise CatalogRefreshError("DeepSeek catalog payload is invalid.") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise CatalogRefreshError("DeepSeek catalog payload is invalid.")
    models = [
        item["id"]
        for item in data
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item["id"].strip()
    ]
    if not models:
        raise CatalogRefreshError("DeepSeek catalog contains no models.")
    return [(model_id, None) for model_id in models]


def _refresh_openrouter(api_key: str) -> list[tuple[str, str | None]]:
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise CatalogRefreshError("OpenRouter catalog request failed.") from exc
    if response.status_code >= 400:
        raise CatalogRefreshError(f"OpenRouter catalog HTTP {response.status_code}")
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise CatalogRefreshError("OpenRouter catalog payload is invalid.") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise CatalogRefreshError("OpenRouter catalog payload is invalid.")
    models = [
        item["id"]
        for item in data
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item["id"].strip()
    ]
    if not models:
        raise CatalogRefreshError("OpenRouter catalog contains no models.")
    return [(model_id, None) for model_id in models]
