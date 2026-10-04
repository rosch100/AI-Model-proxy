"""Static Azure tenant configuration and API-key digest helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from .exceptions import ServiceConfigurationError
from .models import parse_model_deployments

AUTH_MODE_SINGLE = "single"
AUTH_MODE_TENANT = "tenant"
TENANT_CONFIG_ENVIRONMENT = "environment"
TENANT_CONFIG_DATABASE = "database"
VALID_AUTH_MODES = frozenset({AUTH_MODE_SINGLE, AUTH_MODE_TENANT})

_PLACEHOLDER_AZURE_API_KEYS = frozenset({"change_me", "change-me"})
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class TenantConfig:
    """Resolved Azure configuration for one environment-configured tenant."""

    id: str
    api_key_hash: str
    azure_base_url: str
    azure_api_key: str
    azure_model_deployments: Mapping[str, str]
    azure_default_model: str | None = None

    @property
    def azure_responses_api_url(self) -> str:
        """Return the Azure Responses API URL for this tenant."""
        return f"{self.azure_base_url}/openai/v1/responses"


@dataclass(frozen=True)
class DatabaseTenantSnapshot:
    """Consistent persisted tenant and active provider configuration for a request."""

    id: str
    api_key_hash: str
    custom_model_id: str
    provider: str | None
    provider_settings: Mapping[str, Any]
    inference_secret: str | None = field(repr=False)
    default_model: str | None
    profile_id: str | None = None
    profile_name: str | None = None
    history_generation: int | None = None
    profile_deleted: bool = False
    catalog_model_ids: tuple[str, ...] = ()

    @property
    def azure_base_url(self) -> str | None:
        """Return the configured Azure base URL for this active profile."""
        base_url = self.provider_settings.get("base_url")
        return base_url if isinstance(base_url, str) else None

    @property
    def azure_api_key(self) -> str | None:
        """Return the decrypted Azure inference credential for this snapshot."""
        return self.inference_secret

    @property
    def azure_model_deployments(self) -> Mapping[str, str]:
        """Return the configured model-to-deployment map for this profile."""
        deployments = self.provider_settings.get("model_deployments", {})
        if not isinstance(deployments, dict) or any(
            not isinstance(model, str) or not isinstance(deployment, str)
            for model, deployment in deployments.items()
        ):
            raise ServiceConfigurationError(
                "The active Azure profile has invalid model deployments."
            )
        return deployments

    @property
    def azure_responses_api_url(self) -> str:
        """Return the Azure Responses API URL for this active profile."""
        base_url = self.azure_base_url
        if base_url is None:
            raise ServiceConfigurationError(
                "The active tenant profile has no Azure base URL."
            )
        return f"{base_url.rstrip('/')}/openai/v1/responses"


@dataclass(frozen=True)
class DatabaseTenantRoutingSnapshot:
    """Authenticated tenant identity and its validated, ordered provider route."""

    id: str
    api_key_hash: str
    custom_model_id: str
    profiles: tuple[DatabaseTenantSnapshot, ...]


def hash_api_key(api_key: str) -> str:
    """Return the SHA-256 hex digest of an API key."""
    # API keys are generated as 256-bit random bearer tokens, not passwords.
    # codeql[py/weak-sensitive-data-hashing]
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def parse_auth_mode(raw_mode: str | None) -> str:
    """Parse and validate AUTH_MODE."""
    mode = (raw_mode or AUTH_MODE_SINGLE).strip().casefold()
    if mode not in VALID_AUTH_MODES:
        raise ServiceConfigurationError(
            "AUTH_MODE must be either 'single' or 'tenant'.\n" f"Got: {raw_mode!r}"
        )
    return mode


def parse_tenants(raw_tenants: str | None) -> tuple[TenantConfig, ...]:
    """Parse TENANTS JSON and resolve Azure API keys from named env vars."""
    if not raw_tenants or not raw_tenants.strip():
        return ()

    try:
        parsed = json.loads(raw_tenants)
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(
            "TENANTS must be valid JSON: a list of tenant objects."
        ) from exc

    if not isinstance(parsed, list):
        raise ServiceConfigurationError(
            "TENANTS must be a JSON list of tenant objects."
        )

    tenants = tuple(
        _parse_tenant_entry(entry, index) for index, entry in enumerate(parsed)
    )
    _validate_tenant_uniqueness(tenants)
    return tenants


def resolve_tenant_for_api_key(
    api_key: str, tenants: tuple[TenantConfig, ...]
) -> TenantConfig | None:
    """Match an API key against tenant digests with constant-time compares."""
    digest = hash_api_key(api_key)
    matched: TenantConfig | None = None
    for tenant in tenants:
        if hmac.compare_digest(digest, tenant.api_key_hash):
            matched = tenant
    return matched


def validate_tenant_startup(
    *,
    auth_mode: str,
    tenants: tuple[TenantConfig, ...],
) -> None:
    """Validate auth-mode and tenant configuration at application start."""
    mode = parse_auth_mode(auth_mode)
    if mode == AUTH_MODE_SINGLE:
        if tenants:
            raise ServiceConfigurationError(
                "AUTH_MODE=single must not define TENANTS. "
                "Use AUTH_MODE=tenant for per-tenant API keys and Azure credentials."
            )
        return

    if not tenants:
        raise ServiceConfigurationError(
            "AUTH_MODE=tenant requires a non-empty TENANTS configuration."
        )
    _validate_tenant_uniqueness(tenants)


def _parse_tenant_entry(entry: Any, index: int) -> TenantConfig:
    if not isinstance(entry, dict):
        raise ServiceConfigurationError(f"TENANTS[{index}] must be a JSON object.")

    tenant_id = _required_nonempty_str(entry, "id", index)
    api_key_hash = _required_nonempty_str(entry, "api_key_hash", index).casefold()
    if len(api_key_hash) != _SHA256_HEX_LENGTH or any(
        char not in "0123456789abcdef" for char in api_key_hash
    ):
        raise ServiceConfigurationError(
            f"TENANTS[{index}].api_key_hash must be a SHA-256 hex digest."
        )

    azure_base_url = _required_nonempty_str(entry, "azure_base_url", index).rstrip("/")
    azure_api_key_env = _required_nonempty_str(entry, "azure_api_key_env", index)
    azure_api_key = os.environ.get(azure_api_key_env)
    if (
        not isinstance(azure_api_key, str)
        or not azure_api_key.strip()
        or azure_api_key.strip().casefold() in _PLACEHOLDER_AZURE_API_KEYS
    ):
        raise ServiceConfigurationError(
            f"Environment variable {azure_api_key_env!r} referenced by "
            f"TENANTS[{index}] must be set to a non-empty Azure API key, "
            "not an example value."
        )

    deployments_raw = entry.get("azure_model_deployments")
    if not isinstance(deployments_raw, dict) or not deployments_raw:
        raise ServiceConfigurationError(
            f"TENANTS[{index}].azure_model_deployments must be a non-empty JSON object "
            "mapping Cursor model ids to Azure deployment names."
        )

    try:
        deployments = parse_model_deployments(json.dumps(deployments_raw))
    except ServiceConfigurationError as exc:
        raise ServiceConfigurationError(
            f"TENANTS[{index}].azure_model_deployments is invalid: {exc.args[0]}"
        ) from exc

    default_model = entry.get("azure_default_model")
    if default_model is not None and (
        not isinstance(default_model, str) or default_model not in deployments
    ):
        raise ServiceConfigurationError(
            f"TENANTS[{index}].azure_default_model must name a model in "
            "azure_model_deployments."
        )

    return TenantConfig(
        id=tenant_id,
        api_key_hash=api_key_hash,
        azure_base_url=azure_base_url,
        azure_api_key=azure_api_key.strip(),
        azure_model_deployments=dict(deployments),
        azure_default_model=default_model,
    )


def _required_nonempty_str(entry: dict[str, Any], field: str, index: int) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ServiceConfigurationError(
            f"TENANTS[{index}].{field} must be a non-empty string."
        )
    return value.strip()


def _validate_tenant_uniqueness(tenants: tuple[TenantConfig, ...]) -> None:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for tenant in tenants:
        if tenant.id in seen_ids:
            raise ServiceConfigurationError(
                f"Duplicate tenant id in TENANTS: {tenant.id!r}"
            )
        if tenant.api_key_hash in seen_hashes:
            raise ServiceConfigurationError(
                "Duplicate api_key_hash in TENANTS configuration."
            )
        seen_ids.add(tenant.id)
        seen_hashes.add(tenant.api_key_hash)
