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
_API_KEY_HASH_HEX_LENGTH = 64
# Cost parameters for newly stored API-key digests. N=2**14 is the lowest
# power of two that is still expensive relative to a single SHA-256.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
# Deterministic salt so a digest remains a lookup key. Cursor API keys are
# 256-bit bearer tokens, not low-entropy passwords.
_SCRYPT_SALT = b"ai-model-proxy.api-key.v1"


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
    """Return the scrypt hex digest stored for a newly issued API key."""
    digest = hashlib.scrypt(
        api_key.encode("utf-8"),
        salt=_SCRYPT_SALT,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return digest.hex()


def legacy_sha256_api_key_hash(api_key: str) -> str:
    """Return the historical SHA-256 hex digest of an API key.

    Rows and TENANTS entries created before scrypt still store this form.
    Verification accepts it; database lookups upgrade it to :func:`hash_api_key`
    when the session can be written. Environment-configured tenants are not
    rewritten because that configuration is not persisted by the process.
    """
    return _sha256_hex(api_key.encode("utf-8"))


def _rotr(value: int, bits: int) -> int:
    """Rotate a 32-bit word right."""
    return ((value >> bits) | (value << (32 - bits))) & 0xFFFFFFFF


# SHA-256 round constants (FIPS 180-4). Used only to recognize digests that
# were stored before scrypt; new digests go through hash_api_key.
_SHA256_K = (
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
)


def _sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data`` (FIPS 180-4)."""
    bit_length = len(data) * 8
    padded = data + b"\x80"
    padded += b"\x00" * ((56 - len(padded) % 64) % 64)
    padded += bit_length.to_bytes(8, "big")

    state = [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ]
    for offset in range(0, len(padded), 64):
        block = padded[offset : offset + 64]
        words = [
            int.from_bytes(block[index : index + 4], "big") for index in range(0, 64, 4)
        ]
        for index in range(16, 64):
            first = words[index - 15]
            second = words[index - 2]
            small0 = _rotr(first, 7) ^ _rotr(first, 18) ^ (first >> 3)
            small1 = _rotr(second, 17) ^ _rotr(second, 19) ^ (second >> 10)
            words.append(
                (words[index - 16] + small0 + words[index - 7] + small1) & 0xFFFFFFFF
            )
        a, b, c, d, e, f, g, h = state
        for index, word in enumerate(words):
            sigma1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            choose = (e & f) ^ ((~e) & g)
            temp1 = (h + sigma1 + choose + _SHA256_K[index] + word) & 0xFFFFFFFF
            sigma0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (sigma0 + majority) & 0xFFFFFFFF
            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF
        state = [
            (left + right) & 0xFFFFFFFF
            for left, right in zip(state, (a, b, c, d, e, f, g, h))
        ]
    return "".join(f"{word:08x}" for word in state)


def api_key_lookup_digests(api_key: str) -> tuple[str, str]:
    """Return ``(scrypt, legacy SHA-256)`` digests for one presented API key."""
    return hash_api_key(api_key), legacy_sha256_api_key_hash(api_key)


def _digest_matches(candidate: str, stored_hash: str) -> bool:
    folded = stored_hash.casefold()
    if len(candidate) != len(folded):
        return False
    return hmac.compare_digest(candidate, folded)


def matching_api_key_hash(stored_hash: str, current: str, legacy: str) -> str | None:
    """Return the scrypt digest when ``stored_hash`` matches either algorithm."""
    if _digest_matches(current, stored_hash) or _digest_matches(legacy, stored_hash):
        return current
    return None


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
    current, legacy = api_key_lookup_digests(api_key)
    matched: TenantConfig | None = None
    for tenant in tenants:
        if matching_api_key_hash(tenant.api_key_hash, current, legacy) is not None:
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
    if len(api_key_hash) != _API_KEY_HASH_HEX_LENGTH or any(
        char not in "0123456789abcdef" for char in api_key_hash
    ):
        raise ServiceConfigurationError(
            f"TENANTS[{index}].api_key_hash must be a 64-character hex digest."
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
