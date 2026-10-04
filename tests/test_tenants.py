"""Tests for static tenant configuration parsing and startup validation."""

import json

import pytest

from app import create_app
from app.exceptions import ServiceConfigurationError
from app.tenants import (
    TenantConfig,
    hash_api_key,
    legacy_sha256_api_key_hash,
    parse_auth_mode,
    parse_tenants,
    resolve_tenant_for_api_key,
    validate_tenant_startup,
)


def test_parse_auth_mode_defaults_to_single():
    """AUTH_MODE defaults to single-tenant compatibility mode."""
    assert parse_auth_mode(None) == "single"
    assert parse_auth_mode("SINGLE") == "single"
    assert parse_auth_mode("tenant") == "tenant"


def test_parse_auth_mode_rejects_unknown_values():
    """Reject unsupported AUTH_MODE values at parse time."""
    with pytest.raises(ServiceConfigurationError, match="AUTH_MODE"):
        parse_auth_mode("mixed")


def test_parse_tenants_resolves_azure_api_key_from_env(monkeypatch):
    """Resolve Azure credentials from named env vars, not from TENANTS JSON."""
    monkeypatch.setenv("TENANT_ACME_AZURE_API_KEY", "acme-azure-secret")
    api_key = "tenant-acme-cleartext-key"
    raw = json.dumps(
        [
            {
                "id": "acme",
                "api_key_hash": hash_api_key(api_key),
                "azure_base_url": "https://acme.openai.azure.com/",
                "azure_api_key_env": "TENANT_ACME_AZURE_API_KEY",
                "azure_model_deployments": {"gpt-5.4": "acme-gpt54"},
            }
        ]
    )

    tenants = parse_tenants(raw)

    assert len(tenants) == 1
    assert tenants[0].id == "acme"
    assert tenants[0].azure_base_url == "https://acme.openai.azure.com"
    assert tenants[0].azure_api_key == "acme-azure-secret"
    assert tenants[0].azure_model_deployments == {"gpt-5.4": "acme-gpt54"}
    assert (
        tenants[0].azure_responses_api_url
        == "https://acme.openai.azure.com/openai/v1/responses"
    )


def test_parse_tenants_rejects_missing_azure_env(monkeypatch):
    """Fail startup when a tenant references an unset Azure API key env var."""
    monkeypatch.delenv("TENANT_MISSING_AZURE_API_KEY", raising=False)
    raw = json.dumps(
        [
            {
                "id": "acme",
                "api_key_hash": hash_api_key("key"),
                "azure_base_url": "https://acme.openai.azure.com",
                "azure_api_key_env": "TENANT_MISSING_AZURE_API_KEY",
                "azure_model_deployments": {"gpt-5.4": "acme-gpt54"},
            }
        ]
    )

    with pytest.raises(ServiceConfigurationError, match="TENANT_MISSING_AZURE_API_KEY"):
        parse_tenants(raw)


def test_parse_tenants_rejects_duplicate_ids_and_hashes(monkeypatch):
    """Reject duplicate tenant ids or API key hashes."""
    monkeypatch.setenv("TENANT_A_KEY", "azure-a")
    monkeypatch.setenv("TENANT_B_KEY", "azure-b")
    shared_hash = hash_api_key("shared")
    duplicate_ids = json.dumps(
        [
            {
                "id": "acme",
                "api_key_hash": hash_api_key("one"),
                "azure_base_url": "https://a.openai.azure.com",
                "azure_api_key_env": "TENANT_A_KEY",
                "azure_model_deployments": {"gpt-5.4": "a"},
            },
            {
                "id": "acme",
                "api_key_hash": hash_api_key("two"),
                "azure_base_url": "https://b.openai.azure.com",
                "azure_api_key_env": "TENANT_B_KEY",
                "azure_model_deployments": {"gpt-5.4": "b"},
            },
        ]
    )
    duplicate_hashes = json.dumps(
        [
            {
                "id": "acme",
                "api_key_hash": shared_hash,
                "azure_base_url": "https://a.openai.azure.com",
                "azure_api_key_env": "TENANT_A_KEY",
                "azure_model_deployments": {"gpt-5.4": "a"},
            },
            {
                "id": "beta",
                "api_key_hash": shared_hash,
                "azure_base_url": "https://b.openai.azure.com",
                "azure_api_key_env": "TENANT_B_KEY",
                "azure_model_deployments": {"gpt-5.4": "b"},
            },
        ]
    )

    with pytest.raises(ServiceConfigurationError, match="Duplicate tenant id"):
        parse_tenants(duplicate_ids)
    with pytest.raises(ServiceConfigurationError, match="Duplicate api_key_hash"):
        parse_tenants(duplicate_hashes)


def test_resolve_tenant_for_api_key_matches_digest():
    """Match bearer tokens against configured digests only."""
    tenant = TenantConfig(
        id="acme",
        api_key_hash=hash_api_key("correct-key"),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
    )

    assert resolve_tenant_for_api_key("correct-key", (tenant,)).id == "acme"
    assert resolve_tenant_for_api_key("wrong-key", (tenant,)) is None


def test_parse_tenants_rejects_missing_or_empty_deployments(monkeypatch):
    """Require an explicit non-empty deployment map per tenant."""
    monkeypatch.setenv("TENANT_ACME_AZURE_API_KEY", "acme-azure-secret")
    missing = json.dumps(
        [
            {
                "id": "acme",
                "api_key_hash": hash_api_key("key"),
                "azure_base_url": "https://acme.openai.azure.com",
                "azure_api_key_env": "TENANT_ACME_AZURE_API_KEY",
            }
        ]
    )
    empty = json.dumps(
        [
            {
                "id": "acme",
                "api_key_hash": hash_api_key("key"),
                "azure_base_url": "https://acme.openai.azure.com",
                "azure_api_key_env": "TENANT_ACME_AZURE_API_KEY",
                "azure_model_deployments": {},
            }
        ]
    )

    with pytest.raises(ServiceConfigurationError, match="azure_model_deployments"):
        parse_tenants(missing)
    with pytest.raises(ServiceConfigurationError, match="azure_model_deployments"):
        parse_tenants(empty)


def test_validate_tenant_startup_rejects_tenants_in_single_mode():
    """AUTH_MODE=single must not carry unused TENANTS configuration."""
    tenant = TenantConfig(
        id="acme",
        api_key_hash=hash_api_key("key"),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
    )

    with pytest.raises(ServiceConfigurationError, match="AUTH_MODE=single"):
        validate_tenant_startup(auth_mode="single", tenants=(tenant,))


def test_validate_tenant_startup_requires_tenants_in_tenant_mode():
    """AUTH_MODE=tenant refuses to start without tenants."""
    with pytest.raises(ServiceConfigurationError, match="non-empty TENANTS"):
        validate_tenant_startup(
            auth_mode="tenant",
            tenants=(),
        )


def test_create_app_rejects_tenant_mode_without_tenants():
    """Application factory validates AUTH_MODE=tenant before serving."""
    config = type(
        "TenantModeConfig",
        (),
        {
            "SERVICE_API_KEY": None,
            "AUTH_MODE": "tenant",
            "TENANTS": (),
        },
    )

    with pytest.raises(ServiceConfigurationError, match="TENANTS"):
        create_app(config)


def test_create_app_rejects_unknown_tenant_config_source():
    """Runtime source is an explicit, validated configuration choice."""
    config = type(
        "InvalidTenantSourceConfig",
        (),
        {
            "SERVICE_API_KEY": None,
            "AUTH_MODE": "tenant",
            "TENANT_CONFIG_SOURCE": "fallback",
            "TENANTS": (),
        },
    )

    with pytest.raises(ServiceConfigurationError, match="TENANT_CONFIG_SOURCE"):
        create_app(config)


def test_create_app_accepts_database_as_tenant_runtime_source():
    """Database-backed web startup accepts an empty legacy TENANTS setting."""
    config = type(
        "DatabaseTenantConfig",
        (),
        {
            "SERVICE_API_KEY": None,
            "AUTH_MODE": "tenant",
            "TENANT_CONFIG_SOURCE": "database",
            "TENANTS": (),
            "DATABASE_URL": "postgresql+psycopg://user:password@localhost/proxy",
            "PROVIDER_ENCRYPTION_KEY": "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=",
            "ADMIN_SESSION_SECRET": "test-admin-session-secret-bytes-32",
            "WEBAUTHN_RP_ID": "localhost",
            "WEBAUTHN_RP_NAME": "Test Proxy",
            "WEBAUTHN_ORIGINS": "http://localhost",
            "ENABLE_CODEX": False,
        },
    )

    app = create_app(config)

    assert app.config["TENANT_CONFIG_SOURCE"] == "database"
    app.extensions["database"].engine.dispose()


def test_create_app_rejects_tenants_when_database_is_the_runtime_source():
    """Database-backed web startup rejects a conflicting TENANTS configuration."""
    config = type(
        "ConflictingDatabaseTenantConfig",
        (),
        {
            "SERVICE_API_KEY": None,
            "AUTH_MODE": "tenant",
            "TENANT_CONFIG_SOURCE": "database",
            "TENANTS": (),
            "TENANTS_CONFIGURED": True,
            "DATABASE_URL": "postgresql+psycopg://user:password@localhost/proxy",
            "PROVIDER_ENCRYPTION_KEY": "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=",
            "ADMIN_SESSION_SECRET": "test-admin-session-secret-bytes-32",
            "WEBAUTHN_RP_ID": "localhost",
            "WEBAUTHN_RP_NAME": "Test Proxy",
            "WEBAUTHN_ORIGINS": "http://localhost",
            "ENABLE_CODEX": False,
        },
    )

    with pytest.raises(ServiceConfigurationError, match="TENANTS.*database"):
        create_app(config)


def test_create_app_accepts_tenant_mode_without_service_api_key(monkeypatch):
    """Tenant mode does not require SERVICE_API_KEY."""
    monkeypatch.setenv("TENANT_ACME_AZURE_API_KEY", "acme-azure-secret")
    tenant = TenantConfig(
        id="acme",
        api_key_hash=hash_api_key("tenant-key"),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="acme-azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
    )
    config = type(
        "TenantModeConfig",
        (),
        {
            "SERVICE_API_KEY": None,
            "AUTH_MODE": "tenant",
            "TENANTS": (tenant,),
            "AZURE_MODEL_DEPLOYMENTS": {"gpt-5.4": "unused"},
            "CODEX_MODEL_REWRITES": {},
            "CODEX_SUPPORTED_MODELS": (),
        },
    )

    app = create_app(config)

    assert app.config["AUTH_MODE"] == "tenant"
    assert app.config["TENANTS"][0].id == "acme"


def test_hash_api_key_uses_scrypt_and_still_accepts_legacy_sha256():
    """New digests are scrypt, while stored SHA-256 digests still authenticate."""
    api_key = "correct-key"
    current = hash_api_key(api_key)
    legacy = legacy_sha256_api_key_hash(api_key)

    assert current != legacy
    assert len(current) == 64
    assert all(char in "0123456789abcdef" for char in current)
    # FIPS 180-4 vector, plus the historical digest of this bearer token.
    assert legacy_sha256_api_key_hash("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert legacy == "ddb0fd2dede48502669718e09ef1447dba46f3d3822e9fbf05af11d874a0f23b"

    tenant = TenantConfig(
        id="acme",
        api_key_hash=legacy,
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
    )

    assert resolve_tenant_for_api_key(api_key, (tenant,)).id == "acme"
    assert resolve_tenant_for_api_key("wrong-key", (tenant,)) is None
