"""Integration tests for AUTH_MODE=tenant request isolation."""

from __future__ import annotations

import logging

import pytest
from flask import Flask, Response, g
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from webtest import TestApp

from app import create_app
from app.azure.adapter import AzureAdapter
from app.exceptions import ServiceConfigurationError
from app.persistence.database import Database
from app.persistence.models import Base, ProviderProfile, Tenant
from app.persistence.secrets import SecretCipher
from app.tenants import TenantConfig, hash_api_key


def _tenant(
    tenant_id: str,
    api_key: str,
    *,
    azure_base_url: str,
    azure_api_key: str,
    deployments: dict[str, str],
) -> TenantConfig:
    return TenantConfig(
        id=tenant_id,
        api_key_hash=hash_api_key(api_key),
        azure_base_url=azure_base_url,
        azure_api_key=azure_api_key,
        azure_model_deployments=deployments,
    )


@pytest.fixture
def tenant_keys() -> dict[str, str]:
    """Cleartext tenant API keys used only in tests."""
    return {
        "acme": "tenant-acme-cleartext-key",
        "beta": "tenant-beta-cleartext-key",
    }


@pytest.fixture
def tenant_app(tenant_keys) -> Flask:
    """Flask app configured with two isolated Azure tenants."""
    tenants = (
        _tenant(
            "acme",
            tenant_keys["acme"],
            azure_base_url="https://acme.openai.azure.com",
            azure_api_key="acme-azure-key",
            deployments={"gpt-5.4": "acme-gpt54", "gpt-5.5": "acme-gpt55"},
        ),
        _tenant(
            "beta",
            tenant_keys["beta"],
            azure_base_url="https://beta.openai.azure.com",
            azure_api_key="beta-azure-key",
            deployments={"gpt-5.4": "beta-gpt54"},
        ),
    )
    config = type(
        "TenantTestConfig",
        (),
        {
            "TESTING": True,
            "AUTH_MODE": "tenant",
            "SERVICE_API_KEY": "must-not-authenticate-in-tenant-mode",
            "TENANTS": tenants,
            "TRUSTED_HOSTS": [
                "localhost",
                "proxy.altanis.de",
                "proxy.iffm-gmbh.de",
            ],
            "TRUST_PROXY_HEADERS": True,
            "ENABLE_AZURE": True,
            "ENABLE_CODEX": False,
            "AZURE_BASE_URL": "https://global.openai.azure.com",
            "AZURE_API_KEY": "global-azure-key",
            "AZURE_MODEL_DEPLOYMENTS": {"gpt-5.4": "global-gpt54"},
            "AZURE_RESPONSES_API_URL": (
                "https://global.openai.azure.com/openai/v1/responses"
            ),
            "AZURE_SUMMARY_LEVEL": "detailed",
            "AZURE_VERBOSITY_LEVEL": "medium",
            "AZURE_TRUNCATION": "disabled",
            "RECORD_TRAFFIC": False,
            "LOG_CONTEXT": False,
            "LOG_COMPLETION": False,
            "REASONING_DISPLAY_MODE": "mdthinkblocks",
            "CODEX_AUTH_PATH": "~/.codex/auth.json",
            "CODEX_RESPONSES_URL": "https://chatgpt.com/backend-api/codex/responses",
            "CODEX_SUPPORTED_MODELS": ("codex-test-model",),
            "CODEX_MODEL_REWRITES": {},
            "CODEX_ORIGINATOR": "codex_cli_rs",
            "CODEX_USER_AGENT": "test-agent",
            "CODEX_DISCOVERY_MODE": True,
            "CODEX_TOKEN_REFRESH_SKEW_SECONDS": 300,
            "CODEX_REQUEST_TIMEOUT_SECONDS": 600.0,
        },
    )
    app = create_app(config)
    app.logger.setLevel(logging.CRITICAL)
    ctx = app.test_request_context()
    ctx.push()
    yield app
    ctx.pop()


@pytest.fixture
def tenant_testapp(tenant_app) -> TestApp:
    """Create Webtest client for the tenant-mode app."""
    return TestApp(tenant_app)


def test_database_tenant_api_key_authenticates_against_persisted_profile(monkeypatch):
    """Database-backed tenant auth resolves the persisted tenant profile."""
    api_key = "persisted-tenant-cleartext-key"
    database_config = type(
        "DatabaseTenantConfig",
        (),
        {
            "TESTING": True,
            "AUTH_MODE": "tenant",
            "TENANT_CONFIG_SOURCE": "database",
            "TENANTS": (),
            "SERVICE_API_KEY": None,
            "DATABASE_URL": "postgresql+psycopg://user:password@localhost/proxy",
            "PROVIDER_ENCRYPTION_KEY": "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=",
            "ADMIN_SESSION_SECRET": "test-admin-session-secret-bytes-32",
            "WEBAUTHN_RP_ID": "localhost",
            "WEBAUTHN_RP_NAME": "Test Proxy",
            "WEBAUTHN_ORIGINS": "http://localhost",
            "ENABLE_AZURE": True,
            "ENABLE_CODEX": False,
            "AZURE_MODEL_DEPLOYMENTS": {},
            "AZURE_RESPONSES_API_URL": "https://global.openai.azure.com/openai/v1/responses",
            "AZURE_API_KEY": "global-azure-key",
            "AZURE_SUMMARY_LEVEL": "detailed",
            "AZURE_VERBOSITY_LEVEL": "medium",
            "AZURE_TRUNCATION": "disabled",
            "RECORD_TRAFFIC": False,
            "LOG_CONTEXT": False,
            "LOG_COMPLETION": False,
            "REASONING_DISPLAY_MODE": "mdthinkblocks",
            "TRUSTED_HOSTS": ["localhost"],
        },
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(database_config.PROVIDER_ENCRYPTION_KEY)
    database = Database(
        engine, sessionmaker(bind=engine, expire_on_commit=False), cipher
    )
    monkeypatch.setattr(Database, "from_config", lambda config: database)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="profile-persisted",
            tenant_id="persisted",
            provider="azure",
            settings={
                "base_url": "https://persisted.openai.azure.com",
                "model_deployments": {"gpt-5.4": "persisted-deployment"},
            },
            inference_secret_ciphertext=cipher.encrypt("persisted-azure-key"),
        )
        tenant = Tenant(
            id="persisted",
            api_key_hash=hash_api_key(api_key),
            custom_model_id="cursor-persisted-model",
            active_profile_id=profile.id,
        )
        session.add_all((tenant, profile))

    app = create_app(database_config)
    response = app.test_client().get(
        "/v1/models", headers={"Authorization": f"Bearer {api_key}"}
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json["data"]] == ["cursor-persisted-model"]
    engine.dispose()


def test_database_openai_provider_forwards_to_openai_compatible(monkeypatch):
    """Active OpenAI profiles use the OpenAI-compatible forwarder, not Azure."""
    api_key = "openai-tenant-cleartext-key"
    database_config = type(
        "DatabaseOpenAIConfig",
        (),
        {
            "TESTING": True,
            "AUTH_MODE": "tenant",
            "TENANT_CONFIG_SOURCE": "database",
            "TENANTS": (),
            "SERVICE_API_KEY": None,
            "DATABASE_URL": "postgresql+psycopg://user:password@localhost/proxy",
            "PROVIDER_ENCRYPTION_KEY": "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=",
            "ADMIN_SESSION_SECRET": "test-admin-session-secret-bytes-32",
            "WEBAUTHN_RP_ID": "localhost",
            "WEBAUTHN_RP_NAME": "Test Proxy",
            "WEBAUTHN_ORIGINS": "http://localhost",
            "ENABLE_AZURE": True,
            "ENABLE_CODEX": False,
            "AZURE_MODEL_DEPLOYMENTS": {},
            "AZURE_RESPONSES_API_URL": "https://global.openai.azure.com/openai/v1/responses",
            "AZURE_API_KEY": "global-azure-key",
            "AZURE_SUMMARY_LEVEL": "detailed",
            "AZURE_VERBOSITY_LEVEL": "medium",
            "AZURE_TRUNCATION": "disabled",
            "RECORD_TRAFFIC": False,
            "LOG_CONTEXT": False,
            "LOG_COMPLETION": False,
            "REASONING_DISPLAY_MODE": "mdthinkblocks",
            "TRUSTED_HOSTS": ["localhost"],
        },
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(database_config.PROVIDER_ENCRYPTION_KEY)
    database = Database(
        engine, sessionmaker(bind=engine, expire_on_commit=False), cipher
    )
    monkeypatch.setattr(Database, "from_config", lambda config: database)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="profile-openai",
            tenant_id="openai-tenant",
            provider="openai",
            settings={"organization": "", "project": ""},
            inference_secret_ciphertext=cipher.encrypt("sk-test"),
            default_model="gpt-5.4",
        )
        tenant = Tenant(
            id="openai-tenant",
            api_key_hash=hash_api_key(api_key),
            custom_model_id="cursor-openai-model",
            active_profile_id=profile.id,
        )
        session.add_all((tenant, profile))

    seen: dict[str, object] = {}

    def fake_forward(req, snapshot):
        seen["provider"] = snapshot.provider
        seen["default_model"] = snapshot.default_model
        return Response("openai-ok", status=200, mimetype="text/plain")

    monkeypatch.setattr("app.blueprint.forward_openai_compatible", fake_forward)
    app = create_app(database_config)
    response = app.test_client().post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "cursor-openai-model", "messages": []},
    )
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "openai-ok"
    assert seen == {"provider": "openai", "default_model": "gpt-5.4"}
    engine.dispose()


def test_unknown_and_single_service_key_return_same_generic_401(
    tenant_testapp, tenant_keys
):
    """Invalid keys share one generic HTTP 401 without leaking which failed."""
    unknown = tenant_testapp.get(
        "/v1/models",
        headers={"Authorization": "Bearer unknown-key"},
        status=401,
    )
    single_key = tenant_testapp.get(
        "/v1/models",
        headers={"Authorization": "Bearer must-not-authenticate-in-tenant-mode"},
        status=401,
    )
    missing = tenant_testapp.get("/v1/models", status=401)

    assert unknown.text == single_key.text == missing.text
    assert "Authentication with the proxy service failed" in unknown.text
    assert "acme" not in unknown.text
    assert "beta" not in unknown.text


def test_tenant_models_are_isolated(tenant_testapp, tenant_keys):
    """Each tenant only sees its configured Azure model deployments."""
    acme = tenant_testapp.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {tenant_keys['acme']}"},
        status=200,
    ).json
    beta = tenant_testapp.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {tenant_keys['beta']}"},
        status=200,
    ).json

    assert [item["id"] for item in acme["data"]] == ["gpt-5.4", "gpt-5.5"]
    assert [item["id"] for item in beta["data"]] == ["gpt-5.4"]


def test_tenant_resolution_is_independent_of_public_hostname(
    tenant_testapp, tenant_keys
):
    """Both DNS names authenticate and route solely by the tenant API key."""
    responses = [
        tenant_testapp.get(
            "/v1/models",
            headers={
                "Host": hostname,
                "Authorization": f"Bearer {tenant_keys['acme']}",
            },
            status=200,
        ).json
        for hostname in ("proxy.altanis.de", "proxy.iffm-gmbh.de")
    ]

    assert responses[0] == responses[1]
    assert [item["id"] for item in responses[0]["data"]] == ["gpt-5.4", "gpt-5.5"]


def test_tenant_azure_credentials_and_cache_keys_are_isolated(tenant_app, tenant_keys):
    """Tenant A never reuses Tenant B credentials, models, or cache partitions."""
    adapter = AzureAdapter().request_adapter
    conversation_id = "conv-shared-across-tenants"
    payload = {
        "model": "gpt-5.4",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
        "metadata": {"cursorConversationId": conversation_id},
        "stream": True,
    }

    with tenant_app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        json=payload,
        headers={
            "Authorization": f"Bearer {tenant_keys['acme']}",
            "X-Tenant-Id": "spoofed-tenant",
            "Tenant-Id": "spoofed-tenant",
        },
    ) as ctx:
        g.tenant = tenant_app.config["TENANTS"][0]
        g.auth_mode = "tenant"
        acme_kwargs = adapter.adapt(ctx.request)

    with tenant_app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        json=payload,
        headers={"Authorization": f"Bearer {tenant_keys['beta']}"},
    ) as ctx:
        g.tenant = tenant_app.config["TENANTS"][1]
        g.auth_mode = "tenant"
        beta_kwargs = adapter.adapt(ctx.request)

    assert acme_kwargs["url"] == "https://acme.openai.azure.com/openai/v1/responses"
    assert beta_kwargs["url"] == "https://beta.openai.azure.com/openai/v1/responses"
    assert acme_kwargs["headers"]["api-key"] == "acme-azure-key"
    assert beta_kwargs["headers"]["api-key"] == "beta-azure-key"
    assert acme_kwargs["json"]["model"] == "acme-gpt54"
    assert beta_kwargs["json"]["model"] == "beta-gpt54"
    assert acme_kwargs["json"]["prompt_cache_key"] == f"acme:{conversation_id}"
    assert beta_kwargs["json"]["prompt_cache_key"] == f"beta:{conversation_id}"
    assert acme_kwargs["headers"]["session_id"] == f"acme:{conversation_id}"
    assert beta_kwargs["headers"]["session_id"] == f"beta:{conversation_id}"
    assert "Authorization" not in acme_kwargs["headers"]
    assert "X-Tenant-Id" not in acme_kwargs["headers"]
    assert "Tenant-Id" not in acme_kwargs["headers"]
    assert acme_kwargs["headers"]["api-key"] != "global-azure-key"
    assert beta_kwargs["headers"]["api-key"] != "global-azure-key"


def test_create_app_rejects_codex_enabled_in_tenant_mode(tenant_keys):
    """Tenant mode must not start with the shared Codex login enabled."""
    tenant = TenantConfig(
        id="acme",
        api_key_hash=hash_api_key(tenant_keys["acme"]),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="acme-azure-key",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
    )
    config = type(
        "TenantCodexConfig",
        (),
        {
            "AUTH_MODE": "tenant",
            "SERVICE_API_KEY": None,
            "TENANTS": (tenant,),
            "ENABLE_CODEX": True,
            "AZURE_MODEL_DEPLOYMENTS": {"gpt-5.4": "unused"},
            "CODEX_MODEL_REWRITES": {},
            "CODEX_SUPPORTED_MODELS": (),
        },
    )

    with pytest.raises(ServiceConfigurationError, match="cannot enable Codex"):
        create_app(config)


def test_tenant_mode_blocks_codex_routes(
    tenant_testapp, tenant_app, tenant_keys, monkeypatch
):
    """Tenant API keys must not reach the shared Codex login even if Codex is toggled on."""
    tenant_app.config["ENABLE_CODEX"] = True

    def fail_forward(*args, **kwargs):
        raise AssertionError("Codex must not forward for tenant keys")

    monkeypatch.setattr("app.codex.adapter.CodexAdapter.forward", fail_forward)
    monkeypatch.setattr("app.codex.adapter.CodexAdapter.ready", fail_forward)

    models = tenant_testapp.get(
        "/codex/v1/models",
        headers={"Authorization": f"Bearer {tenant_keys['acme']}"},
        status=400,
    )
    proxy = tenant_testapp.post_json(
        "/codex/v1/chat/completions",
        {"model": "codex-test-model"},
        headers={"Authorization": f"Bearer {tenant_keys['acme']}"},
        status=400,
    )
    ready = tenant_testapp.get(
        "/codex/ready",
        headers={"Authorization": f"Bearer {tenant_keys['acme']}"},
        status=400,
    )

    assert "Codex is not available in AUTH_MODE=tenant" in models.text
    assert "Codex is not available in AUTH_MODE=tenant" in proxy.text
    assert "Codex is not available in AUTH_MODE=tenant" in ready.text


def test_tenant_mode_refuses_global_azure_without_request_tenant(tenant_app):
    """Tenant mode must not fall back to global Azure credentials."""
    adapter = AzureAdapter().request_adapter

    with tenant_app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        json={"model": "gpt-5.4", "input": "hi"},
    ) as ctx:
        # Simulate a missing principal despite AUTH_MODE=tenant.
        g.tenant = None
        g.auth_mode = "tenant"
        with pytest.raises(ServiceConfigurationError, match="authenticated tenant"):
            adapter.adapt(ctx.request)


def test_tenant_proxy_route_uses_authenticated_tenant(
    tenant_testapp, tenant_keys, monkeypatch
):
    """Authenticated catch-all traffic keeps the resolved tenant on flask.g."""
    seen = []

    def fake_forward(self, req):
        seen.append(g.tenant.id if g.tenant else None)
        return Response("ok", status=200)

    monkeypatch.setattr("app.azure.adapter.AzureAdapter.forward", fake_forward)

    tenant_testapp.post_json(
        "/v1/chat/completions",
        {"model": "gpt-5.4"},
        headers={"Authorization": f"Bearer {tenant_keys['beta']}"},
        status=200,
    )

    assert seen == ["beta"]
