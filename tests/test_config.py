"""Functional tests using WebTest.

See: http://webtest.readthedocs.org/
"""

import importlib
import json
import sys

import environs
import pytest
from flask import request

from app import create_app
from app.exceptions import ServiceConfigurationError


class TestConfig:
    """Config."""

    def test_test_config_is_set(self, testapp):
        """Ensure that test config is set."""
        app = testapp.app
        assert app.config["AZURE_BASE_URL"] != "change_me"
        assert app.config["AZURE_API_KEY"] != "change_me"

    def test_untrusted_host_is_rejected(self, testapp):
        """Reject Host headers not explicitly configured for the application."""
        testapp.get("/health", headers={"Host": "attacker.example"}, status=400)

    def test_proxy_headers_require_trusted_hosts(self):
        """Refuse proxy-header trust unless the Host allowlist is configured."""
        config = type(
            "ProxyTrustConfig",
            (),
            {
                "SERVICE_API_KEY": "test-service-api-key",
                "TRUST_PROXY_HEADERS": True,
                "TRUSTED_HOSTS": [],
            },
        )

        with pytest.raises(ServiceConfigurationError, match="TRUSTED_HOSTS"):
            create_app(config)

    def test_trusted_hosts_must_be_a_sequence(self):
        """Reject malformed allowlist config instead of splitting into chars."""
        config = type(
            "MalformedHostConfig",
            (),
            {
                "SERVICE_API_KEY": "test-service-api-key",
                "TRUSTED_HOSTS": "proxy.altanis.de",
            },
        )

        with pytest.raises(ServiceConfigurationError, match="TRUSTED_HOSTS"):
            create_app(config)

    @pytest.mark.parametrize("public_host", ["proxy.altanis.de", "proxy.iffm-gmbh.de"])
    def test_configured_proxy_hosts_are_accepted(self, app, public_host):
        """Resolve both DNS names through the same application and tenant config."""
        response = app.test_client().get("/v1/models", headers={"Host": public_host})

        assert response.status_code == 401

    def test_untrusted_forwarded_host_is_rejected(self, testapp):
        """Reject a forwarded Host that is outside the configured allowlist."""
        testapp.get(
            "/health",
            headers={
                "Host": "proxy.altanis.de",
                "X-Forwarded-Host": "attacker.example",
            },
            status=400,
        )

    def test_proxy_headers_reconstruct_public_request(self, testapp, app):
        """Trust forwarding metadata from the configured reverse proxy hop."""
        observed = {}

        @app.before_request
        def capture_request_metadata():
            observed["host"] = request.host
            observed["scheme"] = request.scheme
            observed["remote_addr"] = request.remote_addr

        testapp.get(
            "/health",
            headers={
                "Host": "proxy.iffm-gmbh.de",
                "X-Forwarded-For": "203.0.113.25",
                "X-Forwarded-Proto": "https",
            },
            status=200,
        )

        assert observed == {
            "host": "proxy.iffm-gmbh.de",
            "scheme": "https",
            "remote_addr": "203.0.113.25",
        }

    def test_env_example_loads(self, monkeypatch):
        """Patch Env.read_env to read from .env.example and import settings."""
        orig_read_env = environs.Env.read_env
        monkeypatch.setattr(
            environs.Env,
            "read_env",
            lambda *args, **kwargs: orig_read_env(
                ".env.example", override=True, **kwargs
            ),
        )

        sys.modules.pop("app.settings", None)
        settings = importlib.import_module("app.settings")

        assert (
            settings.AZURE_BASE_URL
            == "https://azureopenai-instanz2.cognitiveservices.azure.com"
        )
        assert (
            settings.AZURE_RESPONSES_API_URL
            == "https://azureopenai-instanz2.cognitiveservices.azure.com/openai/v1/responses"
        )
        expected_azure_deployments = {
            "gpt-6-astra": "gpt-6-astra-api",
            "gpt-6-luna": "gpt-6-luna-api",
            "gpt-6.1-sol": "gpt-6.1-sol-api",
        }
        assert settings.AZURE_MODEL_DEPLOYMENTS == expected_azure_deployments
        assert settings.SERVICE_API_KEY in (None, "")
        assert settings.AUTH_MODE == "single"
        assert settings.TENANTS == ()

    def test_optional_azure_settings_use_defaults_when_missing(self, monkeypatch):
        """Import settings without optional Azure env vars."""
        monkeypatch.setattr(environs.Env, "read_env", lambda *args, **kwargs: None)
        for key in (
            "AZURE_SUMMARY_LEVEL",
            "AZURE_VERBOSITY_LEVEL",
            "AZURE_TRUNCATION",
        ):
            monkeypatch.delenv(key, raising=False)

        sys.modules.pop("app.settings", None)
        settings = importlib.import_module("app.settings")

        assert settings.AZURE_SUMMARY_LEVEL == "detailed"
        assert settings.AZURE_VERBOSITY_LEVEL == "medium"
        assert settings.AZURE_TRUNCATION == "disabled"

    def test_sensitive_request_logging_is_disabled_by_default(self, monkeypatch):
        """Do not log request context or completion content by default."""
        monkeypatch.setattr(environs.Env, "read_env", lambda *args, **kwargs: None)
        for key in ("LOG_CONTEXT", "LOG_COMPLETION"):
            monkeypatch.delenv(key, raising=False)

        sys.modules.pop("app.settings", None)
        settings = importlib.import_module("app.settings")

        assert settings.LOG_CONTEXT is False
        assert settings.LOG_COMPLETION is False

    @pytest.mark.parametrize(
        "service_api_key", (None, "", "change-me", "choose-a-local-secret")
    )
    def test_app_rejects_missing_or_placeholder_service_api_key(self, service_api_key):
        """Require an explicitly configured, non-placeholder service key."""
        config = type("ServiceKeyConfig", (), {"SERVICE_API_KEY": service_api_key})

        with pytest.raises(ServiceConfigurationError, match="SERVICE_API_KEY"):
            create_app(config)

    def test_responses_api_url_uses_v1_without_api_version(self, monkeypatch):
        """Build the modern Azure Responses endpoint without dated api-version."""
        monkeypatch.setattr(environs.Env, "read_env", lambda *args, **kwargs: None)
        monkeypatch.setenv("AZURE_BASE_URL", "https://example.openai.azure.com/")
        monkeypatch.setenv("AZURE_API_VERSION", "2025-04-01-preview")

        sys.modules.pop("app.settings", None)
        settings = importlib.import_module("app.settings")

        assert (
            settings.AZURE_RESPONSES_API_URL
            == "https://example.openai.azure.com/openai/v1/responses"
        )
        assert "api-version" not in settings.AZURE_RESPONSES_API_URL

    def test_env_example_exposes_model_deployments_mapping(self, monkeypatch):
        """Load explicit per-model deployment overrides from the environment."""
        monkeypatch.setenv(
            "AZURE_MODEL_DEPLOYMENTS",
            json.dumps(
                {
                    "gpt-5.4": "prod-gpt54",
                    "gpt-5.4-mini": "team-mini",
                    "gpt-5.5": "gpt-5.5-1",
                    "gpt-6-luna": "luna-test-deployment",
                }
            ),
        )

        sys.modules.pop("app.settings", None)
        settings = importlib.import_module("app.settings")

        assert settings.AZURE_MODEL_DEPLOYMENTS == {
            "gpt-5.4": "prod-gpt54",
            "gpt-5.4-mini": "team-mini",
            "gpt-5.5": "gpt-5.5-1",
            "gpt-6-luna": "luna-test-deployment",
        }

    def test_legacy_single_deployment_env_is_ignored(self, monkeypatch):
        """Do not silently backfill the old single-deployment env var."""
        monkeypatch.setattr(environs.Env, "read_env", lambda *args, **kwargs: None)
        monkeypatch.delenv("AZURE_MODEL_DEPLOYMENTS", raising=False)
        monkeypatch.setenv("AZURE_DEPLOYMENT", "legacy-custom-deployment")

        sys.modules.pop("app.settings", None)
        settings = importlib.import_module("app.settings")

        assert settings.AZURE_MODEL_DEPLOYMENTS["gpt-5"] == "gpt-5"
