"""Root inference failover using real adapters and tenant-owned model targets."""

import json

import pytest
import requests
from flask import Response
from sqlalchemy import select

from app.persistence.admin_ops import (
    activate_provider_profile,
    create_provider_profile,
    replace_catalog_entries,
)
from app.persistence.models import InferenceActivityEvent, Tenant

AUTH = {"Authorization": "Bearer cursor-key"}
AZURE = "https://test-resource.openai.azure.com/openai/v1/responses"
OPENAI = "https://api.openai.com/v1/chat/completions"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK = "https://api.deepseek.com/chat/completions"


def event(data, name=None):
    """Encode a JSON payload as an optionally named SSE event."""
    return (
        (f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n"
    ).encode()


@pytest.fixture
def routed_app(admin_app):
    """Configure a tenant route through Azure, OpenAI, OpenRouter, and DeepSeek."""
    admin_app.config.update(
        AUTH_MODE="tenant", TENANT_CONFIG_SOURCE="database", ENABLE_AZURE=True
    )
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        for provider, model, settings in [
            (
                "azure",
                "gpt-5.4",
                {
                    "base_url": "https://test-resource.openai.azure.com",
                    "model_deployments": {"gpt-5.4": "azure-deployment"},
                },
            ),
            ("openai", "gpt-5.4", {}),
            ("openrouter", "anthropic/claude-sonnet-4", {}),
            ("deepseek", "deepseek-v4-flash", {}),
        ]:
            profile = create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                provider,
                provider,
                settings,
                model,
                provider + "-secret",
                "ada",
            )
            replace_catalog_entries(
                session,
                profile,
                [(model, "azure-deployment" if provider == "azure" else None)],
                None,
            )
            activate_provider_profile(session, tenant, profile.id, "ada")
    return admin_app


def post(app, path="/v1/chat/completions", model="cursor-acme-model"):
    """Submit an authenticated chat request for the selected logical model."""
    return app.test_client().post(
        path,
        headers=AUTH,
        json={"model": model, "messages": [{"role": "user", "content": "Hi"}]},
    )


def successful_chat(model):
    """Encode a successful chat stream with the provider's model name."""
    return (
        event({"model": model, "choices": [{"delta": {"content": "success"}}]})
        + b"data: [DONE]\n\n"
    )


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_azure_error_immediately_uses_second_provider(
    routed_app, requests_mock, status
):
    """Retry transient Azure errors with OpenAI's own model and credentials."""
    requests_mock.post(
        AZURE, status_code=status, json={"error": {"code": "rate_limit_exceeded"}}
    )
    requests_mock.post(
        OPENAI,
        content=successful_chat("gpt-5.4"),
        headers={"Content-Type": "text/event-stream"},
    )
    response = post(routed_app)
    assert response.status_code == 200
    assert b'"model":"cursor-acme-model"' in response.data
    assert [r.url for r in requests_mock.request_history] == [AZURE, OPENAI]
    assert requests_mock.request_history[0].json()["model"] == "azure-deployment"
    assert requests_mock.request_history[1].json()["model"] == "gpt-5.4"
    assert (
        requests_mock.request_history[1].headers["Authorization"]
        == "Bearer openai-secret"
    )
    assert "api-key" not in requests_mock.request_history[1].headers


def test_third_provider_uses_its_own_model(routed_app, requests_mock):
    """Use OpenRouter's model after both earlier providers fail."""
    requests_mock.post(AZURE, status_code=503)
    requests_mock.post(OPENAI, status_code=429)
    requests_mock.post(OPENROUTER, content=successful_chat("anthropic/claude-sonnet-4"))
    response = post(routed_app)
    assert response.status_code == 200
    assert b"success" in response.data
    assert [r.url for r in requests_mock.request_history] == [
        AZURE,
        OPENAI,
        OPENROUTER,
    ]
    assert (
        requests_mock.request_history[-1].json()["model"] == "anthropic/claude-sonnet-4"
    )


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_nonretryable_error_never_uses_fallback(routed_app, requests_mock, status):
    """Return terminal Azure errors without failover or credential leakage."""
    requests_mock.post(
        AZURE, status_code=status, json={"error": {"message": "azure-secret"}}
    )
    response = post(routed_app)
    assert response.status_code == status
    assert requests_mock.call_count == 1
    assert b"azure-secret" not in response.data


def test_early_azure_sse_rate_limit_uses_fallback(routed_app, requests_mock):
    """Retry an Azure SSE rate limit before user-visible output."""
    content = event({"type": "response.created"}, "response.created") + event(
        {"response": {"error": {"code": "rate_limit_exceeded"}}}, "response.failed"
    )
    requests_mock.post(AZURE, content=content)
    requests_mock.post(OPENAI, content=successful_chat("gpt-5.4"))
    response = post(routed_app)
    assert b"success" in response.data
    assert requests_mock.call_count == 2


def test_early_openai_sse_rate_limit_uses_third_provider(routed_app, requests_mock):
    """Retry an early OpenAI SSE rate limit with OpenRouter."""
    requests_mock.post(AZURE, status_code=503)
    requests_mock.post(
        OPENAI, content=event({"error": {"code": "rate_limit_exceeded"}})
    )
    requests_mock.post(OPENROUTER, content=successful_chat("anthropic/claude-sonnet-4"))
    response = post(routed_app)
    assert b"success" in response.data
    assert requests_mock.call_count == 3


def test_partial_azure_output_never_replays(routed_app, requests_mock):
    """Keep partial Azure output and its late error without replaying."""
    content = event({"delta": "partial"}, "response.output_text.delta") + event(
        {"response": {"error": {"code": "rate_limit_exceeded", "message": "busy"}}},
        "response.failed",
    )
    requests_mock.post(AZURE, content=content)
    response = post(routed_app)
    assert b"partial" in response.data
    assert b"rate_limit_exceeded" in response.data
    assert b"busy" not in response.data
    assert requests_mock.call_count == 1


def test_exhausted_route_returns_last_status(routed_app, requests_mock):
    """Return the final provider's status when the route is exhausted."""
    requests_mock.post(AZURE, status_code=503)
    requests_mock.post(OPENAI, status_code=429)
    requests_mock.post(OPENROUTER, status_code=500)
    requests_mock.post(DEEPSEEK, status_code=402)
    response = post(routed_app)
    assert response.status_code == 402
    assert requests_mock.call_count == 4


def test_deepseek_http_402_never_fails_over_and_does_not_leak_credentials(
    routed_app, requests_mock
):
    """The DeepSeek 402 stops before a later tenant-owned profile."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        fallback = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "openai-fallback",
            {},
            "gpt-5.4",
            "openai-fallback-secret",
            "ada",
        )
        replace_catalog_entries(session, fallback, [("gpt-5.4", None)], None)
        activate_provider_profile(session, tenant, fallback.id, "ada")

    requests_mock.post(AZURE, status_code=503)
    requests_mock.post(
        OPENAI,
        status_code=503,
        additional_matcher=lambda request: request.headers["Authorization"]
        == "Bearer openai-secret",
    )
    requests_mock.post(OPENROUTER, status_code=503)
    requests_mock.post(DEEPSEEK, status_code=402, text="deepseek-secret")
    requests_mock.post(
        OPENAI,
        content=successful_chat("gpt-5.4"),
        additional_matcher=lambda request: request.headers["Authorization"]
        == "Bearer openai-fallback-secret",
    )

    response = post(routed_app)

    assert response.status_code == 402
    assert [request.url for request in requests_mock.request_history] == [
        AZURE,
        OPENAI,
        OPENROUTER,
        DEEPSEEK,
    ]
    assert (
        requests_mock.last_request.headers["Authorization"] == "Bearer deepseek-secret"
    )
    assert b"deepseek-secret" not in response.data


def test_deepseek_http_429_uses_the_next_profile(routed_app, requests_mock):
    """Retry DeepSeek throttling and preserve the next profile's credential."""
    requests_mock.post(AZURE, status_code=503)
    requests_mock.post(OPENAI, status_code=503)
    requests_mock.post(OPENROUTER, status_code=503)
    requests_mock.post(DEEPSEEK, status_code=429)
    requests_mock.post(
        OPENAI,
        content=successful_chat("gpt-5.4"),
        additional_matcher=lambda request: request.headers["Authorization"]
        == "Bearer openai-fallback-secret",
    )
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        fallback = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "openai-fallback",
            {},
            "gpt-5.4",
            "openai-fallback-secret",
            "ada",
        )
        replace_catalog_entries(session, fallback, [("gpt-5.4", None)], None)
        activate_provider_profile(session, tenant, fallback.id, "ada")

    response = post(routed_app)

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [
        AZURE,
        OPENAI,
        OPENROUTER,
        DEEPSEEK,
        OPENAI,
    ]
    assert requests_mock.last_request.headers["Authorization"] == (
        "Bearer openai-fallback-secret"
    )


def test_deepseek_chat_completion_records_final_usage_before_done(
    routed_app, requests_mock
):
    """A DeepSeek final usage chunk is saved in its own tenant activity row."""
    requests_mock.post(AZURE, status_code=503)
    requests_mock.post(OPENAI, status_code=503)
    requests_mock.post(OPENROUTER, status_code=503)
    stream = (
        event(
            {
                "model": "deepseek-v4-flash",
                "choices": [{"delta": {"content": "hello"}, "finish_reason": None}],
            }
        )
        + event(
            {
                "model": "deepseek-v4-flash",
                "choices": [],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 5,
                    "total_tokens": 22,
                },
            }
        )
        + b"data: [DONE]\n\n"
    )
    requests_mock.post(
        DEEPSEEK,
        content=stream,
        headers={"Content-Type": "text/event-stream"},
    )

    response = post(routed_app)

    assert response.status_code == 200
    assert response.data.endswith(b"data: [DONE]\n\n")
    assert b'"model":"cursor-acme-model"' in response.data
    database = routed_app.extensions["database"]
    with database.sessions() as session:
        activity = session.scalar(
            select(InferenceActivityEvent).where(
                InferenceActivityEvent.provider == "deepseek"
            )
        )
    assert activity is not None
    assert activity.input_tokens == 17
    assert activity.output_tokens == 5
    assert activity.total_tokens == 22


def test_connect_timeout_is_retryable_but_read_timeout_is_not(
    routed_app, requests_mock
):
    """Retry a connect timeout but stop after an ambiguous read timeout."""
    requests_mock.post(AZURE, exc=requests.ConnectTimeout)
    requests_mock.post(OPENAI, exc=requests.ReadTimeout)
    response = post(routed_app)
    assert response.status_code == 502
    assert requests_mock.call_count == 2


def test_unknown_model_rejected_before_upstream(routed_app, requests_mock):
    """Reject unmapped models without contacting any provider."""
    response = post(routed_app, model="unmapped-model")
    assert response.status_code == 400
    assert requests_mock.call_count == 0


def test_azure_catalog_model_pins_to_azure_without_failover(
    routed_app, monkeypatch, requests_mock
):
    """Keep Cursor Azure catalog IDs on the first matching Azure account."""
    calls: list[str | None] = []

    def fake_forward(self, req, snapshot=None):
        calls.append(None if snapshot is None else snapshot.provider)
        return Response("azure-catalog", status=200)

    def fail_attempt(self, req, snapshot):
        raise AssertionError("catalog model IDs must not enter provider failover")

    monkeypatch.setattr("app.azure.adapter.AzureAdapter.forward", fake_forward)
    monkeypatch.setattr("app.azure.adapter.AzureAdapter.forward_attempt", fail_attempt)
    response = post(routed_app, model="gpt-5.4")
    azure = post(routed_app, path="/azure/v1/chat/completions", model="gpt-5.4")
    assert response.status_code == azure.status_code == 200
    assert calls == ["azure", "azure"]
    assert requests_mock.call_count == 0


def test_explicit_azure_remains_pinned(routed_app, requests_mock, monkeypatch):
    """Keep explicit Azure requests pinned despite retryable errors."""
    requests_mock.post(AZURE, status_code=503)
    response = post(routed_app, path="/azure/v1/chat/completions")
    assert response.status_code == 503
    assert requests_mock.call_count == 1


def test_models_route_does_not_call_upstream(routed_app, requests_mock):
    """List the logical tenant model without upstream requests."""
    response = routed_app.test_client().get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    assert response.json["data"][0]["id"] == "cursor-acme-model"
    assert requests_mock.call_count == 0
