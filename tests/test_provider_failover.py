"""Root inference failover using real adapters and tenant-owned model targets."""

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests
from flask import request
from sqlalchemy import select

from app.persistence.admin_ops import (
    activate_provider_profile,
    create_provider_profile,
    reorder_provider_profile,
    replace_catalog_entries,
)
from app.persistence.models import (
    InferenceActivityEvent,
    ProviderAttemptEvent,
    ProviderCircuitState,
    ProviderProfile,
    Tenant,
)
from app.persistence.provider_circuit_breaker import (
    ProviderCircuitBreakerStore,
    ProviderCircuitStoreError,
)
from app.providers.routing import forward_tenant_route

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
    """Configure a tenant route through Azure, OpenAI, and OpenRouter."""
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
            if provider != "deepseek":
                activate_provider_profile(session, tenant, profile.id, "ada")
    return admin_app


def activate_deepseek_profile(app):
    """Append the configured DeepSeek profile to a test tenant's route."""
    database = app.extensions["database"]
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == "acme",
                ProviderProfile.provider == "deepseek",
            )
        )
        activate_provider_profile(session, tenant, profile.id, "ada")


def post(app, path="/v1/chat/completions", model="cursor-acme-model"):
    """Submit an authenticated chat request for the selected logical model."""
    return app.test_client().post(
        path,
        headers=AUTH,
        json={"model": model, "messages": [{"role": "user", "content": "Hi"}]},
        buffered=True,
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
    with routed_app.extensions["database"].sessions() as session:
        attempts = tuple(
            session.scalars(
                select(ProviderAttemptEvent).order_by(ProviderAttemptEvent.id)
            )
        )
    assert [
        (attempt.provider, attempt.outcome, attempt.status_code) for attempt in attempts
    ] == [
        ("azure", "failure", status),
        ("openai", "success", 200),
    ]


def test_quota_paused_profile_is_skipped_before_attempt_and_uses_fallback(
    routed_app, requests_mock
):
    """A durable quota breaker skips a profile without contacting its upstream."""
    database = routed_app.extensions["database"]
    snapshot = database.get_proxy_snapshot_by_api_key("cursor-key")
    profile = snapshot.profiles[0]
    breaker = ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
    breaker.open_quota(
        breaker.scope(snapshot.id, profile.provider, "profile", profile.profile_id)
    )
    requests_mock.post(
        OPENAI,
        content=successful_chat("gpt-5.4"),
        headers={"Content-Type": "text/event-stream"},
    )

    response = post(routed_app)

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [OPENAI]
    with database.sessions() as session:
        attempts = tuple(
            session.scalars(
                select(ProviderAttemptEvent).order_by(ProviderAttemptEvent.id)
            )
        )
    assert [attempt.provider for attempt in attempts] == ["openai"]


def test_http_quota_failure_is_recorded_once_and_pauses_next_request(
    routed_app, requests_mock
):
    """One upstream quota response advances backoff once and blocks its retry."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        session.get(Tenant, "acme")
        for profile in session.scalars(
            select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
        ):
            profile.route_priority = None
        session.flush()
        openai_profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == "acme",
                ProviderProfile.provider == "openai",
            )
        )
        openai_profile.route_priority = 1

    requests_mock.post(
        OPENAI,
        status_code=429,
        json={"error": {"code": "insufficient_quota", "message": "private"}},
    )
    first = post(routed_app)

    snapshot = database.get_proxy_snapshot_by_api_key("cursor-key")
    profile = snapshot.profiles[0]
    breaker = ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
    scope = breaker.scope(snapshot.id, profile.provider, "profile", profile.profile_id)
    states = breaker.snapshots(snapshot.id, (scope,))
    second = post(routed_app)

    assert first.status_code == 429
    assert states[0].failure_count == 1
    assert second.status_code == 503
    assert requests_mock.call_count == 1


def test_transient_error_is_not_replaced_by_later_blocked_provider(
    routed_app, requests_mock
):
    """Return the attempted provider failure when a later profile is circuit-blocked."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        profiles = tuple(
            session.scalars(
                select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
            )
        )
        for profile in profiles:
            profile.route_priority = None
        session.flush()
        openai_profile = next(
            profile for profile in profiles if profile.provider == "openai"
        )
        openrouter_profile = next(
            profile for profile in profiles if profile.provider == "openrouter"
        )
        openai_profile.route_priority = 1
        openrouter_profile.route_priority = 2

    snapshot = database.get_proxy_snapshot_by_api_key("cursor-key")
    breaker = ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
    breaker.open_quota(
        breaker.scope(snapshot.id, "openrouter", "profile", openrouter_profile.id)
    )
    requests_mock.post(
        OPENAI, status_code=503, json={"error": {"code": "server_error"}}
    )

    response = post(routed_app)

    assert response.status_code == 503
    assert response.json["error"]["code"] != "provider_quota_unavailable"
    assert "Retry-After" not in response.headers
    assert [request.url for request in requests_mock.request_history] == [OPENAI]


def test_circuit_store_failure_fails_closed_with_service_unavailable(
    routed_app, requests_mock, monkeypatch
):
    """Do not call any provider when circuit state cannot be verified."""

    def unavailable(_self, _scopes, *, now=None):
        raise ProviderCircuitStoreError("database unavailable")

    monkeypatch.setattr(ProviderCircuitBreakerStore, "acquire", unavailable)

    response = post(routed_app)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert response.json["error"]["code"] == "provider_circuit_unavailable"
    assert requests_mock.call_count == 0
    with routed_app.extensions["database"].sessions() as session:
        assert session.scalar(select(ProviderAttemptEvent)) is None


def test_unexpected_forwarder_exception_releases_probe_and_completes_attempt(
    routed_app, monkeypatch
):
    """Unexpected adapter failures still resolve leases and attempt telemetry."""
    database = routed_app.extensions["database"]
    initial_snapshot = database.get_proxy_snapshot_by_api_key("cursor-key")
    selected_profile_id = initial_snapshot.profiles[0].profile_id
    with database.sessions.begin() as session:
        profiles = tuple(
            session.scalars(
                select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
            )
        )
        for profile in profiles:
            profile.route_priority = 1 if profile.id == selected_profile_id else None
    snapshot = database.get_proxy_snapshot_by_api_key("cursor-key")
    profile = snapshot.profiles[0]
    breaker = ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
    scope = breaker.scope(snapshot.id, profile.provider, "profile", profile.profile_id)
    breaker.open_quota(scope, now=datetime.now(timezone.utc) - timedelta(hours=2))

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("unexpected adapter failure")

    monkeypatch.setattr("app.providers.routing._forward_profile", fail_forward)
    with routed_app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        headers=AUTH,
        json={"model": "cursor-acme-model", "messages": []},
    ):
        with pytest.raises(RuntimeError, match="unexpected adapter failure"):
            forward_tenant_route(request, snapshot)

    with database.sessions() as session:
        attempt = session.scalar(
            select(ProviderAttemptEvent).where(
                ProviderAttemptEvent.profile_id == profile.profile_id
            )
        )
        state = session.scalar(select(ProviderCircuitState))

    assert attempt.outcome == "failure"
    assert attempt.status_code is None
    assert attempt.completed_at is not None
    assert state.lease_token is None


def test_deepseek_route_persists_attempt_activity_and_breaker(
    routed_app, requests_mock
):
    """Persist routed DeepSeek usage and breaker state through the public path."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == "acme",
                ProviderProfile.provider == "deepseek",
            )
        )
        assert profile is not None
        for existing in session.scalars(
            select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
        ):
            existing.route_priority = None
        session.flush()
        profile.route_priority = 1

    requests_mock.post(
        DEEPSEEK,
        content=successful_chat("deepseek-v4-flash"),
        headers={"Content-Type": "text/event-stream"},
    )
    response = post(routed_app)

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [DEEPSEEK]
    with database.sessions() as session:
        attempt = session.scalar(select(ProviderAttemptEvent))
        activity = session.scalar(select(InferenceActivityEvent))
    assert attempt.provider == "deepseek"
    assert attempt.outcome == "success"
    assert activity.provider == "deepseek"

    breaker = ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
    deepseek_scope = breaker.scope("acme", "deepseek", "profile", profile.id)
    assert breaker.open_quota(deepseek_scope).failure_count == 1
    with database.sessions() as session:
        state = session.scalar(select(ProviderCircuitState))
    assert state.provider == "deepseek"
    assert state.failure_count == 1


def test_late_openrouter_quota_error_pauses_following_request(
    routed_app, requests_mock
):
    """A late structured key limit opens the profile breaker without stream replay."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        for profile in session.scalars(
            select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
        ):
            if profile.provider != "openrouter":
                profile.route_priority = None
        session.flush()
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == "acme",
                ProviderProfile.provider == "openrouter",
            )
        )
        profile.route_priority = 1

    late_error = event(
        {
            "error": {
                "metadata": {
                    "limit_source": "openrouter_key_limit",
                    "secret": "must-not-be-forwarded",
                },
                "message": "private provider details",
            }
        }
    )
    requests_mock.post(
        OPENROUTER,
        content=event({"choices": [{"delta": {"content": "partial"}}]}) + late_error,
    )

    first = post(routed_app)
    second = post(routed_app)

    assert first.status_code == 200
    assert b"partial" in first.data
    assert b"must-not-be-forwarded" not in first.data
    assert b"private provider details" not in first.data
    assert second.status_code == 503
    assert requests_mock.call_count == 1
    with database.sessions() as session:
        attempts = tuple(
            session.scalars(
                select(ProviderAttemptEvent).order_by(ProviderAttemptEvent.id)
            )
        )
    assert len(attempts) == 1


def test_openrouter_inflight_budget_returns_retry_after_to_client(
    routed_app, requests_mock
):
    """A structured transient OpenRouter limit returns its safe retry guidance."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        for profile in session.scalars(
            select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
        ):
            profile.route_priority = None
        session.flush()
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == "acme",
                ProviderProfile.provider == "openrouter",
            )
        )
        profile.route_priority = 1

    requests_mock.post(
        OPENROUTER,
        status_code=402,
        headers={"Retry-After": "120"},
        json={
            "error": {
                "code": "provider_error",
                "metadata": {"limit_source": "openrouter_in_flight_budget"},
            }
        },
    )

    response = post(routed_app)

    assert response.status_code == 402
    assert response.headers["Retry-After"] == "120"
    assert requests_mock.call_count == 1
    with database.sessions() as session:
        assert session.scalar(select(ProviderCircuitState)) is None


def test_all_quota_paused_profiles_return_service_unavailable(
    routed_app, requests_mock
):
    """An entirely open route returns a bounded Retry-After without upstream calls."""
    database = routed_app.extensions["database"]
    snapshot = database.get_proxy_snapshot_by_api_key("cursor-key")
    breaker = ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
    for profile in snapshot.profiles:
        breaker.open_quota(
            breaker.scope(snapshot.id, profile.provider, "profile", profile.profile_id)
        )

    response = post(routed_app)

    assert response.status_code == 503
    assert int(response.headers["Retry-After"]) > 0
    assert "quota" in response.json["error"]["code"]
    assert requests_mock.call_count == 0
    with database.sessions() as session:
        assert session.scalar(select(ProviderAttemptEvent)) is None


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
    activate_deepseek_profile(routed_app)
    requests_mock.post(AZURE, status_code=503)
    requests_mock.post(OPENAI, status_code=429)
    requests_mock.post(OPENROUTER, status_code=500)
    requests_mock.post(DEEPSEEK, status_code=402)
    response = post(routed_app)
    assert response.status_code == 402
    assert requests_mock.call_count == 4


def test_deepseek_http_402_uses_next_profile_without_leaking_credentials(
    routed_app, requests_mock
):
    """Fail over after a DeepSeek 402 without leaking its credentials."""
    activate_deepseek_profile(routed_app)
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

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [
        AZURE,
        OPENAI,
        OPENROUTER,
        DEEPSEEK,
        OPENAI,
    ]
    assert (
        requests_mock.last_request.headers["Authorization"]
        == "Bearer openai-fallback-secret"
    )
    assert b"deepseek-secret" not in response.data


def test_openrouter_http_402_uses_next_provider(routed_app, requests_mock):
    """An OpenRouter payment error cascades to DeepSeek before client output."""
    activate_deepseek_profile(routed_app)
    requests_mock.post(AZURE, status_code=429)
    requests_mock.post(OPENAI, status_code=429)
    requests_mock.post(OPENROUTER, status_code=402, text="openrouter-secret")
    requests_mock.post(
        DEEPSEEK,
        content=successful_chat("deepseek-v4-flash"),
        additional_matcher=lambda request: request.headers["Authorization"]
        == "Bearer deepseek-secret",
    )

    response = post(routed_app)

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [
        AZURE,
        OPENAI,
        OPENROUTER,
        DEEPSEEK,
    ]
    assert b"openrouter-secret" not in response.data
    with routed_app.extensions["database"].sessions() as session:
        assert session.scalar(select(ProviderCircuitState)) is None


def test_openrouter_key_limit_fails_over_and_pauses_provider(routed_app, requests_mock):
    """Fail over on a key limit and skip that profile on the next request."""
    activate_deepseek_profile(routed_app)
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        profiles = tuple(
            session.scalars(
                select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
            )
        )
        openrouter = next(
            profile for profile in profiles if profile.provider == "openrouter"
        )
        deepseek = next(
            profile for profile in profiles if profile.provider == "deepseek"
        )
        for profile in profiles:
            profile.route_priority = None
        session.flush()
        openrouter.route_priority = 1
        deepseek.route_priority = 2

    requests_mock.post(
        OPENROUTER,
        status_code=402,
        json={"error": {"metadata": {"limit_source": "openrouter_key_limit"}}},
    )
    requests_mock.post(DEEPSEEK, content=successful_chat("deepseek-v4-flash"))

    first = post(routed_app)
    second = post(routed_app)

    assert first.status_code == second.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [
        OPENROUTER,
        DEEPSEEK,
        DEEPSEEK,
    ]
    with database.sessions() as session:
        state = session.scalar(select(ProviderCircuitState))
    assert state.provider == "openrouter"
    assert state.failure_category == "quota_exhausted"


def test_deepseek_http_429_uses_the_next_profile(routed_app, requests_mock):
    """Retry DeepSeek throttling and preserve the next profile's credential."""
    activate_deepseek_profile(routed_app)
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
    activate_deepseek_profile(routed_app)
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


def test_azure_catalog_model_uses_provider_cascade_after_rate_limit(
    routed_app, requests_mock
):
    """Fail over from a native Azure catalog ID using each provider's model."""
    requests_mock.post(AZURE, status_code=429)
    requests_mock.post(OPENAI, status_code=429)
    requests_mock.post(
        OPENROUTER,
        content=successful_chat("openrouter-configured-model"),
        headers={"Content-Type": "text/event-stream"},
    )
    with routed_app.extensions["database"].sessions.begin() as session:
        openai_profile = session.scalar(
            select(ProviderProfile).where(ProviderProfile.provider == "openai")
        )
        openai_profile.default_model = "openai-configured-model"
        openrouter_profile = session.scalar(
            select(ProviderProfile).where(ProviderProfile.provider == "openrouter")
        )
        openrouter_profile.default_model = "openrouter-configured-model"
        replace_catalog_entries(
            session, openai_profile, [("openai-configured-model", None)], None
        )
        replace_catalog_entries(
            session, openrouter_profile, [("openrouter-configured-model", None)], None
        )

    response = post(routed_app, model="gpt-5.4")

    assert response.status_code == 200
    assert b"success" in response.data
    assert [r.url for r in requests_mock.request_history] == [
        AZURE,
        OPENAI,
        OPENROUTER,
    ]
    assert [r.json()["model"] for r in requests_mock.request_history] == [
        "azure-deployment",
        "openai-configured-model",
        "openrouter-configured-model",
    ]


def test_native_model_does_not_bypass_higher_priority_provider(
    routed_app, requests_mock
):
    """Keep configured provider order even when a later catalog has the model."""
    with routed_app.extensions["database"].sessions.begin() as session:
        openai_profile = session.scalar(
            select(ProviderProfile).where(ProviderProfile.provider == "openai")
        )
        replace_catalog_entries(
            session,
            openai_profile,
            [("gpt-5.4", None), ("openai-native-model", None)],
            None,
        )

    requests_mock.post(AZURE, status_code=429)
    requests_mock.post(
        OPENAI,
        content=successful_chat("openai-native-model"),
        headers={"Content-Type": "text/event-stream"},
    )

    response = post(routed_app, model="openai-native-model")

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [AZURE, OPENAI]
    assert requests_mock.request_history[0].json()["model"] == "azure-deployment"
    assert requests_mock.request_history[1].json()["model"] == "openai-native-model"


def test_native_openrouter_model_does_not_bypass_higher_priority_providers(
    routed_app, requests_mock
):
    """A later provider's native model must not override route priority."""
    with routed_app.extensions["database"].sessions.begin() as session:
        openrouter_profile = session.scalar(
            select(ProviderProfile).where(ProviderProfile.provider == "openrouter")
        )
        replace_catalog_entries(
            session,
            openrouter_profile,
            [("anthropic/claude-sonnet-4", None), ("native-openrouter-model", None)],
            None,
        )

    requests_mock.post(AZURE, status_code=429)
    requests_mock.post(OPENAI, status_code=503)
    requests_mock.post(
        OPENROUTER,
        content=successful_chat("native-openrouter-model"),
        headers={"Content-Type": "text/event-stream"},
    )

    response = post(routed_app, model="native-openrouter-model")

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [
        AZURE,
        OPENAI,
        OPENROUTER,
    ]
    assert requests_mock.request_history[0].json()["model"] == "azure-deployment"
    assert requests_mock.request_history[1].json()["model"] == "gpt-5.4"
    assert requests_mock.request_history[2].json()["model"] == "native-openrouter-model"


def test_native_openrouter_model_falls_back_after_higher_priority_attempts_fail(
    routed_app, requests_mock
):
    """Preserve native-model selection and continue the ordered cascade on errors."""
    activate_deepseek_profile(routed_app)
    with routed_app.extensions["database"].sessions.begin() as session:
        openrouter_profile = session.scalar(
            select(ProviderProfile).where(ProviderProfile.provider == "openrouter")
        )
        replace_catalog_entries(
            session,
            openrouter_profile,
            [("anthropic/claude-sonnet-4", None), ("native-openrouter-model", None)],
            None,
        )

    requests_mock.post(AZURE, status_code=429)
    requests_mock.post(OPENAI, status_code=503)
    requests_mock.post(OPENROUTER, status_code=429)
    requests_mock.post(
        DEEPSEEK,
        content=successful_chat("deepseek-v4-flash"),
        headers={"Content-Type": "text/event-stream"},
    )

    response = post(routed_app, model="native-openrouter-model")

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [
        AZURE,
        OPENAI,
        OPENROUTER,
        DEEPSEEK,
    ]
    assert [request.json()["model"] for request in requests_mock.request_history] == [
        "azure-deployment",
        "gpt-5.4",
        "native-openrouter-model",
        "deepseek-v4-flash",
    ]


def test_azure_only_route_rejects_other_provider_catalog_model(
    routed_app, requests_mock, monkeypatch
):
    """An explicit Azure route validates the model against Azure profiles only."""

    def fail_forward(*args, **kwargs):
        raise AssertionError("Invalid Azure models must fail before forwarding")

    monkeypatch.setattr("app.providers.routing._forward_profile", fail_forward)
    response = post(
        routed_app,
        path="/azure/v1/chat/completions",
        model="anthropic/claude-sonnet-4",
    )

    assert response.status_code == 400
    assert requests_mock.call_count == 0


def test_openai_catalog_model_routes_without_azure_profile(routed_app, requests_mock):
    """Accept a configured provider model without requiring an Azure route."""
    with routed_app.extensions["database"].sessions.begin() as session:
        for profile in session.scalars(
            select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
        ):
            if profile.provider != "openai":
                profile.route_priority = None

    requests_mock.post(
        OPENAI,
        content=successful_chat("gpt-5.4"),
        headers={"Content-Type": "text/event-stream"},
    )

    response = post(routed_app, model="gpt-5.4")

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [OPENAI]
    assert requests_mock.request_history[0].json()["model"] == "gpt-5.4"


def test_azure_catalog_model_success_does_not_call_fallbacks(routed_app, requests_mock):
    """Keep the native Azure model when its first provider attempt succeeds."""
    requests_mock.post(
        AZURE,
        content=(
            event(
                {"type": "response.output_text.delta", "delta": "azure"},
                "response.output_text.delta",
            )
            + event(
                {"type": "response.completed", "response": {}},
                "response.completed",
            )
        ),
        headers={"Content-Type": "text/event-stream"},
    )

    response = post(routed_app, model="gpt-5.4")

    assert response.status_code == 200
    assert b"azure" in response.data
    assert requests_mock.call_count == 1
    assert requests_mock.request_history[0].json()["model"] == "azure-deployment"


def test_explicit_azure_uses_second_azure_profile_before_failing(
    routed_app, requests_mock
):
    """An explicit Azure route cascades among active Azure profiles only."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        secondary = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "azure",
            "Azure Backup",
            {
                "base_url": "https://backup-resource.openai.azure.com",
                "model_deployments": {"gpt-5.4": "backup-deployment"},
            },
            "gpt-5.4",
            "backup-secret",
            "ada",
        )
        replace_catalog_entries(
            session, secondary, [("gpt-5.4", "backup-deployment")], None
        )
        activate_provider_profile(session, tenant, secondary.id, "ada")
        reorder_provider_profile(session, tenant, secondary.id, "up", "ada")
        reorder_provider_profile(session, tenant, secondary.id, "up", "ada")
    snapshot = database.get_proxy_snapshot_by_api_key("cursor-key")
    assert [profile.profile_name for profile in snapshot.profiles] == [
        "azure",
        "Azure Backup",
        "openai",
        "openrouter",
    ]
    backup_url = "https://backup-resource.openai.azure.com/openai/v1/responses"
    requests_mock.post(AZURE, status_code=429)
    requests_mock.post(backup_url, status_code=503)
    requests_mock.post(OPENAI, status_code=200)

    response = post(routed_app, path="/azure/v1/chat/completions")

    assert response.status_code == 503
    assert [request.url for request in requests_mock.request_history] == [
        AZURE,
        backup_url,
    ]
    assert requests_mock.request_history[0].headers["api-key"] == "azure-secret"
    assert requests_mock.request_history[1].headers["api-key"] == "backup-secret"


def test_root_route_cascades_between_active_azure_profiles(routed_app, requests_mock):
    """Use a second Azure account before moving to the next provider family."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        secondary = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "azure",
            "Azure Backup",
            {
                "base_url": "https://backup-resource.openai.azure.com",
                "model_deployments": {"gpt-5.4": "backup-deployment"},
            },
            "gpt-5.4",
            "backup-secret",
            "ada",
        )
        replace_catalog_entries(
            session, secondary, [("gpt-5.4", "backup-deployment")], None
        )
        activate_provider_profile(session, tenant, secondary.id, "ada")
        reorder_provider_profile(session, tenant, secondary.id, "up", "ada")
        reorder_provider_profile(session, tenant, secondary.id, "up", "ada")

    backup_url = "https://backup-resource.openai.azure.com/openai/v1/responses"
    requests_mock.post(AZURE, status_code=429)
    requests_mock.post(backup_url, status_code=503)
    requests_mock.post(OPENAI, status_code=429)
    requests_mock.post(
        OPENROUTER,
        content=successful_chat("anthropic/claude-sonnet-4"),
        headers={"Content-Type": "text/event-stream"},
    )

    response = post(routed_app)

    assert response.status_code == 200
    assert [request.url for request in requests_mock.request_history] == [
        AZURE,
        backup_url,
        OPENAI,
        OPENROUTER,
    ]


@pytest.mark.parametrize(
    ("provider", "model", "expected_urls"),
    [
        (
            "openai",
            "gpt-5.5",
            [AZURE, OPENAI, OPENAI],
        ),
        (
            "openrouter",
            "google/gemini-2.5-flash",
            [AZURE, OPENAI, OPENROUTER, OPENROUTER],
        ),
    ],
)
def test_same_provider_profiles_are_independent_failover_candidates(
    routed_app, requests_mock, provider, model, expected_urls
):
    """Activate two accounts of one provider and fail over between their secrets."""
    database = routed_app.extensions["database"]
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        secondary = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            provider,
            f"{provider} Backup",
            {},
            model,
            f"{provider}-backup-secret",
            "ada",
        )
        replace_catalog_entries(session, secondary, [(model, None)], None)
        activate_provider_profile(session, tenant, secondary.id, "ada")
        if provider == "openai":
            reorder_provider_profile(session, tenant, secondary.id, "up", "ada")

    assert [
        profile.profile_name
        for profile in database.get_proxy_snapshot_by_api_key("cursor-key").profiles
    ] == (
        ["azure", "openai", "openai Backup", "openrouter"]
        if provider == "openai"
        else ["azure", "openai", "openrouter", "openrouter Backup"]
    )

    requests_mock.post(AZURE, status_code=429)
    if provider == "openai":
        requests_mock.post(
            OPENAI,
            response_list=[
                {"status_code": 429},
                {
                    "status_code": 200,
                    "headers": {"Content-Type": "text/event-stream"},
                    "content": successful_chat(model),
                },
            ],
        )
    else:
        requests_mock.post(OPENAI, status_code=429)
        requests_mock.post(
            OPENROUTER,
            response_list=[
                {"status_code": 429},
                {
                    "status_code": 200,
                    "headers": {"Content-Type": "text/event-stream"},
                    "content": successful_chat(model),
                },
            ],
        )

    response = post(routed_app)

    assert response.status_code == 200
    assert b"success" in response.data
    assert [request.url for request in requests_mock.request_history] == expected_urls
    assert requests_mock.request_history[-1].headers["Authorization"] == (
        f"Bearer {provider}-backup-secret"
    )
    assert requests_mock.request_history[-1].json()["model"] == model


def test_models_route_does_not_call_upstream(routed_app, requests_mock):
    """List the logical tenant model without upstream requests."""
    response = routed_app.test_client().get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    assert response.json["data"][0]["id"] == "cursor-acme-model"
    assert requests_mock.call_count == 0
