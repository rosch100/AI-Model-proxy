"""Tests for safe retries of transient Azure responses."""

from unittest.mock import Mock

import pytest

from app.azure.adapter import AzureAdapter

AZURE_RESPONSES_URL = "https://test-resource.openai.azure.com/openai/v1/responses"


def _request(app):
    return app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        json={
            "model": "gpt-6-luna",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer test-service-api-key"},
    ).request


def test_rate_limit_retry_honors_retry_after_ms(app, requests_mock, monkeypatch):
    """Retry a transient 429 only after Azure's recommended delay."""
    requests_mock.post(
        AZURE_RESPONSES_URL,
        [
            {
                "status_code": 429,
                "headers": {"retry-after-ms": "25"},
                "json": {"error": {"code": "rate_limit_exceeded"}},
            },
            {
                "status_code": 200,
                "headers": {"content-type": "text/event-stream"},
                "content": b"",
            },
        ],
    )
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)

    response = AzureAdapter().forward(_request(app))

    assert response.status_code == 200
    assert requests_mock.call_count == 2
    sleep.assert_called_once_with(0.025)


@pytest.mark.parametrize("error_code", ("insufficient_quota", "quota_exceeded"))
def test_quota_429_is_not_retried(app, requests_mock, monkeypatch, error_code):
    """Do not repeat 429 errors that require quota changes rather than waiting."""
    requests_mock.post(
        AZURE_RESPONSES_URL,
        status_code=429,
        headers={"retry-after-ms": "25"},
        json={"error": {"code": error_code}},
    )
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)

    response = AzureAdapter().forward(_request(app))

    assert response.status_code == 429
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_rate_limit_retries_are_bounded_with_exponential_backoff(
    app, requests_mock, monkeypatch
):
    """Stop after the configured attempt bound when Azure keeps throttling."""
    rate_limited = {
        "status_code": 429,
        "json": {"error": {"code": "rate_limit_exceeded"}},
    }
    requests_mock.post(AZURE_RESPONSES_URL, [rate_limited] * 3)
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr("random.uniform", lambda lower, upper: upper / 2)

    response = AzureAdapter().forward(_request(app))

    assert response.status_code == 429
    assert requests_mock.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]
