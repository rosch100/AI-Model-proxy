"""Tests for safe retries of transient Azure responses."""

import json
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


def _sse_event(event_name, payload):
    """Build one Azure Responses SSE event."""
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


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
    """Stop after five retries with backoff when Azure keeps throttling."""
    rate_limited = {
        "status_code": 429,
        "json": {"error": {"code": "rate_limit_exceeded"}},
    }
    requests_mock.post(AZURE_RESPONSES_URL, [rate_limited] * 6)
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr("random.uniform", lambda lower, upper: upper / 2)

    response = AzureAdapter().forward(_request(app))

    assert response.status_code == 429
    assert requests_mock.call_count == 6
    assert [call.args[0] for call in sleep.call_args_list] == [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ]


def test_token_rate_limit_retries_after_azure_delay_over_30_seconds(
    app, requests_mock, monkeypatch
):
    """Retry Azure token throttles using retry-after-ms values up to one minute."""
    requests_mock.post(
        AZURE_RESPONSES_URL,
        [
            {
                "status_code": 429,
                "headers": {"retry-after-ms": "31000"},
                "json": {
                    "error": {
                        "message": (
                            "Your requests to gpt-6-luna for gpt-6-luna-api in "
                            "germanywestcentral have exceeded token rate limit."
                        ),
                        "type": "too_many_requests",
                        "code": "rate_limit_exceeded",
                    }
                },
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
    sleep.assert_called_once_with(31.0)


def test_rate_limit_retry_delay_is_capped_at_sixty_seconds(
    app, requests_mock, monkeypatch
):
    """Cap Azure retry-after values at the configured maximum wait."""
    requests_mock.post(
        AZURE_RESPONSES_URL,
        [
            {
                "status_code": 429,
                "headers": {"retry-after-ms": "90000"},
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
    sleep.assert_called_once_with(60.0)


def test_stream_rate_limit_retries_before_emitting_output(
    app, requests_mock, monkeypatch
):
    """Retry a response.failed rate limit while the downstream stream is empty."""
    failed_event = _sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "The token rate limit was exceeded.",
                }
            },
        },
    )
    recovered_stream = b"".join(
        (
            _sse_event(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "recovered"},
            ),
            _sse_event(
                "response.completed",
                {"type": "response.completed", "response": {"usage": {}}},
            ),
        )
    )
    requests_mock.post(
        AZURE_RESPONSES_URL,
        [
            {
                "status_code": 200,
                "headers": {
                    "content-type": "text/event-stream",
                    "retry-after-ms": "25",
                },
                "content": failed_event,
            },
            {
                "status_code": 200,
                "headers": {"content-type": "text/event-stream"},
                "content": recovered_stream,
            },
        ],
    )
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr("random.uniform", lambda lower, upper: upper / 2)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert response.status_code == 200
    assert requests_mock.call_count == 2
    assert b"recovered" in response_body
    assert b"rate_limit_exceeded" not in response_body
    sleep.assert_called_once_with(0.025)


def test_stream_rate_limit_does_not_retry_after_output_was_emitted(
    app, requests_mock, monkeypatch
):
    """Do not restart an SSE stream after sending partial output to Cursor."""
    partial_stream = b"".join(
        (
            _sse_event(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "partial"},
            ),
            _sse_event(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": "The token rate limit was exceeded.",
                        }
                    },
                },
            ),
        )
    )
    requests_mock.post(
        AZURE_RESPONSES_URL,
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=partial_stream,
    )
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert b"partial" in response_body
    assert b"rate_limit_exceeded" in response_body
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_stream_rate_limit_retries_are_bounded(app, requests_mock, monkeypatch):
    """Limit repeated SSE rate-limit failures to the configured retry budget."""
    failed_event = _sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "The token rate limit was exceeded.",
                }
            },
        },
    )
    requests_mock.post(
        AZURE_RESPONSES_URL,
        [
            {
                "status_code": 200,
                "headers": {"content-type": "text/event-stream"},
                "content": failed_event,
            }
        ]
        * 6,
    )
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr("random.uniform", lambda lower, upper: upper / 2)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert b"rate_limit_exceeded" in response_body
    assert requests_mock.call_count == 6
    assert sleep.call_count == 5


def test_http_and_stream_rate_limit_retries_share_budget(
    app, requests_mock, monkeypatch
):
    """Do not exceed five total retries across HTTP and SSE rate limits."""
    failed_event = _sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "The token rate limit was exceeded.",
                }
            },
        },
    )
    responses = [
        {
            "status_code": 429,
            "headers": {"retry-after-ms": "25"},
            "json": {"error": {"code": "rate_limit_exceeded"}},
        }
        for _ in range(5)
    ]
    responses.append(
        {
            "status_code": 200,
            "headers": {"content-type": "text/event-stream"},
            "content": failed_event,
        }
    )
    requests_mock.post(AZURE_RESPONSES_URL, responses)
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert b"rate_limit_exceeded" in response_body
    assert requests_mock.call_count == 6
    assert sleep.call_count == 5
