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


def test_rate_limit_retry_floors_short_retry_after_ms(app, requests_mock, monkeypatch):
    """Do not honor sub-15s Azure hints that thrash the TPM window."""
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
    sleep.assert_called_once_with(15.0)


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
    """Stop after eight retries with floored exponential backoff."""
    rate_limited = {
        "status_code": 429,
        "json": {"error": {"code": "rate_limit_exceeded"}},
    }
    requests_mock.post(AZURE_RESPONSES_URL, [rate_limited] * 9)
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr("random.uniform", lambda lower, upper: lower)

    response = AzureAdapter().forward(_request(app))

    assert response.status_code == 429
    assert requests_mock.call_count == 9
    assert [call.args[0] for call in sleep.call_args_list] == [
        15.0,
        15.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
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


def test_rate_limit_retry_clamps_azure_retry_after_to_one_minute(
    app, requests_mock, monkeypatch
):
    """Wait the configured maximum when Azure's delay is longer than one minute."""
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

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert response.status_code == 200
    assert requests_mock.call_count == 2
    assert b"recovered" in response_body
    assert b"rate_limit_exceeded" not in response_body
    sleep.assert_called_once_with(15.0)


def test_stream_error_event_rate_limit_is_retried(app, requests_mock, monkeypatch):
    """Retry bare SSE error events that carry rate_limit_exceeded before output."""
    failed_event = _sse_event(
        "error",
        {
            "type": "error",
            "code": "rate_limit_exceeded",
            "message": "The token rate limit was exceeded.",
            "retry_after_ms": 20000,
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
                "headers": {"content-type": "text/event-stream"},
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

    response = AzureAdapter().forward(_request(app))

    assert b"recovered" in response.get_data()
    assert b"rate_limit_exceeded" not in response.get_data()
    sleep.assert_called_once_with(20.0)


def test_stream_rate_limit_without_retry_after_uses_minimum_backoff(
    app, requests_mock, monkeypatch
):
    """Wait at least 15s when Azure omits retry-after on a streamed 200 failure."""
    failed_event = _sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": (
                        "Your requests to gpt-6-luna for gpt-6-luna-api in "
                        "germanywestcentral have exceeded token rate limit."
                    ),
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
                "headers": {"content-type": "text/event-stream"},
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
    monkeypatch.setattr("random.uniform", lambda lower, upper: lower)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert response.status_code == 200
    assert b"recovered" in response_body
    assert b"rate_limit_exceeded" not in response_body
    sleep.assert_called_once_with(15.0)


def test_stream_rate_limit_honors_retry_after_on_sse_error(
    app, requests_mock, monkeypatch
):
    """Read retry-after-ms from the failed SSE payload when HTTP headers omit it."""
    failed_event = _sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "The token rate limit was exceeded.",
                    "retry_after_ms": 40000,
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
                "headers": {"content-type": "text/event-stream"},
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

    response = AzureAdapter().forward(_request(app))

    assert b"recovered" in response.get_data()
    sleep.assert_called_once_with(40.0)


def test_shared_cooldown_blocks_a_second_request(app, requests_mock, monkeypatch):
    """A later request waits out another stream's Azure token cooldown."""
    from app.azure import adapter as azure_adapter

    azure_adapter._rate_limit_not_before[f"{AZURE_RESPONSES_URL}|gpt-6-luna"] = 1000.0
    monkeypatch.setattr("app.azure.adapter.time.monotonic", lambda: 990.0)
    sleep = Mock()
    monkeypatch.setattr("app.azure.adapter.time.sleep", sleep)
    requests_mock.post(
        AZURE_RESPONSES_URL,
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=b"",
    )

    response = AzureAdapter().forward(_request(app))

    assert response.status_code == 200
    sleep.assert_called_once_with(10.0)


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
        * 9,
    )
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr("random.uniform", lambda lower, upper: lower)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert b"rate_limit_exceeded" in response_body
    assert requests_mock.call_count == 9
    assert sleep.call_count == 8


def test_http_and_stream_rate_limit_retries_share_budget(
    app, requests_mock, monkeypatch
):
    """Do not exceed eight total retries across HTTP and SSE rate limits."""
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
        for _ in range(8)
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
    assert requests_mock.call_count == 9
    assert sleep.call_count == 8
