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


def test_http_rate_limit_retries_until_azure_recovers(app, requests_mock, monkeypatch):
    """Continue after more than eight 429s until Azure accepts the request."""
    rate_limited = {
        "status_code": 429,
        "json": {"error": {"code": "rate_limit_exceeded"}},
    }
    requests_mock.post(
        AZURE_RESPONSES_URL,
        [rate_limited] * 10
        + [
            {
                "status_code": 200,
                "headers": {"content-type": "text/event-stream"},
                "content": _sse_event(
                    "response.completed",
                    {"type": "response.completed", "response": {"usage": {}}},
                ),
            }
        ],
    )
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr("random.uniform", lambda lower, upper: lower)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert response.status_code == 200
    assert b"stream_error" not in response_body
    assert requests_mock.call_count == 11
    assert sleep.call_count == 10
    assert all(call.args[0] <= 60.0 for call in sleep.call_args_list)


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


def test_nested_stream_error_rate_limit_is_retried(app, requests_mock, monkeypatch):
    """Retry a rate limit carried in Azure's nested error SSE object."""
    failed_event = _sse_event(
        "error",
        {
            "type": "error",
            "error": {
                "code": "rate_limit_exceeded",
                "message": "The token rate limit was exceeded.",
                "retry_after_ms": 20000,
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
    response_body = response.get_data()

    assert b"recovered" in response_body
    assert b"rate_limit_exceeded" not in response_body
    assert requests_mock.call_count == 2
    sleep.assert_called_once_with(20.0)


def test_nested_non_retryable_stream_error_is_reported(app, requests_mock, monkeypatch):
    """Surface a non-retryable nested Azure error instead of silently ending."""
    requests_mock.post(
        AZURE_RESPONSES_URL,
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=_sse_event(
            "error",
            {
                "type": "error",
                "error": {
                    "code": "insufficient_quota",
                    "message": "Azure quota is exhausted.",
                },
            },
        ),
    )
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert b"insufficient_quota" in response_body
    assert b"Azure quota is exhausted." in response_body
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_nested_stream_error_logs_azure_error_details(
    app, requests_mock, monkeypatch, capsys
):
    """Log nested Azure error details when an error follows visible output."""
    partial_stream = b"".join(
        (
            _sse_event(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "partial"},
            ),
            _sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "code": "insufficient_quota",
                        "message": "Azure quota is exhausted.",
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
    note_empty_precursor = Mock()
    monkeypatch.setattr(
        AzureAdapter,
        "note_empty_stream_error_precursor",
        staticmethod(note_empty_precursor),
    )

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()
    log_output = capsys.readouterr().out

    assert b"partial" in response_body
    assert b"insufficient_quota" in response_body
    assert "code=insufficient_quota message=Azure quota is exhausted." in log_output
    assert requests_mock.call_count == 1
    sleep.assert_not_called()
    note_empty_precursor.assert_not_called()


def test_late_stream_rate_limit_does_not_record_duplicate_empty_cooldown(
    app, requests_mock, monkeypatch
):
    """Record the actual rate-limit hint once, not an empty-error fallback."""
    partial_stream = b"".join(
        (
            _sse_event(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "partial"},
            ),
            _sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "The token rate limit was exceeded.",
                        "retry_after_ms": 40000,
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
    note_stream_rate_limit = Mock()
    note_empty_precursor = Mock()
    monkeypatch.setattr(
        AzureAdapter, "note_stream_rate_limit", staticmethod(note_stream_rate_limit)
    )
    monkeypatch.setattr(
        AzureAdapter,
        "note_empty_stream_error_precursor",
        staticmethod(note_empty_precursor),
    )

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert b"partial" in response_body
    assert b"rate_limit_exceeded" in response_body
    assert requests_mock.call_count == 1
    note_stream_rate_limit.assert_called_once()
    note_empty_precursor.assert_not_called()


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


def test_stream_rate_limit_after_partial_output_does_not_retry(
    app, requests_mock, monkeypatch
):
    """Do not restart after partial output has already reached Cursor."""
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
                "content": partial_stream,
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
    assert b"partial" in response_body
    assert b"recovered" not in response_body
    assert b"rate_limit_exceeded" in response_body
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_empty_stream_error_followed_by_completion_does_not_retry(
    app, requests_mock, monkeypatch
):
    """Do not retry an empty error precursor after Azure completes successfully."""
    completed_stream = b"".join(
        (
            _sse_event(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "answer"},
            ),
            _sse_event("error", {"type": "error"}),
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
                "content": completed_stream,
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
    response_body = response.get_data()

    assert b"answer" in response_body
    assert b"stream_error" not in response_body
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_empty_stream_error_after_partial_output_does_not_retry(
    app, requests_mock, monkeypatch
):
    """Preserve partial output and report an incomplete stream without retrying."""
    partial_stream = b"".join(
        (
            _sse_event(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "partial"},
            ),
            _sse_event("error", {"type": "error"}),
        )
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
                "content": partial_stream,
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

    assert b"partial" in response_body
    assert b"recovered" not in response_body
    assert b"stream_error" in response_body
    assert b"incomplete" in response_body
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_stream_rate_limit_retries_until_azure_recovers(
    app, requests_mock, monkeypatch
):
    """Continue after more than eight streamed rate limits until recovery."""
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
        * 10
        + [
            {
                "status_code": 200,
                "headers": {"content-type": "text/event-stream"},
                "content": _sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "delta": "recovered",
                    },
                )
                + _sse_event(
                    "response.completed",
                    {"type": "response.completed", "response": {"usage": {}}},
                ),
            }
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
    assert requests_mock.call_count == 11
    assert sleep.call_count == 10


def test_http_and_stream_rate_limits_continue_past_eight_retries(
    app, requests_mock, monkeypatch
):
    """Share backoff state but continue retrying across HTTP and SSE limits."""
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
        for _ in range(10)
    ]
    responses.extend(
        [
            {
                "status_code": 200,
                "headers": {"content-type": "text/event-stream"},
                "content": failed_event,
            },
            {
                "status_code": 200,
                "headers": {"content-type": "text/event-stream"},
                "content": _sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "delta": "recovered",
                    },
                )
                + _sse_event(
                    "response.completed",
                    {"type": "response.completed", "response": {"usage": {}}},
                ),
            },
        ]
    )
    requests_mock.post(AZURE_RESPONSES_URL, responses)
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)

    response = AzureAdapter().forward(_request(app))
    response_body = response.get_data()

    assert response.status_code == 200
    assert b"recovered" in response_body
    assert b"rate_limit_exceeded" not in response_body
    assert requests_mock.call_count == 12
    assert sleep.call_count == 11


def test_empty_error_precursor_avoids_stream_error_and_retries(
    app, requests_mock, monkeypatch
):
    """Absorb empty SSE error noise, cool peers early, then retry on response.failed."""
    from app.azure import adapter as azure_adapter

    failed_stream = b"".join(
        (
            _sse_event("error", {"type": "error", "code": "", "message": ""}),
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
                "content": failed_stream,
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
    precursor = Mock(wraps=AzureAdapter.note_empty_stream_error_precursor)
    monkeypatch.setattr(
        AzureAdapter,
        "note_empty_stream_error_precursor",
        staticmethod(precursor),
    )

    response = AzureAdapter().forward(_request(app))

    assert b"recovered" in response.get_data()
    assert b"rate_limit_exceeded" not in response.get_data()
    precursor.assert_called_once()
    assert sleep.call_count == 1
    assert sleep.call_args.args[0] == 15.0
    assert azure_adapter._rate_limit_not_before


def test_nested_empty_error_alone_before_output_triggers_retry(
    app, requests_mock, monkeypatch
):
    """Retry an empty nested error event before it can look like success."""
    empty_error = _sse_event("error", {"type": "error", "error": {}})
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
                "content": empty_error,
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

    assert b"recovered" in response_body
    assert b"stream_error" not in response_body
    assert requests_mock.call_count == 2
    sleep.assert_called_once_with(15.0)


def test_empty_error_alone_before_output_triggers_retry(
    app, requests_mock, monkeypatch
):
    """Retry when the upstream stream ends on an empty error without response.failed."""
    empty_error = _sse_event("error", {"type": "error"})
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
                "content": empty_error,
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

    assert b"recovered" in response.get_data()
    assert requests_mock.call_count == 2
    sleep.assert_called_once_with(15.0)
