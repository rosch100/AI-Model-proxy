"""Bounded upstream preflight and safe streaming failures."""

import json
from unittest.mock import Mock

import pytest
import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError, ReadTimeoutError

from app.providers import failover_upstream


def event(payload, name=None):
    """Encode a JSON payload as an optionally named SSE event."""
    prefix = f"event: {name}\n" if name else ""
    return (prefix + f"data: {json.dumps(payload)}\n\n").encode()


def upstream(chunks, status=200):
    """Build an upstream response with the given chunks and status."""
    response = Mock()
    response.status_code = status
    response.headers = {"Content-Type": "text/event-stream"}
    response.iter_content.return_value = iter(chunks)
    return response


def test_preview_rate_limit_after_lifecycle_before_output():
    """Allow retry after a rate limit before user-visible output."""
    response = upstream(
        [
            event({"type": "response.created"}, "response.created"),
            event(
                {
                    "response": {
                        "error": {"code": "rate_limit_exceeded", "message": "busy"}
                    }
                },
                "response.failed",
            ),
        ]
    )
    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response)
    assert caught.value.status == 429
    assert caught.value.retryable
    response.close.assert_called_once()


def test_preview_preserves_fragmented_success_exactly():
    """Replay a successful fragmented stream without changing its bytes."""
    content = event(
        {"choices": [{"delta": {"content": "hello"}}], "model": "provider-model"}
    )
    chunks = [content[:8], content[8:], b"data: [DONE]\n\n"]
    response = upstream(chunks)
    prepared = failover_upstream.prepare_upstream(response)
    assert b"".join(prepared.iter_content(chunk_size=128)) == b"".join(chunks)
    response.close.assert_not_called()


def test_late_error_is_not_used_for_failover():
    """Pass through an error that follows user-visible output."""
    chunks = [
        event({"choices": [{"delta": {"content": "partial"}}]}),
        event({"error": {"code": 429}}),
    ]
    prepared = failover_upstream.prepare_upstream(upstream(chunks))
    assert b"".join(prepared.iter_content(chunk_size=128)) == b"".join(chunks)


@pytest.mark.parametrize(
    "status,retryable",
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (408, True),
        (429, True),
        (500, True),
        (503, True),
    ],
)
def test_http_error_retains_classification(status, retryable):
    """Retain HTTP error status and its retry classification."""
    response = upstream([], status)
    response.json.return_value = {
        "error": {"message": "failure", "code": "upstream_error"}
    }
    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response)
    assert caught.value.status == status
    assert caught.value.retryable is retryable
    response.close.assert_called_once()


def test_openrouter_inflight_sse_metadata_is_retryable_transient():
    """Recognize structured OpenRouter in-flight metadata without message text."""
    response = upstream(
        [
            event(
                {
                    "error": {
                        "type": "provider_error",
                        "message": "do not use prose",
                        "metadata": {"limit_source": "openrouter_in_flight_budget"},
                    }
                },
                "error",
            )
        ]
    )

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openrouter")

    assert caught.value.classification.category == "transient"
    assert caught.value.retryable


def test_quota_http_failure_is_reported_by_route_owner():
    """Preflight raises structured HTTP errors without mutating the breaker."""
    response = upstream(
        [json.dumps({"error": {"code": "insufficient_quota"}}).encode()], 429
    )
    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openai")

    assert caught.value.classification.category == "quota_exhausted"


def test_upstream_error_response_exposes_only_safe_retry_after_header(app):
    """Return structured retry guidance without provider error metadata."""
    response = upstream(
        [
            json.dumps(
                {
                    "error": {
                        "code": "insufficient_quota",
                        "message": "private provider message",
                        "metadata": {"secret": "private-provider-metadata"},
                    }
                }
            ).encode()
        ],
        status=429,
    )
    response.headers["Retry-After"] = "12"
    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(
            response,
            provider="openai",
            settings={},
        )

    client_response = caught.value.response()

    assert caught.value.classification.retry_after_seconds == 12
    assert client_response.headers["Retry-After"] == "12"
    assert b"private provider message" not in client_response.get_data()
    assert b"private-provider-metadata" not in client_response.get_data()


def test_auth_sse_error_is_terminal():
    """Treat an SSE authentication failure as nonretryable."""
    response = upstream([event({"error": {"code": 401, "message": "invalid token"}})])
    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response)
    assert caught.value.status == 401
    assert not caught.value.retryable


def test_preview_byte_limit_releases_stream_without_growing_buffer(monkeypatch):
    """Release buffered bytes when preflight reaches its byte limit."""
    monkeypatch.setattr(failover_upstream, "MAX_PREFLIGHT_BYTES", 16)
    chunks = [b": heartbeat\n\n" * 2, event({"error": {"code": 429}})]
    prepared = failover_upstream.prepare_upstream(upstream(chunks))
    assert b"".join(prepared.iter_content(chunk_size=128)) == b"".join(chunks)


def test_new_connection_error_is_safe_but_wrapped_read_error_is_not():
    """Retry refused connections but reject ambiguous transport outcomes."""
    refused = requests.ConnectionError(
        MaxRetryError(None, "/", NewConnectionError(None, "refused"))
    )
    assert failover_upstream.transport_failure(refused).retryable
    uncertain = requests.ConnectionError(ReadTimeoutError(None, "/", "interrupted"))
    assert not failover_upstream.transport_failure(uncertain).retryable
    assert not failover_upstream.transport_failure(
        requests.ConnectionError("unknown outcome")
    ).retryable


def test_read_timeout_closes_upstream_without_marking_safe_retry():
    """Close the upstream on read timeout without allowing a retry."""

    def broken():
        raise requests.ReadTimeout("ambiguous outcome")
        yield b""

    response = upstream([])
    response.iter_content.return_value = broken()
    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response)
    assert caught.value.status == 502
    assert not caught.value.retryable
    response.close.assert_called_once()
