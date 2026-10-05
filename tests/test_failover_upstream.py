"""Bounded upstream preflight and safe streaming failures."""

import json
from unittest.mock import Mock

import pytest
import requests
from flask import g
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
        (402, True),
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


def test_http_failure_keeps_safe_structured_diagnostics_without_provider_text():
    """Retain exact structured error identifiers but never raw error payloads."""
    response = upstream(
        [
            json.dumps(
                {
                    "error": {
                        "code": "invalid_request_error",
                        "type": "invalid_request_error",
                        "param": "messages[0].content",
                        "message": "private prompt fragment must not be logged",
                        "metadata": {
                            "limit_source": "openrouter_key_limit",
                            "raw": "secret raw vendor payload",
                        },
                    }
                }
            ).encode()
        ],
        status=400,
    )
    response.headers["x-request-id"] = "req_1234567890abcdef12345678"

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openrouter")

    failure = caught.value
    assert failure.provider_error_code == "invalid_request_error"
    assert failure.provider_error_type == "invalid_request_error"
    assert failure.provider_error_param == "messages[0].content"
    assert failure.provider_limit_source == "openrouter_key_limit"
    assert failure.provider_request_id == "req_1234567890abcdef12345678"
    assert "private prompt fragment" not in repr(failure)
    assert "secret raw vendor payload" not in repr(failure)


def test_indexed_provider_parameter_path_is_retained():
    """Retain recognized parameter paths with valid message indexes."""
    response = upstream(
        [
            json.dumps(
                {
                    "error": {
                        "code": "invalid_request_error",
                        "param": "messages[7].content",
                    }
                }
            ).encode()
        ],
        status=400,
    )

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openai")

    assert caught.value.provider_error_param == "messages[7].content"


def test_sensitive_shaped_provider_error_identifiers_are_not_retained():
    """Do not log arbitrary token-like values merely because their charset is safe."""
    response = upstream(
        [
            json.dumps(
                {
                    "error": {
                        "code": "sk-secret-credential",
                        "type": "sk-secret-credential",
                        "param": "model.sk-secret-credential",
                    }
                }
            ).encode()
        ],
        status=400,
    )
    response.headers["x-request-id"] = "sk-secret-credential"

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openai")

    failure = caught.value
    assert failure.provider_error_code is None
    assert failure.provider_error_type is None
    assert failure.provider_error_param is None
    assert failure.provider_request_id is None
    assert "sk-secret-credential" not in repr(failure)


@pytest.mark.parametrize("header", ["apim-request-id", "x-ms-request-id"])
def test_azure_request_id_header_is_captured(header):
    """Retain Azure's support correlation header in safe diagnostics."""
    response = upstream(
        [json.dumps({"error": {"code": "invalid_request_error"}}).encode()],
        status=400,
    )
    response.headers[header] = "123e4567-e89b-12d3-a456-426614174000"

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="azure")

    assert caught.value.provider_request_id == "123e4567-e89b-12d3-a456-426614174000"


def test_sse_failure_logs_safe_provider_diagnostics(app, mocker):
    """Log structured failure identifiers discovered in a streamed SSE error."""
    response = upstream(
        [
            event({"choices": [{"delta": {"content": "partial"}}]}),
            event(
                {
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "private provider prose",
                        "metadata": {"limit_source": "openrouter_in_flight_budget"},
                    }
                },
                "error",
            ),
        ]
    )
    warning = mocker.patch.object(app.logger, "warning")

    with app.test_request_context("/v1/chat/completions"):
        g.proxy_request_id = "proxy-correlation-1"
        body = b"".join(
            failover_upstream.chat_stream(
                failover_upstream.prepare_upstream(response),
                "cursor-model",
                provider="openrouter",
            )
        )

    assert warning.call_count == 1
    template, *arguments = warning.call_args.args
    formatted_warning = template % tuple(arguments)
    assert "request_id=proxy-correlation-1" in formatted_warning
    assert "provider_error_code=rate_limit_exceeded" in formatted_warning
    assert "provider_limit_source=openrouter_in_flight_budget" in formatted_warning
    assert "private provider prose" not in formatted_warning
    assert b"private provider prose" not in body


def test_stream_transport_failure_logs_exception_type_without_exception_text(
    app, mocker
):
    """Log the transport failure class without exposing URLs or exception text."""
    response = upstream([])

    def interrupted_stream():
        yield event({"choices": [{"delta": {"content": "partial"}}]})
        raise requests.ReadTimeout("secret-upstream-url-and-token")

    response.iter_content.return_value = interrupted_stream()
    warning = mocker.patch.object(app.logger, "warning")

    with app.test_request_context("/v1/chat/completions"):
        g.proxy_request_id = "proxy-correlation-2"
        body = b"".join(
            failover_upstream.chat_stream(
                failover_upstream.prepare_upstream(response),
                "cursor-model",
                provider="openai",
            )
        )

    assert warning.call_count == 1
    template, *arguments = warning.call_args.args
    formatted_warning = template % tuple(arguments)
    assert "request_id=proxy-correlation-2" in formatted_warning
    assert "provider=openai" in formatted_warning
    assert "error_type=ReadTimeout" in formatted_warning
    assert "secret-upstream-url-and-token" not in formatted_warning
    assert b"stream_interrupted" in body


def test_payment_required_sse_error_is_retryable_without_opening_quota_breaker():
    """Allow a payment error to cascade while keeping ambiguous quota terminal."""
    response = upstream([event({"error": {"code": "payment_required"}})])

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openrouter")

    assert caught.value.status == 402
    assert caught.value.retryable
    assert caught.value.classification.category == "terminal"


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
