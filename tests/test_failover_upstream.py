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


def test_provider_diagnostics_include_unknown_safe_fields_and_rate_headers():
    """Retain useful structured provider diagnostics without raw error content."""
    response = upstream(
        [
            json.dumps(
                {
                    "error": {
                        "code": "account_budget_exceeded",
                        "type": "provider_error",
                        "message": "private prompt content must not be logged",
                        "metadata": {
                            "limit_source": "provider_budget",
                            "provider_name": "Google AI Studio",
                            "is_byok": True,
                            "is_free_tier": False,
                            "billing_multiplier": 1.25,
                            "raw": "sk-secret-provider-payload",
                        },
                    }
                }
            ).encode()
        ],
        status=429,
    )
    response.headers.update(
        {
            "Retry-After": "45",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "2",
            "x-ratelimit-reset-requests": "3s",
            "x-request-id": "req_1234567890abcdef12345678",
            "Authorization": "Bearer sk-secret-header",
        }
    )

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openrouter")

    failure = caught.value
    assert failure.provider_error_code == "account_budget_exceeded"
    assert failure.provider_error_type == "provider_error"
    assert failure.provider_limit_source == "provider_budget"
    assert failure.provider_request_id == "req_1234567890abcdef12345678"
    assert failure.provider_diagnostics == {
        "provider_error_code": "account_budget_exceeded",
        "provider_error_type": "provider_error",
        "provider_limit_source": "provider_budget",
        "provider_request_id": "req_1234567890abcdef12345678",
        "provider_name": "Google AI Studio",
        "is_byok": True,
        "is_free_tier": False,
        "billing_multiplier": 1.25,
        "retry_after_seconds": 45,
        "rate_limit_headers": {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "2",
            "x-ratelimit-reset-requests": "3s",
        },
    }
    assert "private prompt content" not in repr(failure)
    assert "sk-secret-provider-payload" not in repr(failure)
    assert "sk-secret-header" not in repr(failure)


def test_oversized_billing_multiplier_does_not_break_provider_error_handling():
    """Ignore enormous provider metadata instead of raising during diagnostics."""
    response = upstream(
        [
            json.dumps(
                {
                    "error": {
                        "code": "invalid_request_error",
                        "metadata": {"billing_multiplier": 10**1000},
                    }
                }
            ).encode()
        ],
        status=400,
    )

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openai")

    assert "billing_multiplier" not in caught.value.provider_diagnostics


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


def test_common_credential_shaped_provider_codes_are_not_logged():
    """Reject common credential forms even when they match provider-code syntax."""
    response = upstream(
        [
            json.dumps(
                {
                    "error": {
                        "code": "github" + "_pat_" + ("placeholder_" * 2),
                        "metadata": {
                            "limit_source": "xox" + "b-" + ("placeholder-" * 2)
                        },
                    }
                }
            ).encode()
        ],
        status=400,
    )

    with pytest.raises(failover_upstream.UpstreamError) as caught:
        failover_upstream.prepare_upstream(response, provider="openai")

    assert caught.value.provider_error_code is None
    assert caught.value.provider_limit_source is None
    assert caught.value.provider_diagnostics == {}


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
                        "metadata": {
                            "limit_source": "openrouter_in_flight_budget",
                            "provider_name": "OpenAI",
                            "is_byok": True,
                            "is_free_tier": False,
                        },
                    }
                },
                "error",
            ),
        ]
    )
    response.headers["x-ratelimit-remaining-requests"] = "2"
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
    assert '"provider_name": "OpenAI"' in formatted_warning
    assert '"is_byok": true' in formatted_warning
    assert '"x-ratelimit-remaining-requests": "2"' in formatted_warning
    assert "private provider prose" not in formatted_warning
    assert b"private provider prose" not in body


def test_transport_failure_keeps_safe_reason_without_url_or_credentials():
    """Keep actionable transport text but remove private URL and credential data."""
    error = requests.ReadTimeout(
        "Read timed out for https://api.openai.com/v1/chat?token=sk-secret-value"
    )

    failure = failover_upstream.transport_failure(error)

    assert failure.provider_diagnostics["exception_type"] == "ReadTimeout"
    assert failure.provider_diagnostics["message"] == (
        "Read timed out for https://api.openai.com"
    )
    assert "sk-secret-value" not in repr(failure.provider_diagnostics)
    assert "/v1/chat" not in repr(failure.provider_diagnostics)


def test_transport_failure_redacts_scheme_less_url_paths_and_credentials():
    """Redact path and query values in scheme-less transport URLs."""
    error = requests.ReadTimeout(
        "Read timed out url: /v1/responses?sig=signature-value&code=authorization-value"
    )

    failure = failover_upstream.transport_failure(error)
    message = failure.provider_diagnostics["message"]

    assert message == "Read timed out url: <path>"
    assert "/v1/responses" not in message
    assert "signature-value" not in message
    assert "authorization-value" not in message


def test_transport_failure_redacts_complete_scheme_less_url_value():
    """Redact whitespace-separated text after a scheme-less URL label."""
    error = requests.ReadTimeout(
        "Read timed out url: /v1/responses?sig=signature-value extra details"
    )

    failure = failover_upstream.transport_failure(error)

    assert failure.provider_diagnostics["message"] == "Read timed out url: <path>"
    assert "signature-value" not in repr(failure.provider_diagnostics)
    assert "extra details" not in repr(failure.provider_diagnostics)


def test_transport_failure_does_not_redact_curl_diagnostics():
    """A curl diagnostic is not a scheme-less URL label."""
    failure = failover_upstream.transport_failure(requests.ReadTimeout("curl: /path"))

    assert failure.provider_diagnostics["message"] == "curl: /path"


def test_transport_failure_handles_malformed_urls_without_raising():
    """Malformed URLs in transport text must not interrupt error handling."""
    failure = failover_upstream.transport_failure(
        requests.ReadTimeout("Read timed out for https://[::1")
    )

    assert failure.provider_diagnostics["exception_type"] == "ReadTimeout"
    assert failure.provider_diagnostics["message"] == (
        "Read timed out for <upstream-url>"
    )


def test_stream_transport_failure_logs_safe_exception_diagnostics(app, mocker):
    """Log transport cause details while excluding raw URLs and credentials."""
    response = upstream([])

    def interrupted_stream():
        yield event({"choices": [{"delta": {"content": "partial"}}]})
        raise requests.ReadTimeout(
            "Read timed out for https://api.openai.com/v1/chat?token=sk-secret-value"
        )

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
    assert '"exception_type": "ReadTimeout"' in formatted_warning
    assert '"message": "Read timed out for https://api.openai.com"' in formatted_warning
    assert "sk-secret-value" not in formatted_warning
    assert "/v1/chat" not in formatted_warning
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
    refused_failure = failover_upstream.transport_failure(refused)
    assert refused_failure.retryable
    assert not refused_failure.upstream_started
    uncertain = requests.ConnectionError(ReadTimeoutError(None, "/", "interrupted"))
    uncertain_failure = failover_upstream.transport_failure(uncertain)
    assert not uncertain_failure.retryable
    assert uncertain_failure.upstream_started
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
