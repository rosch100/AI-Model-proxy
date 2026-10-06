"""Logical identity, bounded prefixes and interrupted streams across providers."""

import json
from unittest.mock import Mock

import pytest
import requests
from flask import g, request

from app.azure.adapter import AzureAdapter
from app.providers.failover_upstream import (
    UpstreamError,
    chat_stream,
    prepare_upstream,
)
from app.tenants import DatabaseTenantSnapshot


def upstream(chunks):
    """Build a successful upstream response from byte chunks."""
    response = Mock(status_code=200)
    response.headers = {"Content-Type": "text/event-stream"}
    response.iter_content.return_value = iter(chunks)
    return response


def test_chat_stream_keeps_tool_ids_and_model_identity():
    """Preserve tool call IDs while restoring the logical model name."""
    payload = {
        "model": "other-model",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "call-123",
                            "function": {"name": "test", "arguments": "{}"},
                        }
                    ]
                }
            }
        ],
    }
    content = ("data: " + json.dumps(payload) + "\n\n").encode()
    raw = upstream([content, b"data: [DONE]\n\n"])
    body = b"".join(chat_stream(prepare_upstream(raw), "cursor-model"))
    assert b'"model":"cursor-model"' in body
    assert b'"id":"call-123"' in body
    assert body.endswith(b"data: [DONE]\n\n")
    raw.close.assert_called_once()


def test_late_transport_failure_emits_error_without_replay():
    """Expose an interrupted chat stream without leaking transport details."""

    def chunks():
        yield b'data: {"model":"provider","choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise requests.ReadTimeout("secret-upstream-url")

    raw = upstream(chunks())
    budget_attempt = Mock()
    body = b"".join(
        chat_stream(
            prepare_upstream(raw), "cursor-model", budget_attempt=budget_attempt
        )
    )
    assert b"partial" in body
    assert b"stream_interrupted" in body
    assert b"secret-upstream-url" not in body
    budget_attempt.failed.assert_called_once()
    assert isinstance(budget_attempt.failed.call_args.args[0], UpstreamError)
    budget_attempt.release.assert_not_called()
    raw.close.assert_called_once()


def test_late_azure_transport_failure_logs_safe_diagnostics(app, monkeypatch, mocker):
    """Log the Azure transport cause safely when returning stream_interrupted."""

    def chunks():
        yield b'event: response.output_text.delta\ndata: {"delta":"partial"}\n\n'
        raise requests.ReadTimeout(
            "Read timed out for https://resource.openai.azure.com/v1/responses?token=sk-secret-value"
        )

    raw = upstream(chunks())
    raw.headers["Retry-After"] = "45"
    raw.headers["apim-request-id"] = "123e4567-e89b-12d3-a456-426614174000"
    monkeypatch.setattr("app.azure.adapter.requests.request", lambda **kwargs: raw)
    snapshot = DatabaseTenantSnapshot(
        id="tenant",
        api_key_hash="digest",
        custom_model_id="cursor-model",
        provider="azure",
        provider_settings={
            "base_url": "https://resource.openai.azure.com",
            "model_deployments": {"gpt-5.4": "deployment"},
        },
        inference_secret="secret",
        default_model="gpt-5.4",
        profile_id="profile",
    )
    with app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        json={"model": "cursor-model", "messages": []},
    ):
        g.proxy_request_id = "azure-request-correlation"
        warning = mocker.patch.object(app.logger, "warning")
        response = AzureAdapter().forward_attempt(request, snapshot)
        body = response.get_data()
    assert b"partial" in body
    assert b"stream_interrupted" in body
    assert b"sk-secret-value" not in body
    assert warning.call_count == 1
    template, *arguments = warning.call_args.args
    formatted_warning = template % tuple(arguments)
    assert "provider=azure" in formatted_warning
    assert "ReadTimeout" in formatted_warning
    diagnostics = json.loads(arguments[-1])
    assert diagnostics["upstream_host"] == "resource.openai.azure.com"
    assert "Retry-After" not in formatted_warning
    assert '"retry_after_seconds": 45' in formatted_warning
    assert "123e4567-e89b-12d3-a456-426614174000" in formatted_warning
    assert "sk-secret-value" not in formatted_warning
    assert "/v1/responses" not in formatted_warning
    raw.close.assert_called_once()


def test_late_azure_provider_sse_failure_logs_safe_diagnostics(
    app, monkeypatch, mocker
):
    """Log structured provider details for an Azure SSE failure after output."""
    raw = upstream(
        [
            b'event: response.output_text.delta\ndata: {"delta":"partial"}\n\n',
            (
                b"event: response.failed\ndata: "
                b'{"type":"response.failed","response":{"error":'
                b'{"code":"provider_capacity_exhausted",'
                b'"type":"provider_error","param":"model",'
                b'"metadata":{"limit_source":"openrouter_in_flight_budget",'
                b'"provider_name":"OpenRouter","is_byok":true}}}}\n\n'
            ),
        ]
    )
    raw.headers["apim-request-id"] = "123e4567-e89b-12d3-a456-426614174000"
    raw.headers["x-ratelimit-remaining-requests"] = "0"
    monkeypatch.setattr("app.azure.adapter.requests.request", lambda **kwargs: raw)
    snapshot = DatabaseTenantSnapshot(
        id="tenant",
        api_key_hash="digest",
        custom_model_id="cursor-model",
        provider="azure",
        provider_settings={
            "base_url": "https://resource.openai.azure.com",
            "model_deployments": {"gpt-5.4": "deployment"},
        },
        inference_secret="secret",
        default_model="gpt-5.4",
        profile_id="profile",
    )
    with app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        json={"model": "cursor-model", "messages": []},
    ):
        g.proxy_request_id = "azure-request-correlation"
        warning = mocker.patch.object(app.logger, "warning")
        response = AzureAdapter().forward_attempt(request, snapshot)
        body = response.get_data()

    assert b"partial" in body
    assert warning.call_count == 1
    template, *arguments = warning.call_args.args
    formatted_warning = template % tuple(arguments)
    assert "provider=azure" in formatted_warning
    assert '"provider_error_code": "provider_capacity_exhausted"' in formatted_warning
    assert '"provider_error_type": "provider_error"' in formatted_warning
    assert '"provider_limit_source": "openrouter_in_flight_budget"' in formatted_warning
    assert (
        '"provider_request_id": "123e4567-e89b-12d3-a456-426614174000"'
        in formatted_warning
    )
    assert '"provider_name": "OpenRouter"' in formatted_warning
    assert '"is_byok": true' in formatted_warning
    assert '"x-ratelimit-remaining-requests": "0"' in formatted_warning
    assert "private" not in formatted_warning
    raw.close.assert_called_once()


def test_error_code_does_not_reflect_arbitrary_secret():
    """Sanitize arbitrary secrets from upstream error codes and messages."""
    raw = upstream([])
    raw.status_code = 503
    raw.json.return_value = {
        "error": {"code": "sk-secret-credential", "message": "secret"}
    }
    with pytest.raises(UpstreamError) as caught:
        prepare_upstream(raw)
    assert "secret" not in caught.value.code
    assert "secret" not in caught.value.message


def test_heartbeat_and_sse_fields_are_not_json_errors():
    """Accept heartbeat comments and SSE metadata alongside JSON payloads."""
    raw = upstream(
        [
            b": keepalive\r\n\r\n",
            (
                b"event: message\r\nid: event-1\r\nretry: 1000\r\n"
                b'data: {"model":"provider","choices":[{"delta":{"content":"hello"}}]}\r\n\r\n'
            ),
        ]
    )
    prepared = prepare_upstream(raw)
    body = b"".join(chat_stream(prepared, "cursor-model"))
    assert b"hello" in body
    assert b"invalid_upstream_event" not in body


def test_malformed_initial_event_is_explicit_terminal_failure():
    """Reject malformed initial JSON as a terminal upstream error."""
    raw = upstream([b"data: not-json\n\n"])
    with pytest.raises(UpstreamError) as caught:
        prepare_upstream(raw)
    assert caught.value.status == 502
    assert not caught.value.retryable
    raw.close.assert_called_once()
