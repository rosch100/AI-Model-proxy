"""Regressions for upstream error safety and invisible SSE prefix handling."""

import json
from unittest.mock import Mock

import pytest
import requests
from flask import request

from app.azure.adapter import AzureAdapter
from app.providers.failover_upstream import UpstreamError, chat_stream, prepare_upstream
from app.tenants import DatabaseTenantSnapshot


def event(data, name=None):
    """Encode one native or compatible provider SSE event."""
    return (
        (f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n"
    ).encode()


def upstream(chunks, status=200):
    """Create a single-pass HTTP stream without simulating adapter behavior."""
    raw = Mock(status_code=status)
    raw.headers = {"Content-Type": "text/event-stream"}
    raw.iter_content.return_value = iter(chunks)
    return raw


def test_known_http_status_does_not_read_error_body():
    """A body timeout cannot erase an already known retryable status."""
    raw = upstream([], 429)
    raw.json.side_effect = requests.ReadTimeout("unknown outcome")
    with pytest.raises(UpstreamError) as caught:
        prepare_upstream(raw)
    assert caught.value.status == 429
    assert caught.value.retryable
    raw.json.assert_not_called()
    raw.close.assert_called_once()


@pytest.mark.parametrize(
    "prefix",
    [
        event({}, "error"),
        event({"type": "error"}, "error"),
        event({"item": {"type": "reasoning"}}, "response.output_item.added"),
        event({"item": {"type": "message"}}, "response.output_item.added"),
        event({"type": "response.content_part.added"}, "response.content_part.added"),
    ],
)
def test_invisible_prefix_keeps_rate_limit_failover_window(prefix):
    """Lifecycle/empty precursor events cannot commit the client response."""
    raw = upstream(
        [
            prefix,
            event(
                {"response": {"error": {"code": "rate_limit_exceeded"}}},
                "response.failed",
            ),
        ]
    )
    with pytest.raises(UpstreamError) as caught:
        prepare_upstream(raw)
    assert caught.value.status == 429
    assert caught.value.retryable


def test_tool_start_commits_even_before_argument_delta():
    """A visible tool-call ID must forbid any subsequent replay."""
    prefix = event(
        {"item": {"type": "function_call", "call_id": "call-1", "name": "test"}},
        "response.output_item.added",
    )
    raw = upstream([prefix, event({"error": {"code": "rate_limit_exceeded"}}, "error")])
    prepared = prepare_upstream(raw)
    assert b"call-1" in b"".join(prepared.iter_content())


def test_late_chat_error_is_sanitized():
    """Late structured errors stay visible without reflecting provider secrets."""
    raw = upstream(
        [
            event({"choices": [{"delta": {"content": "partial"}}]}),
            event({"error": {"code": "sk-secret", "message": "Bearer real-secret"}}),
        ]
    )
    body = b"".join(chat_stream(prepare_upstream(raw), "cursor-model"))
    assert b"partial" in body
    assert b'"error"' in body
    assert b"secret" not in body


def test_late_azure_error_is_sanitized(app, monkeypatch):
    """Native Azure error chunks preserve the Cursor contract, not raw messages."""
    raw = upstream(
        [
            event({"delta": "partial"}, "response.output_text.delta"),
            event(
                {
                    "response": {
                        "error": {"code": "sk-secret", "message": "Bearer real-secret"}
                    }
                },
                "response.failed",
            ),
        ]
    )
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
        response = AzureAdapter().forward_attempt(request, snapshot)
        body = response.get_data()
    assert b"partial" in body
    assert b"error" in body
    assert b"secret" not in body
