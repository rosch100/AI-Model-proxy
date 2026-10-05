"""Unit tests for Azure response adaptation."""

import json
from unittest.mock import Mock

import pytest
import requests

from app.azure.adapter import AzureAdapter
from app.persistence.inference_activity import start_provider_attempt
from app.persistence.models import ProviderAttemptEvent, ProviderProfile


class _FakeUpstreamResponse:
    """Minimal streaming response stub for ResponseAdapter tests."""

    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False
        self.close_count = 0

    def iter_content(self, chunk_size=8192):
        del chunk_size
        yield from self._chunks

    def close(self):
        self.closed = True
        self.close_count += 1


def _sse(event_name, payload):
    """Build a single SSE event payload."""
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


def _messages_from_response(response):
    """Decode Chat Completions SSE messages from a Flask response."""
    body = b"".join(response.response).decode("utf-8")
    messages = []
    for raw_message in body.strip().split("\n\n"):
        data_lines = [
            line[len("data: ") :]
            for line in raw_message.splitlines()
            if line.startswith("data: ")
        ]
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data == "[DONE]":
            continue
        messages.append(json.loads(data))
    return messages


def _reasoning_messages(app, mode):
    """Run a small reasoning stream through the Azure adapter."""
    app.config["REASONING_DISPLAY_MODE"] = mode
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False

    upstream = _FakeUpstreamResponse(
        [
            _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "item": {"type": "reasoning"},
                },
            ),
            _sse(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "thinking",
                },
            ),
            _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "delta": "answer",
                },
            ),
        ]
    )

    return _messages_from_response(adapter.response_adapter.adapt(upstream))


def _azure_messages(app, events, mode="mdthinkblocks"):
    """Run arbitrary Azure SSE events through the Azure adapter."""
    app.config["REASONING_DISPLAY_MODE"] = mode
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False
    upstream = _FakeUpstreamResponse(events)
    return _messages_from_response(adapter.response_adapter.adapt(upstream))


def test_response_adapter_yields_each_delta_before_reading_next_upstream_chunk(app):
    """Yield visible output immediately instead of draining the whole SSE body."""
    app.config["REASONING_DISPLAY_MODE"] = "mdthinkblocks"
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False
    adapter.response_adapter._chat_completion_id = "chatcmpl-test"
    adapter.response_adapter._reasoning_open = False
    adapter.response_adapter._reasoning_pending_whitespace = ""
    adapter.response_adapter._tool_calls = 0
    adapter.response_adapter._usage = None
    upstream_advanced = False

    def upstream_chunks():
        nonlocal upstream_advanced
        yield _sse(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "first"},
        )
        upstream_advanced = True
        yield _sse(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "second"},
        )

    stream = adapter.response_adapter._adapt_stream(
        _FakeUpstreamResponse(upstream_chunks()), None
    )
    try:
        first_chunk = next(stream)
    finally:
        stream.close()

    assert first_chunk["choices"][0]["delta"]["content"] == "first"
    assert upstream_advanced is False


def test_resume_request_targets_response_and_sequence(monkeypatch):
    """Resume uses the documented Azure response URL and sequence cursor."""
    captured = {}
    expected_response = object()

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return expected_response

    monkeypatch.setattr("app.azure.adapter.requests.get", fake_get)

    response = AzureAdapter._resume_azure_response(
        {
            "url": "https://resource.openai.azure.com/openai/v1/responses",
            "headers": {
                "api-key": "secret",
                "content-type": "application/json",
                "content-length": "36",
            },
            "timeout": (10.0, 300.0),
        },
        "resp_123",
        17,
    )

    assert response is expected_response
    assert captured == {
        "url": "https://resource.openai.azure.com/openai/v1/responses/resp_123",
        "headers": {"api-key": "secret"},
        "params": {"stream": "true", "starting_after": 17},
        "stream": True,
        "timeout": (10.0, 300.0),
    }


def test_cancel_azure_response_uses_response_endpoint_and_bounded_timeout(monkeypatch):
    """Cancel one Azure background response using a bounded request timeout."""
    captured = {}

    class _SuccessfulResponse:
        def raise_for_status(self):
            pass

        def close(self):
            pass

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _SuccessfulResponse()

    monkeypatch.setattr("app.azure.adapter.requests.post", fake_post)

    AzureAdapter._cancel_azure_response(
        {
            "url": "https://resource.openai.azure.com/openai/v1/responses",
            "headers": {
                "api-key": "secret",
                "content-type": "application/json",
                "content-length": "36",
            },
        },
        "resp_123",
    )

    assert captured == {
        "url": "https://resource.openai.azure.com/openai/v1/responses/resp_123/cancel",
        "headers": {"api-key": "secret", "content-type": "application/json"},
        "timeout": (10.0, 10.0),
    }


def test_read_timeout_resumes_azure_response_after_last_emitted_sequence(app):
    """Resume the same Azure response after the last forwarded SSE event."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False

    def interrupted_stream():
        yield _sse(
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_123"},
            },
        )
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "partial",
            },
        )
        raise requests.ReadTimeout("temporary socket interruption")

    def resumed_stream():
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "delta": " continued",
            },
        )
        yield _sse(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 3,
                "response": {"usage": {}},
            },
        )

    resume_calls = []
    original_upstream = _FakeUpstreamResponse(interrupted_stream())
    resumed_upstream = _FakeUpstreamResponse(resumed_stream())

    def resume(response_id, sequence_number):
        resume_calls.append((response_id, sequence_number))
        return resumed_upstream

    response = adapter.response_adapter.adapt(
        original_upstream,
        resume_stream=resume,
    )
    body = b"".join(response.response)

    assert resume_calls == [("resp_123", 1)]
    assert b"partial" in body
    assert b" continued" in body
    assert b"stream_interrupted" not in body
    assert body.endswith(b"data: [DONE]\n\n")
    assert original_upstream.close_count == 1
    assert resumed_upstream.close_count == 1


def test_terminal_event_prevents_resuming_after_later_transport_error(app):
    """Do not resume or replace terminal state after completion was received."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False
    resume_calls = []
    cancelled_response_ids = []

    def terminal_then_interrupted():
        yield _sse(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": {"id": "resp_terminal", "usage": {}},
            },
        )
        raise requests.ReadTimeout("after terminal event")

    def resume(response_id, sequence_number):
        resume_calls.append((response_id, sequence_number))
        raise requests.ConnectionError("unexpected resume")

    response = adapter.response_adapter.adapt(
        _FakeUpstreamResponse(terminal_then_interrupted()),
        resume_stream=resume,
        cancel_response=cancelled_response_ids.append,
    )
    body = b"".join(response.response)

    assert resume_calls == []
    assert cancelled_response_ids == []
    assert b"stream_interrupted" not in body
    assert body.endswith(b"data: [DONE]\n\n")


def test_rate_limit_retry_resets_terminal_response_state(app, monkeypatch):
    """A later retry stream interruption is handled as its own attempt."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False
    monkeypatch.setattr(
        adapter,
        "_retry_stream_rate_limit",
        lambda *_args: _FakeUpstreamResponse(retry_stream()),
    )
    cancelled_response_ids = []
    resume_calls = []

    def rate_limited_stream():
        yield _sse(
            "response.failed",
            {
                "type": "response.failed",
                "sequence_number": 5,
                "response": {
                    "id": "resp_rate_limited",
                    "error": {"code": "rate_limit_exceeded", "status": 429},
                },
            },
        )

    def retry_stream():
        yield _sse(
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_retry"},
            },
        )
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "partial",
            },
        )
        raise requests.ReadTimeout("retry stream interrupted")

    def resume(response_id, sequence_number):
        resume_calls.append((response_id, sequence_number))
        return _FakeUpstreamResponse(
            [
                _sse(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "sequence_number": 2,
                        "response": {"id": response_id, "usage": {}},
                    },
                )
            ]
        )

    response = adapter.response_adapter.adapt(
        _FakeUpstreamResponse(rate_limited_stream()),
        request_context=object(),
        resume_stream=resume,
        cancel_response=cancelled_response_ids.append,
    )
    body = b"".join(response.response)

    assert resume_calls == [("resp_retry", 1)]
    assert cancelled_response_ids == []
    assert b"partial" in body
    assert b"stream_interrupted" not in body


def test_rate_limit_retry_request_failure_reports_stream_interruption(app, monkeypatch):
    """Report a failed retry request instead of treating it as terminal success."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False

    def fail_retry(*_args):
        raise requests.ConnectionError("retry request failed")

    monkeypatch.setattr(adapter, "_retry_stream_rate_limit", fail_retry)
    resume_calls = []

    def rate_limited_stream():
        yield _sse(
            "response.failed",
            {
                "type": "response.failed",
                "sequence_number": 5,
                "response": {
                    "id": "resp_rate_limited",
                    "error": {"code": "rate_limit_exceeded", "status": 429},
                },
            },
        )

    response = adapter.response_adapter.adapt(
        _FakeUpstreamResponse(rate_limited_stream()),
        request_context=object(),
        resume_stream=lambda *args: resume_calls.append(args),
    )
    body = b"".join(response.response)

    assert b"stream_interrupted" in body
    assert resume_calls == []


def test_resume_retries_transient_reconnect_errors(app, monkeypatch):
    """Retry transient failures from the resume request before giving up."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False
    monkeypatch.setattr("app.azure.response_adapter.time.sleep", lambda _delay: None)

    def interrupted_stream():
        yield _sse(
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_retry"},
            },
        )
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "partial",
            },
        )
        raise requests.ReadTimeout("temporary stream interruption")

    def resumed_stream():
        yield _sse(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {"usage": {}},
            },
        )

    resume_calls = []
    resumed_upstream = _FakeUpstreamResponse(resumed_stream())

    def resume(response_id, sequence_number):
        resume_calls.append((response_id, sequence_number))
        if len(resume_calls) < 3:
            raise requests.ConnectionError("temporary reconnect failure")
        return resumed_upstream

    response = adapter.response_adapter.adapt(
        _FakeUpstreamResponse(interrupted_stream()),
        resume_stream=resume,
    )
    body = b"".join(response.response)

    assert resume_calls == [("resp_retry", 1)] * 3
    assert b"partial" in body
    assert b"stream_interrupted" not in body
    assert resumed_upstream.close_count == 1


def test_cancel_response_request_failure_is_swallowed_and_logged(monkeypatch, app):
    """Log cancellation transport errors without exposing their messages."""

    def fail_post(*_args, **_kwargs):
        raise requests.ConnectionError("cancel failed")

    monkeypatch.setattr("app.azure.adapter.requests.post", fail_post)
    adapter = AzureAdapter()
    adapter.activity_settings = {"base_url": "https://resource.openai.azure.com"}
    logger = Mock()
    monkeypatch.setattr("app.azure.adapter.logger.warning", logger)

    adapter._cancel_azure_response(
        {
            "url": "https://resource.openai.azure.com/openai/v1/responses",
            "headers": {"api-key": "secret"},
        },
        "resp_123",
    )

    logger.assert_called_once()
    message = logger.call_args.args[0]
    assert "Azure response cancellation failed" in message
    assert logger.call_args.args[-1] == "ConnectionError"
    assert "cancel failed" not in str(logger.call_args)


def test_aborted_background_stream_cancels_response(app):
    """Cancel the stored Azure response if its downstream client disconnects."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False

    upstream = _FakeUpstreamResponse(
        [
            _sse(
                "response.created",
                {
                    "type": "response.created",
                    "response": {"id": "resp_cancel"},
                },
            ),
            _sse(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "partial"},
            ),
        ]
    )
    cancelled_response_ids = []
    stream = adapter.response_adapter._adapt_stream(
        upstream,
        None,
        cancel_response=cancelled_response_ids.append,
    )
    adapter.response_adapter._chat_completion_id = "chatcmpl-test"
    adapter.response_adapter._reasoning_open = False
    adapter.response_adapter._reasoning_pending_whitespace = ""
    adapter.response_adapter._tool_calls = 0
    adapter.response_adapter._usage = None

    try:
        first_chunk = next(stream)
    finally:
        stream.close()

    assert first_chunk["choices"][0]["delta"]["content"] == "partial"
    assert cancelled_response_ids == ["resp_cancel"]
    assert upstream.close_count == 1


def test_unpinned_stream_does_not_cancel_without_background_response_id(app):
    """Avoid sending cancellation unless the provider supplied a response ID."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False
    cancelled_response_ids = []
    upstream = _FakeUpstreamResponse(
        [_sse("response.output_text.delta", {"delta": "partial"})]
    )

    stream = adapter.response_adapter._adapt_stream(
        upstream,
        None,
        cancel_response=cancelled_response_ids.append,
    )
    adapter.response_adapter._chat_completion_id = "chatcmpl-test"
    adapter.response_adapter._reasoning_open = False
    adapter.response_adapter._reasoning_pending_whitespace = ""
    adapter.response_adapter._tool_calls = 0
    adapter.response_adapter._usage = None

    try:
        next(stream)
    finally:
        stream.close()

    assert cancelled_response_ids == []
    assert upstream.close_count == 1


def test_failed_resume_cancels_background_response_after_retry_limit(app, monkeypatch):
    """Cancel background generation after all transient resume attempts fail."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False
    monkeypatch.setattr("app.azure.response_adapter.time.sleep", lambda _delay: None)

    def interrupted_stream():
        yield _sse(
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_resume_failure"},
            },
        )
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "partial",
            },
        )
        raise requests.ReadTimeout("temporary stream interruption")

    resume_calls = []
    cancelled_response_ids = []

    def resume(response_id, sequence_number):
        resume_calls.append((response_id, sequence_number))
        raise requests.ConnectionError("temporary reconnect failure")

    response = adapter.response_adapter.adapt(
        _FakeUpstreamResponse(interrupted_stream()),
        resume_stream=resume,
        cancel_response=cancelled_response_ids.append,
    )
    body = b"".join(response.response)

    assert resume_calls == [("resp_resume_failure", 1)] * 3
    assert cancelled_response_ids == ["resp_resume_failure"]
    assert b"stream_interrupted" in body


def test_resumed_stream_sanitizes_provider_error_details(app):
    """Do not expose provider-supplied error text after resuming a stream."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = False

    def interrupted_stream():
        yield _sse(
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_123"},
            },
        )
        raise requests.ReadTimeout("temporary socket interruption")

    def resumed_stream():
        yield _sse(
            "response.failed",
            {
                "type": "response.failed",
                "sequence_number": 1,
                "response": {
                    "error": {
                        "code": "server_error",
                        "status": 500,
                        "message": "provider-secret-must-not-leak",
                    }
                },
            },
        )

    response = adapter.response_adapter.adapt(
        _FakeUpstreamResponse(interrupted_stream()),
        resume_stream=lambda _response_id, _sequence_number: _FakeUpstreamResponse(
            resumed_stream()
        ),
    )
    body = b"".join(response.response)

    assert b"provider-secret-must-not-leak" not in body
    assert b"Provider returned an error (HTTP 500)." in body


def test_response_failed_after_partial_error_is_reported(app):
    """Keep the terminal provider failure visible after an earlier stream error."""
    messages = _azure_messages(
        app,
        [
            _sse(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "partial"},
            ),
            _sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "code": "server_error",
                        "message": "intermediate stream error",
                    },
                },
            ),
            _sse(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "code": "server_error",
                            "message": "terminal provider failure",
                        }
                    },
                },
            ),
        ],
    )

    content = "".join(
        message["choices"][0]["delta"].get("content", "") for message in messages
    )
    assert "partial" in content
    assert "terminal provider failure" in content


def test_response_adapter_emits_usage_chunk(app):
    """Emit a terminal usage chunk when Azure reports final token usage."""
    adapter = AzureAdapter()
    adapter.inbound_model = "gpt-5.4"
    adapter.include_usage = True

    upstream = _FakeUpstreamResponse(
        [
            _sse(
                "response.created",
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_123",
                        "usage": None,
                    },
                },
            ),
            _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message"},
                },
            ),
            _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "delta": "pong",
                },
            ),
            _sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_123",
                        "usage": {
                            "input_tokens": 11,
                            "output_tokens": 7,
                            "total_tokens": 18,
                        },
                    },
                },
            ),
        ]
    )

    response = adapter.response_adapter.adapt(upstream)
    messages = _messages_from_response(response)

    assert messages[-1] == {
        "id": messages[-1]["id"],
        "object": "chat.completion.chunk",
        "created": messages[-1]["created"],
        "model": "gpt-5.4",
        "choices": [],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {
                "cached_tokens": 0,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
            },
        },
    }
    assert upstream.closed is True


@pytest.mark.parametrize(
    ("event_name", "payload", "expected_outcome", "expected_status"),
    (
        (
            "response.completed",
            {"type": "response.completed", "response": {"usage": {}}},
            "success",
            200,
        ),
        (
            "response.failed",
            {
                "type": "response.failed",
                "response": {"error": {"status": 429, "code": "rate_limited"}},
            },
            "failure",
            429,
        ),
        (
            "response.failed",
            {
                "type": "response.failed",
                "response": {"error": {"code": "rate_limit_exceeded"}},
            },
            "failure",
            429,
        ),
    ),
)
def test_provider_stream_records_terminal_status(
    admin_app, event_name, payload, expected_outcome, expected_status
):
    """Record success status without overwriting upstream failure status."""
    database = admin_app.extensions["database"]
    profile_id = f"azure-terminal-stream-{expected_outcome}"
    with database.sessions.begin() as session:
        session.add(
            ProviderProfile(
                id=profile_id,
                tenant_id="acme",
                provider="azure",
                display_name="Azure terminal stream",
                settings={},
                default_model="gpt-5.4",
                inference_secret_ciphertext="encrypted-key",
            )
        )

    with admin_app.app_context():
        attempt_id = start_provider_attempt(
            tenant_id="acme",
            provider="azure",
            profile_id=profile_id,
            inbound_model="cursor-acme",
            routed_model="gpt-5.4",
        )
        adapter = AzureAdapter()
        adapter.inbound_model = "gpt-5.4"
        adapter.include_usage = False
        response = adapter.response_adapter.adapt(
            _FakeUpstreamResponse([_sse(event_name, payload)]),
            activity_attempt_id=attempt_id,
        )
        _messages_from_response(response)

    with database.sessions() as session:
        attempt = session.get(ProviderAttemptEvent, attempt_id)

    assert attempt.outcome == expected_outcome
    assert attempt.status_code == expected_status


def test_read_timeout_persists_accepted_http_status_and_diagnostics(admin_app):
    """Retain HTTP 200 and the transport failure when an accepted stream breaks."""
    database = admin_app.extensions["database"]
    profile_id = "azure-interrupted-stream"
    with database.sessions.begin() as session:
        session.add(
            ProviderProfile(
                id=profile_id,
                tenant_id="acme",
                provider="azure",
                display_name="Interrupted Azure stream",
                settings={},
                default_model="gpt-5.4",
                inference_secret_ciphertext="encrypted-key",
            )
        )

    def chunks():
        yield _sse("response.output_text.delta", {"delta": "partial"})
        raise requests.ReadTimeout("secret-hostname-and-api-key")

    with admin_app.app_context():
        attempt_id = start_provider_attempt(
            tenant_id="acme",
            provider="azure",
            profile_id=profile_id,
            inbound_model="cursor-acme",
            routed_model="gpt-5.4",
        )
        adapter = AzureAdapter()
        adapter.inbound_model = "cursor-acme"
        adapter.include_usage = False
        response = adapter.response_adapter.adapt(
            _FakeUpstreamResponse(chunks()),
            activity_attempt_id=attempt_id,
        )
        body = b"".join(response.response)

    with database.sessions() as session:
        attempt = session.get(ProviderAttemptEvent, attempt_id)

    assert b"partial" in body
    assert b"stream_interrupted" in body
    assert attempt.outcome == "failure"
    assert attempt.status_code == 200
    assert attempt.failure_details == {
        "error_code": "stream_interrupted",
        "exception_type": "ReadTimeout",
    }
    assert b"secret-hostname-and-api-key" not in body


def test_response_adapter_emits_reasoning_content_separately(app):
    """Reasoning deltas render visibly while preserving native metadata."""
    messages = _reasoning_messages(app, "mdthinkblocks")

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[0] == {
        "role": "assistant",
        "content": "<details>\n<summary>Thought</summary>\n\nthinking",
        "reasoning": "thinking",
        "reasoning_content": "thinking",
        "reasoning_details": [{"type": "reasoning.text", "text": "thinking"}],
        "thinking_blocks": [{"type": "thinking", "thinking": "thinking"}],
        "provider_specific_fields": {
            "thinking_blocks": [{"type": "thinking", "thinking": "thinking"}]
        },
    }
    assert deltas[1] == {"role": "assistant", "content": "\n\n</details>\n\n"}
    assert deltas[2] == {"role": "assistant", "content": "answer"}
    assert all("<think>" not in str(delta) for delta in deltas)


def test_response_adapter_can_hide_reasoning_content_while_preserving_metadata(app):
    """None mode keeps reasoning metadata without visible thinking text."""
    messages = _reasoning_messages(app, "none")

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[0]["content"] is None
    assert deltas[0]["reasoning_content"] == "thinking"
    assert deltas[1] == {"role": "assistant", "content": "answer"}


def test_response_adapter_can_render_reasoning_as_legacy_think_tags(app):
    """Thinkblocks mode mirrors reasoning as legacy visible think content."""
    messages = _reasoning_messages(app, "thinkblocks")

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[0]["content"] == "<think>\nthinking"
    assert deltas[0]["reasoning_content"] == "thinking"
    assert deltas[1] == {"role": "assistant", "content": "\n</think>\n\n"}
    assert deltas[2] == {"role": "assistant", "content": "answer"}


def test_response_adapter_does_not_render_empty_reasoning_blocks(app):
    """Do not emit visible Thought wrappers when Azure supplies no reasoning text."""
    messages = _azure_messages(
        app,
        [
            _sse(
                "response.output_item.added",
                {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.output_item.done",
                {"type": "response.output_item.done", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.completed",
                {"type": "response.completed", "response": {"usage": {}}},
            ),
        ],
    )

    content = [
        message["choices"][0]["delta"].get("content") for message in messages[:-1]
    ]
    assert content == []


def test_response_adapter_does_not_render_whitespace_only_reasoning(app):
    """Whitespace alone must not create a visible Thought block."""
    messages = _azure_messages(
        app,
        [
            _sse(
                "response.output_item.added",
                {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.reasoning_summary_text.delta",
                {"type": "response.reasoning_summary_text.delta", "delta": " \n"},
            ),
            _sse(
                "response.completed",
                {"type": "response.completed", "response": {"usage": {}}},
            ),
        ],
    )

    content = [
        message["choices"][0]["delta"].get("content") for message in messages[:-1]
    ]
    assert content == []


def test_response_adapter_keeps_adjacent_reasoning_items_in_one_block(app):
    """Render one non-empty block across adjacent Azure reasoning items."""
    messages = _azure_messages(
        app,
        [
            _sse(
                "response.output_item.added",
                {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": " ",
                },
            ),
            _sse(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "first",
                },
            ),
            _sse(
                "response.output_item.done",
                {"type": "response.output_item.done", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.output_item.added",
                {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": " second",
                },
            ),
            _sse(
                "response.output_item.done",
                {"type": "response.output_item.done", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.completed",
                {"type": "response.completed", "response": {"usage": {}}},
            ),
        ],
    )

    content = [
        message["choices"][0]["delta"].get("content")
        for message in messages[:-1]
        if message["choices"][0]["delta"].get("content") is not None
    ]
    assert content == [
        "<details>\n<summary>Thought</summary>\n\n first",
        " second",
        "\n\n</details>\n\n",
    ]


def test_response_adapter_closes_visible_reasoning_before_terminal_event(app):
    """Visible reasoning wrappers are closed even when no text follows."""
    messages = _azure_messages(
        app,
        [
            _sse(
                "response.output_item.added",
                {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.reasoning_summary_text.delta",
                {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
            ),
            _sse(
                "response.completed",
                {"type": "response.completed", "response": {"usage": {}}},
            ),
        ],
    )

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[-1] == {"role": "assistant", "content": "\n\n</details>\n\n"}


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "response.failed", "response": {"error": None}},
        {"type": "response.failed", "response": {}},
        {"type": "response.failed"},
        {"type": "response.failed", "response": None},
        None,
    ],
)
def test_response_adapter_handles_failed_event_without_error_details(app, payload):
    """Missing Azure error details must not crash the downstream stream."""
    messages = _azure_messages(
        app,
        [_sse("response.failed", payload)],
    )

    failure_message = messages[0]["choices"][0]["delta"]["content"]
    assert "failed response without error details" in failure_message


def test_response_adapter_closes_visible_reasoning_before_failure_message(app):
    """Failure text should not be swallowed by an open reasoning wrapper."""
    messages = _azure_messages(
        app,
        [
            _sse(
                "response.output_item.added",
                {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            ),
            _sse(
                "response.reasoning_summary_text.delta",
                {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
            ),
            _sse(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "error": {"code": "bad", "message": "upstream failed"}
                    },
                },
            ),
        ],
    )

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[-2] == {"role": "assistant", "content": "\n\n</details>\n\n"}
    assert "upstream failed" in deltas[-1]["content"]
