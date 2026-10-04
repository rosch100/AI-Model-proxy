"""Unit tests for Azure response adaptation."""

import json

import pytest

from app.azure.adapter import AzureAdapter
from app.persistence.inference_activity import start_provider_attempt
from app.persistence.models import ProviderAttemptEvent, ProviderProfile


class _FakeUpstreamResponse:
    """Minimal streaming response stub for ResponseAdapter tests."""

    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def iter_content(self, chunk_size=8192):
        del chunk_size
        yield from self._chunks

    def close(self):
        self.closed = True


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
    adapter.response_adapter._reasoning_open = False
    adapter.response_adapter._reasoning_pending_whitespace = ""
    adapter.response_adapter._tool_calls = 0
    adapter.response_adapter._usage = None
    adapter.response_adapter._chat_completion_id = "chatcmpl-test"
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
