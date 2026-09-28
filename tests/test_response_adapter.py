"""Unit tests for Azure response adaptation."""

import json

from app.azure.adapter import AzureAdapter


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


def test_response_adapter_emits_reasoning_content_separately(app):
    """Reasoning deltas render visibly while preserving native metadata."""
    messages = _reasoning_messages(app, "mdthinkblocks")

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[0] == {
        "role": "assistant",
        "content": "<details>\n<summary>Thought</summary>\n\n",
    }
    assert deltas[1] == {
        "role": "assistant",
        "content": "thinking",
        "reasoning": "thinking",
        "reasoning_content": "thinking",
        "reasoning_details": [{"type": "reasoning.text", "text": "thinking"}],
        "thinking_blocks": [{"type": "thinking", "thinking": "thinking"}],
        "provider_specific_fields": {
            "thinking_blocks": [{"type": "thinking", "thinking": "thinking"}]
        },
    }
    assert deltas[2] == {"role": "assistant", "content": "\n\n</details>\n\n"}
    assert deltas[3] == {"role": "assistant", "content": "answer"}
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
    assert deltas[0] == {"role": "assistant", "content": "<think>\n"}
    assert deltas[1]["content"] == "thinking"
    assert deltas[1]["reasoning_content"] == "thinking"
    assert deltas[2] == {"role": "assistant", "content": "\n</think>\n\n"}
    assert deltas[3] == {"role": "assistant", "content": "answer"}


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
