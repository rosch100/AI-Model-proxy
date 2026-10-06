"""Unit tests for Codex response adaptation."""

import json

from app.codex.response_adapter import (
    _chat_completion_id,
    adapt_responses_sse_to_chat_sse,
)


def _sse(event_name, payload):
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode()


def _messages(*chunks, reasoning_display_mode="mdthinkblocks"):
    body = b"".join(
        adapt_responses_sse_to_chat_sse(
            chunks,
            model="gpt-5.4",
            reasoning_display_mode=reasoning_display_mode,
        )
    ).decode()
    out = []
    for raw_message in body.strip().split("\n\n"):
        data = "\n".join(
            line[len("data: ") :]
            for line in raw_message.splitlines()
            if line.startswith("data: ")
        )
        if data == "[DONE]":
            out.append("[DONE]")
        elif data:
            out.append(json.loads(data))
    return out


def test_codex_chat_completion_ids_use_secure_random_choice(mocker):
    """Generate opaque chat completion IDs using the secure random source."""
    choice = mocker.patch("app.codex.response_adapter.secrets.choice", return_value="x")

    assert _chat_completion_id() == "chatcmpl-" + "x" * 24
    assert choice.call_count == 24


def test_codex_adapts_output_text_to_chat_completion_chunks():
    """Output text deltas become Chat Completions chunks."""
    messages = _messages(
        _sse(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "hi"},
        ),
        _sse(
            "response.completed",
            {"type": "response.completed", "response": {"usage": None}},
        ),
    )

    assert messages[0]["object"] == "chat.completion.chunk"
    assert messages[0]["model"] == "gpt-5.4"
    assert messages[0]["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "hi",
    }
    assert messages[-2]["choices"][0]["finish_reason"] == "stop"
    assert messages[-1] == "[DONE]"


def test_codex_adapts_reasoning_as_collapsible_details_with_native_metadata():
    """Reasoning deltas render visibly while preserving native metadata."""
    messages = _messages(
        _sse(
            "response.output_item.added",
            {"type": "response.output_item.added", "item": {"type": "reasoning"}},
        ),
        _sse(
            "response.reasoning_summary_text.delta",
            {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
        ),
        _sse(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "answer"},
        ),
    )

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


def test_codex_can_hide_reasoning_content_while_preserving_metadata():
    """None mode keeps reasoning metadata without visible thinking text."""
    messages = _messages(
        _sse(
            "response.output_item.added",
            {"type": "response.output_item.added", "item": {"type": "reasoning"}},
        ),
        _sse(
            "response.reasoning_summary_text.delta",
            {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
        ),
        _sse(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "answer"},
        ),
        reasoning_display_mode="none",
    )

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[0]["content"] is None
    assert deltas[0]["reasoning_content"] == "thinking"
    assert deltas[1] == {"role": "assistant", "content": "answer"}


def test_codex_can_render_reasoning_as_legacy_think_tags():
    """Thinkblocks mode mirrors reasoning as legacy visible think content."""
    messages = _messages(
        _sse(
            "response.output_item.added",
            {"type": "response.output_item.added", "item": {"type": "reasoning"}},
        ),
        _sse(
            "response.reasoning_summary_text.delta",
            {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
        ),
        _sse(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "answer"},
        ),
        reasoning_display_mode="thinkblocks",
    )

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[0] == {"role": "assistant", "content": "<think>\n"}
    assert deltas[1]["content"] == "thinking"
    assert deltas[1]["reasoning_content"] == "thinking"
    assert deltas[2] == {"role": "assistant", "content": "\n</think>\n\n"}
    assert deltas[3] == {"role": "assistant", "content": "answer"}


def test_codex_closes_visible_reasoning_before_terminal_event():
    """Visible reasoning wrappers are closed even when no text follows."""
    messages = _messages(
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
    )

    deltas = [
        msg["choices"][0]["delta"]
        for msg in messages[:-1]
        if msg.get("choices") and msg["choices"][0]["delta"]
    ]
    assert deltas[-1] == {"role": "assistant", "content": "\n\n</details>\n\n"}


def test_codex_closes_visible_reasoning_before_failure_message():
    """Failure text should not be swallowed by an open reasoning wrapper."""
    messages = _messages(
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
                "response": {"error": {"message": "upstream failed"}},
            },
        ),
    )

    deltas = [msg["choices"][0]["delta"] for msg in messages[:-1]]
    assert deltas[-3] == {"role": "assistant", "content": "\n\n</details>\n\n"}
    assert deltas[-2] == {"role": "assistant", "content": "upstream failed"}


def test_codex_adapts_function_call_stream():
    """Function call argument streams become Chat tool call chunks."""
    messages = _messages(
        _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Shell",
                    "arguments": "",
                },
            },
        ),
        _sse(
            "response.function_call_arguments.delta",
            {"type": "response.function_call_arguments.delta", "delta": '{"cmd"'},
        ),
        _sse(
            "response.function_call_arguments.delta",
            {"type": "response.function_call_arguments.delta", "delta": ':"ls"}'},
        ),
    )

    first = messages[0]["choices"][0]["delta"]["tool_calls"][0]
    second = messages[1]["choices"][0]["delta"]["tool_calls"][0]
    assert first["id"] == "call_1"
    assert first["function"]["name"] == "Shell"
    assert second["function"]["arguments"] == '{"cmd"'
    assert messages[-2]["choices"][0]["finish_reason"] == "tool_calls"
