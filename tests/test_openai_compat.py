"""Tests for OpenAI-compatible provider request forwarding."""

import json

import pytest
from flask import request

from app.providers.openai_compat import forward_openai_compatible
from app.tenants import DatabaseTenantSnapshot


def _snapshot(
    provider: str,
    default_model: str,
    provider_settings: dict[str, str] | None = None,
) -> DatabaseTenantSnapshot:
    return DatabaseTenantSnapshot(
        id="acme",
        api_key_hash="unused",
        custom_model_id="cursor-model",
        provider=provider,
        provider_settings=provider_settings or {},
        inference_secret="provider-key",
        default_model=default_model,
        profile_id="profile-1",
    )


@pytest.mark.parametrize(
    ("provider", "model", "tools", "incoming_effort", "expected_effort"),
    [
        (
            "openai",
            "gpt-6-luna",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "high",
            "none",
        ),
        (
            "openai",
            "gpt-6-luna",
            [{"type": "function", "function": {"name": "lookup"}}],
            None,
            "none",
        ),
        (
            "openai",
            "gpt-5.4",
            [{"type": "function", "function": {"name": "lookup"}}],
            "high",
            "high",
        ),
        (
            "openai",
            "gpt-6-luna",
            [{"type": "web_search"}],
            "high",
            "high",
        ),
        (
            "openrouter",
            "gpt-6-luna",
            [{"type": "function", "function": {"name": "lookup"}}],
            "high",
            "high",
        ),
    ],
)
def test_forwarding_applies_luna_tool_reasoning_compatibility_only_when_needed(
    app, requests_mock, provider, model, tools, incoming_effort, expected_effort
):
    """Only OpenAI Luna function tools force the documented no-reasoning mode."""
    payload = {
        "model": "cursor-model",
        "messages": [{"role": "user", "content": "Run a tool."}],
        "tools": tools,
    }
    if incoming_effort is not None:
        payload["reasoning_effort"] = incoming_effort
    requests_mock.post(
        f"https://{'api.openai.com' if provider == 'openai' else 'openrouter.ai/api/v1'}/"
        f"{'v1/' if provider == 'openai' else ''}chat/completions",
        text="data: [DONE]\n\n",
        headers={"Content-Type": "text/event-stream"},
    )

    with app.test_request_context("/v1/chat/completions", json=payload):
        response = forward_openai_compatible(
            request,
            _snapshot(provider, model),
        )
        response.get_data()
        assert response.status_code == 200

    sent_payload = json.loads(requests_mock.last_request.text)
    assert sent_payload["model"] == model
    assert sent_payload["reasoning_effort"] == expected_effort
    assert sent_payload["tools"] == tools
    assert sent_payload["stream_options"]["include_usage"] is True


def test_deepseek_forwarding_preserves_chat_fields_and_uses_native_origin(
    app, requests_mock
):
    """The DeepSeek provider uses shared Chat Completions without OpenAI headers."""
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    requests_mock.post(
        "https://api.deepseek.com/chat/completions",
        text='data: {"model":"deepseek-v4-flash","choices":[{"delta":{"content":"ok"}}]}\n\n'
        "data: [DONE]\n\n",
        headers={"Content-Type": "text/event-stream"},
    )
    payload = {
        "model": "cursor-model",
        "messages": [{"role": "user", "content": "Run a tool."}],
        "tools": tools,
        "reasoning_effort": "high",
    }
    snapshot = _snapshot(
        "deepseek",
        "deepseek-v4-flash",
        {"organization": "ignored", "project": "ignored"},
    )

    with app.test_request_context("/v1/chat/completions", json=payload):
        response = forward_openai_compatible(request, snapshot)
        response.get_data()
        assert response.status_code == 200

    sent = requests_mock.last_request
    assert sent.url == "https://api.deepseek.com/chat/completions"
    assert sent.json()["model"] == "deepseek-v4-flash"
    assert sent.json()["tools"] == tools
    assert sent.json()["reasoning_effort"] == "high"
    assert sent.json()["stream"] is True
    assert sent.json()["stream_options"]["include_usage"] is True
    assert sent.headers["Authorization"] == "Bearer provider-key"
    assert "OpenAI-Organization" not in sent.headers
    assert "OpenAI-Project" not in sent.headers
