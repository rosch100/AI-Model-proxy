"""Tests for OpenAI-compatible provider request forwarding."""

import json

import pytest
from flask import request

from app.providers.openai_compat import forward_openai_compatible
from app.tenants import DatabaseTenantSnapshot


def _snapshot(provider: str, default_model: str) -> DatabaseTenantSnapshot:
    return DatabaseTenantSnapshot(
        id="acme",
        api_key_hash="unused",
        custom_model_id="cursor-model",
        provider=provider,
        provider_settings={},
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
        text="data: [DONE]\\n\\n",
        headers={"Content-Type": "text/event-stream"},
    )

    with app.test_request_context("/v1/chat/completions", json=payload):
        response = forward_openai_compatible(
            request,
            _snapshot(provider, model),
            target_model=model,
        )
        response.get_data()

    sent_payload = json.loads(requests_mock.last_request.text)
    assert sent_payload["model"] == model
    assert sent_payload["reasoning_effort"] == expected_effort
    assert sent_payload["tools"] == tools
    assert sent_payload["stream_options"]["include_usage"] is True


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("inbound_model", ["cursor-model", "gpt-6-luna"])
def test_routed_provider_uses_its_configured_default_model(
    app, requests_mock, provider, inbound_model
):
    """A routed provider always uses its configured model, never caller identity."""
    origin = (
        "https://api.openai.com/v1"
        if provider == "openai"
        else "https://openrouter.ai/api/v1"
    )
    requests_mock.post(
        f"{origin}/chat/completions",
        text="data: [DONE]\\n\\n",
        headers={"Content-Type": "text/event-stream"},
    )
    with app.test_request_context(
        "/v1/chat/completions",
        json={"model": inbound_model, "messages": []},
    ):
        response = forward_openai_compatible(
            request,
            _snapshot(provider, "provider-default-model"),
            target_model="provider-default-model",
        )
        response.get_data()

    assert json.loads(requests_mock.last_request.text)["model"] == (
        "provider-default-model"
    )
