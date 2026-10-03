"""OpenAI Administration API project discovery tests."""

from __future__ import annotations

import pytest
from gevent import sleep

from app.providers import openai_admin
from app.providers.openai_admin import OpenAIProjectLookupError, list_openai_projects


def test_list_openai_projects_returns_active_projects_and_uses_admin_key(requests_mock):
    """The project lookup uses the Admin API and returns validated project IDs."""
    request = requests_mock.get(
        "https://api.openai.com/v1/organization/projects",
        json={
            "object": "list",
            "data": [
                {"id": "proj_alpha-123", "name": "Production"},
                {"id": "proj_beta_456", "name": None},
            ],
            "has_more": False,
            "last_id": "proj_beta_456",
        },
    )

    assert list_openai_projects("secret-admin-key", "org-acme") == [
        ("proj_alpha-123", "Production"),
        ("proj_beta_456", "proj_beta_456"),
    ]
    assert request.last_request.headers["Authorization"] == "Bearer secret-admin-key"
    assert request.last_request.headers["OpenAI-Organization"] == "org-acme"
    assert request.last_request.qs["limit"] == ["100"]


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [], "has_more": "false"},
        {"data": [{"id": "org_wrong", "name": "Bad"}], "has_more": False},
        {"data": [{"id": "proj_alpha", "name": 42}], "has_more": False},
    ],
)
def test_list_openai_projects_rejects_invalid_payloads(requests_mock, payload):
    """Malformed project responses never yield an incomplete selectable scope."""
    requests_mock.get("https://api.openai.com/v1/organization/projects", json=payload)

    with pytest.raises(OpenAIProjectLookupError):
        list_openai_projects("secret-admin-key", "org-acme")


def test_list_openai_projects_follows_pagination(requests_mock):
    """Project discovery follows cursors until OpenAI returns a complete list."""
    request = requests_mock.get(
        "https://api.openai.com/v1/organization/projects",
        response_list=[
            {
                "json": {
                    "data": [{"id": "proj_first", "name": "First"}],
                    "has_more": True,
                    "last_id": "proj_first",
                }
            },
            {
                "json": {
                    "data": [{"id": "proj_second", "name": "Second"}],
                    "has_more": False,
                    "last_id": "proj_second",
                }
            },
        ],
    )

    assert list_openai_projects("secret-admin-key", "org-acme") == [
        ("proj_first", "First"),
        ("proj_second", "Second"),
    ]
    assert [item.qs.get("after") for item in request.request_history] == [
        None,
        ["proj_first"],
    ]


def test_list_openai_projects_enforces_total_deadline_during_streaming(monkeypatch):
    """A trickling response cannot extend a project lookup beyond its time budget."""

    class SlowResponse:
        """Yield chunks slowly enough to exceed the whole-request deadline."""

        status_code = 200

        def iter_content(self, chunk_size):
            while True:
                sleep(0.02)
                yield b" "

        def close(self):
            pass

    monkeypatch.setattr(
        openai_admin.requests,
        "get",
        lambda *args, **kwargs: SlowResponse(),
    )
    monkeypatch.setattr(openai_admin, "OPENAI_PROJECT_REQUEST_BUDGET_SECONDS", 0.08)

    with pytest.raises(OpenAIProjectLookupError, match="Zeitlimit überschritten"):
        list_openai_projects("secret-admin-key", "org-acme")
