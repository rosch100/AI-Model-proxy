"""Live inference activity on the tenant overview."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.admin.view_models import activity_board
from app.persistence.inference_activity import (
    parse_provider_usage,
    record_inference_activity,
)
from app.persistence.models import InferenceActivityEvent
from tests.test_admin_dashboard import _authenticated_client


def test_parse_provider_usage_reads_azure_and_chat_completions_shapes():
    """Accept both Responses and Chat Completions usage without inventing totals."""
    azure = parse_provider_usage(
        {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 1},
        }
    )
    chat = parse_provider_usage(
        {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 1},
        }
    )

    assert azure is not None
    assert azure.cached_tokens == 2
    assert azure.reasoning_tokens == 1
    assert chat is not None
    assert chat.total_tokens == 11
    assert parse_provider_usage({"input_tokens": 3}) is None


def test_activity_board_shows_tokens_per_minute_and_last_query():
    """Bars compare current throughput; catalog IDs stay visible over logical IDs."""
    now = datetime(2026, 10, 3, 16, 20, tzinfo=timezone.utc)
    events = (
        InferenceActivityEvent(
            tenant_id="acme",
            provider="azure",
            inbound_model="cursor-acme-model",
            routed_model="gpt-6-luna",
            input_tokens=14000,
            output_tokens=1000,
            cached_tokens=0,
            reasoning_tokens=0,
            total_tokens=15000,
            occurred_at=now - timedelta(minutes=1),
        ),
        InferenceActivityEvent(
            tenant_id="acme",
            provider="openai",
            inbound_model="gpt-5",
            routed_model="gpt-5",
            input_tokens=1400,
            output_tokens=100,
            cached_tokens=0,
            reasoning_tokens=0,
            total_tokens=1500,
            occurred_at=now - timedelta(minutes=5),
        ),
        InferenceActivityEvent(
            tenant_id="acme",
            provider="openrouter",
            inbound_model="openrouter/free",
            routed_model="openrouter/free",
            input_tokens=None,
            output_tokens=None,
            cached_tokens=None,
            reasoning_tokens=None,
            total_tokens=None,
            occurred_at=now - timedelta(hours=2),
        ),
    )

    board = activity_board(events, custom_model_id="cursor-acme-model", now=now)

    assert board.total_requests_in_window == 2
    assert board.total_tokens_per_minute_label == "1.100"
    azure = board.providers[0]
    assert azure.label == "Azure"
    assert azure.rows[0].model == "gpt-6-luna"
    assert azure.rows[0].inbound_model == "cursor-acme-model"
    assert azure.rows[0].tokens_per_minute_label == "1.000"
    assert azure.rows[0].bar_percent == 100
    assert azure.rows[0].recency == "live"
    openai = board.providers[1]
    assert openai.rows[0].model == "gpt-5"
    assert openai.rows[0].tokens_per_minute_label == "100"
    assert openai.rows[0].bar_percent == 10
    assert openai.rows[0].last_request_label == "vor 5 min"
    openrouter = board.providers[2]
    assert openrouter.rows[0].requests_in_window == 0
    assert openrouter.rows[0].tokens_per_minute_label is None
    assert openrouter.rows[0].bar_percent == 0
    assert openrouter.rows[0].recency == "idle"


def test_dashboard_renders_activity_empty_state(admin_app):
    """An idle tenant still sees the live activity section."""
    body = _authenticated_client(admin_app).get("/admin/").get_data(as_text=True)

    assert "Aktuelle Aktivität" in body
    assert "Keine aktuelle Proxy-Aktivität" in body


def test_dashboard_renders_provider_model_activity(admin_app):
    """The overview shows throughput and recency from recorded inferences."""
    database = admin_app.extensions["database"]
    now = datetime.now(timezone.utc)
    with database.sessions.begin() as session:
        session.add(
            InferenceActivityEvent(
                tenant_id="acme",
                provider="azure",
                inbound_model="gpt-6-luna",
                routed_model="gpt-6-luna",
                input_tokens=14000,
                output_tokens=1000,
                cached_tokens=0,
                reasoning_tokens=0,
                total_tokens=15000,
                occurred_at=now - timedelta(seconds=20),
            )
        )

    client = _authenticated_client(admin_app)
    body = client.get("/admin/").get_data(as_text=True)
    fragment = client.get("/admin/activity").get_data(as_text=True)

    assert "gpt-6-luna" in body
    assert "1.000 Tok/min" in body
    assert "Azure" in body
    assert "Live-Verbindungen" in body
    assert body.index("Live-Verbindungen") < body.index("Aktuelle Aktivität")
    assert "zuletzt" in body
    assert 'id="dashboard-activity"' in fragment
    assert "gpt-6-luna" in fragment


def test_record_inference_activity_stores_tenant_scoped_usage(admin_app):
    """Proxy completions persist the inbound model and token counts."""
    with admin_app.app_context():
        record_inference_activity(
            tenant_id="acme",
            provider="azure",
            profile_id=None,
            inbound_model="gpt-6-astra",
            routed_model="gpt-6-astra",
            usage=parse_provider_usage(
                {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
            ),
        )

    database = admin_app.extensions["database"]
    with database.sessions() as session:
        event = session.scalar(select(InferenceActivityEvent))

    assert event is not None
    assert event.inbound_model == "gpt-6-astra"
    assert event.total_tokens == 20
    assert event.provider == "azure"
