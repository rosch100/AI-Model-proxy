"""Live inference activity on the tenant overview."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.admin.view_models import activity_board, format_relative_time
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


def test_activity_board_shows_average_tokens_per_request_and_last_query():
    """Bars compare average request size; catalog IDs stay visible over logical IDs."""
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
            occurred_at=now - timedelta(minutes=3),
        ),
    )

    board = activity_board(events, custom_model_id="cursor-acme-model", now=now)

    assert board.total_requests_in_window == 3
    assert board.total_requests_with_usage_in_window == 2
    assert board.total_tokens_per_request_label == "8.250"
    assert [row.requested_model for row in board.requests] == [
        "gpt-5",
        "openrouter/free",
        "cursor-acme-model",
    ]
    assert board.requests[-1].total_tokens_label == "15.000"
    assert board.requests[-1].input_tokens_label == "14.000"
    assert board.requests[-1].output_tokens_label == "1.000"
    assert board.requests[-1].provider_label == "Azure"
    assert board.requests[-1].time_label == "2026-10-03T16:19:00+00:00"
    assert board.requests[1].total_tokens_label is None
    azure = board.providers[0]
    assert azure.label == "Azure"
    assert azure.rows[0].model == "gpt-6-luna"
    assert azure.rows[0].inbound_model == "cursor-acme-model"
    assert azure.rows[0].tokens_per_request_label == "15.000"
    assert azure.rows[0].bar_percent == 100
    assert azure.rows[0].recency == "live"
    openai = board.providers[1]
    assert openai.rows[0].model == "gpt-5"
    assert openai.rows[0].tokens_per_request_label == "1.500"
    assert openai.rows[0].bar_percent == 10
    assert openai.rows[0].last_request_label == "vor 5 min"
    openrouter = board.providers[2]
    assert openrouter.rows[0].requests_in_window == 1
    assert openrouter.rows[0].requests_with_usage_in_window == 0
    assert openrouter.rows[0].tokens_per_request_label is None
    assert openrouter.rows[0].bar_percent == 0
    assert openrouter.rows[0].recency == "recent"


def test_relative_time_is_timezone_neutral_for_older_activity():
    """Older live timestamps use relative labels, not a UTC clock rendering."""
    now = datetime(2026, 10, 3, 16, 20, tzinfo=timezone.utc)

    assert format_relative_time(now - timedelta(hours=2), now) == "vor 2 Std."


def test_activity_board_filters_request_list_by_selected_period():
    """Show only list entries inside the selected lookback window."""
    now = datetime(2026, 10, 3, 16, 20, tzinfo=timezone.utc)
    events = tuple(
        InferenceActivityEvent(
            tenant_id="acme",
            provider="azure",
            inbound_model="gpt-6-luna",
            routed_model="gpt-6-luna",
            total_tokens=100,
            occurred_at=now - timedelta(minutes=minutes_old),
        )
        for minutes_old in (120, 30, 5)
    )

    board = activity_board(
        events, custom_model_id="cursor-acme-model", now=now, request_period_hours=1
    )

    assert [row.occurred_at for row in board.requests] == [
        now - timedelta(minutes=30),
        now - timedelta(minutes=5),
    ]


def test_activity_board_limits_request_list_to_most_recent_entries():
    """Keep the overview list compact while retaining the newest requests."""
    now = datetime(2026, 10, 3, 16, 20, tzinfo=timezone.utc)
    events = tuple(
        InferenceActivityEvent(
            tenant_id="acme",
            provider="azure",
            inbound_model="gpt-6-luna",
            routed_model="gpt-6-luna",
            input_tokens=100,
            output_tokens=10,
            cached_tokens=0,
            reasoning_tokens=0,
            total_tokens=110,
            occurred_at=now - timedelta(minutes=minute),
        )
        for minute in range(51)
    )

    board = activity_board(events, custom_model_id="cursor-acme-model", now=now)

    assert len(board.requests) == 50
    assert board.requests[0].occurred_at == now - timedelta(minutes=49)
    assert board.requests[-1].occurred_at == now


def test_dashboard_renders_activity_empty_state(admin_app):
    """An idle tenant still sees the live activity section."""
    body = _authenticated_client(admin_app).get("/admin/").get_data(as_text=True)

    assert "Aktuelle Aktivität" in body
    assert "Keine Proxy-Aktivität im ausgewählten Zeitraum" in body


def test_dashboard_rejects_unsupported_activity_period(admin_app):
    """Reject query values outside the available activity periods."""
    response = _authenticated_client(admin_app).get("/admin/?activity_hours=2")

    assert response.status_code == 400


def test_dashboard_activity_range_includes_older_events(admin_app):
    """The selected list range controls both initial render and HTMX refresh."""
    database = admin_app.extensions["database"]
    now = datetime.now(timezone.utc)
    with database.sessions.begin() as session:
        session.add_all(
            (
                InferenceActivityEvent(
                    tenant_id="acme",
                    provider="azure",
                    inbound_model="within-range-model",
                    routed_model="within-range-model",
                    input_tokens=8,
                    output_tokens=4,
                    total_tokens=12,
                    occurred_at=now - timedelta(days=2),
                ),
                InferenceActivityEvent(
                    tenant_id="acme",
                    provider="azure",
                    inbound_model="outside-range-model",
                    routed_model="outside-range-model",
                    input_tokens=6,
                    output_tokens=4,
                    total_tokens=10,
                    occurred_at=now - timedelta(days=8),
                ),
            )
        )

    client = _authenticated_client(admin_app)
    body = client.get("/admin/?activity_hours=168").get_data(as_text=True)
    fragment = client.get("/admin/activity?activity_hours=168").get_data(as_text=True)

    assert "within-range-model" in body
    assert "outside-range-model" not in body
    assert "/admin/activity?activity_hours=168" in body
    assert "within-range-model" in fragment
    assert "outside-range-model" not in fragment


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
    assert "Tokens / Anfrage" in body
    assert "Zeit (lokal)" in body
    assert "Zeitraum der Liste" in body
    assert "Letzte 7 Tage" in body
    assert "data-local-time" in body
    assert body.index("Live-Verbindungen") < body.index("Kostenübersicht nach Konto")
    assert body.index("Kostenübersicht nach Konto") < body.index("Aktuelle Aktivität")
    assert "15.000" in body
    assert "Eingabe 14.000 · Ausgabe 1.000" in body
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
