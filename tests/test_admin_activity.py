"""Live inference activity on the tenant overview."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import inspect, select

from app.admin.view_models import activity_board, dashboard_view, format_relative_time
from app.persistence.inference_activity import (
    complete_provider_attempt,
    parse_provider_usage,
    record_inference_activity,
    start_provider_attempt,
)
from app.persistence.models import (
    Base,
    InferenceActivityEvent,
    ProviderAttemptEvent,
    ProviderProfile,
    Tenant,
)
from app.persistence.provider_circuit_breaker import ProviderCircuitBreakerStore
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
    assert board.window_summaries[0].minutes == 15
    assert board.window_summaries[0].total_requests == 3
    assert board.window_summaries[0].total_tokens_label == "16.500"
    assert board.window_summaries[1].minutes == 60
    assert board.window_summaries[1].total_requests == 3
    assert board.window_summaries[1].total_tokens_label == "16.500"
    assert [row.requested_model for row in board.requests] == [
        "cursor-acme-model",
        "openrouter/free",
        "gpt-5",
    ]
    assert board.requests[0].total_tokens_label == "15.000"
    assert board.requests[0].input_tokens_label == "14.000"
    assert board.requests[0].output_tokens_label == "1.000"
    assert board.requests[0].provider_label == "Azure"
    assert board.requests[0].time_label == "2026-10-03T16:19:00+00:00"
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
    assert openai.rows[0].last_request_label == "vor 5 Minuten"
    openrouter = board.providers[2]
    assert openrouter.rows[0].requests_in_window == 1
    assert openrouter.rows[0].requests_with_usage_in_window == 0
    assert openrouter.rows[0].tokens_per_request_label is None
    assert openrouter.rows[0].bar_percent == 0
    assert openrouter.rows[0].recency == "recent"


def test_activity_board_summarizes_requests_and_reported_tokens_per_window():
    """Fixed activity windows include counts and only reported token totals."""
    now = datetime(2026, 10, 3, 16, 20, tzinfo=timezone.utc)
    events = tuple(
        InferenceActivityEvent(
            tenant_id="acme",
            provider="azure",
            inbound_model="gpt-6-luna",
            routed_model="gpt-6-luna",
            total_tokens=total_tokens,
            occurred_at=now - timedelta(minutes=minutes_ago),
        )
        for minutes_ago, total_tokens in (
            (8, 100),
            (14, None),
            (45, 200),
            (60, 300),
            (61, 400),
        )
    )

    board = activity_board(events, custom_model_id="cursor-acme-model", now=now)

    short_window, long_window = board.window_summaries
    assert short_window.label == "Letzte 15 Minuten"
    assert short_window.total_requests == 2
    assert short_window.total_requests_with_usage == 1
    assert short_window.total_tokens_label == "100"
    assert long_window.label == "Letzte Stunde"
    assert long_window.total_requests == 4
    assert long_window.total_requests_with_usage == 3
    assert long_window.total_tokens_label == "600"


def test_relative_time_is_timezone_neutral_for_older_activity():
    """Older live timestamps use relative labels, not a UTC clock rendering."""
    now = datetime(2026, 10, 3, 16, 20, tzinfo=timezone.utc)

    assert format_relative_time(now - timedelta(hours=2), now) == "vor 2 Stunden"


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
        now - timedelta(minutes=5),
        now - timedelta(minutes=30),
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
    assert board.requests[0].occurred_at == now
    assert board.requests[-1].occurred_at == now - timedelta(minutes=49)


def test_activity_board_lists_provider_failures_as_attempts_not_requests():
    """Expose failed upstream attempts separately from completed user requests."""
    now = datetime(2026, 10, 3, 16, 20, tzinfo=timezone.utc)
    attempt = ProviderAttemptEvent(
        id=1,
        tenant_id="acme",
        provider="openrouter",
        profile_id="openrouter-main",
        inbound_model="gpt-6-luna",
        routed_model="openai/gpt-6-luna",
        outcome="failure",
        status_code=402,
        occurred_at=now - timedelta(minutes=2),
        completed_at=now - timedelta(minutes=1),
    )

    board = activity_board(
        (),
        custom_model_id="cursor-acme-model",
        now=now,
        failed_attempts=((attempt, "Altanis Proxy"),),
    )

    assert board.requests == ()
    assert len(board.failed_attempts) == 1
    assert board.failed_attempts[0].requested_model == "gpt-6-luna"
    assert board.failed_attempts[0].provider_label == "OpenRouter"
    assert board.failed_attempts[0].profile_name == "Altanis Proxy"
    assert board.failed_attempts[0].routed_model == "openai/gpt-6-luna"
    assert board.failed_attempts[0].status_code == 402
    assert board.failed_attempts[0].occurred_at == now - timedelta(minutes=1)


def test_dashboard_renders_activity_empty_state(admin_app):
    """An idle tenant still sees the current requests section."""
    body = _authenticated_client(admin_app).get("/admin/").get_data(as_text=True)

    assert "Aktuelle Anfragen" in body
    assert "Keine Anfragen im gewählten Zeitraum" in body
    assert "In diesem Zeitraum gab es noch keine Anfragen." in body
    assert "Letzte 15 Minuten" in body
    assert "Letzte Stunde" in body
    assert "Für 0 von 0 Anfragen liegen Tokenangaben vor" in body
    assert "Tokens insgesamt" in body


def test_dashboard_lists_failed_provider_attempts_separately(admin_app):
    """Render failed upstream attempts without counting them as user requests."""
    database = admin_app.extensions["database"]
    occurred_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="openrouter-failed-attempt-dashboard",
            tenant_id="acme",
            provider="openrouter",
            display_name="Altanis Proxy",
            settings={},
            default_model="openai/gpt-6-luna",
            route_priority=3,
            inference_secret_ciphertext="encrypted-key",
        )
        session.add(profile)
        session.flush()
        session.add(
            ProviderAttemptEvent(
                tenant_id="acme",
                provider="openrouter",
                profile_id=profile.id,
                inbound_model="gpt-6-luna",
                routed_model="openai/gpt-6-luna",
                outcome="failure",
                status_code=402,
                occurred_at=occurred_at,
                completed_at=occurred_at + timedelta(seconds=1),
            )
        )

    body = _authenticated_client(admin_app).get("/admin/").get_data(as_text=True)

    assert "Fehlgeschlagene Provider-Versuche" in body
    assert "OpenRouter · Altanis Proxy" in body
    assert "Angefragt: gpt-6-luna" in body
    assert "An Provider gesendet: <code>openai/gpt-6-luna</code>" in body
    assert "Provider-Versuch fehlgeschlagen · HTTP 402" in body
    assert "Keine Anfragen im gewählten Zeitraum" in body


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
    """The page shows understandable request, provider, and cost information."""
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
    assert "Tokens pro Anfrage" in body
    assert "Uhrzeiten werden in deiner Zeitzone angezeigt" in body
    assert "Anfragen anzeigen für" in body
    assert "Letzte 7 Tage" in body
    assert "data-local-time" in body
    assert body.index("Anbieterstatus") < body.index("Kosten pro Konto")
    assert body.index("Kosten pro Konto") < body.index("Aktuelle Anfragen")
    assert "15.000" in body
    assert "Eingabe: 14.000 · Ausgabe: 1.000" in body
    assert "Letzte 15 Minuten" in body
    assert "Letzte Stunde" in body
    assert "Tokens insgesamt" in body
    assert "Neueste Anfragen zuerst" in body
    assert "Azure" in body
    assert "Anbieterstatus" in body
    assert body.index("Anbieterstatus") < body.index("Aktuelle Anfragen")
    assert "zuletzt" in body
    assert 'id="dashboard-activity"' in fragment
    assert "gpt-6-luna" in fragment


def test_single_recent_openrouter_failure_is_reported_without_hysteresis(
    admin_app,
):
    """Show a transient OpenRouter 402 without treating it as quota exhaustion."""
    now = datetime.now(timezone.utc)
    tenant = Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="cursor-acme")
    profile = ProviderProfile(
        id="openrouter-single-inflight-limit",
        tenant_id="acme",
        provider="openrouter",
        display_name="In-flight limited",
        settings={},
        default_model="gpt-5.4",
        route_priority=1,
        inference_secret_ciphertext="encrypted-key",
    )

    status = dashboard_view(
        tenant,
        (profile,),
        provider_attempts=(
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="failure",
                status_code=402,
                occurred_at=now - timedelta(seconds=2),
                completed_at=now - timedelta(seconds=2),
            ),
        ),
        now=now,
    ).providers[0]

    assert status.has_recent_failure
    assert not status.is_overloaded
    assert status.outcome == "success"
    assert status.last_failure_status_code == 402


def test_pending_attempt_does_not_hide_active_failure_hysteresis(admin_app):
    """An in-flight attempt does not replace a still-active failure state."""
    now = datetime.now(timezone.utc)
    tenant = Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="cursor-acme")
    profile = ProviderProfile(
        id="openai-failure-with-pending",
        tenant_id="acme",
        provider="openai",
        display_name="Failure with pending attempt",
        settings={},
        default_model="gpt-5.4",
        route_priority=1,
        inference_secret_ciphertext="encrypted-key",
    )

    status = dashboard_view(
        tenant,
        (profile,),
        provider_attempts=(
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="failure",
                occurred_at=now - timedelta(seconds=8),
            ),
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="failure",
                occurred_at=now - timedelta(seconds=2),
            ),
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="pending",
                occurred_at=now - timedelta(seconds=1),
            ),
        ),
        now=now,
    ).providers[0]

    assert status.outcome == "failure"
    assert status.has_current_activity


def test_provider_status_hysteresis_requires_two_failures_then_quiets(admin_app):
    """Require two recent failures and clear only after a quiet recovery window."""
    now = datetime.now(timezone.utc)
    tenant = Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="cursor-acme")
    profile = ProviderProfile(
        id="openai-hysteresis",
        tenant_id="acme",
        provider="openai",
        display_name="Hysteresis",
        settings={},
        default_model="gpt-5.4",
        route_priority=1,
        inference_secret_ciphertext="encrypted-key",
    )
    profile_status = dashboard_view(
        tenant,
        (profile,),
        provider_attempts=(
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="failure",
                occurred_at=now - timedelta(seconds=8),
            ),
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="failure",
                occurred_at=now - timedelta(seconds=2),
            ),
        ),
        now=now,
    ).providers[0]
    quiet_status = dashboard_view(
        tenant,
        (profile,),
        provider_attempts=(
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="failure",
                occurred_at=now - timedelta(seconds=25),
            ),
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="failure",
                occurred_at=now - timedelta(seconds=22),
            ),
        ),
        now=now,
    ).providers[0]
    single_failure_status = dashboard_view(
        tenant,
        (profile,),
        provider_attempts=(
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="failure",
                occurred_at=now - timedelta(seconds=2),
            ),
        ),
        now=now,
    ).providers[0]
    long_running_status = dashboard_view(
        tenant,
        (profile,),
        provider_attempts=(
            ProviderAttemptEvent(
                profile_id=profile.id,
                outcome="pending",
                occurred_at=now - timedelta(seconds=60),
            ),
        ),
        now=now,
    ).providers[0]

    assert profile_status.outcome == "failure"
    assert quiet_status.outcome == "success"
    assert single_failure_status.outcome == "success"
    assert long_running_status.outcome == "pending"
    assert long_running_status.has_current_activity


def test_provider_status_needs_consecutive_failures_and_recovers_after_quiet_period(
    admin_app,
):
    """A success breaks failure streaks; another failure needs a fresh pair."""
    now = datetime.now(timezone.utc)
    tenant = Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="cursor-acme")
    profile = ProviderProfile(
        id="openai-hysteresis-reset",
        tenant_id="acme",
        provider="openai",
        display_name="Hysteresis reset",
        settings={},
        default_model="gpt-5.4",
        route_priority=1,
        inference_secret_ciphertext="encrypted-key",
    )
    attempts = (
        ProviderAttemptEvent(
            profile_id=profile.id,
            outcome="failure",
            occurred_at=now - timedelta(seconds=12),
        ),
        ProviderAttemptEvent(
            profile_id=profile.id,
            outcome="success",
            occurred_at=now - timedelta(seconds=8),
        ),
        ProviderAttemptEvent(
            profile_id=profile.id,
            outcome="failure",
            occurred_at=now - timedelta(seconds=2),
        ),
    )

    status = dashboard_view(
        tenant, (profile,), provider_attempts=attempts, now=now
    ).providers[0]

    assert status.outcome == "success"


def test_provider_attempt_lifecycle_prunes_events_older_than_one_day(admin_app):
    """Keep provider-attempt telemetry bounded to one day of history."""
    database = admin_app.extensions["database"]
    old_attempt = ProviderAttemptEvent(
        tenant_id="acme",
        provider="openai",
        profile_id="openai-retention",
        inbound_model="cursor-acme",
        routed_model="gpt-5.4",
        outcome="success",
        status_code=200,
        occurred_at=datetime.now(timezone.utc) - timedelta(days=2),
        completed_at=datetime.now(timezone.utc) - timedelta(days=2),
    )
    with database.sessions.begin() as session:
        session.add(
            ProviderProfile(
                id="openai-retention",
                tenant_id="acme",
                provider="openai",
                display_name="Retention",
                settings={},
                default_model="gpt-5.4",
                inference_secret_ciphertext="encrypted-key",
            )
        )
        session.flush()
        session.add(old_attempt)

    with database.sessions() as session:
        assert (
            session.scalar(
                select(ProviderAttemptEvent.id).where(
                    ProviderAttemptEvent.occurred_at
                    < datetime.now(timezone.utc) - timedelta(hours=24)
                )
            )
            == old_attempt.id
        )

    with admin_app.app_context():
        attempt_id = start_provider_attempt(
            tenant_id="acme",
            provider="openai",
            profile_id="openai-retention",
            inbound_model="cursor-acme",
            routed_model="gpt-5.4",
        )
    assert attempt_id is not None

    with database.sessions() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        expired = tuple(
            session.scalars(
                select(ProviderAttemptEvent).where(
                    ProviderAttemptEvent.occurred_at < cutoff
                )
            )
        )
        assert expired == ()
        assert session.get(ProviderAttemptEvent, attempt_id).outcome == "pending"


def test_provider_attempt_lifecycle_records_pending_then_terminal_outcome(admin_app):
    """Keep one attempt row while the adapter finalizes its stream result."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        session.add(
            ProviderProfile(
                id="openai-attempt-lifecycle",
                tenant_id="acme",
                provider="openai",
                display_name="Attempt lifecycle",
                settings={},
                default_model="gpt-5.4",
                route_priority=1,
                inference_secret_ciphertext="encrypted-key",
            )
        )

    with admin_app.app_context():
        attempt_id = start_provider_attempt(
            tenant_id="acme",
            provider="openai",
            profile_id="openai-attempt-lifecycle",
            inbound_model="cursor-acme",
            routed_model="gpt-5.4",
        )

    with database.sessions() as session:
        attempt = session.get(ProviderAttemptEvent, attempt_id)
        assert attempt.outcome == "pending"

    with admin_app.app_context():
        complete_provider_attempt(attempt_id, outcome="failure", status_code=429)

    with database.sessions() as session:
        attempt = session.get(ProviderAttemptEvent, attempt_id)

    assert attempt.outcome == "failure"
    assert attempt.status_code == 429
    assert attempt.completed_at is not None


def test_dashboard_provider_status_poll_renders_hysteresis_state(admin_app):
    """Refresh the provider icons with orange active status per profile."""
    database = admin_app.extensions["database"]
    now = datetime.now(timezone.utc)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="openai-warning-dashboard",
            tenant_id="acme",
            provider="openai",
            display_name="Warning profile",
            settings={},
            default_model="gpt-5.4",
            route_priority=1,
            inference_secret_ciphertext="encrypted-key",
        )
        session.add(profile)
        session.flush()
        session.add_all(
            (
                ProviderAttemptEvent(
                    tenant_id="acme",
                    provider="openai",
                    profile_id=profile.id,
                    inbound_model="cursor-acme",
                    routed_model="gpt-5.4",
                    outcome="failure",
                    status_code=429,
                    occurred_at=now - timedelta(seconds=8),
                    completed_at=now - timedelta(seconds=8),
                ),
                ProviderAttemptEvent(
                    tenant_id="acme",
                    provider="openai",
                    profile_id=profile.id,
                    inbound_model="cursor-acme",
                    routed_model="gpt-5.4",
                    outcome="failure",
                    status_code=429,
                    occurred_at=now - timedelta(seconds=2),
                    completed_at=now - timedelta(seconds=2),
                ),
                ProviderAttemptEvent(
                    tenant_id="acme",
                    provider="openai",
                    profile_id=profile.id,
                    inbound_model="cursor-acme",
                    routed_model="gpt-5.4",
                    outcome="success",
                    status_code=200,
                    occurred_at=now - timedelta(seconds=1),
                    completed_at=now - timedelta(seconds=1),
                ),
            )
        )

    response = _authenticated_client(admin_app).get("/admin/provider-status")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-provider-account="openai-warning-dashboard"' in body
    assert 'data-activity-outcome="failure"' in body
    assert "HTTP 429" in body
    card = re.search(
        r'<li class="provider-status-card[^\"]*" '
        r'data-provider-account="openai-warning-dashboard">([\s\S]*?)</li>',
        body,
    )
    assert card is not None
    assert 'class="provider-status-indicator is-warning"' in card.group(1)
    assert "in der konfigurierten Reihenfolge versucht" in body
    assert "durch Circuit-Breaker gesperrte Anbieter werden übersprungen" in body
    assert "nach wiederholbaren Upstreamfehlern wird mit dem nächsten" in body
    assert 'hx-trigger="every 5s"' in body


def test_provider_status_poll_uses_activity_without_building_history_board(
    admin_app, monkeypatch
):
    """The status fragment loads only status inputs, not the history board."""

    def unexpected_activity_query(*_args, **_kwargs):
        raise AssertionError(
            "provider status polling must not load full activity history"
        )

    def unexpected_activity_board(*_args, **_kwargs):
        raise AssertionError("provider status polling must not build the history board")

    monkeypatch.setattr("app.admin.views._activity_events", unexpected_activity_query)
    monkeypatch.setattr(
        "app.admin.view_models.activity_board", unexpected_activity_board
    )
    database = admin_app.extensions["database"]
    occurred_at = datetime.now(timezone.utc) - timedelta(hours=2)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="openai-provider-poll-history",
            tenant_id="acme",
            provider="openai",
            display_name="Poll history",
            settings={},
            default_model="gpt-5.4",
            inference_secret_ciphertext="encrypted-key",
        )
        session.add(profile)
        session.flush()
        session.add(
            InferenceActivityEvent(
                tenant_id="acme",
                provider="openai",
                profile_id=profile.id,
                inbound_model="cursor-acme",
                routed_model="gpt-5.4",
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                occurred_at=occurred_at,
            )
        )

    response = _authenticated_client(admin_app).get("/admin/provider-status")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "letzte Anfrage vor 2 Stunden" in body
    assert occurred_at.isoformat() in body


def test_provider_status_poll_keeps_attempts_running_beyond_fifteen_minutes(
    admin_app,
):
    """A provider attempt remains visible while a long-running stream is pending."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="openai-long-running-poll",
            tenant_id="acme",
            provider="openai",
            display_name="Long running",
            settings={},
            default_model="gpt-5.4",
            inference_secret_ciphertext="encrypted-key",
        )
        session.add(profile)
        session.flush()
        session.add(
            ProviderAttemptEvent(
                tenant_id="acme",
                provider="openai",
                profile_id=profile.id,
                inbound_model="cursor-acme",
                routed_model="gpt-5.4",
                outcome="pending",
                occurred_at=datetime.now(timezone.utc) - timedelta(minutes=20),
            )
        )

    body = (
        _authenticated_client(admin_app)
        .get("/admin/provider-status")
        .get_data(as_text=True)
    )

    assert 'data-provider-account="openai-long-running-poll"' in body
    assert 'data-activity-outcome="pending"' in body
    assert "Provideranfrage läuft" in body


def test_dashboard_renders_shared_quota_pause_for_configured_openai_profiles(
    admin_app,
):
    """Project organization breaker state to matching profiles without exposing IDs."""
    database = admin_app.extensions["database"]
    probe_at = datetime.now(timezone.utc) + timedelta(hours=2)
    with database.sessions.begin() as session:
        session.add_all(
            (
                ProviderProfile(
                    id="openai-shared-one",
                    tenant_id="acme",
                    provider="openai",
                    display_name="Shared One",
                    settings={"organization": "org-secret-123"},
                    default_model="gpt-5.4",
                    route_priority=1,
                    inference_secret_ciphertext="encrypted-one",
                ),
                ProviderProfile(
                    id="openai-shared-two",
                    tenant_id="acme",
                    provider="openai",
                    display_name="Shared Two",
                    settings={"organization": "org-secret-123"},
                    default_model="gpt-5.4",
                    route_priority=2,
                    inference_secret_ciphertext="encrypted-two",
                ),
                ProviderProfile(
                    id="openai-different-org",
                    tenant_id="acme",
                    provider="openai",
                    display_name="Different Org",
                    settings={"organization": "org-other"},
                    default_model="gpt-5.4",
                    route_priority=3,
                    inference_secret_ciphertext="encrypted-three",
                ),
            )
        )
    store = ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
    store.open_quota(
        store.scope("acme", "openai", "organization", "org-secret-123"),
        now=probe_at - timedelta(hours=2),
    )

    body = (
        _authenticated_client(admin_app)
        .get("/admin/provider-status")
        .get_data(as_text=True)
    )

    assert 'data-provider-account="openai-shared-one"' in body
    assert 'data-provider-account="openai-shared-two"' in body
    assert 'data-provider-account="openai-different-org"' in body
    assert body.count('class="provider-quota-status"') == 2
    assert "org-secret-123" not in body
    assert 'aria-label="Quota-Limit – pausiert bis' in body
    assert 'hx-trigger="every 5s"' in body


def test_provider_attempt_table_stores_outcome_and_profile(admin_app):
    """A provider attempt records its profile-level result without usage counts."""
    assert (
        "provider_attempt_events"
        in inspect(admin_app.extensions["database"].engine).get_table_names()
    )
    assert Base.metadata.tables["provider_attempt_events"].c.outcome.nullable is False


def test_deepseek_activity_is_grouped_and_missing_usage_stays_unknown():
    """The DeepSeek activity has its provider label without estimated counts."""
    now = datetime(2026, 10, 3, 16, 20, tzinfo=timezone.utc)
    event = InferenceActivityEvent(
        tenant_id="acme",
        provider="deepseek",
        inbound_model="cursor-acme-model",
        routed_model="deepseek-v4-flash",
        total_tokens=None,
        input_tokens=None,
        output_tokens=None,
        occurred_at=now - timedelta(minutes=1),
    )

    board = activity_board(
        [event],
        custom_model_id="cursor-acme-model",
        now=now,
    )

    assert board.providers[0].provider == "deepseek"
    assert board.providers[0].label == "DeepSeek"
    assert board.providers[0].rows[0].tokens_per_request is None
    assert board.requests[0].provider_label == "DeepSeek"
    assert board.requests[0].total_tokens_label is None


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
