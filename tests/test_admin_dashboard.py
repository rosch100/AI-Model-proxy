"""Dashboard cost and usage presentation tests."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.admin.security import ADMIN_COOKIE_NAME
from app.persistence.admin_auth import authenticate_admin, create_admin_session
from app.persistence.models import (
    CostRefreshJob,
    CostUsageRecord,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
)
from app.persistence.passkeys import insert_passkey


def _authenticated_client(admin_app):
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        insert_passkey(
            session,
            account_id=account.id,
            credential_id=secrets.token_bytes(32),
            public_key=secrets.token_bytes(64),
            sign_count=0,
            user_handle=secrets.token_bytes(32),
            label="Primary",
            aaguid=None,
            backed_up=False,
        )
        principal = create_admin_session(session, account, enrollment_only=False)
    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    return client


def _add_refresh(session, *, job_id, binding_id, status, created_at, amount=None):
    period_end = datetime(2026, 10, 1, tzinfo=timezone.utc)
    period_start = period_end - timedelta(days=30)
    session.add(
        CostRefreshJob(
            id=job_id,
            tenant_id="acme",
            provider="openai",
            binding_id=binding_id,
            period_start=period_start,
            period_end=period_end,
            source_api="openai",
            operation_key=job_id,
            status=status,
            retry_at=None,
            limitation="Billing API nicht erreichbar" if status == "failed" else None,
            created_at=created_at,
            completed_at=created_at,
        )
    )
    if amount is not None:
        session.add(
            CostUsageRecord(
                job_id=job_id,
                tenant_id="acme",
                provider="openai",
                binding_id=binding_id,
                kind="actual",
                metric="cost",
                value=Decimal(amount),
                unit="currency",
                currency="USD",
                bucket_start=period_start,
                bucket_end=period_end,
                source="openai.organization.costs",
                granularity="window",
                dimensions={},
            )
        )


def test_dashboard_shows_status_for_each_same_provider_account(admin_app):
    """Show every account and its independent route status on the dashboard."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        session.add_all(
            (
                ProviderProfile(
                    id="openai-primary-dashboard",
                    tenant_id="acme",
                    provider="openai",
                    display_name="Primary OpenAI",
                    settings={},
                    default_model="gpt-5.4",
                    route_priority=1,
                    inference_secret_ciphertext="encrypted-primary",
                ),
                ProviderProfile(
                    id="openai-backup-dashboard",
                    tenant_id="acme",
                    provider="openai",
                    display_name="Backup OpenAI",
                    settings={},
                    default_model="gpt-5.4",
                    route_priority=2,
                    inference_secret_ciphertext="encrypted-backup",
                ),
            )
        )

    response = _authenticated_client(admin_app).get("/admin/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-provider-account="openai-primary-dashboard"' in body
    assert 'data-provider-account="openai-backup-dashboard"' in body
    assert "Primary OpenAI" in body
    assert "Backup OpenAI" in body
    assert "Position 1" in body
    assert "Position 2" in body


def test_dashboard_explains_missing_cost_configuration(admin_app):
    """An unconfigured tenant sees a direct path to billing setup."""
    response = _authenticated_client(admin_app).get("/admin/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Noch keine Kosten- oder Verbrauchsdaten" in body
    assert 'href="/admin/settings/costs"' in body
    assert "Die Abrechnung ist noch nicht eingerichtet" in body
    assert "Anbieterstatus" in body
    assert "<title>Anfragen und Kosten · Altanis Proxy</title>" in body
    assert '<h1 id="dashboard-title">Anfragen und Kosten</h1>' in body
    assert body.index("Anbieterstatus") < body.index("Kosten pro Konto")
    assert body.index("Kosten pro Konto") < body.index("Aktuelle Anfragen")
    assert "in den letzten 24 Stunden gab es keine Anfragen" in body
    assert (
        "Hier siehst du, welche Anbieter Anfragen bearbeiten, wie viele Anfragen eingehen und welche Kosten entstehen."
        in body
    )
    assert "Nicht eingerichtet" in body
    assert "Anfragen &amp; Kosten" not in body
    assert "Mandant acme" not in body
    assert "Custom-Model-ID" not in body
    assert "Billing-Scope" not in body
    assert "Snapshot" not in body


def test_dashboard_distinguishes_empty_successful_refresh_from_no_refresh(admin_app):
    """A successful empty cost result is not reported as an unrun refresh."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="openai-empty",
            tenant_id="acme",
            provider="openai",
            display_name="Empty Snapshot",
            settings={},
            billing_secret_ciphertext="encrypted",
        )
        session.add(profile)
        session.flush()
        node = ProviderScopeNode(
            id="openai-empty-node",
            tenant_id="acme",
            provider="openai",
            scope_type="project",
            canonical_scope_id="project-empty",
        )
        session.add(node)
        session.flush()
        binding = ProviderScopeBinding(
            id="openai-empty-binding",
            tenant_id="acme",
            provider="openai",
            profile_id=profile.id,
            purpose="billing",
            node_id=node.id,
        )
        session.add(binding)
        session.flush()
        _add_refresh(
            session,
            job_id="empty-success",
            binding_id=binding.id,
            status="success",
            created_at=datetime(2026, 10, 3, 5, tzinfo=timezone.utc),
        )
        pending_profile = ProviderProfile(
            id="openai-not-refreshed",
            tenant_id="acme",
            provider="openai",
            display_name="Not Refreshed",
            settings={},
            billing_secret_ciphertext="encrypted",
        )
        session.add(pending_profile)
        session.flush()
        pending_node = ProviderScopeNode(
            id="openai-not-refreshed-node",
            tenant_id="acme",
            provider="openai",
            scope_type="project",
            canonical_scope_id="project-not-refreshed",
        )
        session.add(pending_node)
        session.flush()
        session.add(
            ProviderScopeBinding(
                id="openai-not-refreshed-binding",
                tenant_id="acme",
                provider="openai",
                profile_id=pending_profile.id,
                purpose="billing",
                node_id=pending_node.id,
            )
        )

    response = _authenticated_client(admin_app).get("/admin/")
    body = response.get_data(as_text=True)

    assert (
        "Für einige Konten liegen noch keine erfolgreich abgerufenen Daten vor; "
        "bei den übrigen Konten fehlen Kosten- und Verbrauchsdaten."
    ) in body
    assert "Die letzten erfolgreichen Abrufe enthielten keine Kostendaten." not in body


def test_dashboard_uses_latest_successful_refresh_without_double_counting(admin_app):
    """Current totals use the latest successful retrieval, even after a later failure."""
    database = admin_app.extensions["database"]
    now = datetime(2026, 10, 3, 5, tzinfo=timezone.utc)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="openai-production",
            tenant_id="acme",
            provider="openai",
            display_name="Production",
            settings={},
            billing_secret_ciphertext="encrypted",
        )
        session.add(profile)
        session.flush()
        node = ProviderScopeNode(
            id="openai-node",
            tenant_id="acme",
            provider="openai",
            scope_type="project",
            canonical_scope_id="project-production",
        )
        session.add(node)
        session.flush()
        binding = ProviderScopeBinding(
            id="openai-binding",
            tenant_id="acme",
            provider="openai",
            profile_id=profile.id,
            purpose="billing",
            node_id=node.id,
        )
        session.add(binding)
        session.flush()
        _add_refresh(
            session,
            job_id="old-success",
            binding_id=binding.id,
            status="success",
            created_at=now - timedelta(days=2),
            amount="100.00",
        )
        _add_refresh(
            session,
            job_id="latest-success",
            binding_id=binding.id,
            status="success",
            created_at=now - timedelta(days=1),
            amount="0.0042",
        )
        session.add(
            CostUsageRecord(
                job_id="latest-success",
                tenant_id="acme",
                provider="openai",
                binding_id=binding.id,
                kind="actual",
                metric="cost",
                value=Decimal("2.00"),
                unit="currency",
                currency="EUR",
                bucket_start=now - timedelta(days=30),
                bucket_end=now,
                source="test.costs",
                granularity="window",
                dimensions={},
            )
        )
        session.add(
            CostUsageRecord(
                job_id="latest-success",
                tenant_id="acme",
                provider="openai",
                binding_id=binding.id,
                kind="usage",
                metric="input_tokens",
                value=Decimal("1200"),
                unit="tokens",
                currency=None,
                bucket_start=now - timedelta(days=30),
                bucket_end=now,
                source="openai.organization.usage.completions",
                granularity="day",
                dimensions={},
            )
        )
        _add_refresh(
            session,
            job_id="latest-failure",
            binding_id=binding.id,
            status="failed",
            created_at=now,
        )

    response = _authenticated_client(admin_app).get("/admin/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "0,0042 USD" in body
    assert "2,00 EUR" in body
    assert "100,00 USD" not in body
    assert "1.200" in body
    assert "Production" in body
    assert "Abruf fehlgeschlagen" in body
    assert "Beim letzten Abruf gab es ein Problem." in body
    assert "Billing API nicht erreichbar" not in body
    assert "Anbieterstatus" in body

    costs_body = (
        _authenticated_client(admin_app)
        .get("/admin/settings/costs")
        .get_data(as_text=True)
    )
    assert "Fehlgeschlagen (Billing API nicht erreichbar)" in costs_body
    database.engine.dispose()
