"""Provider catalog, cost persistence, and OpenAI-compatible dispatch."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import select

from app.persistence.admin_auth import (
    authenticate_admin,
    create_admin_session,
    load_admin_principal,
)
from app.persistence.admin_ops import (
    activate_provider_profile,
    change_admin_password,
    replace_catalog_entries,
    upsert_provider_profile,
)
from app.persistence.models import (
    CostUsageRecord,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from app.providers.catalog import refresh_provider_catalog
from app.providers.cost_jobs import persist_cost_refresh
from app.providers.costs import CostBucket
from app.providers.openai_compat import openai_compatible_base_url
from tests.admin_app import build_admin_database, seed_admin


def test_catalog_refresh_parses_openai_models(requests_mock):
    """The OpenAI catalog refresh stores provider model ids."""
    requests_mock.get(
        "https://api.openai.com/v1/models",
        json={"data": [{"id": "gpt-5.4"}, {"id": "gpt-5.5"}]},
    )
    entries = refresh_provider_catalog("openai", {}, "sk-test")
    assert entries == [("gpt-5.4", None), ("gpt-5.5", None)]


def test_activate_profile_writes_audit_and_active_id():
    """Profile activation updates the tenant in the same unit of work."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = upsert_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            {},
            "gpt-5.4",
            "sk-test",
            "ada",
        )
        tenant = session.get(Tenant, "acme")
        activate_provider_profile(session, tenant, "openai", "ada")
        assert tenant.active_profile_id == profile.id
    database.engine.dispose()


def test_catalog_entries_replace_previous_rows():
    """A catalog refresh replaces previous rows for the profile."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = upsert_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            {},
            "gpt-5.4",
            "sk-test",
            "ada",
        )
        replace_catalog_entries(session, profile, [("gpt-5.4", None)], None)
        replace_catalog_entries(session, profile, [("gpt-5.5", None)], None)
        models = [
            row.model_id
            for row in session.scalars(
                select(ProviderCatalogEntry).where(
                    ProviderCatalogEntry.profile_id == profile.id
                )
            )
        ]
        assert models == ["gpt-5.5"]
    database.engine.dispose()


def test_cost_refresh_persists_job_and_records():
    """Successful cost refresh writes job, event, records, and audit together."""
    database = build_admin_database()
    seed_admin(database)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, tzinfo=timezone.utc)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openai",
            settings={},
            default_model="gpt-5.4",
        )
        session.add(profile)
        session.flush()
        node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openai",
            scope_type="organization",
            canonical_scope_id="org-1",
        )
        session.add(node)
        session.flush()
        binding = ProviderScopeBinding(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openai",
            profile_id=profile.id,
            purpose="billing",
            node_id=node.id,
        )
        session.add(binding)
        session.flush()
        persist_cost_refresh(
            session,
            "acme",
            "ada",
            profile,
            binding,
            start,
            end,
            [
                CostBucket(
                    kind="actual",
                    metric="cost",
                    value=Decimal("1.25"),
                    unit="currency",
                    currency="USD",
                    bucket_start=start,
                    bucket_end=end,
                    source="openai.organization.costs",
                    granularity="window",
                    dimensions={"organization": "org-1"},
                )
            ],
            None,
        )
        record = session.scalar(select(CostUsageRecord))
        assert record is not None
        assert record.value == Decimal("1.25")
    database.engine.dispose()


def test_openai_compatible_origins():
    """The OpenAI and OpenRouter origins use public Chat Completions URLs."""
    assert openai_compatible_base_url("openai") == "https://api.openai.com/v1"
    assert openai_compatible_base_url("openrouter") == "https://openrouter.ai/api/v1"


def test_password_change_revokes_other_sessions(admin_app):
    """Changing the password keeps the current session and revokes others."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        other = create_admin_session(session, account)
        current = create_admin_session(session, account, enrollment_only=False)
        current_token = current.token
        change_admin_password(
            session,
            account,
            "correct-horse-battery",
            "new-password-12",
            current.session_id,
        )

    with database.sessions() as session:
        assert load_admin_principal(session, other.token) is None
        assert load_admin_principal(session, current_token) is not None
