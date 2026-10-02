"""Provider catalog, cost persistence, and OpenAI-compatible dispatch."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
import requests
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
    CostRefreshEvent,
    CostRefreshJob,
    CostUsageRecord,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from app.providers import cost_jobs
from app.providers.catalog import refresh_provider_catalog
from app.providers.cost_jobs import collect_provider_costs, persist_cost_refresh
from app.providers.costs import (
    CostBucket,
    CostRefreshError,
    fetch_openai_costs,
    fetch_openrouter_costs,
)
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


def test_catalog_refresh_parses_azure_deployments(requests_mock):
    """Azure catalog refresh uses the legacy deployments API version."""
    requests_mock.get(
        "https://example.openai.azure.com/openai/deployments" "?api-version=2022-12-01",
        json={
            "data": [
                {"id": "gpt-6-astra", "model": "gpt-6-astra"},
                {"id": "gpt-6-luna", "model": "gpt-6-luna"},
            ]
        },
    )
    entries = refresh_provider_catalog(
        "azure",
        {"base_url": "https://example.openai.azure.com"},
        "azure-key",
    )
    assert entries == [("gpt-6-astra", "gpt-6-astra"), ("gpt-6-luna", "gpt-6-luna")]
    assert requests_mock.request_history[0].headers["api-key"] == "azure-key"


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
        job = cost_jobs.start_cost_refresh(session, "acme", "openai", start, end)[3]
        persist_cost_refresh(
            session,
            "acme",
            "ada",
            profile,
            binding,
            job,
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


def test_cost_refresh_reserves_running_job_before_provider_io():
    """Persist a running refresh attempt before making the provider request."""
    database = build_admin_database()
    seed_admin(database)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, tzinfo=timezone.utc)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            settings={},
            default_model="openai/gpt-5",
        )
        node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            scope_type="workspace",
            canonical_scope_id="550e8400-e29b-41d4-a716-446655440000",
        )
        session.add_all((profile, node))
        session.flush()
        binding = ProviderScopeBinding(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            profile_id=profile.id,
            purpose="billing",
            node_id=node.id,
        )
        session.add(binding)

    with database.sessions.begin() as session:
        _, _, _, job = cost_jobs.start_cost_refresh(
            session, "acme", "openrouter", start, end
        )
        job_id = job.id

    with pytest.raises(LookupError, match="already running"):
        with database.sessions.begin() as session:
            cost_jobs.start_cost_refresh(session, "acme", "openrouter", start, end)

    with database.sessions() as session:
        job = session.get(CostRefreshJob, job_id)
        event = session.scalar(
            select(CostRefreshEvent).where(CostRefreshEvent.job_id == job_id)
        )
        assert job is not None and job.status == "running"
        assert event is not None and event.new_status == "running"
    database.engine.dispose()


def test_cost_refresh_expires_stale_running_job_before_new_attempt():
    """Recover a crashed refresh after its bounded runtime plus grace period."""
    database = build_admin_database()
    seed_admin(database)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, tzinfo=timezone.utc)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            settings={},
            default_model="openai/gpt-5",
        )
        node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            scope_type="workspace",
            canonical_scope_id="550e8400-e29b-41d4-a716-446655440000",
        )
        session.add_all((profile, node))
        session.flush()
        binding = ProviderScopeBinding(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            profile_id=profile.id,
            purpose="billing",
            node_id=node.id,
        )
        session.add(binding)

    with database.sessions.begin() as session:
        _, _, _, stale_job = cost_jobs.start_cost_refresh(
            session, "acme", "openrouter", start, end
        )
        stale_job_id = stale_job.id
    with database.sessions.begin() as session:
        stale_job = session.get(CostRefreshJob, stale_job_id)
        stale_job.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)

    with database.sessions.begin() as session:
        _, _, _, next_job = cost_jobs.start_cost_refresh(
            session, "acme", "openrouter", start, end
        )
        next_job_id = next_job.id
    with database.sessions() as session:
        stale_job = session.get(CostRefreshJob, stale_job_id)
        next_job = session.get(CostRefreshJob, next_job_id)
        events = session.scalars(
            select(CostRefreshEvent)
            .where(CostRefreshEvent.job_id == stale_job_id)
            .order_by(CostRefreshEvent.id)
        ).all()
        profile = session.scalar(
            select(ProviderProfile).where(ProviderProfile.provider == "openrouter")
        )
        binding = session.scalar(select(ProviderScopeBinding))
        assert stale_job.status == "failed"
        assert next_job.status == "running"
        assert [event.new_status for event in events] == ["running", "failed"]

    with database.sessions.begin() as session:
        stale_job = session.get(CostRefreshJob, stale_job_id)
        profile = session.get(ProviderProfile, profile.id)
        binding = session.get(ProviderScopeBinding, binding.id)
        result = persist_cost_refresh(
            session,
            "acme",
            "ada",
            profile,
            binding,
            stale_job,
            [
                CostBucket(
                    kind="actual",
                    metric="cost",
                    value=Decimal("1.25"),
                    unit="currency",
                    currency="USD",
                    bucket_start=start,
                    bucket_end=end,
                    source="openrouter.activity",
                    granularity="day",
                    dimensions={},
                )
            ],
            None,
        )
        assert result is None
    with database.sessions() as session:
        assert session.scalar(select(CostUsageRecord)) is None
    database.engine.dispose()


def test_cost_refresh_reports_unreadable_billing_credentials():
    """An unreadable stored credential becomes a terminal refresh error."""
    database = build_admin_database()
    profile = ProviderProfile(
        id=str(uuid4()),
        tenant_id="acme",
        provider="openrouter",
        settings={},
        billing_secret_ciphertext="malformed-ciphertext",
    )

    with pytest.raises(CostRefreshError, match="credentials could not be decrypted"):
        collect_provider_costs(
            database.secret_cipher,
            profile,
            "550e8400-e29b-41d4-a716-446655440000",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
    database.engine.dispose()


def test_cost_refresh_marks_provider_transport_errors_failed(requests_mock):
    """Provider transport errors complete the reserved job as failed."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            settings={},
            default_model="openai/gpt-5",
            billing_secret_ciphertext=database.secret_cipher.encrypt("management-key"),
        )
        node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            scope_type="workspace",
            canonical_scope_id="550e8400-e29b-41d4-a716-446655440000",
        )
        session.add_all((profile, node))
        session.flush()
        binding = ProviderScopeBinding(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openrouter",
            profile_id=profile.id,
            purpose="billing",
            node_id=node.id,
        )
        session.add(binding)

    requests_mock.get(
        "https://openrouter.ai/api/v1/activity",
        exc=requests.Timeout,
    )
    with database.sessions.begin() as session:
        profile, binding, node, job = cost_jobs.start_cost_refresh(
            session,
            "acme",
            "openrouter",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
        job_id = job.id
    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile.id)
        with pytest.raises(CostRefreshError, match="billing request failed"):
            collect_provider_costs(
                database.secret_cipher,
                profile,
                node.canonical_scope_id,
                datetime(2026, 9, 1, tzinfo=timezone.utc),
                datetime(2026, 10, 1, tzinfo=timezone.utc),
            )
    with database.sessions.begin() as session:
        profile = session.get(ProviderProfile, profile.id)
        binding = session.get(ProviderScopeBinding, binding.id)
        job = session.get(CostRefreshJob, job_id)
        persist_cost_refresh(
            session,
            "acme",
            "ada",
            profile,
            binding,
            job,
            None,
            CostRefreshError("Provider billing request timed out."),
        )
    with database.sessions() as session:
        job = session.get(CostRefreshJob, job_id)
        events = session.scalars(
            select(CostRefreshEvent)
            .where(CostRefreshEvent.job_id == job_id)
            .order_by(CostRefreshEvent.id)
        ).all()
        assert job.status == "failed"
        assert [event.new_status for event in events] == ["running", "failed"]
    database.engine.dispose()


def test_openai_compatible_origins():
    """The OpenAI and OpenRouter origins use public Chat Completions URLs."""
    assert openai_compatible_base_url("openai") == "https://api.openai.com/v1"
    assert openai_compatible_base_url("openrouter") == "https://openrouter.ai/api/v1"


def test_openai_billing_scopes_costs_and_tokens_to_project(requests_mock):
    """Fetch project-scoped costs and completion token usage from OpenAI."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    requests_mock.get(
        "https://api.openai.com/v1/organization/costs",
        json={
            "data": [
                {
                    "start_time": int(start.timestamp()),
                    "end_time": int(end.timestamp()),
                    "results": [
                        {
                            "project_id": "project-1",
                            "amount": {"value": 1.25, "currency": "usd"},
                        }
                    ],
                }
            ],
            "has_more": False,
            "next_page": None,
        },
    )
    requests_mock.get(
        "https://api.openai.com/v1/organization/usage/completions",
        json={
            "data": [
                {
                    "start_time": int(start.timestamp()),
                    "end_time": int(end.timestamp()),
                    "results": [
                        {
                            "project_id": "project-1",
                            "model": "gpt-5",
                            "input_tokens": 100,
                            "output_tokens": 40,
                        }
                    ],
                }
            ],
            "has_more": False,
            "next_page": None,
        },
    )

    buckets = fetch_openai_costs("admin-key", "project-1", start, end)

    assert [(bucket.kind, bucket.metric, bucket.value) for bucket in buckets] == [
        ("actual", "cost", Decimal("1.25")),
        ("usage", "input_tokens", Decimal("100")),
        ("usage", "output_tokens", Decimal("40")),
    ]
    for request in requests_mock.request_history:
        params = parse_qs(urlparse(request.url).query)
        assert params["project_ids"] == ["project-1"]
        assert (
            params["group_by"] == ["project_id", "model"]
            if "usage" in request.url
            else ["project_id"]
        )
        assert request.headers["Authorization"] == "Bearer admin-key"
        assert "OpenAI-Organization" not in request.headers


def test_openai_billing_follows_pagination_for_costs_and_usage(requests_mock):
    """Fetch all pages for both OpenAI costs and completion usage."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    bucket = {
        "start_time": int(start.timestamp()),
        "end_time": int(end.timestamp()),
    }
    costs_url = "https://api.openai.com/v1/organization/costs"
    usage_url = "https://api.openai.com/v1/organization/usage/completions"
    requests_mock.get(
        costs_url,
        response_list=[
            {
                "json": {
                    "data": [
                        {
                            **bucket,
                            "results": [
                                {
                                    "project_id": "project-1",
                                    "amount": {"value": 1.25, "currency": "usd"},
                                }
                            ],
                        }
                    ],
                    "has_more": True,
                    "next_page": "costs-cursor",
                }
            },
            {
                "json": {
                    "data": [
                        {
                            **bucket,
                            "results": [
                                {
                                    "project_id": "project-1",
                                    "amount": {"value": 2.50, "currency": "usd"},
                                }
                            ],
                        }
                    ],
                    "has_more": False,
                    "next_page": None,
                }
            },
        ],
    )
    requests_mock.get(
        usage_url,
        response_list=[
            {
                "json": {
                    "data": [
                        {
                            **bucket,
                            "results": [
                                {
                                    "project_id": "project-1",
                                    "model": "gpt-5",
                                    "input_tokens": 100,
                                    "output_tokens": 40,
                                }
                            ],
                        }
                    ],
                    "has_more": True,
                    "next_page": "usage-cursor",
                }
            },
            {
                "json": {
                    "data": [
                        {
                            **bucket,
                            "results": [
                                {
                                    "project_id": "project-1",
                                    "model": "gpt-5",
                                    "input_tokens": 200,
                                    "output_tokens": 80,
                                }
                            ],
                        }
                    ],
                    "has_more": False,
                    "next_page": None,
                }
            },
        ],
    )

    buckets = fetch_openai_costs("admin-key", "project-1", start, end)

    assert [bucket.value for bucket in buckets if bucket.metric == "cost"] == [
        Decimal("1.25"),
        Decimal("2.50"),
    ]
    assert [bucket.value for bucket in buckets if bucket.metric == "input_tokens"] == [
        Decimal("100"),
        Decimal("200"),
    ]
    assert [
        parse_qs(urlparse(request.url).query).get("page")
        for request in requests_mock.request_history
    ] == [None, ["costs-cursor"], None, ["usage-cursor"]]


def test_openai_billing_rejects_unbounded_pagination(requests_mock):
    """Fail instead of leaving a refresh unbounded on endless provider pages."""

    def endless_pages(request, context):
        page_number = len(requests_mock.request_history)
        return {
            "data": [],
            "has_more": True,
            "next_page": f"cursor-{page_number}",
        }

    requests_mock.get(
        "https://api.openai.com/v1/organization/costs", json=endless_pages
    )

    with pytest.raises(CostRefreshError, match="exceeded the maximum page count"):
        fetch_openai_costs(
            "admin-key",
            "project-1",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )

    assert len(requests_mock.request_history) == 5


def test_openai_billing_rejects_invalid_currency_code(requests_mock):
    """Reject malformed currency codes rather than overflowing storage."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    requests_mock.get(
        "https://api.openai.com/v1/organization/costs",
        json={
            "data": [
                {
                    "start_time": int(start.timestamp()),
                    "end_time": int(end.timestamp()),
                    "results": [
                        {
                            "project_id": "project-1",
                            "amount": {"value": 1.25, "currency": "usdollars"},
                        }
                    ],
                }
            ],
            "has_more": False,
        },
    )
    requests_mock.get(
        "https://api.openai.com/v1/organization/usage/completions",
        json={"data": [], "has_more": False},
    )

    with pytest.raises(CostRefreshError, match="currency code is unsupported"):
        fetch_openai_costs("admin-key", "project-1", start, end)


def test_openai_billing_rejects_more_pages_without_cursor(requests_mock):
    """Fail billing refresh when OpenAI pagination omits its next cursor."""
    requests_mock.get(
        "https://api.openai.com/v1/organization/costs",
        json={"data": [], "has_more": True, "next_page": None},
    )
    requests_mock.get(
        "https://api.openai.com/v1/organization/usage/completions",
        json={"data": [], "has_more": False, "next_page": None},
    )

    with pytest.raises(CostRefreshError, match="no next page cursor"):
        fetch_openai_costs(
            "admin-key",
            "project-1",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    "invalid_value",
    ["not-a-number", "NaN", "Infinity", -1, "1e18", "0.00000000001"],
)
def test_openai_billing_rejects_invalid_cost_values(requests_mock, invalid_value):
    """Reject malformed or negative cost values from OpenAI."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    requests_mock.get(
        "https://api.openai.com/v1/organization/costs",
        json={
            "data": [
                {
                    "start_time": int(start.timestamp()),
                    "end_time": int(end.timestamp()),
                    "results": [
                        {
                            "project_id": "project-1",
                            "amount": {"value": invalid_value, "currency": "usd"},
                        }
                    ],
                }
            ],
            "has_more": False,
        },
    )
    requests_mock.get(
        "https://api.openai.com/v1/organization/usage/completions",
        json={"data": [], "has_more": False},
    )

    with pytest.raises(
        CostRefreshError, match=r"OpenAI costs cost (is invalid|exceeds storage)"
    ):
        fetch_openai_costs("admin-key", "project-1", start, end)


@pytest.mark.parametrize("invalid_value", ["not-a-number", "NaN", -1, 1.5])
def test_openai_billing_rejects_invalid_token_values(requests_mock, invalid_value):
    """Reject malformed, negative, or fractional token counts from OpenAI."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    requests_mock.get(
        "https://api.openai.com/v1/organization/costs",
        json={"data": [], "has_more": False},
    )
    requests_mock.get(
        "https://api.openai.com/v1/organization/usage/completions",
        json={
            "data": [
                {
                    "start_time": int(start.timestamp()),
                    "end_time": int(end.timestamp()),
                    "results": [
                        {
                            "project_id": "project-1",
                            "input_tokens": invalid_value,
                        }
                    ],
                }
            ],
            "has_more": False,
        },
    )

    with pytest.raises(CostRefreshError, match="OpenAI usage input_tokens is invalid"):
        fetch_openai_costs("admin-key", "project-1", start, end)


def test_openrouter_activity_returns_daily_costs_and_token_usage(requests_mock):
    """Fetch daily costs and token usage from OpenRouter activity."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    requests_mock.get(
        "https://openrouter.ai/api/v1/activity",
        json={
            "data": [
                {
                    "date": "2026-09-01",
                    "model": "openai/gpt-5",
                    "endpoint_id": "endpoint-1",
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "reasoning_tokens": 12,
                    "usage": 0.75,
                    "byok_usage_inference": 0.12,
                }
            ]
        },
    )

    buckets = fetch_openrouter_costs(
        "management-key", "550e8400-e29b-41d4-a716-446655440000", start, end
    )

    assert {(bucket.kind, bucket.metric, bucket.value) for bucket in buckets} == {
        ("usage", "prompt_tokens", Decimal("100")),
        ("usage", "completion_tokens", Decimal("40")),
        ("usage", "reasoning_tokens", Decimal("12")),
        ("actual", "cost", Decimal("0.75")),
        ("actual", "byok_inference_cost", Decimal("0.12")),
    }
    assert all(
        bucket.bucket_start == start
        and bucket.bucket_end == end
        and bucket.granularity == "day"
        and bucket.source == "openrouter.activity"
        for bucket in buckets
    )
    assert all(
        bucket.dimensions == {"model": "openai/gpt-5", "endpoint_id": "endpoint-1"}
        for bucket in buckets
    )
    request = requests_mock.request_history[0]
    assert request.method == "GET"
    assert urlparse(request.url).path == "/api/v1/activity"
    assert len(requests_mock.request_history) == 1
    assert parse_qs(urlparse(request.url).query) == {
        "workspace_id": ["550e8400-e29b-41d4-a716-446655440000"]
    }
    assert request.headers["Authorization"] == "Bearer management-key"


@pytest.mark.parametrize("invalid_value", ["1e18", "0.00000000001"])
def test_openrouter_activity_rejects_values_outside_storage_precision(
    requests_mock, invalid_value
):
    """Reject activity values that cannot fit the persisted Numeric column."""
    requests_mock.get(
        "https://openrouter.ai/api/v1/activity",
        json={
            "data": [
                {
                    "date": "2026-09-01",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "reasoning_tokens": 0,
                    "usage": invalid_value,
                    "byok_usage_inference": 0,
                }
            ]
        },
    )

    with pytest.raises(CostRefreshError, match="exceeds storage precision"):
        fetch_openrouter_costs(
            "management-key",
            "550e8400-e29b-41d4-a716-446655440000",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )


def test_openrouter_activity_rejects_partial_day_windows(requests_mock):
    """Require completed UTC-day boundaries for activity snapshots."""
    with pytest.raises(CostRefreshError, match="must use full UTC days"):
        fetch_openrouter_costs(
            "management-key",
            "550e8400-e29b-41d4-a716-446655440000",
            datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )

    assert requests_mock.request_history == []


def test_openrouter_activity_requires_workspace_uuid(requests_mock):
    """Reject an account ID before sending it as an activity workspace filter."""
    with pytest.raises(CostRefreshError, match="workspace ID must be a UUID"):
        fetch_openrouter_costs(
            "management-key",
            "account-id",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )

    assert requests_mock.request_history == []


def test_openrouter_activity_excludes_records_outside_requested_period(requests_mock):
    """Exclude OpenRouter activity outside the refresh window."""
    start = datetime(2026, 9, 2, tzinfo=timezone.utc)
    end = datetime(2026, 9, 3, tzinfo=timezone.utc)
    requests_mock.get(
        "https://openrouter.ai/api/v1/activity",
        json={
            "data": [
                {
                    "date": "2026-09-01",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "reasoning_tokens": 0,
                    "usage": 0.01,
                    "byok_usage_inference": 0,
                },
                {
                    "date": "2026-09-02",
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "reasoning_tokens": 3,
                    "usage": 0.25,
                    "byok_usage_inference": 0.05,
                },
            ]
        },
    )

    buckets = fetch_openrouter_costs(
        "management-key", "550e8400-e29b-41d4-a716-446655440000", start, end
    )

    assert len(buckets) == 5
    assert {bucket.bucket_start for bucket in buckets} == {start}
    assert parse_qs(urlparse(requests_mock.last_request.url).query) == {
        "workspace_id": ["550e8400-e29b-41d4-a716-446655440000"]
    }


def test_cost_refresh_requires_separate_billing_credential(requests_mock):
    """Cost refresh never sends the inference credential to a billing API."""
    database = build_admin_database()
    seed_admin(database)
    try:
        with database.sessions.begin() as session:
            profile = ProviderProfile(
                id=str(uuid4()),
                tenant_id="acme",
                provider="openrouter",
                settings={},
                default_model="openai/gpt-5",
                inference_secret_ciphertext=database.secret_cipher.encrypt(
                    "inference-only-secret"
                ),
            )
            session.add(profile)
            node = ProviderScopeNode(
                id=str(uuid4()),
                tenant_id="acme",
                provider="openrouter",
                scope_type="workspace",
                canonical_scope_id="550e8400-e29b-41d4-a716-446655440000",
            )
            session.add(node)
            session.flush()
            binding = ProviderScopeBinding(
                id=str(uuid4()),
                tenant_id="acme",
                provider="openrouter",
                profile_id=profile.id,
                purpose="billing",
                node_id=node.id,
            )
            session.add(binding)

        profile_id = profile.id
        with database.sessions() as session:
            profile = session.get(ProviderProfile, profile_id)
            with pytest.raises(
                CostRefreshError, match="Billing credentials are missing"
            ):
                collect_provider_costs(
                    database.secret_cipher,
                    profile,
                    "account-1",
                    datetime(2026, 9, 1, tzinfo=timezone.utc),
                    datetime(2026, 10, 1, tzinfo=timezone.utc),
                )
        assert requests_mock.request_history == []
    finally:
        database.engine.dispose()


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
