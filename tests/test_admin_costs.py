"""Provider account and immutable cost-scope schema constraints."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
import requests
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.admin.security import ADMIN_COOKIE_NAME
from app.persistence.admin_auth import authenticate_admin, create_admin_session
from app.persistence.admin_ops import create_provider_profile, save_billing_secret
from app.persistence.models import (
    AuditEvent,
    CostRefreshJob,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
)
from app.persistence.passkeys import insert_passkey
from app.providers.cost_jobs import collect_provider_costs, load_billing_binding
from app.providers.costs import CostBucket, CostRefreshError, fetch_azure_costs
from tests.admin_app import build_admin_database, seed_admin


def test_profiles_allow_same_provider_with_casefold_unique_active_names():
    """Named same-provider profiles coexist while active names remain unique."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        session.add_all(
            (
                ProviderProfile(
                    id="azure-production",
                    tenant_id="acme",
                    provider="azure",
                    display_name="Production",
                    settings={},
                ),
                ProviderProfile(
                    id="azure-staging",
                    tenant_id="acme",
                    provider="azure",
                    display_name="Staging",
                    settings={},
                ),
            )
        )

    with database.sessions.begin() as session:
        session.add(
            ProviderProfile(
                id="azure-duplicate-name",
                tenant_id="acme",
                provider="azure",
                display_name="pRODUCTION",
                settings={},
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
    database.engine.dispose()


def test_scope_node_cannot_be_bound_to_multiple_profiles():
    """A globally exclusive scope node has at most one profile binding."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profiles = [
            ProviderProfile(
                id=f"openai-{suffix}",
                tenant_id="acme",
                provider="openai",
                display_name=f"Account {suffix}",
                settings={},
            )
            for suffix in ("one", "two")
        ]
        session.add_all(profiles)
        node = ProviderScopeNode(
            id=str(uuid4()),
            tenant_id="acme",
            provider="openai",
            scope_type="organization",
            canonical_scope_id="org-scope",
        )
        session.add(node)
        session.flush()
        session.add(
            ProviderScopeBinding(
                id=str(uuid4()),
                tenant_id="acme",
                provider="openai",
                profile_id=profiles[0].id,
                purpose="billing",
                node_id=node.id,
            )
        )
        session.flush()
        session.add(
            ProviderScopeBinding(
                id=str(uuid4()),
                tenant_id="acme",
                provider="openai",
                profile_id=profiles[1].id,
                purpose="billing",
                node_id=node.id,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
    database.engine.dispose()


def test_billing_binding_selects_requested_profile():
    """Billing lookup stays attached to the explicitly requested account."""
    database = build_admin_database()
    seed_admin(database)
    profile_ids = ("openai-production", "openai-staging")
    with database.sessions.begin() as session:
        for profile_id, suffix in zip(profile_ids, ("production", "staging")):
            profile = ProviderProfile(
                id=profile_id,
                tenant_id="acme",
                provider="openai",
                display_name=suffix,
                settings={},
            )
            session.add(profile)
            session.flush()
            node = ProviderScopeNode(
                id=f"node-{suffix}",
                tenant_id="acme",
                provider="openai",
                scope_type="organization",
                canonical_scope_id=f"org-{suffix}",
            )
            session.add(node)
            session.flush()
            session.add(
                ProviderScopeBinding(
                    id=f"binding-{suffix}",
                    tenant_id="acme",
                    provider="openai",
                    profile_id=profile_id,
                    purpose="billing",
                    node_id=node.id,
                )
            )

    with database.sessions() as session:
        profile, binding, node, usage_node = load_billing_binding(
            session, "acme", profile_ids[1]
        )
        assert profile.id == profile_ids[1]
        assert binding.profile_id == profile_ids[1]
        assert node.canonical_scope_id == "org-staging"
        assert usage_node is None
    database.engine.dispose()


def test_billing_secret_is_saved_to_requested_profile_only():
    """Updating a billing credential cannot select another same-provider account."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        session.add_all(
            (
                ProviderProfile(
                    id="openai-production",
                    tenant_id="acme",
                    provider="openai",
                    display_name="Production",
                    settings={},
                ),
                ProviderProfile(
                    id="openai-staging",
                    tenant_id="acme",
                    provider="openai",
                    display_name="Staging",
                    settings={},
                ),
            )
        )

    with database.sessions.begin() as session:
        save_billing_secret(
            session,
            database.secret_cipher,
            "acme",
            "openai-staging",
            "billing-secret",
            "ada",
        )
    with database.sessions() as session:
        production = session.get(ProviderProfile, "openai-production")
        staging = session.get(ProviderProfile, "openai-staging")
        assert production.billing_secret_ciphertext is None
        assert database.secret_cipher.decrypt(staging.billing_secret_ciphertext) == (
            "billing-secret"
        )
    database.engine.dispose()


def test_azure_cost_response_preserves_daily_currency_buckets(
    requests_mock, monkeypatch
):
    """Azure daily costs retain date/currency and only accept the bound resource."""
    monkeypatch.setattr("app.providers.costs._azure_arm_token", lambda secret: "token")
    resource_id = (
        "/subscriptions/sub/resourcegroups/rg/providers/"
        "microsoft.cognitiveservices/accounts/llm"
    )
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, tzinfo=timezone.utc)
    request = requests_mock.post(
        "https://management.azure.com/subscriptions/sub/resourceGroups/rg"
        "/providers/Microsoft.CostManagement/query?api-version=2023-11-01",
        json={
            "properties": {
                "columns": [
                    {"name": "ResourceId", "type": "String"},
                    {"name": "Currency", "type": "String"},
                    {"name": "PreTaxCost", "type": "Number"},
                    {"name": "UsageDate", "type": "Number"},
                ],
                "rows": [
                    [resource_id, "EUR", 12.5, 20260901],
                    [resource_id, "EUR", 2.5, 20260901],
                    [resource_id, "EUR", 7, 20260902],
                    [resource_id, "GBP", 3, 20260902],
                ],
            }
        },
    )
    buckets = fetch_azure_costs(
        "service-principal",
        "/subscriptions/sub/resourceGroups/rg",
        resource_id,
        start,
        end,
    )
    request_body = request.last_request.json()
    dataset = request_body["dataset"]
    assert dataset["aggregation"] == {
        "totalCost": {"name": "PreTaxCost", "function": "Sum"}
    }
    assert request_body["timePeriod"] == {
        "from": "2026-09-01",
        "to": "2026-09-30",
    }
    assert dataset["filter"]["dimensions"]["values"] == [resource_id]
    assert {group["name"] for group in dataset["grouping"]} == {
        "ResourceId",
        "Currency",
    }
    assert [
        (
            bucket.currency,
            bucket.value,
            bucket.bucket_start,
            bucket.bucket_end,
            bucket.granularity,
        )
        for bucket in buckets
    ] == [
        (
            "EUR",
            Decimal("15.0"),
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
            "day",
        ),
        (
            "EUR",
            Decimal("7"),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
            datetime(2026, 9, 3, tzinfo=timezone.utc),
            "day",
        ),
        (
            "GBP",
            Decimal("3"),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
            datetime(2026, 9, 3, tzinfo=timezone.utc),
            "day",
        ),
    ]


@pytest.mark.parametrize(
    ("cost_value", "currency"),
    [("1000000000000000000", "USD"), ("1.00000000001", "USD"), ("12", "EURO")],
)
def test_azure_cost_response_rejects_values_outside_storage_schema(
    requests_mock, monkeypatch, cost_value, currency
):
    """Azure values that cannot fit the cost record schema fail before persistence."""
    monkeypatch.setattr("app.providers.costs._azure_arm_token", lambda secret: "token")
    resource_id = (
        "/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.CognitiveServices/accounts/llm"
    )
    requests_mock.post(
        "https://management.azure.com/subscriptions/sub/resourceGroups/rg"
        "/providers/Microsoft.CostManagement/query?api-version=2023-11-01",
        json={
            "properties": {
                "columns": [
                    {"name": "ResourceId", "type": "String"},
                    {"name": "Currency", "type": "String"},
                    {"name": "PreTaxCost", "type": "Number"},
                    {"name": "UsageDate", "type": "Number"},
                ],
                "rows": [[resource_id, currency, cost_value, 20260901]],
            }
        },
    )

    with pytest.raises(CostRefreshError, match="invalid|precision"):
        fetch_azure_costs(
            "service-principal",
            "/subscriptions/sub/resourceGroups/rg",
            resource_id,
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )


def test_azure_cost_response_rejects_unexpected_resource(requests_mock, monkeypatch):
    """Grouped costs for any resource other than the bound one are rejected."""
    monkeypatch.setattr("app.providers.costs._azure_arm_token", lambda secret: "token")
    requests_mock.post(
        "https://management.azure.com/subscriptions/sub/resourceGroups/rg"
        "/providers/Microsoft.CostManagement/query?api-version=2023-11-01",
        json={
            "properties": {
                "columns": [
                    {"name": "ResourceId", "type": "String"},
                    {"name": "Currency", "type": "String"},
                    {"name": "PreTaxCost", "type": "Number"},
                    {"name": "UsageDate", "type": "Number"},
                ],
                "rows": [
                    [
                        "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/other",
                        "USD",
                        8,
                        20260901,
                    ]
                ],
            }
        },
    )
    with pytest.raises(CostRefreshError, match="unexpected resource"):
        fetch_azure_costs(
            "service-principal",
            "/subscriptions/sub/resourceGroups/rg",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/llm",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )


def test_azure_billing_secret_is_saved_to_requested_profile():
    """Azure service-principal credentials are stored on the selected profile."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        session.add(
            ProviderProfile(
                id="azure-production",
                tenant_id="acme",
                provider="azure",
                display_name="Production",
                settings={},
            )
        )
    with database.sessions.begin() as session:
        save_billing_secret(
            session,
            database.secret_cipher,
            "acme",
            "azure-production",
            '{"tenant_id":"tenant","client_id":"client",' '"client_secret":"secret"}',
            "ada",
        )
    with database.sessions() as session:
        profile = session.get(ProviderProfile, "azure-production")
        assert database.secret_cipher.decrypt(profile.billing_secret_ciphertext) == (
            '{"tenant_id":"tenant","client_id":"client",' '"client_secret":"secret"}'
        )
    database.engine.dispose()


@pytest.mark.parametrize("invalid_amount", ["NaN", "Infinity", "-Infinity"])
def test_azure_cost_response_rejects_non_finite_cost(
    requests_mock, monkeypatch, invalid_amount
):
    """Non-finite decimal values are rejected rather than persisted."""
    monkeypatch.setattr("app.providers.costs._azure_arm_token", lambda secret: "token")
    requests_mock.post(
        "https://management.azure.com/subscriptions/sub/resourceGroups/rg"
        "/providers/Microsoft.CostManagement/query?api-version=2023-11-01",
        json={
            "properties": {
                "columns": [
                    {"name": "ResourceId", "type": "String"},
                    {"name": "PreTaxCost", "type": "Number"},
                    {"name": "Currency", "type": "String"},
                    {"name": "UsageDate", "type": "Number"},
                ],
                "rows": [
                    [
                        "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/llm",
                        invalid_amount,
                        "USD",
                        20260901,
                    ]
                ],
            }
        },
    )
    with pytest.raises(CostRefreshError, match="non-finite cost"):
        fetch_azure_costs(
            "service-principal",
            "/subscriptions/sub/resourceGroups/rg",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/llm",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )


def test_azure_cost_response_treats_no_content_as_unavailable(
    requests_mock, monkeypatch
):
    """A 204 response is persisted as unavailable, not as malformed JSON."""
    monkeypatch.setattr("app.providers.costs._azure_arm_token", lambda secret: "token")
    requests_mock.post(
        "https://management.azure.com/subscriptions/sub/resourceGroups/rg"
        "/providers/Microsoft.CostManagement/query?api-version=2023-11-01",
        status_code=204,
    )

    with pytest.raises(CostRefreshError, match="not available") as failure:
        fetch_azure_costs(
            "service-principal",
            "/subscriptions/sub/resourceGroups/rg",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/llm",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )

    assert failure.value.status == "unavailable"


def test_azure_cost_response_rejects_invalid_json(requests_mock, monkeypatch):
    """Invalid Cost Management JSON uses the persisted refresh error path."""
    monkeypatch.setattr("app.providers.costs._azure_arm_token", lambda secret: "token")
    requests_mock.post(
        "https://management.azure.com/subscriptions/sub/resourceGroups/rg"
        "/providers/Microsoft.CostManagement/query?api-version=2023-11-01",
        text="not-json",
    )
    with pytest.raises(CostRefreshError, match="Azure costs payload is invalid"):
        fetch_azure_costs(
            "service-principal",
            "/subscriptions/sub/resourceGroups/rg",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/llm",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )


def test_azure_billing_credentials_reject_valid_non_object_json(requests_mock):
    """JSON primitives are reported as invalid credentials instead of crashing."""
    with pytest.raises(CostRefreshError, match="service principal object"):
        fetch_azure_costs(
            "[]",
            "/subscriptions/sub/resourceGroups/rg",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/llm",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
    assert not requests_mock.request_history


def test_azure_billing_token_network_error_is_a_cost_refresh_error(requests_mock):
    """Token endpoint transport failures use the persisted refresh error path."""
    requests_mock.post(
        "https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
        exc=requests.exceptions.ConnectTimeout,
    )
    with pytest.raises(CostRefreshError, match="authentication request failed"):
        fetch_azure_costs(
            '{"tenant_id":"tenant","client_id":"client",' '"client_secret":"secret"}',
            "/subscriptions/sub/resourceGroups/rg",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/llm",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )


def test_cost_collection_requires_profile_billing_secret():
    """Inference credentials are not silently reused for billing APIs."""
    database = build_admin_database()
    profile = ProviderProfile(
        id="openai-production",
        tenant_id="acme",
        provider="openai",
        display_name="Production",
        settings={},
        inference_secret_ciphertext=database.secret_cipher.encrypt("inference-key"),
    )

    with pytest.raises(CostRefreshError, match="Billing credentials are missing"):
        collect_provider_costs(
            database.secret_cipher,
            profile,
            "org-production",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
    database.engine.dispose()


def test_cost_routes_keep_same_provider_accounts_isolated(admin_app, monkeypatch):
    """Billing save and refresh use the explicitly selected profile and scope."""
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
        profiles = {}
        for suffix in ("production", "staging"):
            profile = create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                "openai",
                suffix.title(),
                {},
                "gpt-5.4",
                f"inference-{suffix}",
                "ada",
            )
            node = ProviderScopeNode(
                id=f"node-{suffix}",
                tenant_id="acme",
                provider="openai",
                scope_type="project",
                canonical_scope_id=f"project-{suffix}",
            )
            session.add(node)
            session.flush()
            binding = ProviderScopeBinding(
                id=f"binding-{suffix}",
                tenant_id="acme",
                provider="openai",
                profile_id=profile.id,
                purpose="billing",
                node_id=node.id,
            )
            session.add(binding)
            save_billing_secret(
                session,
                database.secret_cipher,
                "acme",
                profile.id,
                f"billing-{suffix}",
                "ada",
            )
            profiles[suffix] = (profile.id, binding.id)

    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    costs_page = client.get("/admin/settings/costs")
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"',
        costs_page.get_data(as_text=True),
    ).group(1)
    staging_profile_id, staging_binding_id = profiles["staging"]
    saved = client.post(
        "/admin/settings/costs/billing",
        data={
            "csrf_token": csrf,
            "profile_id": staging_profile_id,
            "billing_secret": "billing-staging-updated",
        },
    )
    assert saved.status_code == 302

    with database.sessions() as session:
        assert (
            database.secret_cipher.decrypt(
                session.get(
                    ProviderProfile, profiles["production"][0]
                ).billing_secret_ciphertext
            )
            == "billing-production"
        )
        assert (
            database.secret_cipher.decrypt(
                session.get(
                    ProviderProfile, staging_profile_id
                ).billing_secret_ciphertext
            )
            == "billing-staging-updated"
        )

    calls = []
    monkeypatch.setattr(
        "app.providers.cost_jobs.fetch_openai_costs",
        lambda key, scope, start, end: calls.append((key, scope))
        or [
            CostBucket(
                kind="actual",
                metric="cost",
                value=Decimal("3.25"),
                unit="currency",
                currency="USD",
                bucket_start=start,
                bucket_end=end,
                source="test.costs",
                granularity="window",
                dimensions={},
            )
        ],
    )
    refreshed = client.post(
        "/admin/settings/costs/refresh",
        data={"csrf_token": csrf, "profile_id": staging_profile_id},
    )
    assert refreshed.status_code == 302
    assert calls == [("billing-staging-updated", "project-staging")]

    with database.sessions() as session:
        job = session.scalar(select(CostRefreshJob))
        assert job.binding_id == staging_binding_id
        assert job.operation_key.startswith(
            f"{staging_profile_id}:{staging_binding_id}:"
        )
        refresh_audit = session.scalar(
            select(AuditEvent).where(AuditEvent.action == "costs.refresh")
        )
        assert refresh_audit.target == f"costs:{staging_profile_id}"
        assert refresh_audit.details["profile_id"] == staging_profile_id
    database.engine.dispose()


def test_openrouter_refresh_uses_selected_profiles_workspace_and_binding(
    admin_app, requests_mock
):
    """The selected OpenRouter profile supplies both key and workspace scope."""
    database = admin_app.extensions["database"]
    workspaces = {
        "production": "550e8400-e29b-41d4-a716-446655440000",
        "staging": "550e8400-e29b-41d4-a716-446655440001",
    }
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
        profiles = {}
        for suffix, workspace_id in workspaces.items():
            profile = create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                "openrouter",
                suffix.title(),
                {},
                "openai/gpt-5",
                f"inference-{suffix}",
                "ada",
            )
            node = ProviderScopeNode(
                id=f"workspace-node-{suffix}",
                tenant_id="acme",
                provider="openrouter",
                scope_type="workspace",
                canonical_scope_id=workspace_id,
            )
            session.add(node)
            session.flush()
            binding = ProviderScopeBinding(
                id=f"workspace-binding-{suffix}",
                tenant_id="acme",
                provider="openrouter",
                profile_id=profile.id,
                purpose="billing",
                node_id=node.id,
            )
            session.add(binding)
            save_billing_secret(
                session,
                database.secret_cipher,
                "acme",
                profile.id,
                f"management-{suffix}",
                "ada",
            )
            profiles[suffix] = (profile.id, binding.id)

    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    page = client.get("/admin/settings/costs")
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^\"]+)"',
        page.get_data(as_text=True),
    ).group(1)
    requests_mock.get(
        "https://openrouter.ai/api/v1/activity",
        json={"data": []},
    )
    selected_profile_id, selected_binding_id = profiles["staging"]
    response = client.post(
        "/admin/settings/costs/refresh",
        data={"csrf_token": csrf, "profile_id": selected_profile_id},
    )

    assert response.status_code == 302
    request = requests_mock.last_request
    assert parse_qs(urlparse(request.url).query) == {
        "workspace_id": [workspaces["staging"]]
    }
    assert request.headers["Authorization"] == "Bearer management-staging"
    with database.sessions() as session:
        job = session.scalar(select(CostRefreshJob))
        assert job.binding_id == selected_binding_id
        audit = session.scalar(
            select(AuditEvent).where(AuditEvent.action == "costs.refresh")
        )
        assert audit.target == f"costs:{selected_profile_id}"
        assert audit.details["profile_id"] == selected_profile_id
    database.engine.dispose()
