"""Provider catalog, cost persistence, and OpenAI-compatible dispatch."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
import requests
from sqlalchemy import select

from app.admin.forms import ProviderProfileForm
from app.admin.security import ADMIN_COOKIE_NAME
from app.persistence.admin_auth import (
    authenticate_admin,
    create_admin_session,
    load_admin_principal,
)
from app.persistence.admin_ops import (
    activate_provider_profile,
    change_admin_password,
    create_provider_profile,
    replace_catalog_entries,
    update_provider_profile,
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
from app.persistence.passkeys import insert_passkey
from app.providers.catalog import CatalogRefreshError, refresh_provider_catalog
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


def test_catalog_refresh_wraps_openrouter_transport_errors(requests_mock):
    """Catalog transport failures become provider-scoped OpenRouter errors."""
    requests_mock.get(
        "https://openrouter.ai/api/v1/models",
        exc=requests.exceptions.ConnectTimeout,
    )

    with pytest.raises(CatalogRefreshError, match="OpenRouter catalog request failed"):
        refresh_provider_catalog("openrouter", {}, "sk-or-test")


def test_catalog_refresh_rejects_invalid_openrouter_json(requests_mock):
    """Catalog parsing failures become safe OpenRouter errors."""
    requests_mock.get(
        "https://openrouter.ai/api/v1/models",
        text="not-json",
    )

    with pytest.raises(
        CatalogRefreshError, match="OpenRouter catalog payload is invalid"
    ):
        refresh_provider_catalog("openrouter", {}, "sk-or-secret")


def test_catalog_refresh_rejects_empty_openrouter_catalog(requests_mock):
    """An empty OpenRouter response must not look like a successful refresh."""
    requests_mock.get(
        "https://openrouter.ai/api/v1/models",
        json={"data": []},
    )

    with pytest.raises(CatalogRefreshError, match="contains no models"):
        refresh_provider_catalog("openrouter", {}, "sk-or-secret")


def test_catalog_refresh_parses_openrouter_models(requests_mock):
    """Refresh the OpenRouter catalog from its versioned models endpoint."""
    requests_mock.get(
        "https://openrouter.ai/api/v1/models",
        json={"data": [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-sonnet-4"}]},
    )

    entries = refresh_provider_catalog("openrouter", {}, "sk-or-test")

    assert entries == [
        ("openai/gpt-4o", None),
        ("anthropic/claude-sonnet-4", None),
    ]
    assert (
        requests_mock.request_history[0].headers["Authorization"] == "Bearer sk-or-test"
    )


def test_openrouter_account_key_survives_save_and_catalog_refresh(
    admin_app, requests_mock
):
    """Saving then refreshing an OpenRouter account keeps its key and catalog."""
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
    create_page = client.get("/admin/settings/connection/create?provider=openrouter")
    create_body = create_page.get_data(as_text=True)
    assert 'data-provider-fields="azure" hidden' in create_body
    assert 'data-provider-fields="openrouter"' in create_body
    assert 'name="base_url" required' not in create_body
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"',
        create_page.get_data(as_text=True),
    ).group(1)
    key = "sk-or-v1-integration-test"
    saved = client.post(
        "/admin/settings/connection/create",
        data={
            "csrf_token": csrf,
            "provider": "openrouter",
            "display_name": "Production",
            "default_model": "",
            "api_key": key,
        },
    )
    assert saved.status_code == 302

    requests_mock.get(
        "https://openrouter.ai/api/v1/models",
        json={"data": [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-sonnet-4"}]},
    )
    with database.sessions() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == "acme",
                ProviderProfile.provider == "openrouter",
            )
        )
        profile_id = profile.id
        assert (
            database.secret_cipher.decrypt(profile.inference_secret_ciphertext) == key
        )

    refreshed = client.post(
        f"/admin/settings/connection/{profile_id}/catalog",
        data={"csrf_token": csrf},
    )
    assert refreshed.status_code == 302
    assert requests_mock.request_history[-1].headers["Authorization"] == f"Bearer {key}"

    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile_id)
        models = list(
            session.scalars(
                select(ProviderCatalogEntry.model_id).where(
                    ProviderCatalogEntry.profile_id == profile_id
                )
            )
        )
        assert (
            database.secret_cipher.decrypt(profile.inference_secret_ciphertext) == key
        )
        assert profile.default_model == "openai/gpt-4o"
    assert sorted(models) == ["anthropic/claude-sonnet-4", "openai/gpt-4o"]
    page = client.get("/admin/settings/connection")
    page_body = page.get_data(as_text=True)
    assert "openai/gpt-4o" in page_body
    assert "Schlüssel gespeichert" in page_body
    assert key not in page_body

    edit_page = client.get(f"/admin/settings/connection/{profile_id}/edit")
    edit_body = edit_page.get_data(as_text=True)
    assert "gespeicherten Schlüssel beizubehalten" in edit_body
    assert "API-Schlüssel ist gespeichert" in edit_body
    assert key not in edit_body

    activated = client.post(
        f"/admin/settings/connection/{profile_id}/activate",
        data={"csrf_token": csrf, "profile_id": profile_id},
    )
    assert activated.status_code == 302
    active_page = client.get("/admin/settings/connection")
    assert "Aktiver Account: <strong>OpenRouter · Production</strong>" in (
        active_page.get_data(as_text=True)
    )


def test_catalog_refresh_does_not_store_results_for_changed_profile(
    admin_app, monkeypatch
):
    """A catalog response is discarded if profile settings change during I/O."""
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
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openrouter",
            "Production",
            {},
            "openai/gpt-4o",
            "sk-or-test",
            "ada",
        )
        profile_id = profile.id

    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    page = client.get("/admin/settings/connection")
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"',
        page.get_data(as_text=True),
    ).group(1)

    def refresh_with_profile_change(provider, settings, inference_secret):
        with database.sessions.begin() as session:
            changed_profile = session.get(ProviderProfile, profile_id)
            changed_profile.settings = {"changed_during_refresh": True}
        return [("openai/gpt-4o", None)]

    monkeypatch.setattr(
        "app.admin.views.refresh_provider_catalog", refresh_with_profile_change
    )
    response = client.post(
        f"/admin/settings/connection/{profile_id}/catalog",
        data={"csrf_token": csrf},
    )

    assert response.status_code == 302
    with database.sessions() as session:
        changed_profile = session.get(ProviderProfile, profile_id)
        assert changed_profile.settings == {"changed_during_refresh": True}
        assert changed_profile.catalog_refreshed_at is None
        assert not list(
            session.scalars(
                select(ProviderCatalogEntry).where(
                    ProviderCatalogEntry.profile_id == profile_id
                )
            )
        )


def test_older_catalog_refresh_cannot_overwrite_newer_refresh(admin_app, monkeypatch):
    """A later-started refresh supersedes an earlier response on same config."""
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
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openrouter",
            "Production",
            {},
            "openai/gpt-4o",
            "sk-or-test",
            "ada",
        )
        profile_id = profile.id

    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    page = client.get("/admin/settings/connection")
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"',
        page.get_data(as_text=True),
    ).group(1)
    older_request_generation = None

    def start_older_refresh(provider, settings, inference_secret):
        nonlocal older_request_generation
        with database.sessions() as session:
            older_request_generation = session.get(
                ProviderProfile, profile_id
            ).catalog_generation
        with database.sessions.begin() as session:
            session.get(ProviderProfile, profile_id).catalog_generation += 1
        return [("older/model", None)]

    monkeypatch.setattr("app.admin.views.refresh_provider_catalog", start_older_refresh)
    first_response = client.post(
        f"/admin/settings/connection/{profile_id}/catalog",
        data={"csrf_token": csrf},
    )
    assert first_response.status_code == 302
    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile_id)
        assert profile.catalog_generation == older_request_generation + 1
        assert profile.catalog_refreshed_at is None

    monkeypatch.setattr(
        "app.admin.views.refresh_provider_catalog",
        lambda provider, settings, inference_secret: [("newer/model", None)],
    )
    second_response = client.post(
        f"/admin/settings/connection/{profile_id}/catalog",
        data={"csrf_token": csrf},
    )
    assert second_response.status_code == 302
    with database.sessions() as session:
        assert (
            session.scalar(
                select(ProviderCatalogEntry.model_id).where(
                    ProviderCatalogEntry.profile_id == profile_id
                )
            )
            == "newer/model"
        )


def test_provider_profile_form_requires_azure_base_url_but_not_for_openrouter(
    admin_app,
):
    """Only Azure submissions require the Azure endpoint field."""
    with admin_app.app_context():
        azure_form = ProviderProfileForm(
            data={
                "provider": "azure",
                "display_name": "Azure account",
                "base_url": "",
                "default_model": "",
                "api_key": "azure-key",
            },
            meta={"csrf": False},
        )
        azure_form.provider.choices = [("azure", "Azure")]
        assert not azure_form.validate()
        assert "Azure Base-URL ist erforderlich." in azure_form.errors["base_url"]

        openrouter_form = ProviderProfileForm(
            data={
                "provider": "openrouter",
                "display_name": "OpenRouter account",
                "base_url": "",
                "default_model": "",
                "api_key": "openrouter-key",
            },
            meta={"csrf": False},
        )
        openrouter_form.provider.choices = [("openrouter", "OpenRouter")]
        assert openrouter_form.validate()


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
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Primary",
            {},
            "gpt-5.4",
            "sk-test",
            "ada",
        )
        tenant = session.get(Tenant, "acme")
        with pytest.raises(ValueError, match="not in its available catalog"):
            activate_provider_profile(session, tenant, profile.id, "ada")
        replace_catalog_entries(session, profile, [("gpt-5.4", None)], None)
        activate_provider_profile(session, tenant, profile.id, "ada")
        assert tenant.active_profile_id == profile.id
    database.engine.dispose()


def test_create_provider_profile_is_inactive_and_tenant_scoped():
    """Creating another named account never changes the active profile."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        original = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Personal",
            {"organization": "org-1"},
            "gpt-5.4",
            "sk-first",
            "ada",
        )
        tenant.active_profile_id = original.id
        second = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Work",
            {"organization": "org-2"},
            "gpt-5.5",
            "sk-second",
            "ada",
        )
        assert original.id != second.id
        assert original.display_name == "Personal"
        assert second.display_name == "Work"
        assert tenant.active_profile_id == original.id
        assert (
            session.scalar(
                select(ProviderProfile).where(
                    ProviderProfile.tenant_id == "other",
                    ProviderProfile.id == second.id,
                )
            )
            is None
        )
    database.engine.dispose()


def test_azure_profile_updates_validate_url_and_keep_model_deployments():
    """Azure profile updates validate endpoints and retain catalog mappings."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "azure",
            "Production",
            {
                "base_url": "https://resource.openai.azure.com",
                "model_deployments": {"gpt-6-astra": "deployment-a"},
            },
            "gpt-6-astra",
            "azure-key",
            "ada",
        )
        tenant = session.get(Tenant, "acme")
        tenant.active_profile_id = profile.id
        replace_catalog_entries(
            session, profile, [("gpt-6-astra", "deployment-a")], None
        )
        with pytest.raises(ValueError, match="HTTPS-Adresse"):
            create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                "azure",
                "Invalid",
                {"base_url": "http://resource.openai.azure.com"},
                "gpt-6-astra",
                "azure-key",
                "ada",
            )
        updated = update_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            profile.id,
            "Production",
            {"base_url": "https://new-resource.openai.azure.com/"},
            "gpt-6-astra",
            None,
            "ada",
        )
        assert updated.settings == {"base_url": "https://new-resource.openai.azure.com"}
        assert updated.default_model is None
        assert updated.catalog_refreshed_at is None
        assert tenant.active_profile_id is None
        assert not list(
            session.scalars(
                select(ProviderCatalogEntry).where(
                    ProviderCatalogEntry.profile_id == profile.id
                )
            )
        )
        with pytest.raises(ValueError, match="HTTPS-Adresse"):
            update_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                profile.id,
                "Production",
                {"base_url": "https://user:password@resource.openai.azure.com"},
                "gpt-6-astra",
                None,
                "ada",
            )
    database.engine.dispose()


def test_update_provider_profile_preserves_secret_and_provider_identity():
    """Blank secret updates preserve ciphertext and provider identity is fixed."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Work",
            {},
            "gpt-5.4",
            "sk-original",
            "ada",
        )
        ciphertext = profile.inference_secret_ciphertext
        updated = update_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            profile.id,
            "Work account",
            {"project": "proj-1"},
            "gpt-5.5",
            None,
            "ada",
        )
        assert updated.provider == "openai"
        assert updated.display_name == "Work account"
        assert updated.inference_secret_ciphertext == ciphertext
        assert database.secret_cipher.decrypt(ciphertext) == "sk-original"
    database.engine.dispose()


def test_provider_profile_rejects_duplicate_names_and_foreign_ids():
    """Names are case insensitive and profile access remains tenant scoped."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Work",
            {},
            "gpt-5.4",
            "sk-original",
            "ada",
        )
        with pytest.raises(ValueError, match="already exists"):
            create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                "openai",
                "work",
                {},
                "gpt-5.4",
                "sk-other",
                "ada",
            )
        create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Straße",
            {},
            "gpt-5.4",
            "sk-german",
            "ada",
        )
        with pytest.raises(ValueError, match="already exists"):
            create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                "openai",
                "STRASSE",
                {},
                "gpt-5.4",
                "sk-other-german",
                "ada",
            )
        with pytest.raises(LookupError):
            update_provider_profile(
                session,
                database.secret_cipher,
                "other",
                profile.id,
                "Other",
                {},
                "gpt-5.4",
                None,
                "ada",
            )
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


def test_failed_catalog_refresh_preserves_last_successful_entries():
    """A failed provider query keeps the last known selectable catalog."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Work",
            {},
            "gpt-5.4",
            "sk-test",
            "ada",
        )
        replace_catalog_entries(session, profile, [("gpt-5.4", None)], None)
        replace_catalog_entries(session, profile, [], "Provider catalog HTTP 503")
        models = list(
            session.scalars(
                select(ProviderCatalogEntry.model_id).where(
                    ProviderCatalogEntry.profile_id == profile.id
                )
            )
        )
        assert models == ["gpt-5.4"]
        assert profile.catalog_error == "Provider catalog HTTP 503"
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
