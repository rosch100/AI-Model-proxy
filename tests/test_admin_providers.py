"""Provider catalog, cost persistence, and OpenAI-compatible dispatch."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import parse_qs, urlparse
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
    delete_provider_profile,
    replace_catalog_entries,
    save_billing_secret,
    update_provider_profile,
    upsert_provider_profile,
)
from app.persistence.models import (
    CostRefreshEvent,
    CostRefreshJob,
    CostUsageRecord,
    InferenceActivityEvent,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from app.persistence.passkeys import insert_passkey
from app.providers import cost_jobs, costs
from app.providers.catalog import CatalogRefreshError, refresh_provider_catalog
from app.providers.cost_jobs import collect_provider_costs, persist_cost_refresh
from app.providers.costs import (
    CostBucket,
    CostRefreshError,
    fetch_azure_costs,
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


def test_deepseek_profile_can_be_created_without_provider_specific_settings(
    admin_app,
):
    """Persist DeepSeek profiles with their model target and encrypted key."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "deepseek",
            "Production",
            {},
            "deepseek-v4-flash",
            "deepseek-key",
            "ada",
        )
        profile_id = profile.id

    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile_id)
        assert profile.provider == "deepseek"
        assert profile.default_model == "deepseek-v4-flash"
        assert database.secret_cipher.decrypt(profile.inference_secret_ciphertext) == (
            "deepseek-key"
        )


def test_deepseek_admin_profile_creation_and_catalog_refresh(admin_app, requests_mock):
    """Create a tenant DeepSeek account, then load its model catalog."""
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
    create_page = client.get("/admin/settings/connection/create?provider=deepseek")
    create_body = create_page.get_data(as_text=True)
    assert create_page.status_code == 200
    assert 'data-provider-fields="deepseek"' in create_body
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"', create_body
    ).group(1)
    saved = client.post(
        "/admin/settings/connection/create",
        data={
            "csrf_token": csrf,
            "provider": "deepseek",
            "display_name": "Production",
            "default_model": "",
            "api_key": "deepseek-inference-key",
        },
    )
    assert saved.status_code == 302
    with database.sessions() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == "acme",
                ProviderProfile.provider == "deepseek",
            )
        )
        profile_id = profile.id
        assert profile.settings == {}
        assert database.secret_cipher.decrypt(profile.inference_secret_ciphertext) == (
            "deepseek-inference-key"
        )

    request = requests_mock.get(
        "https://api.deepseek.com/models",
        json={"data": [{"id": "deepseek-v4-flash"}, {"id": "deepseek-v4-pro"}]},
    )
    refreshed = client.post(
        f"/admin/settings/connection/{profile_id}/catalog",
        data={"csrf_token": csrf},
    )
    assert refreshed.status_code == 302
    assert request.last_request.headers["Authorization"] == (
        "Bearer deepseek-inference-key"
    )
    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile_id)
        assert profile.default_model == "deepseek-v4-flash"
    page = client.get("/admin/settings/connection")
    assert "DeepSeek" in page.get_data(as_text=True)
    assert "deepseek-v4-flash" in page.get_data(as_text=True)

    activated = client.post(
        f"/admin/settings/connection/{profile_id}/activate",
        data={"csrf_token": csrf, "profile_id": profile_id},
    )
    assert activated.status_code == 302

    admin_app.config.update(
        AUTH_MODE="tenant", TENANT_CONFIG_SOURCE="database", ENABLE_AZURE=True
    )
    completion = requests_mock.post(
        "https://api.deepseek.com/chat/completions",
        content=(
            b'data: {"model":"deepseek-v4-flash","choices":[{"delta":'
            b'{"content":"hello"}}]}\n\n'
            b'data: {"model":"deepseek-v4-flash","choices":[],"usage":'
            b'{"prompt_tokens":17,"completion_tokens":5,"total_tokens":22}}'
            b"\n\ndata: [DONE]\n\n"
        ),
        headers={"Content-Type": "text/event-stream"},
    )
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer cursor-key"},
        json={
            "model": "cursor-acme-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert b'"content":"hello"' in response.data
    assert response.data.endswith(b"data: [DONE]\n\n")
    assert completion.last_request.headers["Authorization"] == (
        "Bearer deepseek-inference-key"
    )
    assert completion.last_request.json()["model"] == "deepseek-v4-flash"
    with database.sessions() as session:
        activity = session.scalar(
            select(InferenceActivityEvent).where(
                InferenceActivityEvent.provider == "deepseek"
            )
        )
    assert activity is not None
    assert activity.tenant_id == "acme"
    assert activity.profile_id == profile_id
    assert activity.inbound_model == "cursor-acme-model"
    assert activity.routed_model == "deepseek-v4-flash"
    assert activity.input_tokens == 17
    assert activity.output_tokens == 5
    assert activity.total_tokens == 22


def test_catalog_refresh_parses_deepseek_models(requests_mock):
    """Refresh the DeepSeek catalog using its OpenAI-compatible models endpoint."""
    request = requests_mock.get(
        "https://api.deepseek.com/models",
        json={"data": [{"id": "deepseek-v4-flash"}, {"id": "deepseek-v4-pro"}]},
    )

    entries = refresh_provider_catalog("deepseek", {}, "deepseek-test")

    assert entries == [("deepseek-v4-flash", None), ("deepseek-v4-pro", None)]
    assert request.last_request.headers["Authorization"] == "Bearer deepseek-test"


def test_catalog_refresh_rejects_empty_deepseek_catalog(requests_mock):
    """An empty DeepSeek catalog is not a successful refresh."""
    requests_mock.get("https://api.deepseek.com/models", json={"data": []})

    with pytest.raises(
        CatalogRefreshError, match="DeepSeek catalog contains no models"
    ):
        refresh_provider_catalog("deepseek", {}, "deepseek-test")


def test_catalog_refresh_rejects_invalid_deepseek_payload(requests_mock):
    """Invalid DeepSeek JSON cannot silently remove the profile catalog."""
    requests_mock.get("https://api.deepseek.com/models", text="not-json")

    with pytest.raises(
        CatalogRefreshError, match="DeepSeek catalog payload is invalid"
    ):
        refresh_provider_catalog("deepseek", {}, "deepseek-test")


def test_catalog_refresh_surfaces_deepseek_http_errors(requests_mock):
    """A non-success DeepSeek model-list response is reported explicitly."""
    requests_mock.get("https://api.deepseek.com/models", status_code=403)

    with pytest.raises(CatalogRefreshError, match="DeepSeek catalog HTTP 403"):
        refresh_provider_catalog("deepseek", {}, "deepseek-test")


def test_catalog_refresh_surfaces_deepseek_transport_errors(requests_mock):
    """The DeepSeek model-list transport errors become safe provider errors."""
    requests_mock.get("https://api.deepseek.com/models", exc=requests.ConnectTimeout)

    with pytest.raises(CatalogRefreshError, match="DeepSeek catalog request failed"):
        refresh_provider_catalog("deepseek", {}, "deepseek-test")


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
    assert "leer lassen oder unverändert lassen" in edit_body
    assert 'name="api_key"' in edit_body
    assert 'value="••••••••"' in edit_body
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


def test_connection_page_lists_every_same_provider_account(admin_app):
    """The connection page must not collapse profiles by provider."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        profiles = {
            display_name: create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                "openrouter",
                display_name,
                {},
                "openai/gpt-5",
                f"inference-{display_name.casefold()}",
                "ada",
            )
            for display_name in ("Production", "Staging")
        }
        node = ProviderScopeNode(
            id="bound-workspace-node",
            tenant_id="acme",
            provider="openrouter",
            scope_type="workspace",
            canonical_scope_id="a1b2c3d4-1111-4111-8111-123456789abc",
        )
        session.add(node)
        session.flush()
        session.add(
            ProviderScopeBinding(
                id="bound-workspace-binding",
                tenant_id="acme",
                provider="openrouter",
                profile_id=profiles["Production"].id,
                purpose="billing",
                node_id=node.id,
            )
        )
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
    response = client.get("/admin/settings/connection")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert page.count("<h3>Production</h3>") == 1
    assert page.count("<h3>Staging</h3>") == 1
    assert "Dieser Account kann nicht entfernt werden" in page
    assert page.count(">Account entfernen</button>") == 1
    database.engine.dispose()


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
        assert profile.route_priority == 1
    database.engine.dispose()


def test_create_provider_profile_is_inactive_and_tenant_scoped():
    """Creating another named account never changes the active profile."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
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
        original.route_priority = 1
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
        assert original.route_priority == 1
        assert second.route_priority is None
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


def test_bound_provider_profile_cannot_be_deleted():
    """Immutable scope ownership prevents deleting a profile with bindings."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openrouter",
            "Production",
            {},
            "openai/gpt-5",
            "sk-test",
            "ada",
        )
        node = ProviderScopeNode(
            id="bound-workspace-node",
            tenant_id="acme",
            provider="openrouter",
            scope_type="workspace",
            canonical_scope_id="a1b2c3d4-1111-4111-8111-123456789abc",
        )
        session.add(node)
        session.flush()
        session.add(
            ProviderScopeBinding(
                id="bound-workspace-binding",
                tenant_id="acme",
                provider="openrouter",
                profile_id=profile.id,
                purpose="billing",
                node_id=node.id,
            )
        )
        session.flush()

        with pytest.raises(
            ValueError, match="cannot be deleted while scopes are bound"
        ):
            delete_provider_profile(session, "acme", profile.id, "ada")

        assert session.get(ProviderProfile, profile.id).deleted_at is None
    database.engine.dispose()


def test_deleted_unbound_profile_erases_stored_credentials():
    """Soft deletion also removes credentials that must no longer be used."""
    database = build_admin_database()
    seed_admin(database)
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openrouter",
            "Retired",
            {},
            "openai/gpt-5",
            "sk-inference",
            "ada",
        )
        profile.billing_secret_ciphertext = database.secret_cipher.encrypt("sk-billing")
        profile_id = profile.id

    with database.sessions.begin() as session:
        delete_provider_profile(session, "acme", profile_id, "ada")

    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile_id)
        assert profile.deleted_at is not None
        assert profile.inference_secret_ciphertext is None
        assert profile.billing_secret_ciphertext is None
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
        profile.route_priority = 1
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
        assert profile.route_priority is None
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
        job = cost_jobs.start_cost_refresh(session, "acme", profile.id, start, end)[4]
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
        _, _, _, _, job = cost_jobs.start_cost_refresh(
            session, "acme", profile.id, start, end
        )
        job_id = job.id

    with pytest.raises(LookupError, match="already running"):
        with database.sessions.begin() as session:
            cost_jobs.start_cost_refresh(session, "acme", profile.id, start, end)

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
        _, _, _, _, stale_job = cost_jobs.start_cost_refresh(
            session, "acme", profile.id, start, end
        )
        stale_job_id = stale_job.id
    with database.sessions.begin() as session:
        stale_job = session.get(CostRefreshJob, stale_job_id)
        stale_job.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)

    with database.sessions.begin() as session:
        _, _, _, _, next_job = cost_jobs.start_cost_refresh(
            session, "acme", profile.id, start, end
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
        profile, binding, node, usage_node, job = cost_jobs.start_cost_refresh(
            session,
            "acme",
            profile.id,
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


def test_azure_cost_query_filters_and_verifies_the_bound_resource(
    requests_mock, monkeypatch
):
    """Azure actual costs use a filtered daily query and return exact dimensions."""
    resource_group_id = (
        "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/" "resourcegroups/acme-rg"
    )
    resource_id = (
        f"{resource_group_id}/providers/microsoft.cognitiveservices/accounts/acme-ai"
    )
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 3, tzinfo=timezone.utc)
    monkeypatch.setattr("app.providers.costs._azure_arm_token", lambda: "arm-token")
    requests_mock.post(
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2026-06-01",
        json={
            "properties": {
                "columns": [
                    {"name": "PreTaxCost"},
                    {"name": "ResourceId"},
                    {"name": "UsageDate"},
                    {"name": "Currency"},
                ],
                "rows": [
                    ["0.125", resource_id, 20260901, "USD"],
                    ["1.50", resource_id, 20260902, "EUR"],
                ],
            }
        },
    )

    buckets = fetch_azure_costs(resource_group_id, resource_id, start, end)

    assert [(bucket.value, bucket.currency) for bucket in buckets] == [
        (Decimal("0.125"), "USD"),
        (Decimal("1.50"), "EUR"),
    ]
    assert [bucket.bucket_start for bucket in buckets] == [
        start,
        datetime(2026, 9, 2, tzinfo=timezone.utc),
    ]
    assert all(
        bucket.kind == "actual" and bucket.granularity == "day" for bucket in buckets
    )
    assert all(bucket.dimensions == {"resource_id": resource_id} for bucket in buckets)
    request = requests_mock.last_request
    assert request.headers["Authorization"] == "Bearer arm-token"
    assert request.json() == {
        "type": "ActualCost",
        "timeframe": "Custom",
        "timePeriod": {"from": "2026-09-01", "to": "2026-09-02"},
        "dataset": {
            "granularity": "Daily",
            "aggregation": {"totalCost": {"name": "PreTaxCost", "function": "Sum"}},
            "filter": {
                "dimensions": {
                    "name": "ResourceId",
                    "operator": "In",
                    "values": [resource_id],
                }
            },
            "grouping": [
                {"type": "Dimension", "name": "ResourceId"},
                {"type": "Dimension", "name": "Currency"},
            ],
        },
    }


@pytest.mark.parametrize("currency", ["ßaa", "1SD", "U$D"])
def test_azure_cost_query_rejects_non_ascii_or_non_alphabetic_currency(
    requests_mock, monkeypatch, currency
):
    """Azure currency must be three ASCII letters before persistence."""
    resource_group_id = "/subscriptions/sub/resourcegroups/acme-rg"
    resource_id = (
        f"{resource_group_id}/providers/microsoft.cognitiveservices/accounts/acme-ai"
    )
    monkeypatch.setattr(costs, "_azure_arm_token", lambda: "arm-token")
    requests_mock.post(
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2026-06-01",
        json={
            "properties": {
                "columns": [
                    {"name": "PreTaxCost"},
                    {"name": "ResourceId"},
                    {"name": "UsageDate"},
                    {"name": "Currency"},
                ],
                "rows": [[1.25, resource_id, 20260901, currency]],
            }
        },
    )

    with pytest.raises(CostRefreshError, match="currency is invalid"):
        fetch_azure_costs(
            resource_group_id,
            resource_id,
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("usage_date", ["202691", "2026-09-01", 202691, 20260901.0])
def test_azure_cost_query_rejects_noncanonical_usage_date(
    requests_mock, monkeypatch, usage_date
):
    """Azure UsageDate must be an exact eight-digit calendar date."""
    resource_group_id = "/subscriptions/sub/resourcegroups/acme-rg"
    resource_id = (
        f"{resource_group_id}/providers/microsoft.cognitiveservices/accounts/acme-ai"
    )
    monkeypatch.setattr(costs, "_azure_arm_token", lambda: "arm-token")
    requests_mock.post(
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2026-06-01",
        json={
            "properties": {
                "columns": [
                    {"name": "PreTaxCost"},
                    {"name": "ResourceId"},
                    {"name": "UsageDate"},
                    {"name": "Currency"},
                ],
                "rows": [[1.25, resource_id, usage_date, "USD"]],
            }
        },
    )

    with pytest.raises(CostRefreshError, match="invalid values"):
        fetch_azure_costs(
            resource_group_id,
            resource_id,
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )


def test_azure_cost_query_rejects_unbound_resource_rows(requests_mock, monkeypatch):
    """A scope-level row without the exact bound resource is unavailable."""
    resource_group_id = (
        "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/" "resourcegroups/acme-rg"
    )
    resource_id = (
        f"{resource_group_id}/providers/microsoft.cognitiveservices/accounts/acme-ai"
    )
    monkeypatch.setattr("app.providers.costs._azure_arm_token", lambda: "arm-token")
    requests_mock.post(
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2026-06-01",
        json={
            "properties": {
                "columns": [
                    {"name": "PreTaxCost"},
                    {"name": "ResourceId"},
                    {"name": "UsageDate"},
                    {"name": "Currency"},
                ],
                "rows": [
                    [
                        1.25,
                        f"{resource_group_id}/providers/other/resource",
                        20260901,
                        "USD",
                    ]
                ],
            }
        },
    )

    with pytest.raises(CostRefreshError, match="unexpected resource"):
        fetch_azure_costs(
            resource_group_id,
            resource_id,
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )


def test_azure_cost_query_treats_no_content_as_no_costs(requests_mock, monkeypatch):
    """A successful Azure 204 response represents an empty cost result."""
    resource_group_id = "/subscriptions/sub/resourcegroups/acme-rg"
    resource_id = (
        f"{resource_group_id}/providers/microsoft.cognitiveservices/accounts/acme-ai"
    )
    monkeypatch.setattr(costs, "_azure_arm_token", lambda: "arm-token")
    requests_mock.post(
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2026-06-01",
        status_code=204,
    )

    assert (
        fetch_azure_costs(
            resource_group_id,
            resource_id,
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )
        == []
    )


def test_azure_cost_query_rejects_non_string_column_names(requests_mock, monkeypatch):
    """Malformed Azure column metadata fails as a typed provider error."""
    resource_group_id = "/subscriptions/sub/resourcegroups/acme-rg"
    resource_id = (
        f"{resource_group_id}/providers/microsoft.cognitiveservices/accounts/acme-ai"
    )
    monkeypatch.setattr(costs, "_azure_arm_token", lambda: "arm-token")
    requests_mock.post(
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2026-06-01",
        json={
            "properties": {
                "columns": [
                    {"name": ["PreTaxCost"]},
                    {"name": "ResourceId"},
                    {"name": "UsageDate"},
                    {"name": "Currency"},
                ],
                "rows": [],
            }
        },
    )

    with pytest.raises(CostRefreshError, match="column names are invalid") as exc_info:
        fetch_azure_costs(
            resource_group_id,
            resource_id,
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )

    assert exc_info.value.status == "unavailable"


def test_azure_cost_query_marks_missing_resource_dimension_unavailable(
    requests_mock, monkeypatch
):
    """Do not report costs when the response cannot prove the bound resource."""
    resource_group_id = "/subscriptions/sub/resourcegroups/acme-rg"
    resource_id = (
        f"{resource_group_id}/providers/microsoft.cognitiveservices/accounts/acme-ai"
    )
    monkeypatch.setattr(costs, "_azure_arm_token", lambda: "arm-token")
    requests_mock.post(
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2026-06-01",
        json={
            "properties": {
                "columns": [
                    {"name": "PreTaxCost"},
                    {"name": "UsageDate"},
                    {"name": "Currency"},
                ],
                "rows": [[1.25, 20260901, "USD"]],
            }
        },
    )

    with pytest.raises(
        CostRefreshError, match="missing required dimensions"
    ) as exc_info:
        fetch_azure_costs(
            resource_group_id,
            resource_id,
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )

    assert exc_info.value.status == "unavailable"


def test_azure_cost_refresh_uses_host_identity_without_tenant_secret(monkeypatch):
    """Azure costs use the two bound scopes and never decrypt a tenant secret."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    resource_group_id = "/subscriptions/sub/resourcegroups/acme-rg"
    resource_id = (
        f"{resource_group_id}/providers/microsoft.cognitiveservices/accounts/acme-ai"
    )
    profile = ProviderProfile(
        id=str(uuid4()),
        tenant_id="acme",
        provider="azure",
        settings={},
    )
    requested = []

    def fetch_costs(group_id, bound_resource_id, requested_start, requested_end):
        requested.append((group_id, bound_resource_id, requested_start, requested_end))
        return []

    monkeypatch.setattr(cost_jobs, "fetch_azure_costs", fetch_costs)

    result = collect_provider_costs(
        build_admin_database().secret_cipher,
        profile,
        resource_group_id,
        start,
        end,
        usage_scope_id=resource_id,
    )

    assert result == []
    assert requested == [(resource_group_id, resource_id, start, end)]


def test_azure_profile_rejects_tenant_billing_secret():
    """Azure billing credentials are owned by the operator host identity."""
    database = build_admin_database()
    seed_admin(database)
    try:
        with database.sessions.begin() as session:
            session.add(
                ProviderProfile(
                    id="azure-profile",
                    tenant_id="acme",
                    provider="azure",
                    display_name="Production",
                    settings={},
                )
            )
        with database.sessions.begin() as session:
            with pytest.raises(ValueError, match="operator host identity"):
                save_billing_secret(
                    session,
                    database.secret_cipher,
                    "acme",
                    "azure-profile",
                    "tenant-secret",
                    "ada",
                )
    finally:
        database.engine.dispose()


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


def test_azure_token_uses_user_assigned_managed_identity(monkeypatch):
    """The operator-selected client ID is passed to managed identity auth."""
    for name in (
        "AZURE_FEDERATED_TOKEN_FILE",
        "IDENTITY_ENDPOINT",
        "MSI_ENDPOINT",
        "WEBSITE_HOSTNAME",
        "CONTAINER_APP_HOSTNAME",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_CERTIFICATE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "user-assigned-client-id")
    credentials = []

    class FakeCredential:
        def __init__(self, **kwargs):
            credentials.append(kwargs)

        def get_token(self, scope):
            return type("Token", (), {"token": "arm-token"})()

        def close(self):
            pass

    monkeypatch.setattr(costs, "ManagedIdentityCredential", FakeCredential)

    assert costs._azure_arm_token() == "arm-token"
    assert credentials == [{"client_id": "user-assigned-client-id"}]


def test_azure_token_uses_managed_identity_without_platform_markers(monkeypatch):
    """Managed identity credential can discover an Azure VM through IMDS."""
    for name in (
        "AZURE_FEDERATED_TOKEN_FILE",
        "IDENTITY_ENDPOINT",
        "MSI_ENDPOINT",
        "WEBSITE_HOSTNAME",
        "CONTAINER_APP_HOSTNAME",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_CERTIFICATE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    calls = []

    class FakeCredential:
        def __init__(self, client_id=None):
            assert client_id is None

        def get_token(self, scope):
            calls.append(("get_token", scope))
            return type("Token", (), {"token": "arm-token"})()

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(costs, "ManagedIdentityCredential", FakeCredential)

    assert costs._azure_arm_token() == "arm-token"
    assert calls == [
        ("get_token", "https://management.azure.com/.default"),
        ("close",),
    ]


def test_azure_token_uses_operator_certificate_identity(monkeypatch):
    """Off-Azure certificate identity uses the Azure Identity SDK credential."""
    for name in (
        "AZURE_FEDERATED_TOKEN_FILE",
        "IDENTITY_ENDPOINT",
        "MSI_ENDPOINT",
        "WEBSITE_HOSTNAME",
        "CONTAINER_APP_HOSTNAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-id")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-id")
    monkeypatch.setenv("AZURE_CLIENT_CERTIFICATE_PATH", "/operator/id.pem")
    credentials = []

    class FakeCredential:
        def __init__(self, **kwargs):
            credentials.append(kwargs)

        def get_token(self, scope):
            return type("Token", (), {"token": "arm-token"})()

        def close(self):
            pass

    monkeypatch.setattr(costs, "CertificateCredential", FakeCredential)

    assert costs._azure_arm_token() == "arm-token"
    assert credentials == [
        {
            "tenant_id": "tenant-id",
            "client_id": "client-id",
            "certificate_path": "/operator/id.pem",
            "password": None,
        }
    ]


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
