"""Tenant-facing provider billing configuration routes."""

from __future__ import annotations

import re
import secrets

from sqlalchemy import select

from app.admin.security import ADMIN_COOKIE_NAME, SAVED_SECRET_MASK
from app.persistence.admin_auth import authenticate_admin, create_admin_session
from app.persistence.admin_ops import create_provider_profile, save_billing_secret
from app.persistence.models import (
    AuditEvent,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
)
from app.persistence.passkeys import insert_passkey


def _authenticated_admin_client(admin_app):
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


def _csrf(body: str) -> str:
    match = re.search(r'name="csrf_token" type="hidden" value="([^"]+)"', body)
    assert match is not None
    return match.group(1)


def test_openai_billing_setup_is_available_without_server_access(
    admin_app, requests_mock
):
    """Tenant admins can discover OpenAI projects and bind scopes from the web UI."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Production",
            {"organization": "org-from-profile"},
            "gpt-5.4",
            "inference-secret",
            "ada",
        )
        save_billing_secret(
            session,
            database.secret_cipher,
            "acme",
            profile.id,
            "sk-admin-private",
            "ada",
        )
        profile_id = profile.id

    client = _authenticated_admin_client(admin_app)
    page = client.get("/admin/settings/costs")
    body = page.get_data(as_text=True)
    assert page.status_code == 200
    assert f'value="{SAVED_SECRET_MASK}"' in body
    assert "sk-admin-private" not in body
    assert 'name="billing_secret"' in body
    assert 'name="organization_id"' in body
    assert 'name="project_id"' in body
    assert "flask tenants bind-billing-scope" not in body

    unchanged_key_response = client.post(
        "/admin/settings/costs/billing",
        data={
            "csrf_token": _csrf(body),
            "profile_id": profile_id,
            "billing_secret": SAVED_SECRET_MASK,
        },
    )
    assert unchanged_key_response.status_code == 302
    with database.sessions() as session:
        saved_profile = session.get(ProviderProfile, profile_id)
        assert (
            database.secret_cipher.decrypt(saved_profile.billing_secret_ciphertext)
            == "sk-admin-private"
        )

    requests_mock.get(
        "https://api.openai.com/v1/organization/projects",
        status_code=403,
        json={"error": "missing permission"},
    )
    failed_lookup = client.post(
        "/admin/settings/costs/openai-projects",
        data={
            "csrf_token": _csrf(body),
            "profile_id": profile_id,
            "organization_id": "org-acme",
        },
    )
    assert failed_lookup.status_code == 400
    assert 'value="org-acme"' in failed_lookup.get_data(as_text=True)

    projects_request = requests_mock.get(
        "https://api.openai.com/v1/organization/projects",
        json={
            "object": "list",
            "data": [{"id": "proj_live", "name": "Production project"}],
            "has_more": False,
            "last_id": "proj_live",
        },
    )
    loaded = client.post(
        "/admin/settings/costs/openai-projects",
        data={
            "csrf_token": _csrf(body),
            "profile_id": profile_id,
            "organization_id": "org-acme",
        },
    )
    loaded_body = loaded.get_data(as_text=True)
    assert loaded.status_code == 200
    assert "Production project" in loaded_body
    assert 'value="proj_live"' in loaded_body
    assert 'value="org-acme"' in loaded_body
    assert projects_request.last_request.headers["Authorization"] == (
        "Bearer sk-admin-private"
    )

    saved = client.post(
        "/admin/settings/costs/openai-scope",
        data={
            "csrf_token": _csrf(loaded_body),
            "profile_id": profile_id,
            "organization_id": "org-acme",
            "project_id": "proj_live",
            "exclusive_scope_confirmation": "y",
        },
    )
    assert saved.status_code == 302
    with database.sessions() as session:
        binding = session.scalar(
            select(ProviderScopeBinding).where(
                ProviderScopeBinding.profile_id == profile_id
            )
        )
        project_node = session.get(ProviderScopeNode, binding.node_id)
        organization_node = session.get(ProviderScopeNode, project_node.parent_node_id)
        audit = session.scalar(
            select(AuditEvent).where(AuditEvent.action == "billing_scope.bind")
        )
        assert project_node.canonical_scope_id == "proj_live"
        assert organization_node.canonical_scope_id == "org-acme"
        assert audit.details == {"provider": "openai", "purpose": "billing"}
    database.engine.dispose()


def test_openrouter_workspace_can_be_bound_from_costs_page(admin_app):
    """Configure an OpenRouter Workspace ID with explicit tenant exclusivity in the UI."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openrouter",
            "OpenRouter Production",
            {},
            "openai/gpt-5",
            "inference-secret",
            "ada",
        )
        save_billing_secret(
            session,
            database.secret_cipher,
            "acme",
            profile.id,
            "management-private",
            "ada",
        )
        profile_id = profile.id

    client = _authenticated_admin_client(admin_app)
    page = client.get("/admin/settings/costs")
    body = page.get_data(as_text=True)
    assert 'name="workspace_id"' in body
    assert "flask tenants bind-billing-scope" not in body
    response = client.post(
        "/admin/settings/costs/openrouter-scope",
        data={
            "csrf_token": _csrf(body),
            "profile_id": profile_id,
            "workspace_id": "550e8400-e29b-41d4-a716-446655440000",
            "exclusive_scope_confirmation": "y",
        },
    )
    assert response.status_code == 302
    with database.sessions() as session:
        binding = session.scalar(
            select(ProviderScopeBinding).where(
                ProviderScopeBinding.profile_id == profile_id
            )
        )
        node = session.get(ProviderScopeNode, binding.node_id)
        assert node.scope_type == "workspace"
        assert node.canonical_scope_id == "550e8400-e29b-41d4-a716-446655440000"
    database.engine.dispose()
