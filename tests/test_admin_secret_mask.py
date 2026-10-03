"""Connection form secret masking and catalog model selection."""

from __future__ import annotations

import re
import secrets

from sqlalchemy import select

from app.admin.security import ADMIN_COOKIE_NAME, mask_secret
from app.persistence.admin_auth import authenticate_admin, create_admin_session
from app.persistence.admin_ops import (
    create_provider_profile,
    replace_catalog_entries,
)
from app.persistence.models import (
    AuditEvent,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from app.persistence.passkeys import insert_passkey
from app.providers.catalog import (
    azure_deployments_from_catalog,
    selectable_catalog_models,
)


def test_mask_secret_hides_middle():
    """Long secrets keep short edges; short secrets become bullets only."""
    assert mask_secret("abcdefghijklmnop") == "abcd••••••••mnop"
    assert mask_secret("short") == "•••••"
    assert mask_secret("1234567890") == "12••••••••7890"


def test_selectable_azure_catalog_keeps_supported_models_only():
    """Azure selectable models are the intersection with SUPPORTED_MODELS."""
    entries = [
        ("gpt-6-astra", "gpt-6-astra"),
        ("some-other-model", "other-deploy"),
        ("gpt-4o", "gpt-6-luna"),
    ]
    assert selectable_catalog_models("azure", entries) == [
        ("gpt-6-astra", "gpt-6-astra"),
        ("gpt-6-luna", "gpt-6-luna"),
    ]
    assert azure_deployments_from_catalog(entries) == {
        "gpt-6-astra": "gpt-6-astra",
        "gpt-6-luna": "gpt-6-luna",
    }


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


def test_connection_page_renders_same_provider_profiles_without_exposing_secrets(
    admin_app,
):
    """Both same-provider profiles remain visible and their keys stay private."""
    database = admin_app.extensions["database"]
    secret_values = ("azure-production-secret", "azure-staging-secret")
    with database.sessions.begin() as session:
        profiles = []
        for name, model, secret in (
            ("Production", "gpt-6-astra", secret_values[0]),
            ("Staging", "gpt-6-luna", secret_values[1]),
        ):
            profile = create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                "azure",
                name,
                {
                    "base_url": "https://example.openai.azure.com",
                    "model_deployments": {model: model},
                },
                model,
                secret,
                "ada",
            )
            replace_catalog_entries(session, profile, [(model, model)], None)
            profiles.append(profile)

    client = _authenticated_admin_client(admin_app)
    response = client.get("/admin/settings/connection")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Production" in body
    assert "Staging" in body
    assert "gpt-6-astra" in body
    assert "gpt-6-luna" in body
    assert "Schlüssel gespeichert" in body
    for secret in secret_values:
        assert secret not in body
    for profile in profiles:
        assert f"/admin/settings/connection/{profile.id}/edit" in body


def test_delete_connection_soft_deletes_only_the_tenant_profile(admin_app):
    """Profile deletion is CSRF-protected, audited, and tenant-scoped."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        session.add(
            Tenant(
                id="other",
                api_key_hash="b" * 64,
                custom_model_id="cursor-other-model",
            )
        )
        own_profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "openai",
            "Own account",
            {},
            "gpt-5.4",
            "sk-own-secret",
            "ada",
        )
        foreign_profile = create_provider_profile(
            session,
            database.secret_cipher,
            "other",
            "openai",
            "Foreign account",
            {},
            "gpt-5.4",
            "sk-foreign-secret",
            "operator",
        )
        own_profile_id = own_profile.id
        foreign_profile_id = foreign_profile.id
        own_profile.route_priority = 1

    client = _authenticated_admin_client(admin_app)
    page = client.get("/admin/settings/connection")
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"',
        page.get_data(as_text=True),
    ).group(1)
    forbidden = client.post(
        f"/admin/settings/connection/{foreign_profile_id}/delete",
        data={"csrf_token": csrf, "profile_id": foreign_profile_id},
    )
    assert forbidden.status_code == 404

    deleted = client.post(
        f"/admin/settings/connection/{own_profile_id}/delete",
        data={"csrf_token": csrf, "profile_id": own_profile_id},
    )
    assert deleted.status_code == 302
    with database.sessions() as session:
        own_profile = session.get(ProviderProfile, own_profile_id)
        assert own_profile.deleted_at is not None
        assert session.get(ProviderProfile, foreign_profile_id).deleted_at is None
        assert own_profile.route_priority is None
        event = session.scalar(
            select(AuditEvent).where(AuditEvent.action == "profile.delete")
        )
        assert event.target == f"profile:{own_profile_id}"
    database.engine.dispose()


def test_costs_page_explains_provider_specific_credentials_and_limits(admin_app):
    """Provider-specific billing requirements are visible in the admin UI."""
    client = _authenticated_admin_client(admin_app)
    response = client.get("/admin/settings/costs")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Billing-Zugangsdaten sind vom Inference-Schlüssel getrennt." in body
    assert "Admin API Key" in body
    assert "Management Key" in body
    assert "Workspace-ID dieses Kontos enthalten" in body
    assert "als Filter verwendet" in body
    assert "keine Tenant-Credential eingegeben" in body
    assert "letzten 30 abgeschlossenen UTC-Tage" in body
    assert "OpenRouter- und BYOK-Kosten sowie Tokenverbrauch" in body
    assert "Kosten und Input-/Output-Tokens" in body


def test_azure_cost_refresh_rejects_usage_scope_outside_billing_scope(
    admin_app, monkeypatch
):
    """Azure refresh does not query a usage node from another resource group."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        session.add(
            ProviderProfile(
                id="azure-profile",
                tenant_id="acme",
                provider="azure",
                settings={},
            )
        )
        session.add_all(
            (
                ProviderScopeNode(
                    id="azure-rg-node",
                    tenant_id="acme",
                    provider="azure",
                    scope_type="resource_group",
                    canonical_scope_id="/subscriptions/sub/resourceGroups/acme-rg",
                ),
                ProviderScopeNode(
                    id="other-rg-node",
                    tenant_id="acme",
                    provider="azure",
                    scope_type="resource_group",
                    canonical_scope_id="/subscriptions/sub/resourceGroups/other-rg",
                ),
                ProviderScopeNode(
                    id="azure-resource-node",
                    tenant_id="acme",
                    provider="azure",
                    scope_type="cognitive_resource",
                    canonical_scope_id=(
                        "/subscriptions/sub/resourceGroups/other-rg/providers/"
                        "microsoft.cognitiveservices/accounts/acme-ai"
                    ),
                    parent_node_id="other-rg-node",
                ),
            )
        )
        session.flush()
        session.add_all(
            (
                ProviderScopeBinding(
                    id="azure-billing-binding",
                    tenant_id="acme",
                    provider="azure",
                    profile_id="azure-profile",
                    purpose="billing",
                    node_id="azure-rg-node",
                ),
                ProviderScopeBinding(
                    id="azure-usage-binding",
                    tenant_id="acme",
                    provider="azure",
                    profile_id="azure-profile",
                    purpose="usage",
                    node_id="azure-resource-node",
                    parent_binding_id="azure-billing-binding",
                ),
            )
        )

    queried_scopes = []
    monkeypatch.setattr(
        "app.admin.views.collect_provider_costs",
        lambda *args, **kwargs: queried_scopes.append(kwargs.get("usage_scope_id"))
        or [],
    )
    client = _authenticated_admin_client(admin_app)
    page = client.get("/admin/settings/costs").get_data(as_text=True)
    csrf_token = re.search(r'name="csrf_token" type="hidden" value="([^"]+)"', page)
    assert csrf_token is not None

    client.post(
        "/admin/settings/costs/refresh",
        data={
            "profile_id": "azure-profile",
            "csrf_token": csrf_token.group(1),
        },
    )

    assert queried_scopes == []


def test_costs_page_shows_azure_host_identity_and_bound_scope_controls(admin_app):
    """Azure uses read-only operator identity information, not tenant secrets."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        session.add(
            ProviderProfile(
                id="azure-profile",
                tenant_id="acme",
                provider="azure",
                settings={},
            )
        )
        resource_group = ProviderScopeNode(
            id="azure-rg-node",
            tenant_id="acme",
            provider="azure",
            scope_type="resource_group",
            canonical_scope_id="/subscriptions/sub/resourcegroups/acme-rg",
        )
        resource = ProviderScopeNode(
            id="azure-resource-node",
            tenant_id="acme",
            provider="azure",
            scope_type="cognitive_services",
            canonical_scope_id=(
                "/subscriptions/sub/resourcegroups/acme-rg/providers/"
                "microsoft.cognitiveservices/accounts/acme-ai"
            ),
            parent_node_id="azure-rg-node",
        )
        session.add_all((resource_group, resource))
        session.flush()
        billing_binding = ProviderScopeBinding(
            id="azure-billing-binding",
            tenant_id="acme",
            provider="azure",
            profile_id="azure-profile",
            purpose="billing",
            node_id=resource_group.id,
        )
        usage_binding = ProviderScopeBinding(
            id="azure-usage-binding",
            tenant_id="acme",
            provider="azure",
            profile_id="azure-profile",
            purpose="usage",
            node_id=resource.id,
            parent_binding_id="azure-billing-binding",
        )
        session.add_all((billing_binding, usage_binding))

    body = (
        _authenticated_admin_client(admin_app)
        .get("/admin/settings/costs")
        .get_data(as_text=True)
    )

    assert "Hostidentität" in body
    assert "Managed Identity" in body
    assert "Workload Identity" in body
    assert "Azure-Kosten aktualisieren" in body
    assert 'name="client_secret"' not in body
    assert "Azure-Service-Principal" not in body
    assert "acme-rg" in body
