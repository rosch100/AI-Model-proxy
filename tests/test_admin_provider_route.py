"""Accessible, tenant-authorized administration of the provider order."""

import re
import secrets

import pytest

from app.admin.security import ADMIN_COOKIE_NAME
from app.persistence.admin_auth import authenticate_admin, create_admin_session
from app.persistence.admin_ops import (
    activate_provider_profile,
    create_provider_profile,
    replace_catalog_entries,
)
from app.persistence.models import ProviderProfile, Tenant
from app.persistence.passkeys import insert_passkey


@pytest.fixture
def route_client(admin_app):
    """Create an authenticated client with two active provider profiles."""
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
        tenant = session.get(Tenant, "acme")
        ids = []
        for index in range(2):
            profile = create_provider_profile(
                session,
                database.secret_cipher,
                "acme",
                "openai",
                f"Account {index}",
                {},
                "model-a",
                "secret",
                "ada",
            )
            replace_catalog_entries(
                session, profile, [("model-a", None), ("model-b", None)], None
            )
            activate_provider_profile(session, tenant, profile.id, "ada")
            ids.append(profile.id)
    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    page = client.get("/admin/settings/connection")
    csrf = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', page.text).group(1)
    return client, database, ids, csrf


def test_tenant_routing_policy_is_editable_and_persisted(route_client):
    """Expose tenant strategy controls and persist valid policy changes."""
    client, database, _ids, csrf = route_client
    page = client.get("/admin/settings/connection")
    assert "Routingstrategie" in page.text
    response = client.post(
        "/admin/settings/connection/routing",
        data={
            "csrf_token": csrf,
            "strategy": "load_balanced",
            "load_balancing_method": "weighted_least_loaded",
            "cost_policy": "cost_tiers",
            "headroom_weight": "0",
            "max_retry_wait_seconds": "18",
            "tie_breaker": "profile_id",
        },
    )

    assert response.status_code == 302
    with database.sessions() as session:
        tenant = session.get(Tenant, "acme")
        assert tenant.routing_strategy == "load_balanced"
        assert tenant.routing_cost_policy == "cost_tiers"
        assert tenant.routing_headroom_weight == 0
        assert tenant.routing_max_retry_wait_seconds == 18


def test_connection_shows_priority_order(route_client):
    """Show provider profiles in route order with reorder controls."""
    client, _, ids, _ = route_client
    response = client.get("/admin/settings/connection")
    assert response.status_code == 200
    assert "Bevorzugte Reihenfolge" in response.text
    assert response.text.index(f'data-route-profile="{ids[0]}"') < response.text.index(
        f'data-route-profile="{ids[1]}"'
    )
    assert "Nach oben" in response.text
    assert "Nach unten" in response.text


def test_move_route_is_csrf_protected_and_changes_priority(route_client):
    """Require CSRF validation before updating route priorities."""
    client, database, ids, csrf = route_client
    url = f"/admin/settings/connection/{ids[1]}/move"
    assert (
        client.post(url, data={"profile_id": ids[1], "direction": "up"}).status_code
        == 400
    )
    response = client.post(
        url, data={"csrf_token": csrf, "profile_id": ids[1], "direction": "up"}
    )
    assert response.status_code == 302
    with database.sessions() as session:
        assert session.get(ProviderProfile, ids[1]).route_priority == 1
        assert session.get(ProviderProfile, ids[0]).route_priority == 2


def test_priority_endpoint_joins_equal_priority_group(route_client):
    """The tenant admin can explicitly assign a profile to a priority group."""
    client, database, ids, csrf = route_client
    response = client.post(
        f"/admin/settings/connection/{ids[1]}/priority",
        data={"csrf_token": csrf, "profile_id": ids[1], "priority": "1"},
    )

    assert response.status_code == 302
    with database.sessions() as session:
        assert session.get(ProviderProfile, ids[0]).route_priority == 1
        assert session.get(ProviderProfile, ids[1]).route_priority == 1


def test_individual_deactivation_preserves_other_profile(route_client):
    """Deactivate one profile while retaining the remaining route."""
    client, database, ids, csrf = route_client
    response = client.post(
        f"/admin/settings/connection/{ids[0]}/deactivate",
        data={"csrf_token": csrf, "profile_id": ids[0]},
    )
    assert response.status_code == 302
    with database.sessions() as session:
        assert session.get(ProviderProfile, ids[0]).route_priority is None
        assert session.get(ProviderProfile, ids[1]).route_priority == 1


def test_model_edit_uses_catalog_select_and_rejects_unknown(route_client):
    """Offer catalog models and reject an unknown default model."""
    client, database, ids, csrf = route_client
    page = client.get(f"/admin/settings/connection/{ids[0]}/edit")
    assert "<select" in page.text
    assert 'name="default_model"' in page.text
    response = client.post(
        f"/admin/settings/connection/{ids[0]}/edit",
        data={
            "csrf_token": csrf,
            "provider": "openai",
            "display_name": "Account 0",
            "default_model": "invented",
            "api_key": "",
        },
    )
    assert response.status_code == 400
    with database.sessions() as session:
        assert session.get(ProviderProfile, ids[0]).default_model == "model-a"


def test_invalid_direction_is_rejected(route_client):
    """Reject route moves with an unsupported direction."""
    client, _, ids, csrf = route_client
    response = client.post(
        f"/admin/settings/connection/{ids[0]}/move",
        data={"csrf_token": csrf, "profile_id": ids[0], "direction": "invalid"},
    )
    assert response.status_code == 400
