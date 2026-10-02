"""Connection form secret masking and catalog model selection."""

from __future__ import annotations

import secrets

from app.admin.security import ADMIN_COOKIE_NAME, mask_secret
from app.persistence.admin_auth import authenticate_admin, create_admin_session
from app.persistence.admin_ops import replace_catalog_entries, upsert_provider_profile
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


def test_connection_page_puts_secret_in_password_field(admin_app):
    """Stored keys fill a password input with a Klartext reveal control."""
    database = admin_app.extensions["database"]
    full_key = "azure-secret-key-value-xyz9"
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
        profile = upsert_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            "azure",
            {
                "base_url": "https://example.openai.azure.com",
                "model_deployments": {"gpt-6-astra": "gpt-6-astra"},
            },
            "gpt-6-astra",
            full_key,
            "ada",
        )
        replace_catalog_entries(
            session,
            profile,
            [("gpt-6-astra", "gpt-6-astra"), ("gpt-6-luna", "gpt-6-luna")],
            None,
        )
        principal = create_admin_session(session, account, enrollment_only=False)

    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    response = client.get("/admin/settings/connection")
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'id="azure-api-key"' in body
    assert 'type="password"' in body
    assert f'value="{full_key}"' in body
    assert 'data-reveal-secret="azure-api-key"' in body
    assert "Klartext anzeigen" in body
    assert "Wählbare Modelle" in body
    assert "gpt-6-astra" in body
    assert "Modell-Deployments" not in body
    assert "Pflichtfelder sind mit" in body
