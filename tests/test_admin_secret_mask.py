"""Masked secret display for stored provider credentials."""

from __future__ import annotations

import secrets

from app.admin.security import ADMIN_COOKIE_NAME, mask_secret
from app.persistence.admin_auth import authenticate_admin, create_admin_session
from app.persistence.admin_ops import upsert_provider_profile
from app.persistence.passkeys import insert_passkey


def test_mask_secret_hides_middle():
    """Long secrets keep short edges; short secrets become bullets only."""
    assert mask_secret("abcdefghijklmnop") == "abcd••••••••mnop"
    assert mask_secret("short") == "•••••"
    assert mask_secret("1234567890") == "12••••••••7890"


def test_connection_page_shows_masked_inference_key(admin_app):
    """Stored Azure keys render masked and never appear in full in HTML."""
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
        upsert_provider_profile(
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
        principal = create_admin_session(session, account, enrollment_only=False)

    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    response = client.get("/admin/settings/connection")
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert mask_secret(full_key) in body
    assert full_key not in body
    assert "Gespeicherter Schlüssel" in body
