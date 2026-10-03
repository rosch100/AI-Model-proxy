"""Admin passkey persistence, enrollment gate, and WebAuthn routes."""

from __future__ import annotations

import secrets
from types import SimpleNamespace

from sqlalchemy import select
from webauthn.helpers.exceptions import InvalidAuthenticationResponse

from app.admin.security import GENERIC_LOGIN_ERROR
from app.admin.webauthn_service import webauthn_config_from_app
from app.persistence.admin_auth import (
    authenticate_admin,
    create_admin_session,
    load_admin_principal,
)
from app.persistence.models import AdminPasskey, AdminSession, AuditEvent
from app.persistence.passkeys import (
    account_has_passkeys,
    consume_challenge,
    delete_passkey,
    insert_passkey,
    store_challenge,
)


def _csrf_token(response_text: str) -> str:
    marker = 'name="csrf_token" type="hidden" value="'
    start = response_text.index(marker) + len(marker)
    end = response_text.index('"', start)
    return response_text[start:end]


def _password_login(
    client, username: str = "ada", password: str = "correct-horse-battery"
):
    login_page = client.get("/admin/login")
    token = _csrf_token(login_page.get_data(as_text=True))
    return client.post(
        "/admin/login",
        data={"username": username, "password": password, "csrf_token": token},
        follow_redirects=False,
    )


def _insert_dummy_passkey(
    session, account_id: int, label: str = "Primary"
) -> AdminPasskey:
    return insert_passkey(
        session,
        account_id=account_id,
        credential_id=secrets.token_bytes(32),
        public_key=secrets.token_bytes(64),
        sign_count=0,
        user_handle=secrets.token_bytes(32),
        label=label,
        aaguid=None,
        backed_up=False,
    )


def test_challenge_is_single_use(admin_app):
    """A WebAuthn challenge can be consumed exactly once."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        challenge_id = store_challenge(
            session, purpose="authentication", challenge=b"abc"
        )
        assert consume_challenge(session, challenge_id, "authentication") == b"abc"
        try:
            consume_challenge(session, challenge_id, "authentication")
            assert False, "expected LookupError"
        except LookupError:
            pass


def test_delete_refuses_last_passkey(admin_app):
    """Accounts must keep at least one passkey."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        first = _insert_dummy_passkey(session, account.id, "One")
        second = _insert_dummy_passkey(session, account.id, "Two")
        delete_passkey(session, account.id, first.id)
        try:
            delete_passkey(session, account.id, second.id)
            assert False, "expected ValueError"
        except ValueError:
            pass
        assert account_has_passkeys(session, account.id)


def test_password_login_sets_enrollment_only_session(admin_app):
    """Bootstrap password login creates an enrollment-only session."""
    client = admin_app.test_client()
    response = _password_login(client)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/admin/passkeys/enroll")

    database = admin_app.extensions["database"]
    with database.sessions() as session:
        row = session.scalar(select(AdminSession))
        assert row is not None
        assert row.enrollment_only is True


def test_enrollment_only_blocks_dashboard(admin_app):
    """Enrollment sessions cannot reach the dashboard."""
    client = admin_app.test_client()
    _password_login(client)
    redirected = client.get("/admin/")
    assert redirected.status_code == 302
    assert redirected.headers["Location"].endswith("/admin/passkeys/enroll")
    enroll = client.get("/admin/passkeys/enroll")
    assert enroll.status_code == 200
    assert "Passkey erforderlich" in enroll.get_data(as_text=True)


def test_full_session_without_passkeys_is_forced_to_enroll(admin_app):
    """Passkey mandate is enforced by credential count, not only the session flag."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        principal = create_admin_session(session, account, enrollment_only=False)
        token = principal.token
    client = admin_app.test_client()
    client.set_cookie("admin_session", token, path="/admin")
    redirected = client.get("/admin/")
    assert redirected.status_code == 302
    assert redirected.headers["Location"].endswith("/admin/passkeys/enroll")
    enroll = client.get("/admin/passkeys/enroll")
    assert enroll.status_code == 200


def test_registration_challenge_rejects_foreign_account(admin_app):
    """Registration complete refuses challenges that belong to another account."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        challenge_id = store_challenge(
            session,
            purpose="registration",
            challenge=b"foreign",
            account_id=None,
        )
        try:
            consume_challenge(
                session,
                challenge_id,
                "registration",
                expected_account_id=1,
            )
            assert False, "expected LookupError"
        except LookupError:
            pass


def test_password_rejected_when_passkey_exists(admin_app):
    """Password login fails once the account has a passkey."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        _insert_dummy_passkey(session, account.id)

    client = admin_app.test_client()
    response = _password_login(client)
    assert response.status_code == 401
    assert GENERIC_LOGIN_ERROR in response.get_data(as_text=True)


def test_register_complete_clears_enrollment_only(admin_app, monkeypatch):
    """Successful registration clears enrollment_only and writes an audit event."""
    client = admin_app.test_client()
    _password_login(client)
    enroll = client.get("/admin/passkeys/enroll")
    token = enroll.get_data(as_text=True)
    csrf = None
    for marker in (
        'name="csrf-token" content="',
        'name="csrf_token" type="hidden" value="',
    ):
        if marker in token:
            start = token.index(marker) + len(marker)
            csrf = token[start : token.index('"', start)]
            break
    assert csrf

    verified = SimpleNamespace(
        credential_id=b"cred-1",
        credential_public_key=b"pubkey-1",
        sign_count=0,
        aaguid="00000000-0000-0000-0000-000000000000",
        credential_backed_up=True,
    )
    monkeypatch.setattr(
        "app.admin.views.complete_registration",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(
        "app.admin.views.consume_challenge",
        lambda *args, **kwargs: b"challenge",
    )

    response = client.post(
        "/admin/webauthn/register/complete",
        json={"challenge_id": "ignored", "credential": {"id": "x"}, "label": "Laptop"},
        headers={"X-CSRFToken": csrf},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["redirect"].endswith("/admin/")

    database = admin_app.extensions["database"]
    with database.sessions() as session:
        keys = list(session.scalars(select(AdminPasskey)))
        assert len(keys) == 1
        assert keys[0].label == "Laptop"
        row = session.scalar(select(AdminSession))
        assert row.enrollment_only is False
        events = list(session.scalars(select(AuditEvent)))
        assert any(event.action == "passkey.register" for event in events)

    dashboard = client.get("/admin/")
    assert dashboard.status_code == 200


def test_additional_passkey_redirects_to_account(admin_app, monkeypatch):
    """A second passkey stays on the account page and excludes the first credential."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        account_id = account.id
        existing = _insert_dummy_passkey(session, account_id, "Primary")
        existing_credential_id = existing.credential_id
        principal = create_admin_session(session, account, enrollment_only=False)
        token = principal.token

    client = admin_app.test_client()
    client.set_cookie("admin_session", token, path="/admin")
    account_page = client.get("/admin/account")
    assert account_page.status_code == 200
    html = account_page.get_data(as_text=True)
    assert "Mehrere Passkeys sind erlaubt" in html
    marker = 'name="csrf-token" content="'
    start = html.index(marker) + len(marker)
    csrf = html[start : html.index('"', start)]

    begin = client.post(
        "/admin/webauthn/register/begin",
        json={},
        headers={"X-CSRFToken": csrf},
    )
    assert begin.status_code == 200
    begin_payload = begin.get_json()
    exclude = begin_payload["options"]["excludeCredentials"]
    assert len(exclude) == 1
    assert begin_payload["options"]["hints"] == [
        "security-key",
        "client-device",
        "hybrid",
    ]

    verified = SimpleNamespace(
        credential_id=b"cred-2-additional",
        credential_public_key=b"pubkey-2",
        sign_count=0,
        aaguid="00000000-0000-0000-0000-000000000000",
        credential_backed_up=False,
    )
    monkeypatch.setattr(
        "app.admin.views.complete_registration",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(
        "app.admin.views.consume_challenge",
        lambda *args, **kwargs: b"challenge-2",
    )

    complete = client.post(
        "/admin/webauthn/register/complete",
        json={
            "challenge_id": begin_payload["challenge_id"],
            "credential": {"id": "y"},
            "label": "YubiKey",
        },
        headers={"X-CSRFToken": csrf},
    )
    assert complete.status_code == 200
    payload = complete.get_json()
    assert payload["ok"] is True
    assert payload["redirect"].endswith("/admin/account")

    with database.sessions() as session:
        keys = list(
            session.scalars(
                select(AdminPasskey).where(AdminPasskey.account_id == account_id)
            )
        )
        assert len(keys) == 2
        labels = {key.label for key in keys}
        assert labels == {"Primary", "YubiKey"}
        assert existing_credential_id in {key.credential_id for key in keys}


def test_webauthn_config_from_test_settings(admin_app):
    """Test settings expose a usable WebAuthn relying-party config."""
    config = webauthn_config_from_app(admin_app)
    assert config.rp_id == "localhost"
    assert "http://localhost" in config.origins


def test_login_page_promotes_passkey_cta(admin_app):
    """Login HTML exposes the passkey primary call to action."""
    page = admin_app.test_client().get("/admin/login").get_data(as_text=True)
    assert "Mit Passkey anmelden" in page
    assert "admin-passkeys.js" in page


def test_passkey_login_failure_writes_audit(admin_app, monkeypatch):
    """Failed passkey verification writes a generic login_failed audit event."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        passkey = _insert_dummy_passkey(session, account.id)
        credential_id = passkey.credential_id
        challenge_id = store_challenge(
            session, purpose="authentication", challenge=b"chal"
        )

    client = admin_app.test_client()
    login_page = client.get("/admin/login")
    html = login_page.get_data(as_text=True)
    marker = 'name="csrf-token" content="'
    start = html.index(marker) + len(marker)
    csrf = html[start : html.index('"', start)]

    monkeypatch.setattr(
        "app.admin.views.parse_authentication_credential_json",
        lambda _payload: SimpleNamespace(raw_id=credential_id),
    )

    def _fail_verify(*args, **kwargs):
        raise InvalidAuthenticationResponse("bad assertion")

    monkeypatch.setattr("app.admin.views.complete_authentication", _fail_verify)

    response = client.post(
        "/admin/webauthn/login/complete",
        json={
            "challenge_id": challenge_id,
            "credential": {"id": "x", "rawId": "x", "type": "public-key"},
        },
        headers={"X-CSRFToken": csrf},
    )
    assert response.status_code == 401
    with database.sessions() as session:
        events = list(session.scalars(select(AuditEvent)))
        assert any(event.action == "passkey.login_failed" for event in events)


def test_enrollment_only_principal_flag_roundtrip(admin_app):
    """create_admin_session persists enrollment_only for load_admin_principal."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        principal = create_admin_session(session, account, enrollment_only=True)
    with database.sessions() as session:
        loaded = load_admin_principal(session, principal.token)
        assert loaded is not None
        assert loaded.enrollment_only is True
