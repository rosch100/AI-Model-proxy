"""Administrator authentication, sessions, and login rate limits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.admin.security import GENERIC_LOGIN_ERROR, LOGIN_MAX_FAILURES
from app.persistence.admin_auth import (
    authenticate_admin,
    create_admin_session,
    is_login_locked,
    load_admin_principal,
    login_subject_hash,
    record_login_failure,
    revoke_admin_session,
)
from app.persistence.models import AdminSession


def _csrf_token(response_text: str) -> str:
    marker = 'name="csrf_token" type="hidden" value="'
    start = response_text.index(marker) + len(marker)
    end = response_text.index('"', start)
    return response_text[start:end]


def test_login_creates_httponly_admin_cookie(admin_app):
    """A valid login issues a Path=/admin HttpOnly session cookie."""
    client = admin_app.test_client()
    login_page = client.get("/admin/login")
    token = _csrf_token(login_page.get_data(as_text=True))

    response = client.post(
        "/admin/login",
        data={
            "username": "ada",
            "password": "correct-horse-battery",
            "csrf_token": token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    cookies = response.headers.getlist("Set-Cookie")
    admin_cookie = next(
        cookie for cookie in cookies if cookie.startswith("admin_session=")
    )
    assert "HttpOnly" in admin_cookie
    assert "Path=/admin" in admin_cookie
    assert "SameSite=Lax" in admin_cookie


def test_login_failures_are_generic_and_rate_limited(admin_app):
    """Invalid credentials stay generic and lock after too many failures."""
    client = admin_app.test_client()
    for _ in range(LOGIN_MAX_FAILURES):
        login_page = client.get("/admin/login")
        token = _csrf_token(login_page.get_data(as_text=True))
        response = client.post(
            "/admin/login",
            data={"username": "ada", "password": "wrong-password", "csrf_token": token},
        )
        assert response.status_code == 401
        assert GENERIC_LOGIN_ERROR in response.get_data(as_text=True)
        assert "ada" not in response.get_data(
            as_text=True
        ) or "Benutzername" in response.get_data(as_text=True)

    login_page = client.get("/admin/login")
    token = _csrf_token(login_page.get_data(as_text=True))
    locked = client.post(
        "/admin/login",
        data={
            "username": "ada",
            "password": "correct-horse-battery",
            "csrf_token": token,
        },
    )
    assert locked.status_code == 429


def test_logout_revokes_the_session(admin_app):
    """Logout deletes the cookie and refuses the stored session token."""
    client = admin_app.test_client()
    login_page = client.get("/admin/login")
    token = _csrf_token(login_page.get_data(as_text=True))
    client.post(
        "/admin/login",
        data={
            "username": "ada",
            "password": "correct-horse-battery",
            "csrf_token": token,
        },
    )
    enroll = client.get("/admin/passkeys/enroll")
    logout_token = _csrf_token(enroll.get_data(as_text=True))
    client.post("/admin/logout", data={"csrf_token": logout_token})

    redirected = client.get("/admin/")
    assert redirected.status_code == 302
    assert redirected.headers["Location"].endswith("/admin/login")


def test_session_loader_rejects_revoked_and_expired_tokens(admin_app):
    """Revoked and expired session rows do not authenticate."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        principal = create_admin_session(session, account)
        session_row = session.get(AdminSession, principal.session_id)
        session_row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    with database.sessions() as session:
        assert load_admin_principal(session, principal.token) is None

    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        principal = create_admin_session(session, account)
        revoke_admin_session(session, principal.session_id)

    with database.sessions() as session:
        assert load_admin_principal(session, principal.token) is None


def test_rate_limit_is_persisted_per_username_and_address(admin_app):
    """Login lockouts are stored against a username and remote address hash."""
    database = admin_app.extensions["database"]
    subject = login_subject_hash("ada", "203.0.113.9")
    with database.sessions.begin() as session:
        for _ in range(LOGIN_MAX_FAILURES):
            record_login_failure(session, subject)
        assert is_login_locked(session, subject)


def test_html_root_redirects_to_admin(admin_app):
    """Browsers requesting HTML at / are sent to the admin UI."""
    client = admin_app.test_client()
    response = client.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/admin")
