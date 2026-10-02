"""Admin CSRF and session-secret configuration."""

import base64

import pytest
from flask import Flask

from app import create_app
from app.admin.security import configure_admin_security
from app.exceptions import ServiceConfigurationError


def test_admin_session_secret_is_required():
    """Startup refuses a missing or too-short ADMIN_SESSION_SECRET."""
    encryption_key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    config = type(
        "MissingAdminSecret",
        (),
        {
            "SERVICE_API_KEY": "test-service-api-key",
            "AUTH_MODE": "tenant",
            "TENANT_CONFIG_SOURCE": "database",
            "ENABLE_CODEX": False,
            "DATABASE_URL": "postgresql+psycopg://proxy:proxy@localhost:5432/proxy",
            "PROVIDER_ENCRYPTION_KEY": encryption_key,
            "ADMIN_SESSION_SECRET": "short",
            "WEBAUTHN_RP_ID": "localhost",
            "WEBAUTHN_RP_NAME": "Test",
            "WEBAUTHN_ORIGINS": "http://localhost",
        },
    )
    with pytest.raises(ServiceConfigurationError, match="ADMIN_SESSION_SECRET"):
        create_app(config)


def test_admin_csrf_cookie_is_scoped_to_admin_path():
    """Flask CSRF cookies stay on the administrator path."""
    app = Flask("csrf-config")
    app.config["ADMIN_SESSION_SECRET"] = "test-admin-session-secret-bytes-32"
    app.config["ENV"] = "production"
    configure_admin_security(app)
    assert app.config["SESSION_COOKIE_PATH"] == "/admin"
    assert app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert app.config["SESSION_COOKIE_SECURE"] is True
    assert app.config["WTF_CSRF_CHECK_DEFAULT"] is False


def test_login_post_without_csrf_is_rejected(admin_app):
    """Administrator POST routes require a CSRF token."""
    response = admin_app.test_client().post(
        "/admin/login",
        data={"username": "ada", "password": "correct-horse-battery"},
    )
    assert response.status_code == 400
