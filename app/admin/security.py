"""Admin cookie, CSRF, and session-secret configuration."""

from __future__ import annotations

from flask import Flask, Request, request
from flask_wtf.csrf import CSRFProtect

ADMIN_COOKIE_NAME = "admin_session"
ADMIN_COOKIE_PATH = "/admin"
MIN_ADMIN_SESSION_SECRET_BYTES = 32
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 900
LOGIN_LOCK_SECONDS = 900
SESSION_IDLE_SECONDS = 1800
SESSION_ABSOLUTE_SECONDS = 28800
GENERIC_LOGIN_ERROR = "Anmeldung fehlgeschlagen."
SAVED_SECRET_MASK = "••••••••"

csrf = CSRFProtect()


def admin_secret_bytes(secret: object) -> bytes:
    """Return UTF-8 bytes for a configured admin session secret."""
    if not isinstance(secret, str) or not secret:
        raise ValueError("ADMIN_SESSION_SECRET is required")
    encoded = secret.encode("utf-8")
    if len(encoded) < MIN_ADMIN_SESSION_SECRET_BYTES:
        raise ValueError(
            "ADMIN_SESSION_SECRET must contain at least "
            f"{MIN_ADMIN_SESSION_SECRET_BYTES} bytes"
        )
    return encoded


def configure_admin_security(app: Flask) -> None:
    """Bind Flask CSRF and cookie policy to the admin session secret."""
    secret = admin_secret_bytes(app.config.get("ADMIN_SESSION_SECRET"))
    app.config["SECRET_KEY"] = secret
    app.config["SESSION_COOKIE_NAME"] = "admin_csrf"
    app.config["SESSION_COOKIE_PATH"] = ADMIN_COOKIE_PATH
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = app.config.get("ENV") == "production"
    app.config["WTF_CSRF_CHECK_DEFAULT"] = False
    app.config["WTF_CSRF_SSL_STRICT"] = app.config.get("ENV") == "production"
    csrf.init_app(app)


def prefers_html(incoming: Request | None = None) -> bool:
    """Return True when the client prefers an HTML document over JSON."""
    current = incoming if incoming is not None else request
    accepted = current.accept_mimetypes
    html = accepted["text/html"]
    json_type = accepted["application/json"]
    return html > json_type or (
        html == json_type
        and html > 0
        and "text/html" in current.headers.get("Accept", "")
    )


def admin_cookie_secure(app: Flask) -> bool:
    """Return whether the opaque admin session cookie must be marked Secure."""
    return bool(app.config.get("SESSION_COOKIE_SECURE"))


def mask_secret(secret: str) -> str:
    """Return a display-only mask that reveals short prefix and suffix edges."""
    if len(secret) <= 8:
        return "•" * max(len(secret), 4)
    prefix_len = 4 if len(secret) >= 12 else 2
    return f"{secret[:prefix_len]}{'•' * 8}{secret[-4:]}"
