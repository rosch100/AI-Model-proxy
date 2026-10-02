"""Login, session, and rate-limit persistence for tenant administrators."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.admin.passwords import verify_admin_password
from app.admin.security import (
    LOGIN_LOCK_SECONDS,
    LOGIN_MAX_FAILURES,
    LOGIN_WINDOW_SECONDS,
    SESSION_ABSOLUTE_SECONDS,
    SESSION_IDLE_SECONDS,
)

from .models import AdminAccount, AdminSession, LoginRateLimit, Tenant


@dataclass(frozen=True)
class AdminPrincipal:
    """Authenticated administrator bound to exactly one tenant."""

    session_id: str
    account_id: int
    tenant_id: str
    username: str
    token: str
    enrollment_only: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def login_subject_hash(username: str, remote_addr: str) -> str:
    """Return the durable rate-limit key for one username and client address."""
    material = f"{username.strip().casefold()}|{remote_addr}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def is_login_locked(
    session: Session, subject_hash: str, now: datetime | None = None
) -> bool:
    """Return True when the subject is inside an active lockout window."""
    current = now or _utc_now()
    row = session.get(LoginRateLimit, subject_hash)
    return (
        row is not None
        and row.locked_until is not None
        and _as_utc(row.locked_until) > current
    )


def record_login_failure(
    session: Session, subject_hash: str, now: datetime | None = None
) -> None:
    """Count a failed attempt and lock the subject after too many failures."""
    current = now or _utc_now()
    row = session.get(LoginRateLimit, subject_hash)
    if row is None:
        session.add(
            LoginRateLimit(
                subject_hash=subject_hash,
                failures=1,
                window_started_at=current,
                locked_until=None,
            )
        )
        return
    window_end = _as_utc(row.window_started_at) + timedelta(
        seconds=LOGIN_WINDOW_SECONDS
    )
    if current > window_end:
        row.failures = 1
        row.window_started_at = current
        row.locked_until = None
        return
    row.failures += 1
    if row.failures >= LOGIN_MAX_FAILURES:
        row.locked_until = current + timedelta(seconds=LOGIN_LOCK_SECONDS)


def clear_login_failures(session: Session, subject_hash: str) -> None:
    """Drop the rate-limit row after a successful authentication."""
    row = session.get(LoginRateLimit, subject_hash)
    if row is not None:
        session.delete(row)


def authenticate_admin(
    session: Session, username: str, password: str
) -> AdminAccount | None:
    """Return the unique matching administrator, or None on failure."""
    accounts = list(
        session.scalars(
            select(AdminAccount).where(AdminAccount.username == username.strip())
        )
    )
    matched: AdminAccount | None = None
    for account in accounts:
        if not verify_admin_password(account.password_hash, password):
            continue
        if matched is not None:
            return None
        matched = account
    return matched


def create_admin_session(
    session: Session, account: AdminAccount, *, enrollment_only: bool = False
) -> AdminPrincipal:
    """Create a new opaque session and return the plaintext cookie token once."""
    now = _utc_now()
    token = secrets.token_urlsafe(32)
    record = AdminSession(
        id=str(uuid4()),
        account_id=account.id,
        token_hash=_token_hash(token),
        created_at=now,
        expires_at=now + timedelta(seconds=SESSION_IDLE_SECONDS),
        revoked_at=None,
        enrollment_only=enrollment_only,
    )
    session.add(record)
    session.flush()
    return AdminPrincipal(
        session_id=record.id,
        account_id=account.id,
        tenant_id=account.tenant_id,
        username=account.username,
        token=token,
        enrollment_only=enrollment_only,
    )


def load_admin_principal(session: Session, token: str | None) -> AdminPrincipal | None:
    """Resolve a presented cookie token to a live administrator principal."""
    if not isinstance(token, str) or not token:
        return None
    digest = _token_hash(token)
    row = session.scalar(select(AdminSession).where(AdminSession.token_hash == digest))
    if row is None or not hmac.compare_digest(row.token_hash, digest):
        return None
    now = _utc_now()
    if row.revoked_at is not None or _as_utc(row.expires_at) <= now:
        return None
    absolute = _as_utc(row.created_at) + timedelta(seconds=SESSION_ABSOLUTE_SECONDS)
    if now >= absolute:
        row.revoked_at = now
        return None
    account = session.get(AdminAccount, row.account_id)
    tenant = session.get(Tenant, account.tenant_id) if account is not None else None
    if account is None or tenant is None:
        return None
    idle_expiry = now + timedelta(seconds=SESSION_IDLE_SECONDS)
    row.expires_at = idle_expiry if idle_expiry < absolute else absolute
    return AdminPrincipal(
        session_id=row.id,
        account_id=account.id,
        tenant_id=account.tenant_id,
        username=account.username,
        token=token,
        enrollment_only=bool(row.enrollment_only),
    )


def revoke_admin_session(session: Session, session_id: str) -> None:
    """Revoke one administrator session immediately."""
    row = session.get(AdminSession, session_id)
    if row is not None and row.revoked_at is None:
        row.revoked_at = _utc_now()


def clear_enrollment_only(session: Session, session_id: str) -> None:
    """Mark an enrollment-only session as fully authenticated."""
    row = session.get(AdminSession, session_id)
    if row is not None:
        row.enrollment_only = False


def revoke_account_sessions(
    session: Session, account_id: int, except_session_id: str | None = None
) -> None:
    """Revoke every live session for an account, optionally keeping the current one."""
    now = _utc_now()
    rows = session.scalars(
        select(AdminSession).where(
            AdminSession.account_id == account_id,
            AdminSession.revoked_at.is_(None),
        )
    )
    for row in rows:
        if except_session_id is not None and row.id == except_session_id:
            continue
        row.revoked_at = now
