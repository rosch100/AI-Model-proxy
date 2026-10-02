"""Passkey credential and WebAuthn challenge persistence."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import AdminAccount, AdminPasskey, AdminWebAuthnChallenge

CHALLENGE_TTL_SECONDS = 120


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_user_handle(session: Session, account: AdminAccount) -> bytes:
    """Return a stable WebAuthn user handle, creating one if missing."""
    if account.webauthn_user_handle is not None:
        return account.webauthn_user_handle
    handle = secrets.token_bytes(32)
    account.webauthn_user_handle = handle
    session.flush()
    return handle


def account_has_passkeys(session: Session, account_id: int) -> bool:
    """Return True when the account has at least one registered passkey."""
    count = session.scalar(
        select(func.count())
        .select_from(AdminPasskey)
        .where(AdminPasskey.account_id == account_id)
    )
    return bool(count)


def store_challenge(
    session: Session,
    *,
    purpose: str,
    challenge: bytes,
    account_id: int | None = None,
    ttl_seconds: int = CHALLENGE_TTL_SECONDS,
) -> str:
    """Persist a one-time ceremony challenge and return its id."""
    if purpose not in {"registration", "authentication"}:
        raise ValueError(f"Unsupported WebAuthn purpose {purpose!r}")
    row = AdminWebAuthnChallenge(
        id=str(uuid4()),
        account_id=account_id,
        purpose=purpose,
        challenge=challenge,
        expires_at=_utc_now() + timedelta(seconds=ttl_seconds),
        consumed_at=None,
    )
    session.add(row)
    session.flush()
    return row.id


def consume_challenge(
    session: Session,
    challenge_id: str,
    purpose: str,
    *,
    expected_account_id: int | None = None,
) -> bytes:
    """Consume a live challenge exactly once and return its bytes."""
    row = session.get(AdminWebAuthnChallenge, challenge_id)
    now = _utc_now()
    if (
        row is None
        or row.purpose != purpose
        or row.consumed_at is not None
        or _as_utc(row.expires_at) <= now
    ):
        raise LookupError("WebAuthn challenge is missing or expired")
    if (
        expected_account_id is not None
        and row.account_id is not None
        and row.account_id != expected_account_id
    ):
        raise LookupError("WebAuthn challenge does not belong to this account")
    if expected_account_id is not None and row.account_id is None:
        raise LookupError("WebAuthn challenge does not belong to this account")
    row.consumed_at = now
    session.flush()
    return row.challenge


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_passkey_by_credential_id(
    session: Session, credential_id: bytes
) -> AdminPasskey | None:
    """Resolve a stored passkey by its credential id."""
    return session.scalar(
        select(AdminPasskey).where(AdminPasskey.credential_id == credential_id)
    )


def insert_passkey(
    session: Session,
    *,
    account_id: int,
    credential_id: bytes,
    public_key: bytes,
    sign_count: int,
    user_handle: bytes,
    label: str,
    aaguid: str | None,
    backed_up: bool,
    transports: list[object] | None = None,
) -> AdminPasskey:
    """Insert a verified registration credential."""
    row = AdminPasskey(
        id=str(uuid4()),
        account_id=account_id,
        credential_id=credential_id,
        public_key=public_key,
        sign_count=sign_count,
        user_handle=user_handle,
        transports=transports,
        label=label.strip() or "Passkey",
        aaguid=aaguid,
        backed_up=backed_up,
    )
    session.add(row)
    session.flush()
    return row


def list_passkeys(session: Session, account_id: int) -> list[AdminPasskey]:
    """Return passkeys for an account ordered by creation time."""
    return list(
        session.scalars(
            select(AdminPasskey)
            .where(AdminPasskey.account_id == account_id)
            .order_by(AdminPasskey.created_at.asc())
        )
    )


def delete_passkey(session: Session, account_id: int, passkey_id: str) -> None:
    """Delete one passkey unless it is the account's last credential."""
    keys = list_passkeys(session, account_id)
    if len(keys) < 2:
        raise ValueError("The last passkey cannot be deleted")
    target = next((key for key in keys if key.id == passkey_id), None)
    if target is None:
        raise LookupError("Passkey not found")
    session.delete(target)
    session.flush()
