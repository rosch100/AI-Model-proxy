"""Argon2id hashing for tenant administrator passwords."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError, VerifyMismatchError

_HASHER = PasswordHasher()


def hash_admin_password(password: str) -> str:
    """Return an Argon2id hash for a newly chosen administrator password."""
    return _HASHER.hash(password)


def verify_admin_password(password_hash: str, password: str) -> bool:
    """Return True when the password matches the stored Argon2id hash."""
    try:
        return _HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHash):
        return False
