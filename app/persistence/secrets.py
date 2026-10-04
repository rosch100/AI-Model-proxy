"""Authenticated encryption for provider credentials stored in the database."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True)
class SecretCipher:
    """Encrypt secrets with AES-256-GCM and an independently generated nonce."""

    _key: bytes = field(repr=False)

    @classmethod
    def from_key(cls, encoded_key: str) -> SecretCipher:
        """Build a cipher from a URL-safe base64-encoded 32-byte key."""
        if not isinstance(encoded_key, str):
            raise ValueError("PROVIDER_ENCRYPTION_KEY must encode exactly 32 bytes")
        try:
            key = base64.b64decode(encoded_key, altchars=b"-_", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "PROVIDER_ENCRYPTION_KEY must encode exactly 32 bytes"
            ) from exc
        if len(key) != 32:
            raise ValueError("PROVIDER_ENCRYPTION_KEY must encode exactly 32 bytes")
        return cls(key)

    def scope_fingerprint(
        self, tenant_id: str, provider: str, scope_type: str, scope_id: str
    ) -> str:
        """HMAC a canonical provider scope with a domain-separated derived key."""
        parts = (tenant_id, provider, scope_type, scope_id)
        if any(not isinstance(part, str) or not part for part in parts):
            raise ValueError(
                "Provider circuit scope components must be non-empty strings"
            )
        key = hmac.new(
            self._key, b"provider-circuit-breaker-scope-v1", hashlib.sha256
        ).digest()
        message = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        return hmac.new(key, message, hashlib.sha256).hexdigest()

    def encrypt(self, plaintext: str) -> str:
        """Return a base64-encoded nonce and authenticated ciphertext envelope."""
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._key).encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def decrypt(self, envelope: str) -> str:
        """Authenticate and decrypt a base64-encoded secret envelope."""
        try:
            payload = base64.b64decode(envelope, altchars=b"-_", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Encrypted secret envelope is malformed") from exc
        if len(payload) < 12 + 16:
            raise ValueError("Encrypted secret envelope is malformed")
        plaintext = AESGCM(self._key).decrypt(payload[:12], payload[12:], None)
        return plaintext.decode("utf-8")
