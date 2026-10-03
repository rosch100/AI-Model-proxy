"""WebAuthn registration and authentication helpers for admin passkeys."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any

from flask import Flask
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.authentication.verify_authentication_response import (
    VerifiedAuthentication,
)
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    PublicKeyCredentialHint,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)
from webauthn.registration.verify_registration_response import VerifiedRegistration


@dataclass(frozen=True)
class WebAuthnConfig:
    """Relying-party settings for admin passkey ceremonies."""

    rp_id: str
    rp_name: str
    origins: tuple[str, ...]


def webauthn_config_from_app(app: Flask) -> WebAuthnConfig:
    """Load and validate WebAuthn settings from the Flask config."""
    rp_id = app.config.get("WEBAUTHN_RP_ID")
    rp_name = app.config.get("WEBAUTHN_RP_NAME")
    origins = app.config.get("WEBAUTHN_ORIGINS")
    if not isinstance(rp_id, str) or not rp_id.strip():
        raise ValueError("WEBAUTHN_RP_ID is required")
    if not isinstance(rp_name, str) or not rp_name.strip():
        raise ValueError("WEBAUTHN_RP_NAME is required")
    if isinstance(origins, str):
        parsed = tuple(part.strip() for part in origins.split(",") if part.strip())
    elif isinstance(origins, (list, tuple)):
        parsed = tuple(str(origin).strip() for origin in origins if str(origin).strip())
    else:
        parsed = ()
    if not parsed:
        raise ValueError("WEBAUTHN_ORIGINS must list at least one origin")
    return WebAuthnConfig(rp_id=rp_id.strip(), rp_name=rp_name.strip(), origins=parsed)


def begin_registration(
    config: WebAuthnConfig,
    *,
    user_id: bytes,
    user_name: str,
    exclude_credential_ids: list[bytes],
) -> tuple[bytes, dict[str, Any]]:
    """Return challenge bytes and JSON-serializable creation options."""
    challenge = secrets.token_bytes(32)
    # Additional passkeys should prefer a different authenticator (security key /
    # another device) once the platform credential is already excluded.
    hints = (
        [
            PublicKeyCredentialHint.SECURITY_KEY,
            PublicKeyCredentialHint.CLIENT_DEVICE,
            PublicKeyCredentialHint.HYBRID,
        ]
        if exclude_credential_ids
        else None
    )
    options = generate_registration_options(
        rp_id=config.rp_id,
        rp_name=config.rp_name,
        user_name=user_name,
        user_id=user_id,
        challenge=challenge,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED,
            user_verification=UserVerificationRequirement.REQUIRED,
        ),
        exclude_credentials=[
            PublicKeyCredentialDescriptor(id=credential_id)
            for credential_id in exclude_credential_ids
        ]
        or None,
        hints=hints,
    )
    return challenge, json.loads(options_to_json(options))


def complete_registration(
    config: WebAuthnConfig, *, challenge: bytes, credential_json: str
) -> VerifiedRegistration:
    """Verify a registration credential against the expected challenge."""
    return verify_registration_response(
        credential=credential_json,
        expected_challenge=challenge,
        expected_rp_id=config.rp_id,
        expected_origin=list(config.origins),
        require_user_verification=True,
    )


def begin_authentication(config: WebAuthnConfig) -> tuple[bytes, dict[str, Any]]:
    """Return challenge bytes and discoverable authentication options."""
    challenge = secrets.token_bytes(32)
    options = generate_authentication_options(
        rp_id=config.rp_id,
        challenge=challenge,
        user_verification=UserVerificationRequirement.REQUIRED,
        allow_credentials=None,
    )
    return challenge, json.loads(options_to_json(options))


def complete_authentication(
    config: WebAuthnConfig,
    *,
    challenge: bytes,
    credential_json: str,
    credential_public_key: bytes,
    credential_current_sign_count: int,
) -> VerifiedAuthentication:
    """Verify an authentication assertion for a stored credential."""
    return verify_authentication_response(
        credential=credential_json,
        expected_challenge=challenge,
        expected_rp_id=config.rp_id,
        expected_origin=list(config.origins),
        credential_public_key=credential_public_key,
        credential_current_sign_count=credential_current_sign_count,
        require_user_verification=True,
    )
