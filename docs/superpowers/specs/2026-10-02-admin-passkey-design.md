# Admin Passkey (WebAuthn) Authentication

**Date:** 2026-10-02  
**Status:** Draft for review  
**Worktree:** `feature/tenant-provider-admin`  
**Library:** Duo Labs `webauthn` (py_webauthn) on PyPI

## Goal

Replace password as the steady-state admin login with mandatory passkeys. Passwords remain only for bootstrap until the first passkey is enrolled. Cursor Bearer API keys are unchanged.

## Decisions (locked)

| Topic | Choice |
| --- | --- |
| Relation to password | Passkey replaces password after enrollment |
| Login UX | Username-free discoverable passkey (Conditional UI); username+password only for bootstrap |
| Mandate | Passkey is required; no optional long-term password login |
| Enrollment gate | After password login, force passkey setup; no dashboard/settings until ≥1 passkey exists |
| Recovery | One passkey minimum; more managed under Account; lost last passkey → host operator (DB), no UI self-reset, no CLI recovery command in this scope |
| Approach | Own Flask routes + `webauthn` package (not Flask-Security, not external IdP) |

## Architecture

```text
Browser ──► /admin/login (passkey get) ──► AdminSession cookie ──► /admin/*
   │
   └── bootstrap: username+password ──► enrollment-only session ──► register passkey
                                                                      │
Postgres ◄── admin_passkeys + admin_webauthn_challenges ◄─────────────┘
```

- Opaque `admin_session` cookie (`Path=/admin`) stays the session authority after either bootstrap or passkey login.
- WebAuthn ceremonies use short-lived challenges stored in the database (shared across gunicorn workers).
- Tenant identity continues to come only from the authenticated `AdminAccount` / session, never from Host or form fields.

## Data model

### `admin_passkeys`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID string PK | |
| `account_id` | FK → `admin_accounts.id` | Cascade on account delete |
| `credential_id` | bytes / base64url unique | WebAuthn credential ID |
| `public_key` | bytes | COSE key from registration |
| `sign_count` | int | Updated on each successful auth; reject regressions |
| `user_handle` | bytes | Stable per account (discoverable) |
| `transports` | JSON list nullable | Optional hint for UI |
| `label` | string | User-visible name |
| `aaguid` | string nullable | |
| `backed_up` | bool | From registration flags |
| `created_at` / `last_used_at` | timestamptz | |

### `admin_webauthn_challenges`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID string PK | |
| `account_id` | FK nullable | Set for registration; null for usernameless auth begin |
| `purpose` | `registration` \| `authentication` | |
| `challenge` | bytes | |
| `expires_at` | timestamptz | ~2 minutes |
| `consumed_at` | timestamptz nullable | One-time use |

### Account enrollment state

Derived, not a separate enum column: an account is **enrolled** iff it has ≥1 row in `admin_passkeys`.

Optional session marker: store `enrollment_required=true` in a server-side session row flag or a dedicated column on `admin_sessions` so middleware can restrict routes without re-querying passkeys on every request. Prefer a boolean `admin_sessions.enrollment_only` set at password-bootstrap login and cleared when the first passkey is verified.

## Configuration

New settings (validated at startup when admin/database mode is active):

- `WEBAUTHN_RP_ID` — e.g. `proxy.altanis.de` (no scheme/port)
- `WEBAUTHN_RP_NAME` — display name, e.g. `Altanis Proxy`
- `WEBAUTHN_ORIGINS` — comma-separated exact origins, e.g. `https://proxy.altanis.de,https://proxy.iffm-gmbh.de`

Registration and authentication verification must use these origins and RP ID. Local tests use `http://localhost` / `localhost` via test settings.

Dependency: add `webauthn` (current stable) to `requirements/prod.txt`.

## HTTP surface

All under `/admin`, CSRF: HTML forms keep Flask-WTF; JSON WebAuthn POSTs require the CSRF header/token like other admin POSTs (same cookie session).

| Method | Path | Auth | Behavior |
| --- | --- | --- | --- |
| GET | `/admin/login` | anon | Passkey primary UI; password bootstrap form secondary/collapsed |
| POST | `/admin/login` | anon | Password bootstrap only if account has **zero** passkeys; else generic failure |
| POST | `/admin/webauthn/login/begin` | anon | `generate_authentication_options` (empty `allow_credentials` for discoverable) |
| POST | `/admin/webauthn/login/complete` | anon | Verify assertion → full session cookie |
| GET | `/admin/passkeys/enroll` | enrollment session | Forced enrollment page |
| POST | `/admin/webauthn/register/begin` | enrollment or full session | Registration options for current account |
| POST | `/admin/webauthn/register/complete` | enrollment or full session | Persist passkey; clear `enrollment_only`; audit |
| GET | `/admin/account` | full session | List passkeys; add/remove |
| POST | `/admin/account/passkeys/<id>/delete` | full session | Delete if ≥2 remain; refuse deleting last |

### Route guard

Authenticated requests with `enrollment_only`:

- Allow: enroll page, register begin/complete, logout
- Deny (redirect to enroll): dashboard, settings, costs, key rotation, password change (password change disabled once enrolled)

Unauthenticated `/admin/*` (except login + webauthn login) → login.

## Ceremony details

- `user_verification=required` for registration and authentication
- Discoverable / resident key preferred for registration (`resident_key=required` or equivalent authenticator selection)
- `user_handle`: random 32 bytes per `AdminAccount`, stored once (new column `admin_accounts.webauthn_user_handle`), used for all of that account’s passkeys
- Update `sign_count` after successful auth; if authenticator reports lower count than stored (and previous was non-zero), reject and audit
- Challenges: single use, expire in 120 seconds, deleted or marked consumed

## UI

- Login: brand-first page already present; primary CTA „Mit Passkey anmelden“; password fields only for accounts still in bootstrap (UI always offers both, server enforces policy)
- Enroll: single purpose page — register first passkey; no navigation chrome to settings
- Account: passkey list with label, created/last used, add and delete controls
- Progressively enhance with local JS (`admin-passkeys.js`); no external CDN for WebAuthn helpers beyond vendoring if needed — prefer native `navigator.credentials` + JSON from `webauthn.options_to_json`

## Audit

Append `audit_events` without secrets:

- `passkey.register` / `passkey.delete` / `passkey.login` / `passkey.login_failed` (generic)
- Never store credential private material or raw challenges in audit details

## Testing

- Unit: challenge lifecycle, enrollment gate, last-passkey delete refusal, password rejected when enrolled
- Integration: registration + authentication with mocked/faked WebAuthn verification hooks or library test helpers
- Existing admin auth/session tests remain green; extend fixtures with passkey tables

## Out of scope

- Self-service password reset or passkey recovery email
- Flask CLI recovery command (document operator DB steps only in DEPLOYMENT)
- Passkey as second factor alongside password
- Host-based tenant selection
- Changes to Cursor Bearer auth

## Rollout on proxy.altanis.de

1. Migrate schema; set `WEBAUTHN_RP_ID` / `WEBAUTHN_ORIGINS`
2. Existing `admin` accounts have zero passkeys → password bootstrap still works once
3. Operator logs in with password, enrolls passkey, subsequent logins are passkey-only
4. Verify both public hostnames in `WEBAUTHN_ORIGINS` if both serve `/admin`
