# Admin Passkey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mandatory WebAuthn passkeys for `/admin`, with password only for bootstrap until first enrollment.

**Architecture:** Duo Labs `webauthn` 3.x ceremonies over JSON under the existing Flask admin blueprint; credentials and challenges in PostgreSQL; `admin_sessions.enrollment_only` gates routes until the first passkey exists.

**Tech Stack:** Flask 3, SQLAlchemy 2, Alembic, `webauthn==3.0.1`, existing opaque admin cookies + Flask-WTF CSRF.

## Global Constraints

- Work only in worktree `/Users/roschmac/Entwicklung/Cursor-Azure-GPT-5-tenant-provider-admin` on `feature/tenant-provider-admin`.
- Spec SSOT: `docs/superpowers/specs/2026-10-02-admin-passkey-design.md`.
- Passkey mandatory after enrollment; password rejected when account has ≥1 passkey.
- Username-free discoverable authentication; `user_verification=required`; resident key required on registration.
- No self-service recovery CLI; document operator DB recovery in DEPLOYMENT only.
- Cursor Bearer auth unchanged.
- After substantive code changes: `source .venv/bin/activate && flask --app 'app:create_app("tests.settings")' lint` and `pytest -k ""`.

## File map

| File | Responsibility |
| --- | --- |
| `app/persistence/models.py` | `AdminPasskey`, `AdminWebAuthnChallenge`, `AdminAccount.webauthn_user_handle`, `AdminSession.enrollment_only` |
| `migrations/versions/20261002_admin_passkeys.py` | Alembic upgrade/downgrade |
| `app/persistence/passkeys.py` | Challenge + credential persistence helpers |
| `app/admin/webauthn_service.py` | py_webauthn option/verify wrappers |
| `app/admin/views.py` | Routes + enrollment guard |
| `app/admin/security.py` | WebAuthn config constants helpers |
| `app/settings.py`, `tests/settings.py`, `.env.example` | `WEBAUTHN_*` |
| `app/templates/admin/login.html`, `passkeys_enroll.html`, `account.html` | UI |
| `app/static/admin/admin-passkeys.js` | Browser WebAuthn client |
| `tests/test_admin_passkeys.py` | Auth/enrollment/passkey tests |
| `DEPLOYMENT.md`, `README.md` | Bootstrap + operator recovery note |

---

### Task 1: Schema and migration

**Files:**
- Modify: `app/persistence/models.py`
- Create: `migrations/versions/20261002_admin_passkeys.py`
- Modify: `tests/test_persistence.py` (table presence assert)
- Test: `tests/test_persistence.py`

**Interfaces:**
- Produces: models `AdminPasskey`, `AdminWebAuthnChallenge`; columns `AdminAccount.webauthn_user_handle: Mapped[bytes | None]`, `AdminSession.enrollment_only: Mapped[bool]`

- [ ] **Step 1: Failing test for new tables**

```python
def test_persistence_schema_defines_passkey_tables():
    assert {"admin_passkeys", "admin_webauthn_challenges"}.issubset(Base.metadata.tables)
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_persistence.py::test_persistence_schema_defines_passkey_tables -v`

- [ ] **Step 3: Add models**

On `AdminAccount` add:

```python
webauthn_user_handle: Mapped[bytes | None] = mapped_column(LargeBinary(64), unique=True, nullable=True)
```

On `AdminSession` add:

```python
enrollment_only: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
```

Add classes (use `LargeBinary`, `Boolean` from SQLAlchemy):

```python
class AdminPasskey(Base):
    __tablename__ = "admin_passkeys"
    __table_args__ = (UniqueConstraint("credential_id", name="uq_admin_passkey_credential"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("admin_accounts.id", ondelete="CASCADE"), nullable=False)
    credential_id: Mapped[bytes] = mapped_column(LargeBinary(1024), nullable=False)
    public_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    sign_count: Mapped[int] = mapped_column(nullable=False, default=0)
    user_handle: Mapped[bytes] = mapped_column(LargeBinary(64), nullable=False)
    transports: Mapped[list[object] | None] = mapped_column(JSON, nullable=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    aaguid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    backed_up: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class AdminWebAuthnChallenge(Base):
    __tablename__ = "admin_webauthn_challenges"
    __table_args__ = (
        CheckConstraint("purpose IN ('registration', 'authentication')", name="ck_webauthn_purpose"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("admin_accounts.id", ondelete="CASCADE"), nullable=True)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    challenge: Mapped[bytes] = mapped_column(LargeBinary(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 4: Alembic revision `20261002_admin_passkeys`** creating columns/tables; downgrade drops them.

- [ ] **Step 5: Run persistence tests**

Run: `pytest tests/test_persistence.py -q` — Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/persistence/models.py migrations/versions/20261002_admin_passkeys.py tests/test_persistence.py
git commit -m "feat: add admin passkey persistence schema"
```

---

### Task 2: Passkey persistence helpers

**Files:**
- Create: `app/persistence/passkeys.py`
- Modify: `app/persistence/admin_auth.py` (`AdminPrincipal.enrollment_only`, `create_admin_session(..., enrollment_only=False)`)
- Test: `tests/test_admin_passkeys.py`

**Interfaces:**
- Produces:
  - `ensure_user_handle(session, account) -> bytes`
  - `account_has_passkeys(session, account_id) -> bool`
  - `store_challenge(session, *, purpose, challenge, account_id=None, ttl_seconds=120) -> str`
  - `consume_challenge(session, challenge_id, purpose) -> bytes`
  - `get_passkey_by_credential_id(session, credential_id: bytes) -> AdminPasskey | None`
  - `insert_passkey(...) -> AdminPasskey`
  - `list_passkeys(session, account_id) -> list[AdminPasskey]`
  - `delete_passkey(session, account_id, passkey_id) -> None` raises `ValueError` if last
  - `create_admin_session(..., enrollment_only: bool = False) -> AdminPrincipal`
  - `AdminPrincipal.enrollment_only: bool`

- [ ] **Step 1: Write failing tests** for challenge consume-once, last-passkey delete refusal, enrollment_only on session.

- [ ] **Step 2: Implement helpers + session flag**

- [ ] **Step 3: `pytest tests/test_admin_passkeys.py -k "challenge or delete or enrollment" -v` — PASS**

- [ ] **Step 4: Commit** `feat: add passkey persistence helpers`

---

### Task 3: WebAuthn service + settings

**Files:**
- Create: `app/admin/webauthn_service.py`
- Modify: `app/settings.py`, `tests/settings.py`, `.env.example`, `requirements/prod.txt`
- Modify: `app/app.py` if startup validation of `WEBAUTHN_*` needed when database extension present
- Test: `tests/test_admin_passkeys.py`

**Interfaces:**
- Produces:
  - `WebAuthnConfig(rp_id: str, rp_name: str, origins: tuple[str, ...])`
  - `webauthn_config_from_app(app) -> WebAuthnConfig`
  - `begin_registration(config, *, user_id: bytes, user_name: str, exclude_credential_ids: list[bytes]) -> tuple[str, dict]`
  - `complete_registration(config, *, challenge: bytes, credential_json: str) -> VerifiedRegistration`
  - `begin_authentication(config) -> tuple[str, dict]`  # returns (raw_challenge, options_json_dict)
  - `complete_authentication(config, *, challenge: bytes, credential_json: str, credential_public_key: bytes, credential_current_sign_count: int, credential_id: bytes) -> VerifiedAuthentication`

Use `webauthn==3.0.1`:
- `generate_registration_options`, `verify_registration_response`
- `generate_authentication_options`, `verify_authentication_response`
- `options_to_json` / parse helpers
- `ResidentKeyRequirement.REQUIRED`, `UserVerificationRequirement.REQUIRED`

Test settings:

```python
WEBAUTHN_RP_ID = "localhost"
WEBAUTHN_RP_NAME = "Test Proxy"
WEBAUTHN_ORIGINS = ("http://localhost", "http://127.0.0.1")
```

- [ ] **Step 1: Add dependency and config parsing tests**

- [ ] **Step 2: Implement service with monkeypatchable verify functions for unit tests**

- [ ] **Step 3: Commit** `feat: add webauthn service and WEBAUTHN settings`

---

### Task 4: Routes, enrollment gate, password policy

**Files:**
- Modify: `app/admin/views.py`
- Modify: `app/persistence/admin_auth.py` (`load_admin_principal` returns enrollment_only)
- Test: `tests/test_admin_passkeys.py`, update `tests/test_admin_auth.py` as needed

**Interfaces:**
- Produces routes listed in the spec table
- `login_required` redirects enrollment_only sessions (except enroll/register/logout) to `admin.passkeys_enroll`
- Password `POST /admin/login`: if `account_has_passkeys` → generic 401; else session with `enrollment_only=True` → redirect enroll

- [ ] **Step 1: Failing tests** — password rejected when passkey exists; enrollment blocks dashboard; register complete clears enrollment_only (mock verify)

- [ ] **Step 2: Implement routes**

JSON responses for webauthn begin/complete: `{"options": ...}` / `{"ok": true}` / `{"error": "..."}` with CSRF from form/header `X-CSRFToken`.

- [ ] **Step 3: Full admin auth + passkey pytest subset PASS**

- [ ] **Step 4: Commit** `feat: enforce mandatory passkey enrollment and login`

---

### Task 5: Templates and client JS

**Files:**
- Modify: `app/templates/admin/login.html`, `account.html`, `base.html` (nav if needed)
- Create: `app/templates/admin/passkeys_enroll.html`
- Create: `app/static/admin/admin-passkeys.js`
- Modify: `app/static/admin/admin.css` lightly for CTA stacking

**Interfaces:**
- JS functions: `passkeyLogin()`, `passkeyRegister(label)`, base64url helpers for ArrayBuffer ↔ JSON matching py_webauthn expectations

- [ ] **Step 1: Update login** — primary button Mit Passkey anmelden; bootstrap password form secondary

- [ ] **Step 2: Enroll template** — no full nav; register CTA

- [ ] **Step 3: Account passkey list** + add/delete forms

- [ ] **Step 4: Manual smoke with test client asserting HTML contains strings; commit** `feat: add passkey admin UI`

---

### Task 6: Docs, live migration note, regression

**Files:**
- Modify: `DEPLOYMENT.md`, `README.md` (admin/database section)
- Run full: `pytest -k ""` and lint

- [ ] **Step 1: Document** `WEBAUTHN_*`, forced enroll, operator recovery (SQL delete passkeys + note password works again when zero passkeys)

- [ ] **Step 2: Lint + full pytest**

- [ ] **Step 3: Commit** `docs: document admin passkey bootstrap and recovery`

- [ ] **Step 4: On proxy (if deploying):** `flask db upgrade`, set env, recreate flask, enroll passkey for `admin`

---

## Spec coverage checklist

- [x] Mandatory passkey after enroll — Task 4
- [x] Password bootstrap + enrollment gate — Task 4
- [x] Discoverable login — Task 3–4
- [x] Schema + challenges — Task 1–2
- [x] Account manage / last delete refusal — Task 2, 4, 5
- [x] Config origins/RP — Task 3
- [x] Audit events — Task 4
- [x] No recovery CLI — Task 6 docs only
- [x] Tests — Tasks 1–4, 6
