"""Server-rendered tenant administrator routes."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Callable

from flask import (
    Blueprint,
    Flask,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from webauthn.helpers import parse_authentication_credential_json
from webauthn.helpers.exceptions import (
    InvalidAuthenticationResponse,
    InvalidJSONStructure,
    InvalidRegistrationResponse,
    WebAuthnException,
)

from app.admin.forms import (
    ActivateProviderForm,
    AzureScopeForm,
    BillingCredentialsForm,
    DeactivateProviderForm,
    DeleteProviderForm,
    LoginForm,
    PasswordChangeForm,
    ProviderProfileForm,
    ReorderProviderForm,
)
from app.admin.security import (
    ADMIN_COOKIE_NAME,
    ADMIN_COOKIE_PATH,
    GENERIC_LOGIN_ERROR,
    admin_cookie_secure,
    csrf,
    mask_secret,
    prefers_html,
)
from app.admin.view_models import (
    ConnectionView,
    CostsView,
    dashboard_view,
)
from app.admin.webauthn_service import (
    begin_authentication,
    begin_registration,
    complete_authentication,
    complete_registration,
    webauthn_config_from_app,
)
from app.persistence.admin_auth import (
    authenticate_admin,
    clear_enrollment_only,
    clear_login_failures,
    create_admin_session,
    is_login_locked,
    load_admin_principal,
    login_subject_hash,
    record_login_failure,
    revoke_admin_session,
)
from app.persistence.admin_ops import (
    activate_provider_profile,
    bind_azure_cost_scopes,
    change_admin_password,
    create_provider_profile,
    deactivate_provider_profile,
    delete_provider_profile,
    reorder_provider_profile,
    replace_catalog_entries,
    rotate_api_key,
    save_billing_secret,
    update_provider_profile,
)
from app.persistence.database import Database
from app.persistence.models import (
    AdminAccount,
    AuditEvent,
    CostRefreshJob,
    CostUsageRecord,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from app.persistence.passkeys import (
    account_has_passkeys,
    consume_challenge,
    delete_passkey,
    ensure_user_handle,
    get_passkey_by_credential_id,
    insert_passkey,
    list_passkeys,
    store_challenge,
)
from app.providers.azure_url import validate_azure_base_url
from app.providers.catalog import (
    CatalogRefreshError,
    azure_deployments_from_catalog,
    refresh_provider_catalog,
    selectable_catalog_models,
)
from app.providers.cost_jobs import (
    collect_provider_costs,
    persist_cost_refresh,
    start_cost_refresh,
)
from app.providers.costs import CostRefreshError

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

_ENROLLMENT_ALLOWED_ENDPOINTS = frozenset(
    {
        "admin.passkeys_enroll",
        "admin.webauthn_register_begin",
        "admin.webauthn_register_complete",
        "admin.logout",
    }
)


def _database() -> Database:
    database = current_app.extensions.get("database")
    if not isinstance(database, Database):
        raise RuntimeError("Admin routes require a configured database.")
    return database


def _set_session_cookie(response, token: str):
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        token,
        httponly=True,
        samesite="Lax",
        secure=admin_cookie_secure(current_app),
        path=ADMIN_COOKIE_PATH,
    )
    return response


def _clear_session_cookie(response):
    response.delete_cookie(ADMIN_COOKIE_NAME, path=ADMIN_COOKIE_PATH)
    return response


def _webauthn_config():
    return webauthn_config_from_app(current_app)


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _request_json() -> dict[str, object]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def login_required(view: Callable):
    """Redirect anonymous browsers to the login page."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        token = request.cookies.get(ADMIN_COOKIE_NAME)
        database = _database()
        with database.sessions() as session:
            principal = load_admin_principal(session, token)
            needs_enrollment = False
            if principal is not None:
                needs_enrollment = (
                    principal.enrollment_only
                    or not account_has_passkeys(session, principal.account_id)
                )
            session.commit()
        if principal is None:
            return redirect(url_for("admin.login"))
        g.admin = principal
        if needs_enrollment and request.endpoint not in _ENROLLMENT_ALLOWED_ENDPOINTS:
            return redirect(url_for("admin.passkeys_enroll"))
        return view(*args, **kwargs)

    return wrapped


@admin_bp.before_request
def _protect_admin_csrf() -> None:
    csrf.protect()


@admin_bp.get("/login")
def login():
    """Render the administrator login form."""
    form = LoginForm()
    return render_template("admin/login.html", form=form)


@admin_bp.post("/login")
def login_submit():
    """Authenticate an administrator and issue a rotated session cookie."""
    form = LoginForm()
    if not form.validate_on_submit():
        return render_template("admin/login.html", form=form), 400
    username = form.username.data or ""
    password = form.password.data or ""
    subject = login_subject_hash(username, request.remote_addr or "")
    database = _database()
    with database.sessions.begin() as session:
        if is_login_locked(session, subject):
            flash(GENERIC_LOGIN_ERROR, "error")
            return render_template("admin/login.html", form=form), 429
        account = authenticate_admin(session, username, password)
        if account is None:
            record_login_failure(session, subject)
            flash(GENERIC_LOGIN_ERROR, "error")
            return render_template("admin/login.html", form=form), 401
        if account_has_passkeys(session, account.id):
            record_login_failure(session, subject)
            flash(GENERIC_LOGIN_ERROR, "error")
            return render_template("admin/login.html", form=form), 401
        clear_login_failures(session, subject)
        principal = create_admin_session(session, account, enrollment_only=True)
    response = redirect(url_for("admin.passkeys_enroll"))
    return _set_session_cookie(response, principal.token)


@admin_bp.post("/webauthn/login/begin")
def webauthn_login_begin():
    """Start a discoverable passkey authentication ceremony."""
    database = _database()
    config = _webauthn_config()
    challenge, options = begin_authentication(config)
    with database.sessions.begin() as session:
        challenge_id = store_challenge(
            session, purpose="authentication", challenge=challenge
        )
    return jsonify({"challenge_id": challenge_id, "options": options})


@admin_bp.post("/webauthn/login/complete")
def webauthn_login_complete():
    """Finish passkey authentication and issue a full admin session."""
    payload = _request_json()
    challenge_id = payload.get("challenge_id")
    credential = payload.get("credential")
    if not isinstance(challenge_id, str) or credential is None:
        return _json_error("Ungültige Passkey-Antwort.")
    credential_json = (
        credential if isinstance(credential, str) else json.dumps(credential)
    )
    database = _database()
    config = _webauthn_config()
    subject = login_subject_hash("passkey", request.remote_addr or "")
    try:
        parsed = parse_authentication_credential_json(credential_json)
    except (InvalidJSONStructure, WebAuthnException, TypeError, ValueError, KeyError):
        return _json_error(GENERIC_LOGIN_ERROR, 401)
    with database.sessions.begin() as session:
        if is_login_locked(session, subject):
            return _json_error(GENERIC_LOGIN_ERROR, 429)
        audit_tenant_id: str | None = None
        audit_actor = "anonymous"
        try:
            challenge = consume_challenge(session, challenge_id, "authentication")
            passkey = get_passkey_by_credential_id(session, parsed.raw_id)
            if passkey is None:
                raise LookupError("unknown credential")
            account = session.get(AdminAccount, passkey.account_id)
            if account is None:
                raise LookupError("unknown account")
            audit_tenant_id = account.tenant_id
            audit_actor = account.username
            verified = complete_authentication(
                config,
                challenge=challenge,
                credential_json=credential_json,
                credential_public_key=passkey.public_key,
                credential_current_sign_count=passkey.sign_count,
            )
            if passkey.sign_count > 0 and verified.new_sign_count < passkey.sign_count:
                raise InvalidAuthenticationResponse("sign count decreased")
            passkey.sign_count = verified.new_sign_count
            passkey.last_used_at = datetime.now(timezone.utc)
            clear_login_failures(session, subject)
            principal = create_admin_session(session, account, enrollment_only=False)
            session.add(
                AuditEvent(
                    tenant_id=account.tenant_id,
                    actor_id=account.username,
                    target=f"passkey:{passkey.id}",
                    action="passkey.login",
                    outcome="success",
                    details={},
                )
            )
        except (
            LookupError,
            InvalidAuthenticationResponse,
            WebAuthnException,
            TypeError,
            ValueError,
            KeyError,
        ):
            record_login_failure(session, subject)
            if audit_tenant_id is not None:
                session.add(
                    AuditEvent(
                        tenant_id=audit_tenant_id,
                        actor_id=audit_actor,
                        target="passkey",
                        action="passkey.login_failed",
                        outcome="failure",
                        details={},
                    )
                )
            return _json_error(GENERIC_LOGIN_ERROR, 401)
    response = jsonify({"ok": True, "redirect": url_for("admin.dashboard")})
    return _set_session_cookie(response, principal.token)


@admin_bp.post("/logout")
@login_required
def logout():
    """Revoke the current session and clear the admin cookie."""
    database = _database()
    with database.sessions.begin() as session:
        revoke_admin_session(session, g.admin.session_id)
    response = redirect(url_for("admin.login"))
    return _clear_session_cookie(response)


@admin_bp.get("/passkeys/enroll")
@login_required
def passkeys_enroll():
    """Force first-passkey enrollment before the rest of the admin UI."""
    database = _database()
    with database.sessions() as session:
        enrolled = account_has_passkeys(session, g.admin.account_id)
    if enrolled:
        return redirect(url_for("admin.dashboard"))
    return render_template(
        "admin/passkeys_enroll.html",
        logout_form=LoginForm(),
        username=g.admin.username,
    )


@admin_bp.post("/webauthn/register/begin")
@login_required
def webauthn_register_begin():
    """Start passkey registration for the authenticated administrator."""
    database = _database()
    config = _webauthn_config()
    with database.sessions.begin() as session:
        account = session.get(AdminAccount, g.admin.account_id)
        user_handle = ensure_user_handle(session, account)
        exclude = [key.credential_id for key in list_passkeys(session, account.id)]
        challenge, options = begin_registration(
            config,
            user_id=user_handle,
            user_name=account.username,
            exclude_credential_ids=exclude,
        )
        challenge_id = store_challenge(
            session,
            purpose="registration",
            challenge=challenge,
            account_id=account.id,
        )
    return jsonify({"challenge_id": challenge_id, "options": options})


@admin_bp.post("/webauthn/register/complete")
@login_required
def webauthn_register_complete():
    """Persist a verified passkey and clear enrollment-only sessions."""
    payload = _request_json()
    challenge_id = payload.get("challenge_id")
    credential = payload.get("credential")
    label = payload.get("label")
    if not isinstance(challenge_id, str) or credential is None:
        return _json_error("Ungültige Passkey-Antwort.")
    credential_json = (
        credential if isinstance(credential, str) else json.dumps(credential)
    )
    label_text = label.strip() if isinstance(label, str) else "Passkey"
    database = _database()
    config = _webauthn_config()
    already_enrolled = False
    try:
        with database.sessions.begin() as session:
            account = session.get(AdminAccount, g.admin.account_id)
            already_enrolled = account_has_passkeys(session, account.id)
            challenge = consume_challenge(
                session,
                challenge_id,
                "registration",
                expected_account_id=account.id,
            )
            verified = complete_registration(
                config, challenge=challenge, credential_json=credential_json
            )
            user_handle = ensure_user_handle(session, account)
            passkey = insert_passkey(
                session,
                account_id=account.id,
                credential_id=verified.credential_id,
                public_key=verified.credential_public_key,
                sign_count=verified.sign_count,
                user_handle=user_handle,
                label=label_text or "Passkey",
                aaguid=verified.aaguid,
                backed_up=verified.credential_backed_up,
            )
            clear_enrollment_only(session, g.admin.session_id)
            session.add(
                AuditEvent(
                    tenant_id=account.tenant_id,
                    actor_id=account.username,
                    target=f"passkey:{passkey.id}",
                    action="passkey.register",
                    outcome="success",
                    details={"label": passkey.label},
                )
            )
    except IntegrityError:
        return _json_error(
            "Dieser Passkey ist bereits registriert. Nutze ein anderes Gerät "
            "oder einen Security Key."
        )
    except (
        LookupError,
        InvalidRegistrationResponse,
        WebAuthnException,
        TypeError,
        ValueError,
        KeyError,
    ):
        return _json_error("Passkey-Registrierung fehlgeschlagen.")
    redirect_to = (
        url_for("admin.account") if already_enrolled else url_for("admin.dashboard")
    )
    return jsonify({"ok": True, "redirect": redirect_to})


@admin_bp.get("/")
@login_required
def dashboard():
    """Show provider status and the latest verified cost records."""
    tenant, profiles = _load_tenant_profiles()
    dashboard = dashboard_view(tenant, profiles)
    database = _database()
    with database.sessions() as session:
        counts = dict(
            session.execute(
                select(CostUsageRecord.kind, func.count(CostUsageRecord.id))
                .where(CostUsageRecord.tenant_id == g.admin.tenant_id)
                .group_by(CostUsageRecord.kind)
            ).all()
        )
        records = tuple(
            session.scalars(
                select(CostUsageRecord)
                .where(CostUsageRecord.tenant_id == g.admin.tenant_id)
                .order_by(
                    CostUsageRecord.bucket_start.desc(), CostUsageRecord.id.desc()
                )
                .limit(10)
            )
        )
        record_bindings = {
            binding.id: (profile.display_name or profile.provider)
            for profile, binding in session.execute(
                select(ProviderProfile, ProviderScopeBinding)
                .join(
                    ProviderScopeBinding,
                    (ProviderScopeBinding.profile_id == ProviderProfile.id)
                    & (ProviderScopeBinding.tenant_id == ProviderProfile.tenant_id),
                )
                .where(
                    ProviderProfile.tenant_id == g.admin.tenant_id,
                    ProviderProfile.deleted_at.is_(None),
                )
            )
        }
    dashboard = replace(
        dashboard,
        cost_record_counts={
            kind: counts.get(kind, 0) for kind in ("actual", "usage", "estimate")
        },
        cost_records=records,
        cost_record_profile_names={
            record.binding_id: record_bindings.get(record.binding_id, record.provider)
            for record in records
        },
    )
    return render_template(
        "admin/dashboard.html",
        view=dashboard,
        logout_form=LoginForm(),
    )


@admin_bp.get("/settings/general")
@login_required
def settings_general():
    """Show the stable custom model id and API-key rotation control."""
    tenant, _profiles = _load_tenant_profiles()
    return render_template(
        "admin/settings/general.html",
        tenant=tenant,
        issued_api_key=None,
        logout_form=LoginForm(),
    )


@admin_bp.post("/settings/general/rotate-key")
@login_required
def rotate_key():
    """Rotate the Cursor API key and show the plaintext once."""
    database = _database()
    with database.sessions.begin() as session:
        tenant = session.get(Tenant, g.admin.tenant_id)
        issued = rotate_api_key(session, tenant, g.admin.username)
    flash("Neuer API-Schlüssel erzeugt. Er wird nur einmal angezeigt.", "info")
    return render_template(
        "admin/settings/general.html",
        tenant=tenant,
        issued_api_key=issued,
        logout_form=LoginForm(),
    )


@admin_bp.get("/settings/connection")
@login_required
def settings_connection():
    """Render provider accounts grouped by provider."""
    return render_template(
        "admin/settings/connection.html",
        **_connection_context(),
    )


@admin_bp.get("/settings/connection/create")
@login_required
def create_connection_form():
    """Choose a provider, then show its account-specific configuration."""
    form = ProviderProfileForm()
    form.provider.choices = [
        ("", "Anbieter wählen"),
        ("azure", "Azure"),
        ("openai", "OpenAI"),
        ("openrouter", "OpenRouter"),
    ]
    form.default_model.choices = [("", "Nach dem Katalogabruf auswählen")]
    provider = request.args.get("provider", "")
    if provider in {"azure", "openai", "openrouter"}:
        form.provider.data = provider
    return render_template(
        "admin/settings/profile_form.html",
        form=form,
        profile=None,
        logout_form=LoginForm(),
    )


@admin_bp.get("/settings/connection/<profile_id>/edit")
@login_required
def edit_connection_form(profile_id: str):
    """Render an edit form only for an account owned by this tenant."""
    database = _database()
    with database.sessions() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.id == profile_id,
                ProviderProfile.tenant_id == g.admin.tenant_id,
                ProviderProfile.deleted_at.is_(None),
            )
        )
        if profile is None:
            return "Not Found", 404
        catalog_rows = tuple(
            session.scalars(
                select(ProviderCatalogEntry)
                .where(ProviderCatalogEntry.profile_id == profile.id)
                .order_by(ProviderCatalogEntry.model_id)
            )
        )
        session.expunge(profile)
    form = _profile_form(profile, catalog_rows)
    return render_template(
        "admin/settings/profile_form.html",
        form=form,
        profile=profile,
        logout_form=LoginForm(),
    )


@admin_bp.post("/settings/connection/create")
@login_required
def create_connection():
    """Create a named, inactive profile with provider-specific settings."""
    form = ProviderProfileForm()
    _set_profile_form_choices(form)
    if not form.validate_on_submit():
        flash("Accountdaten sind unvollständig.", "error")
        return (
            render_template(
                "admin/settings/profile_form.html",
                form=form,
                profile=None,
                logout_form=LoginForm(),
            ),
            400,
        )
    provider = form.provider.data or ""
    settings = _provider_settings(form)
    try:
        with _database().sessions.begin() as session:
            create_provider_profile(
                session,
                _database().secret_cipher,
                g.admin.tenant_id,
                provider,
                form.display_name.data or "",
                settings,
                form.default_model.data or "",
                form.api_key.data or None,
                g.admin.username,
            )
    except (LookupError, ValueError, IntegrityError):
        flash("Account konnte nicht angelegt werden. Name und Angaben prüfen.", "error")
        return (
            render_template(
                "admin/settings/profile_form.html",
                form=form,
                profile=None,
                logout_form=LoginForm(),
            ),
            400,
        )
    flash("Account gespeichert. Er ist noch nicht aktiv.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/<profile_id>/edit")
@login_required
def update_connection(profile_id: str):
    """Update account settings without changing provider identity."""
    database = _database()
    with database.sessions() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.id == profile_id,
                ProviderProfile.tenant_id == g.admin.tenant_id,
                ProviderProfile.deleted_at.is_(None),
            )
        )
        if profile is None:
            return "Not Found", 404
        catalog_rows = tuple(
            session.scalars(
                select(ProviderCatalogEntry)
                .where(ProviderCatalogEntry.profile_id == profile.id)
                .order_by(ProviderCatalogEntry.model_id)
            )
        )
        session.expunge(profile)
    form = ProviderProfileForm()
    _set_profile_form_choices(form, provider=profile.provider)
    _set_default_model_choices(form, profile, catalog_rows)
    if not form.validate_on_submit() or form.provider.data != profile.provider:
        flash("Accountdaten sind ungültig.", "error")
        return (
            render_template(
                "admin/settings/profile_form.html",
                form=form,
                profile=profile,
                logout_form=LoginForm(),
            ),
            400,
        )
    try:
        with database.sessions.begin() as session:
            update_provider_profile(
                session,
                database.secret_cipher,
                g.admin.tenant_id,
                profile_id,
                form.display_name.data or "",
                _provider_settings(form),
                form.default_model.data or "",
                form.api_key.data or None,
                g.admin.username,
            )
    except (LookupError, ValueError, IntegrityError):
        flash(
            "Account konnte nicht gespeichert werden. Name und Angaben prüfen.", "error"
        )
        return (
            render_template(
                "admin/settings/profile_form.html",
                form=form,
                profile=profile,
                logout_form=LoginForm(),
            ),
            400,
        )
    flash("Account aktualisiert.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/<profile_id>/delete")
@login_required
def delete_connection(profile_id: str):
    """Soft-delete one provider account owned by the current tenant."""
    form = DeleteProviderForm()
    if not form.validate_on_submit() or form.profile_id.data != profile_id:
        return "Bad Request", 400
    try:
        with _database().sessions.begin() as session:
            delete_provider_profile(
                session, g.admin.tenant_id, profile_id, g.admin.username
            )
    except LookupError:
        return "Not Found", 404
    except ValueError:
        flash(
            "Account kann nicht entfernt werden, solange Provider-Scopes gebunden sind.",
            "error",
        )
        return redirect(url_for("admin.settings_connection"))
    flash("Account entfernt.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/<profile_id>/activate")
@login_required
def activate_connection(profile_id: str):
    """Activate one tenant-owned account by its immutable profile ID."""
    form = ActivateProviderForm()
    if not form.validate_on_submit() or form.profile_id.data != profile_id:
        return "Bad Request", 400
    database = _database()
    try:
        with database.sessions.begin() as session:
            tenant = session.scalar(
                select(Tenant).where(Tenant.id == g.admin.tenant_id).with_for_update()
            )
            activate_provider_profile(session, tenant, profile_id, g.admin.username)
    except LookupError:
        return "Not Found", 404
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.settings_connection"))
    flash("Account aktiviert.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/deactivate")
@login_required
def deactivate_connection():
    """Explicitly disable proxy forwarding by clearing the active profile."""
    form = DeactivateProviderForm()
    if not form.validate_on_submit():
        return "Bad Request", 400
    database = _database()
    with database.sessions.begin() as session:
        tenant = session.scalar(
            select(Tenant).where(Tenant.id == g.admin.tenant_id).with_for_update()
        )
        profile_ids = tuple(
            session.scalars(
                select(ProviderProfile.id)
                .where(
                    ProviderProfile.tenant_id == tenant.id,
                    ProviderProfile.route_priority.is_not(None),
                    ProviderProfile.deleted_at.is_(None),
                )
                .order_by(ProviderProfile.route_priority)
            )
        )
        for profile_id in profile_ids:
            deactivate_provider_profile(session, tenant, profile_id, g.admin.username)
    flash(
        "Kein Account aktiv. Proxy-Anfragen sind bis zur nächsten Aktivierung nicht verfügbar.",
        "warning",
    )
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/<profile_id>/deactivate")
@login_required
def deactivate_connection_profile(profile_id: str):
    """Remove only this account from the tenant failover order."""
    form = ActivateProviderForm()
    if not form.validate_on_submit() or form.profile_id.data != profile_id:
        return "Bad Request", 400
    with _database().sessions.begin() as session:
        tenant = session.get(Tenant, g.admin.tenant_id)
        try:
            deactivate_provider_profile(session, tenant, profile_id, g.admin.username)
        except LookupError:
            return "Not Found", 404
    flash("Account aus der Failover-Reihenfolge entfernt.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/<profile_id>/move")
@login_required
def move_connection(profile_id: str):
    """Move one tenant-owned account without accepting arbitrary positions."""
    form = ReorderProviderForm()
    if not form.validate_on_submit() or form.profile_id.data != profile_id:
        return "Bad Request", 400
    try:
        with _database().sessions.begin() as session:
            tenant = session.get(Tenant, g.admin.tenant_id)
            reorder_provider_profile(
                session, tenant, profile_id, form.direction.data, g.admin.username
            )
    except LookupError:
        return "Not Found", 404
    except ValueError:
        return "Bad Request", 400
    flash("Failover-Reihenfolge aktualisiert.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/<profile_id>/catalog")
@login_required
def refresh_catalog(profile_id: str):
    """Refresh one profile's catalog without holding a DB transaction on I/O."""
    database = _database()
    with database.sessions.begin() as session:
        profile = session.scalar(
            select(ProviderProfile)
            .where(
                ProviderProfile.id == profile_id,
                ProviderProfile.tenant_id == g.admin.tenant_id,
                ProviderProfile.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if profile is None:
            return "Not Found", 404
        if not profile.inference_secret_ciphertext:
            flash("Für den Katalog-Refresh fehlt ein gespeicherter Schlüssel.", "error")
            return redirect(url_for("admin.settings_connection"))
        provider = profile.provider
        inference_secret_ciphertext = profile.inference_secret_ciphertext
        secret = database.secret_cipher.decrypt(inference_secret_ciphertext)
        settings = dict(profile.settings)
        profile.catalog_generation += 1
        catalog_generation = profile.catalog_generation
    try:
        entries = refresh_provider_catalog(provider, settings, secret)
        error = None
    except CatalogRefreshError as exc:
        entries = []
        error = str(exc)
    with database.sessions.begin() as session:
        tenant = session.scalar(
            select(Tenant).where(Tenant.id == g.admin.tenant_id).with_for_update()
        )
        if tenant is None:
            return "Not Found", 404
        profile = session.scalar(
            select(ProviderProfile)
            .where(
                ProviderProfile.id == profile_id,
                ProviderProfile.tenant_id == g.admin.tenant_id,
                ProviderProfile.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if profile is None:
            return "Not Found", 404
        if (
            profile.provider != provider
            or profile.inference_secret_ciphertext != inference_secret_ciphertext
            or dict(profile.settings) != settings
            or profile.catalog_generation != catalog_generation
        ):
            flash(
                "Accountdaten wurden während des Katalog-Refreshs geändert. "
                "Bitte den Katalog erneut aktualisieren.",
                "warning",
            )
            return redirect(url_for("admin.settings_connection"))
        was_routed = profile.route_priority is not None
        replace_catalog_entries(session, profile, entries, error)
        if error is None and provider == "azure":
            deployments = azure_deployments_from_catalog(entries)
            profile.settings = {**profile.settings, "model_deployments": deployments}
            if profile.default_model not in deployments:
                profile.default_model = (
                    None if was_routed else next(iter(deployments), None)
                )
        elif error is None and provider in {"openai", "openrouter"}:
            selectable = selectable_catalog_models(provider, entries)
            model_ids = [model_id for model_id, _ in selectable]
            if profile.default_model not in model_ids:
                profile.default_model = (
                    None if was_routed or not model_ids else model_ids[0]
                )
    flash(
        (
            f"Katalog aktualisiert — {len(selectable_catalog_models(provider, entries))} wählbare Modelle übernommen."
            if error is None
            else error
        ),
        "info" if error is None else "error",
    )
    return redirect(url_for("admin.settings_connection"))


@admin_bp.get("/settings/costs")
@login_required
def settings_costs():
    """Show bound billing scopes and cost records."""
    return render_template("admin/settings/costs.html", **_costs_context())


@admin_bp.post("/settings/costs/azure-scope")
@login_required
def save_azure_cost_scopes():
    """Bind tenant-confirmed Azure billing and usage scopes to one profile."""
    form = AzureScopeForm()
    if not form.validate_on_submit():
        flash("Azure-Scope-Angaben sind ungültig oder unvollständig.", "error")
        return (
            render_template(
                "admin/settings/costs.html",
                **_costs_context(
                    azure_scope_form=form,
                    azure_scope_form_profile_id=form.profile_id.data,
                ),
            ),
            400,
        )
    try:
        with _database().sessions.begin() as session:
            bind_azure_cost_scopes(
                session,
                g.admin.tenant_id,
                form.profile_id.data or "",
                form.subscription_id.data or "",
                form.resource_group_arm_id.data or "",
                form.cognitive_resource_arm_id.data or "",
                g.admin.username,
            )
    except (LookupError, ValueError, IntegrityError) as exc:
        flash(f"Azure-Scopes konnten nicht gebunden werden: {exc}", "error")
        return (
            render_template(
                "admin/settings/costs.html",
                **_costs_context(
                    azure_scope_form=form,
                    azure_scope_form_profile_id=form.profile_id.data,
                ),
            ),
            400,
        )
    flash("Azure-Billing- und Usage-Scope wurden gespeichert.", "info")
    return redirect(url_for("admin.settings_costs"))


@admin_bp.post("/settings/costs/billing")
@login_required
def save_billing():
    """Store billing credentials for one account."""
    form = BillingCredentialsForm()
    if not form.validate_on_submit() or not form.billing_secret.data:
        flash("Billing-Schlüssel fehlt.", "error")
        return redirect(url_for("admin.settings_costs"))
    database = _database()
    try:
        with database.sessions.begin() as session:
            save_billing_secret(
                session,
                database.secret_cipher,
                g.admin.tenant_id,
                form.profile_id.data or "",
                form.billing_secret.data,
                g.admin.username,
            )
    except (LookupError, ValueError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.settings_costs"))
    flash("Billing-Schlüssel gespeichert.", "info")
    return redirect(url_for("admin.settings_costs"))


@admin_bp.post("/settings/costs/refresh")
@login_required
def refresh_costs():
    """Refresh one profile's costs; provider I/O stays outside the write transaction."""
    profile_id = request.form.get("profile_id", "")
    database = _database()
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)
    try:
        with database.sessions.begin() as session:
            profile, binding, node, usage_node, job = start_cost_refresh(
                session, g.admin.tenant_id, profile_id, start, end
            )
            profile_id = profile.id
            binding_id = binding.id
            job_id = job.id
            canonical_scope_id = node.canonical_scope_id
            usage_scope_id = usage_node.canonical_scope_id if usage_node else None
    except LookupError as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.settings_costs"))

    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile_id)
        try:
            buckets = collect_provider_costs(
                database.secret_cipher,
                profile,
                canonical_scope_id,
                start,
                end,
                usage_scope_id=usage_scope_id,
            )
            error = None
        except CostRefreshError as exc:
            buckets = None
            error = exc
    with database.sessions.begin() as session:
        profile = session.get(ProviderProfile, profile_id)
        binding = session.get(ProviderScopeBinding, binding_id)
        job = session.get(CostRefreshJob, job_id)
        persisted_job = persist_cost_refresh(
            session,
            g.admin.tenant_id,
            g.admin.username,
            profile,
            binding,
            job,
            buckets,
            error,
        )
    if persisted_job is None:
        flash(
            "Der Refresh ist abgelaufen; das verspätete Ergebnis wurde verworfen.",
            "error",
        )
    else:
        flash(
            "Kosten aktualisiert." if error is None else str(error),
            "info" if error is None else "error",
        )
    return redirect(url_for("admin.settings_costs"))


@admin_bp.get("/account")
@login_required
def account():
    """Render password change, passkey management, and session end controls."""
    database = _database()
    with database.sessions() as session:
        enrolled = account_has_passkeys(session, g.admin.account_id)
        passkeys = list_passkeys(session, g.admin.account_id) if enrolled else []
        session.expunge_all()
    return render_template(
        "admin/account.html",
        form=PasswordChangeForm(),
        logout_form=LoginForm(),
        username=g.admin.username,
        passkeys=passkeys,
        password_change_allowed=not enrolled,
    )


@admin_bp.post("/account/password")
@login_required
def change_password():
    """Change the administrator password and revoke other sessions."""
    database = _database()
    with database.sessions() as session:
        if account_has_passkeys(session, g.admin.account_id):
            flash("Passwortänderung ist nach Passkey-Einrichtung deaktiviert.", "error")
            return redirect(url_for("admin.account"))
    form = PasswordChangeForm()
    if not form.validate_on_submit():
        return (
            render_template(
                "admin/account.html",
                form=form,
                logout_form=LoginForm(),
                username=g.admin.username,
                passkeys=[],
                password_change_allowed=True,
            ),
            400,
        )
    try:
        with database.sessions.begin() as session:
            account_row = session.get(AdminAccount, g.admin.account_id)
            change_admin_password(
                session,
                account_row,
                form.current_password.data or "",
                form.new_password.data or "",
                g.admin.session_id,
            )
    except ValueError:
        flash("Aktuelles Passwort ist ungültig.", "error")
        return redirect(url_for("admin.account"))
    flash("Passwort geändert. Andere Sitzungen wurden beendet.", "info")
    return redirect(url_for("admin.account"))


@admin_bp.post("/account/passkeys/<passkey_id>/delete")
@login_required
def delete_account_passkey(passkey_id: str):
    """Delete one passkey when at least two remain."""
    database = _database()
    try:
        with database.sessions.begin() as session:
            delete_passkey(session, g.admin.account_id, passkey_id)
            session.add(
                AuditEvent(
                    tenant_id=g.admin.tenant_id,
                    actor_id=g.admin.username,
                    target=f"passkey:{passkey_id}",
                    action="passkey.delete",
                    outcome="success",
                    details={},
                )
            )
    except ValueError:
        flash("Der letzte Passkey kann nicht gelöscht werden.", "error")
        return redirect(url_for("admin.account"))
    except LookupError:
        flash("Passkey wurde nicht gefunden.", "error")
        return redirect(url_for("admin.account"))
    flash("Passkey entfernt.", "info")
    return redirect(url_for("admin.account"))


def _load_tenant_profiles() -> tuple[Tenant, tuple[ProviderProfile, ...]]:
    """Load every active tenant profile without collapsing same-provider accounts."""
    database = _database()
    with database.sessions() as session:
        tenant = session.get(Tenant, g.admin.tenant_id)
        profiles = tuple(
            session.scalars(
                select(ProviderProfile)
                .where(
                    ProviderProfile.tenant_id == tenant.id,
                    ProviderProfile.deleted_at.is_(None),
                )
                .order_by(ProviderProfile.provider, ProviderProfile.display_name)
            )
        )
        session.expunge_all()
        return tenant, profiles


def _set_profile_form_choices(
    form: ProviderProfileForm, provider: str | None = None
) -> None:
    """Keep the submitted provider fixed on edit and render only supported choices."""
    providers = [("azure", "Azure"), ("openai", "OpenAI"), ("openrouter", "OpenRouter")]
    form.provider.choices = (
        [(provider, dict(providers)[provider])]
        if provider in dict(providers)
        else [("", "Anbieter wählen"), *providers]
    )


def _profile_form(
    profile: ProviderProfile,
    catalog_rows: tuple[ProviderCatalogEntry, ...],
) -> ProviderProfileForm:
    """Populate profile fields and catalog-backed model choices."""
    form = ProviderProfileForm(
        provider=profile.provider,
        display_name=profile.display_name or "",
        base_url=str(profile.settings.get("base_url", "")),
        default_model=profile.default_model or "",
        organization=str(profile.settings.get("organization", "")),
        project=str(profile.settings.get("project", "")),
        api_key="",
    )
    _set_default_model_choices(form, profile, catalog_rows)
    return form


def _set_default_model_choices(
    form: ProviderProfileForm,
    profile: ProviderProfile,
    catalog_rows: tuple[ProviderCatalogEntry, ...],
) -> None:
    """Use only this profile's catalog entries in its scrollbox."""
    models = selectable_catalog_models(
        profile.provider,
        [(entry.model_id, entry.deployment_id) for entry in catalog_rows],
    )
    choices = [(model_id, model_id) for model_id, _deployment_id in models]
    saved_model = profile.default_model
    if saved_model and saved_model not in {model_id for model_id, _ in choices}:
        choices.insert(0, (saved_model, f"{saved_model} (nicht im aktuellen Katalog)"))
    form.default_model.choices = [("", "Standardmodell wählen"), *choices]


def _provider_settings(form: ProviderProfileForm) -> dict[str, object]:
    """Extract only fields belonging to the chosen provider."""
    provider = form.provider.data
    if provider == "azure":
        return {"base_url": validate_azure_base_url(form.base_url.data)}
    if provider == "openai":
        return {
            "organization": form.organization.data or "",
            "project": form.project.data or "",
        }
    if provider == "openrouter":
        return {}
    raise ValueError("Unsupported provider")


def _connection_context() -> dict[str, object]:
    database = _database()
    provider_names = ("azure", "openai", "openrouter")
    with database.sessions() as session:
        tenant = session.get(Tenant, g.admin.tenant_id)
        profiles = tuple(
            session.scalars(
                select(ProviderProfile)
                .where(
                    ProviderProfile.tenant_id == tenant.id,
                    ProviderProfile.deleted_at.is_(None),
                )
                .order_by(ProviderProfile.provider, ProviderProfile.display_name)
            )
        )
        profiles_by_provider = {
            provider: tuple(
                profile for profile in profiles if profile.provider == provider
            )
            for provider in provider_names
        }
        profiles_with_bindings = frozenset(
            session.scalars(
                select(ProviderScopeBinding.profile_id).where(
                    ProviderScopeBinding.tenant_id == tenant.id
                )
            )
        )
        catalogs: dict[str, tuple[ProviderCatalogEntry, ...]] = {}
        selectable_models: dict[str, tuple[tuple[str, str | None], ...]] = {}
        selectable_model_ids: dict[str, tuple[str, ...]] = {}
        for profile in profiles:
            rows = tuple(
                session.scalars(
                    select(ProviderCatalogEntry)
                    .where(ProviderCatalogEntry.profile_id == profile.id)
                    .order_by(ProviderCatalogEntry.model_id)
                )
            )
            catalogs[profile.id] = rows
            selectable_models[profile.id] = tuple(
                selectable_catalog_models(
                    profile.provider,
                    [(row.model_id, row.deployment_id) for row in rows],
                )
            )
            selectable_model_ids[profile.id] = tuple(
                model_id for model_id, _deployment_id in selectable_models[profile.id]
            )
        session.expunge_all()
        return {
            "cursor_model_id": tenant.custom_model_id,
            "view": ConnectionView(
                tenant_id=tenant.id,
                profiles_by_provider=profiles_by_provider,
                profiles_with_bindings=profiles_with_bindings,
                catalogs=catalogs,
                selectable_models=selectable_models,
                selectable_model_ids=selectable_model_ids,
                routed_profiles=tuple(
                    sorted(
                        (
                            profile
                            for profile in profiles
                            if profile.route_priority is not None
                        ),
                        key=lambda profile: profile.route_priority,
                    )
                ),
            ),
            "logout_form": LoginForm(),
        }


def _costs_context(
    azure_scope_form: AzureScopeForm | None = None,
    azure_scope_form_profile_id: str | None = None,
) -> dict[str, object]:
    database = _database()
    with database.sessions() as session:
        profiles = tuple(
            session.scalars(
                select(ProviderProfile)
                .where(
                    ProviderProfile.tenant_id == g.admin.tenant_id,
                    ProviderProfile.deleted_at.is_(None),
                )
                .order_by(ProviderProfile.provider, ProviderProfile.display_name)
            )
        )
        billing_key_masks = {
            profile.id: (
                mask_secret(
                    database.secret_cipher.decrypt(profile.billing_secret_ciphertext)
                )
                if profile.provider in {"openai", "openrouter"}
                and profile.billing_secret_ciphertext
                else None
            )
            for profile in profiles
        }
        bindings = tuple(
            session.execute(
                select(ProviderProfile, ProviderScopeBinding, ProviderScopeNode)
                .join(
                    ProviderScopeBinding,
                    (ProviderScopeBinding.profile_id == ProviderProfile.id)
                    & (ProviderScopeBinding.tenant_id == ProviderProfile.tenant_id),
                )
                .join(
                    ProviderScopeNode,
                    ProviderScopeNode.id == ProviderScopeBinding.node_id,
                )
                .where(
                    ProviderProfile.tenant_id == g.admin.tenant_id,
                    ProviderProfile.deleted_at.is_(None),
                    ProviderScopeBinding.purpose.in_(("billing", "usage")),
                )
                .order_by(ProviderProfile.provider, ProviderProfile.display_name)
            ).all()
        )
        jobs = tuple(
            session.scalars(
                select(CostRefreshJob)
                .where(CostRefreshJob.tenant_id == g.admin.tenant_id)
                .order_by(CostRefreshJob.created_at.desc())
            )
        )
        records = tuple(
            session.scalars(
                select(CostUsageRecord)
                .where(CostUsageRecord.tenant_id == g.admin.tenant_id)
                .order_by(CostUsageRecord.id.desc())
            )
        )
        binding_profile_names = {
            binding.id: profile.display_name or profile.provider
            for profile, binding, _node in bindings
        }
        azure_bindings: dict[
            str, dict[str, tuple[ProviderScopeBinding, ProviderScopeNode]]
        ] = {}
        for _profile, binding, node in bindings:
            if binding.provider == "azure":
                azure_bindings.setdefault(binding.profile_id, {})[binding.purpose] = (
                    binding,
                    node,
                )
        azure_costs_ready_profile_ids = frozenset(
            profile_id
            for profile_id, profile_bindings in azure_bindings.items()
            if "billing" in profile_bindings
            and "usage" in profile_bindings
            and profile_bindings["usage"][0].parent_binding_id
            == profile_bindings["billing"][0].id
            and profile_bindings["usage"][1].parent_node_id
            == profile_bindings["billing"][0].node_id
        )
        azure_scope_bound_profile_ids = frozenset(azure_bindings)
        session.expunge_all()
        return {
            "view": CostsView(
                tenant_id=g.admin.tenant_id,
                profiles=profiles,
                bindings=bindings,
                binding_profile_names=binding_profile_names,
                jobs=jobs,
                records=records,
                billing_key_masks=billing_key_masks,
                azure_costs_ready_profile_ids=azure_costs_ready_profile_ids,
                azure_scope_bound_profile_ids=azure_scope_bound_profile_ids,
            ),
            "billing_form": BillingCredentialsForm(),
            "azure_scope_forms": {
                profile.id: (
                    azure_scope_form
                    if profile.id == azure_scope_form_profile_id
                    and azure_scope_form is not None
                    else AzureScopeForm(formdata=None)
                )
                for profile in profiles
                if profile.provider == "azure"
                and profile.id not in azure_scope_bound_profile_ids
            },
            "logout_form": LoginForm(),
        }


def register_admin(app: Flask) -> None:
    """Register the admin blueprint and HTML root alias when a database exists."""
    if app.extensions.get("database") is None:
        return
    app.register_blueprint(admin_bp)

    @app.before_request
    def redirect_browser_root_to_admin():
        if request.path != "/" or request.method != "GET":
            return None
        if not prefers_html():
            return None
        return redirect("/admin")
