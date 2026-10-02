"""Server-rendered tenant administrator routes."""

from __future__ import annotations

import json
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
from sqlalchemy import select
from webauthn.helpers import parse_authentication_credential_json
from webauthn.helpers.exceptions import (
    InvalidAuthenticationResponse,
    InvalidJSONStructure,
    InvalidRegistrationResponse,
    WebAuthnException,
)

from app.admin.forms import (
    ActivateProviderForm,
    AzureConnectionForm,
    BillingCredentialsForm,
    LoginForm,
    OpenAIConnectionForm,
    OpenRouterConnectionForm,
    PasswordChangeForm,
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
from app.admin.view_models import ConnectionView, CostsView, dashboard_view
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
    change_admin_password,
    replace_catalog_entries,
    rotate_api_key,
    save_billing_secret,
    upsert_provider_profile,
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
    with database.sessions.begin() as session:
        account = session.get(AdminAccount, g.admin.account_id)
        try:
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
        except (
            LookupError,
            InvalidRegistrationResponse,
            WebAuthnException,
            TypeError,
            ValueError,
            KeyError,
        ):
            return _json_error("Passkey-Registrierung fehlgeschlagen.")
    return jsonify({"ok": True, "redirect": url_for("admin.dashboard")})


@admin_bp.get("/")
@login_required
def dashboard():
    """Show tenant and provider status from the database."""
    tenant, profiles = _load_tenant_profiles()
    return render_template(
        "admin/dashboard.html",
        view=dashboard_view(tenant, profiles),
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
    """Render provider connection forms and catalog status."""
    return render_template(
        "admin/settings/connection.html",
        **_connection_context(),
    )


@admin_bp.post("/settings/connection/azure")
@login_required
def save_azure_connection():
    """Save Azure inference settings without activating the profile."""
    database = _database()
    with database.sessions() as session:
        catalog_rows = _catalog_tuples(session, g.admin.tenant_id, "azure")
    selectable = selectable_catalog_models("azure", catalog_rows)
    form = AzureConnectionForm()
    form.default_model.choices = _model_choices(selectable)
    if not form.default_model.choices:
        flash("Zuerst den Azure-Katalog aktualisieren.", "error")
        return redirect(url_for("admin.settings_connection"))
    if not form.validate_on_submit():
        flash("Azure-Verbindung ist unvollständig.", "error")
        return redirect(url_for("admin.settings_connection"))
    deployments = azure_deployments_from_catalog(catalog_rows)
    if form.default_model.data not in deployments:
        flash("Standardmodell ist im Azure-Katalog nicht enthalten.", "error")
        return redirect(url_for("admin.settings_connection"))
    _save_profile(
        "azure",
        {
            "base_url": (form.base_url.data or "").rstrip("/"),
            "model_deployments": deployments,
        },
        form.default_model.data or "",
        form.api_key.data,
    )
    flash("Azure-Verbindung gespeichert.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/openai")
@login_required
def save_openai_connection():
    """Save OpenAI inference settings without activating the profile."""
    database = _database()
    with database.sessions() as session:
        catalog_rows = _catalog_tuples(session, g.admin.tenant_id, "openai")
    selectable = selectable_catalog_models("openai", catalog_rows)
    form = OpenAIConnectionForm()
    form.default_model.choices = _model_choices(selectable)
    if not form.default_model.choices:
        flash("Zuerst den OpenAI-Katalog aktualisieren.", "error")
        return redirect(url_for("admin.settings_connection"))
    if not form.validate_on_submit():
        flash("OpenAI-Verbindung ist unvollständig.", "error")
        return redirect(url_for("admin.settings_connection"))
    _save_profile(
        "openai",
        {
            "organization": form.organization.data or "",
            "project": form.project.data or "",
        },
        form.default_model.data or "",
        form.api_key.data,
    )
    flash("OpenAI-Verbindung gespeichert.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/openrouter")
@login_required
def save_openrouter_connection():
    """Save OpenRouter inference settings without activating the profile."""
    database = _database()
    with database.sessions() as session:
        catalog_rows = _catalog_tuples(session, g.admin.tenant_id, "openrouter")
    selectable = selectable_catalog_models("openrouter", catalog_rows)
    form = OpenRouterConnectionForm()
    form.default_model.choices = _model_choices(selectable)
    if not form.default_model.choices:
        flash("Zuerst den OpenRouter-Katalog aktualisieren.", "error")
        return redirect(url_for("admin.settings_connection"))
    if not form.validate_on_submit():
        flash("OpenRouter-Verbindung ist unvollständig.", "error")
        return redirect(url_for("admin.settings_connection"))
    _save_profile(
        "openrouter",
        {},
        form.default_model.data or "",
        form.api_key.data,
    )
    flash("OpenRouter-Verbindung gespeichert.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/activate")
@login_required
def activate_connection():
    """Activate one saved provider profile for proxy traffic."""
    form = ActivateProviderForm()
    if not form.validate_on_submit():
        flash("Aktivierung fehlgeschlagen.", "error")
        return redirect(url_for("admin.settings_connection"))
    database = _database()
    try:
        with database.sessions.begin() as session:
            tenant = session.get(Tenant, g.admin.tenant_id)
            activate_provider_profile(
                session, tenant, form.provider.data or "", g.admin.username
            )
    except (LookupError, ValueError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.settings_connection"))
    flash("Anbieter aktiviert.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/refresh-catalog")
@login_required
def refresh_catalog():
    """Refresh the model catalog for one provider outside the database write."""
    provider = request.form.get("provider", "")
    database = _database()
    with database.sessions() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == g.admin.tenant_id,
                ProviderProfile.provider == provider,
            )
        )
        if profile is None or not profile.inference_secret_ciphertext:
            flash("Für den Katalog-Refresh fehlt ein gespeicherter Schlüssel.", "error")
            return redirect(url_for("admin.settings_connection"))
        secret = database.secret_cipher.decrypt(profile.inference_secret_ciphertext)
        settings = dict(profile.settings)
        profile_id = profile.id
    try:
        entries = refresh_provider_catalog(provider, settings, secret)
        error = None
    except CatalogRefreshError as exc:
        entries = []
        error = str(exc)
    with database.sessions.begin() as session:
        profile = session.get(ProviderProfile, profile_id)
        replace_catalog_entries(session, profile, entries, error)
        if error is None and provider == "azure":
            deployments = azure_deployments_from_catalog(entries)
            updated = dict(profile.settings)
            updated["model_deployments"] = deployments
            profile.settings = updated
            if deployments and (
                not profile.default_model or profile.default_model not in deployments
            ):
                profile.default_model = next(iter(deployments))
        elif error is None and provider in {"openai", "openrouter"}:
            selectable = selectable_catalog_models(provider, entries)
            model_ids = [model_id for model_id, _ in selectable]
            if model_ids and (
                not profile.default_model or profile.default_model not in model_ids
            ):
                profile.default_model = model_ids[0]
    if error is None:
        selectable = selectable_catalog_models(provider, entries)
        flash(
            f"Katalog aktualisiert — {len(selectable)} wählbare Modelle übernommen.",
            "info",
        )
    else:
        flash(error, "error")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.get("/settings/costs")
@login_required
def settings_costs():
    """Show bound billing scopes and cost records."""
    return render_template("admin/settings/costs.html", **_costs_context())


@admin_bp.post("/settings/costs/billing")
@login_required
def save_billing():
    """Store OpenAI or OpenRouter billing credentials."""
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
                form.provider.data or "",
                form.billing_secret.data,
                g.admin.username,
            )
    except LookupError as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.settings_costs"))
    flash("Billing-Schlüssel gespeichert.", "info")
    return redirect(url_for("admin.settings_costs"))


@admin_bp.post("/settings/costs/refresh")
@login_required
def refresh_costs():
    """Refresh provider costs; provider I/O stays outside the write transaction."""
    provider = request.form.get("provider", "")
    database = _database()
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)
    try:
        with database.sessions.begin() as session:
            profile, binding, node, job = start_cost_refresh(
                session, g.admin.tenant_id, provider, start, end
            )
            profile_id = profile.id
            binding_id = binding.id
            job_id = job.id
            canonical_scope_id = node.canonical_scope_id
    except LookupError as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.settings_costs"))

    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile_id)
        try:
            buckets = collect_provider_costs(
                database.secret_cipher, profile, canonical_scope_id, start, end
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


def _load_tenant_profiles() -> tuple[Tenant, dict[str, ProviderProfile]]:
    database = _database()
    with database.sessions() as session:
        tenant = session.get(Tenant, g.admin.tenant_id)
        profiles = {
            profile.provider: profile
            for profile in session.scalars(
                select(ProviderProfile).where(ProviderProfile.tenant_id == tenant.id)
            )
        }
        session.expunge_all()
        return tenant, profiles


def _save_profile(
    provider: str,
    settings: dict[str, object],
    default_model: str,
    inference_secret: str | None,
) -> None:
    database = _database()
    with database.sessions.begin() as session:
        upsert_provider_profile(
            session,
            database.secret_cipher,
            g.admin.tenant_id,
            provider,
            settings,
            default_model,
            inference_secret or None,
            g.admin.username,
        )


def _catalog_tuples(
    session, tenant_id: str, provider: str
) -> list[tuple[str, str | None]]:
    profile = session.scalar(
        select(ProviderProfile).where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.provider == provider,
        )
    )
    if profile is None:
        return []
    rows = session.scalars(
        select(ProviderCatalogEntry)
        .where(ProviderCatalogEntry.profile_id == profile.id)
        .order_by(ProviderCatalogEntry.model_id)
    )
    return [(row.model_id, row.deployment_id) for row in rows]


def _model_choices(
    selectable: list[tuple[str, str | None]],
) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for model_id, deployment_id in selectable:
        if deployment_id and deployment_id != model_id:
            label = f"{model_id} → {deployment_id}"
        else:
            label = model_id
        choices.append((model_id, label))
    return choices


def _connection_context() -> dict[str, object]:
    database = _database()
    with database.sessions() as session:
        tenant = session.get(Tenant, g.admin.tenant_id)
        profiles = {
            profile.provider: profile
            for profile in session.scalars(
                select(ProviderProfile).where(ProviderProfile.tenant_id == tenant.id)
            )
        }
        catalogs: dict[str, tuple[ProviderCatalogEntry, ...]] = {}
        selectable_models: dict[str, tuple[tuple[str, str | None], ...]] = {}
        for name in ("azure", "openai", "openrouter"):
            profile = profiles.get(name)
            if profile is None:
                catalogs[name] = ()
                selectable_models[name] = ()
                continue
            rows = tuple(
                session.scalars(
                    select(ProviderCatalogEntry)
                    .where(ProviderCatalogEntry.profile_id == profile.id)
                    .order_by(ProviderCatalogEntry.model_id)
                )
            )
            catalogs[name] = rows
            selectable_models[name] = tuple(
                selectable_catalog_models(
                    name, [(row.model_id, row.deployment_id) for row in rows]
                )
            )
        azure = profiles.get("azure")
        openai = profiles.get("openai")
        openrouter = profiles.get("openrouter")
        inference_secrets = {
            name: (
                database.secret_cipher.decrypt(profile.inference_secret_ciphertext)
                if profile is not None and profile.inference_secret_ciphertext
                else None
            )
            for name, profile in (
                ("azure", azure),
                ("openai", openai),
                ("openrouter", openrouter),
            )
        }
        azure_form = AzureConnectionForm(
            base_url=(azure.settings.get("base_url") if azure else "") or "",
            api_key=inference_secrets["azure"] or "",
            default_model=azure.default_model if azure else "",
        )
        azure_form.default_model.choices = _model_choices(
            list(selectable_models["azure"])
        ) or [("", "— Katalog aktualisieren —")]
        openai_form = OpenAIConnectionForm(
            api_key=inference_secrets["openai"] or "",
            default_model=openai.default_model if openai else "",
            organization=(openai.settings.get("organization") if openai else "") or "",
            project=(openai.settings.get("project") if openai else "") or "",
        )
        openai_form.default_model.choices = _model_choices(
            list(selectable_models["openai"])
        ) or [("", "— Katalog aktualisieren —")]
        openrouter_form = OpenRouterConnectionForm(
            api_key=inference_secrets["openrouter"] or "",
            default_model=openrouter.default_model if openrouter else "",
        )
        openrouter_form.default_model.choices = _model_choices(
            list(selectable_models["openrouter"])
        ) or [("", "— Katalog aktualisieren —")]
        _set_api_key_labels(azure_form, openai_form, openrouter_form, inference_secrets)
        activate_form = ActivateProviderForm()
        session.expunge_all()
        return {
            "view": ConnectionView(
                tenant_id=tenant.id,
                azure=azure,
                openai=openai,
                openrouter=openrouter,
                catalogs=catalogs,
                selectable_models=selectable_models,
                active_provider=next(
                    (
                        profile.provider
                        for profile in profiles.values()
                        if profile.id == tenant.active_profile_id
                    ),
                    None,
                ),
                inference_secrets=inference_secrets,
            ),
            "azure_form": azure_form,
            "openai_form": openai_form,
            "openrouter_form": openrouter_form,
            "activate_form": activate_form,
            "logout_form": LoginForm(),
        }


def _set_api_key_labels(
    azure_form: AzureConnectionForm,
    openai_form: OpenAIConnectionForm,
    openrouter_form: OpenRouterConnectionForm,
    secrets: dict[str, str | None],
) -> None:
    azure_form.api_key.label.text = (
        "Azure API-Schlüssel (optional — leer lässt den gespeicherten Wert)"
        if secrets["azure"]
        else "Azure API-Schlüssel *"
    )
    openai_form.api_key.label.text = (
        "OpenAI API-Schlüssel (optional — leer lässt den gespeicherten Wert)"
        if secrets["openai"]
        else "OpenAI API-Schlüssel *"
    )
    openrouter_form.api_key.label.text = (
        "OpenRouter API-Schlüssel (optional — leer lässt den gespeicherten Wert)"
        if secrets["openrouter"]
        else "OpenRouter API-Schlüssel *"
    )


def _costs_context() -> dict[str, object]:
    database = _database()
    with database.sessions() as session:
        profiles = {
            profile.provider: profile
            for profile in session.scalars(
                select(ProviderProfile).where(
                    ProviderProfile.tenant_id == g.admin.tenant_id
                )
            )
        }
        billing_key_masks = {
            name: (
                mask_secret(
                    database.secret_cipher.decrypt(profile.billing_secret_ciphertext)
                )
                if profile is not None and profile.billing_secret_ciphertext
                else None
            )
            for name, profile in (
                ("openai", profiles.get("openai")),
                ("openrouter", profiles.get("openrouter")),
            )
        }
        bindings = tuple(
            session.execute(
                select(ProviderScopeBinding, ProviderScopeNode)
                .join(
                    ProviderScopeNode,
                    ProviderScopeNode.id == ProviderScopeBinding.node_id,
                )
                .where(ProviderScopeBinding.tenant_id == g.admin.tenant_id)
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
        session.expunge_all()
        return {
            "view": CostsView(
                tenant_id=g.admin.tenant_id,
                bindings=bindings,
                jobs=jobs,
                records=records,
                billing_key_masks=billing_key_masks,
            ),
            "billing_form": BillingCredentialsForm(),
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
