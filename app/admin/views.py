"""Server-rendered tenant administrator routes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Callable

from cryptography.exceptions import InvalidTag
from flask import (
    Blueprint,
    Flask,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from sqlalchemy import func, or_, select
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
    OpenAIProjectLookupForm,
    OpenAIScopeForm,
    OpenRouterScopeForm,
    PasswordChangeForm,
    ProviderProfileForm,
    ReorderProviderForm,
)
from app.admin.security import (
    ADMIN_COOKIE_NAME,
    ADMIN_COOKIE_PATH,
    GENERIC_LOGIN_ERROR,
    SAVED_SECRET_MASK,
    admin_cookie_secure,
    csrf,
    prefers_html,
)
from app.admin.view_models import (
    ConnectionView,
    CostsView,
    activity_board,
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
    bind_provider_cost_scope,
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
from app.persistence.inference_activity import (
    ACTIVITY_LOOKBACK_HOURS,
    ACTIVITY_LOOKBACK_OPTIONS,
)
from app.persistence.models import (
    AdminAccount,
    AuditEvent,
    CostRefreshJob,
    CostUsageRecord,
    InferenceActivityEvent,
    ProviderAttemptEvent,
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
from app.persistence.provider_circuit_breaker import (
    CircuitSnapshot,
    ProviderCircuitBreakerStore,
)
from app.providers.azure_url import validate_azure_base_url
from app.providers.catalog import (
    CatalogRefreshError,
    azure_deployments_from_catalog,
    refresh_provider_catalog_with_pricing,
    selectable_catalog_models,
)
from app.providers.cost_jobs import (
    collect_provider_costs,
    persist_cost_refresh,
    start_cost_refresh,
)
from app.providers.costs import CostRefreshError
from app.providers.deepseek_balance import (
    DeepSeekBalanceError,
    fetch_deepseek_balance,
)
from app.providers.openai_admin import OpenAIProjectLookupError, list_openai_projects

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


def _activity_period_hours() -> int:
    """Validate the requested activity period against the visible UI options."""
    raw_period = request.args.get("activity_hours")
    if raw_period is None:
        return ACTIVITY_LOOKBACK_HOURS
    try:
        period = int(raw_period)
    except ValueError:
        abort(400)
    if period not in {hours for hours, _label in ACTIVITY_LOOKBACK_OPTIONS}:
        abort(400)
    return period


def _provider_circuit_scopes_by_profile(
    database: Database, tenant_id: str, profiles: tuple[ProviderProfile, ...]
) -> dict[str, tuple[CircuitSnapshot, ...]]:
    """Load only tenant-owned profile and configured shared circuit scopes."""
    store = ProviderCircuitBreakerStore(database.sessions, database.secret_cipher)
    profile_scopes = {
        profile.id: store.scopes_for_profile(
            tenant_id,
            profile.provider,
            profile.id,
            profile.settings,
        )
        for profile in profiles
    }
    all_scopes = tuple(scope for scopes in profile_scopes.values() for scope in scopes)
    snapshots = store.snapshots(tenant_id, all_scopes)
    snapshots_by_identity = {
        (
            snapshot.scope.provider,
            snapshot.scope.scope_type,
            snapshot.scope.fingerprint,
        ): snapshot
        for snapshot in snapshots
    }
    return {
        profile_id: tuple(
            snapshots_by_identity[identity]
            for scope in scopes
            if (identity := (scope.provider, scope.scope_type, scope.fingerprint))
            in snapshots_by_identity
        )
        for profile_id, scopes in profile_scopes.items()
    }


def _latest_provider_activity_events(session, tenant_id: str):
    """Load one last-request record per profile for the provider status fragment."""
    ranked_events = (
        select(
            InferenceActivityEvent.id.label("event_id"),
            func.row_number()
            .over(
                partition_by=InferenceActivityEvent.profile_id,
                order_by=(
                    InferenceActivityEvent.occurred_at.desc(),
                    InferenceActivityEvent.id.desc(),
                ),
            )
            .label("event_rank"),
        )
        .where(
            InferenceActivityEvent.tenant_id == tenant_id,
            InferenceActivityEvent.profile_id.is_not(None),
            InferenceActivityEvent.occurred_at
            >= datetime.now(timezone.utc) - timedelta(hours=ACTIVITY_LOOKBACK_HOURS),
        )
        .subquery()
    )
    statement = (
        select(InferenceActivityEvent)
        .join(ranked_events, InferenceActivityEvent.id == ranked_events.c.event_id)
        .where(ranked_events.c.event_rank == 1)
        .order_by(InferenceActivityEvent.profile_id)
    )
    return tuple(session.scalars(statement))


def _provider_attempt_events(session, tenant_id: str):
    """Load recent status history and pending attempts within telemetry retention."""
    now = datetime.now(timezone.utc)
    status_start = now - timedelta(minutes=15)
    pending_start = now - timedelta(hours=24)
    statement = (
        select(ProviderAttemptEvent)
        .where(
            ProviderAttemptEvent.tenant_id == tenant_id,
            or_(
                ProviderAttemptEvent.occurred_at >= status_start,
                ProviderAttemptEvent.completed_at >= status_start,
                (
                    (ProviderAttemptEvent.outcome == "pending")
                    & (ProviderAttemptEvent.occurred_at >= pending_start)
                ),
            ),
        )
        .order_by(
            ProviderAttemptEvent.occurred_at.asc(),
            ProviderAttemptEvent.id.asc(),
        )
    )
    return tuple(session.scalars(statement))


def _failed_provider_attempts(
    session, tenant_id: str, lookback_hours: int
) -> tuple[tuple[ProviderAttemptEvent, str | None], ...]:
    """Load recent failed upstream attempts with safe profile display names."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    statement = (
        select(ProviderAttemptEvent, ProviderProfile.display_name)
        .join(ProviderProfile, ProviderProfile.id == ProviderAttemptEvent.profile_id)
        .where(
            ProviderAttemptEvent.tenant_id == tenant_id,
            ProviderAttemptEvent.outcome == "failure",
            func.coalesce(
                ProviderAttemptEvent.completed_at, ProviderAttemptEvent.occurred_at
            )
            >= cutoff,
        )
        .order_by(
            func.coalesce(
                ProviderAttemptEvent.completed_at, ProviderAttemptEvent.occurred_at
            ).desc(),
            ProviderAttemptEvent.id.desc(),
        )
        .limit(50)
    )
    return tuple(session.execute(statement).all())


def _catalog_model_ids_by_profile(
    session, profiles: tuple[ProviderProfile, ...]
) -> dict[str, tuple[str, ...]]:
    """Load each profile's selectable model IDs for readiness status."""
    profile_by_id = {profile.id: profile for profile in profiles}
    if not profile_by_id:
        return {}
    rows = session.execute(
        select(
            ProviderCatalogEntry.profile_id,
            ProviderCatalogEntry.model_id,
            ProviderCatalogEntry.deployment_id,
        )
        .where(ProviderCatalogEntry.profile_id.in_(profile_by_id))
        .order_by(ProviderCatalogEntry.model_id)
    )
    entries_by_profile: dict[str, list[tuple[str, str | None]]] = {}
    for profile_id, model_id, deployment_id in rows:
        entries_by_profile.setdefault(profile_id, []).append((model_id, deployment_id))
    return {
        profile_id: tuple(
            model_id
            for model_id, _deployment_id in selectable_catalog_models(
                profile_by_id[profile_id].provider,
                entries_by_profile.get(profile_id, ()),
            )
        )
        for profile_id in profile_by_id
    }


def _activity_events(
    session, tenant_id: str, lookback_hours: int, *, limit: int | None = None
):
    """Load recent tenant activity, optionally limiting the display list."""
    lookback_start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    statement = (
        select(InferenceActivityEvent)
        .where(
            InferenceActivityEvent.tenant_id == tenant_id,
            InferenceActivityEvent.occurred_at >= lookback_start,
        )
        .order_by(
            InferenceActivityEvent.occurred_at.desc(),
            InferenceActivityEvent.id.desc(),
        )
    )
    if limit is not None:
        statement = statement.limit(limit)
    return tuple(session.scalars(statement))


@admin_bp.get("/")
@login_required
def dashboard():
    """Show provider status and the latest successful cost snapshot per account."""
    activity_hours = _activity_period_hours()
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
        catalog_model_ids_by_profile = _catalog_model_ids_by_profile(session, profiles)
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
                    ProviderProfile.tenant_id == tenant.id,
                    ProviderProfile.deleted_at.is_(None),
                    ProviderScopeBinding.purpose == "billing",
                )
                .order_by(ProviderProfile.provider, ProviderProfile.display_name)
            ).all()
        )
        binding_ids = tuple(binding.id for _profile, binding, _node in bindings)
        ranked_jobs = (
            select(
                CostRefreshJob.id.label("job_id"),
                func.row_number()
                .over(
                    partition_by=CostRefreshJob.binding_id,
                    order_by=(
                        CostRefreshJob.created_at.desc(),
                        CostRefreshJob.id.desc(),
                    ),
                )
                .label("job_rank"),
            )
            .where(
                CostRefreshJob.tenant_id == tenant.id,
                CostRefreshJob.binding_id.in_(binding_ids),
            )
            .subquery()
        )
        ranked_successes = (
            select(
                CostRefreshJob.id.label("job_id"),
                func.row_number()
                .over(
                    partition_by=CostRefreshJob.binding_id,
                    order_by=(
                        CostRefreshJob.created_at.desc(),
                        CostRefreshJob.id.desc(),
                    ),
                )
                .label("success_rank"),
            )
            .where(
                CostRefreshJob.tenant_id == tenant.id,
                CostRefreshJob.binding_id.in_(binding_ids),
                CostRefreshJob.status == "success",
            )
            .subquery()
        )
        latest_attempts = tuple(
            session.scalars(
                select(CostRefreshJob)
                .join(ranked_jobs, ranked_jobs.c.job_id == CostRefreshJob.id)
                .where(ranked_jobs.c.job_rank == 1)
            )
        )
        latest_successes = tuple(
            session.scalars(
                select(CostRefreshJob)
                .join(ranked_successes, ranked_successes.c.job_id == CostRefreshJob.id)
                .where(ranked_successes.c.success_rank == 1)
            )
        )
        jobs = tuple(
            sorted(
                {job.id: job for job in latest_attempts + latest_successes}.values(),
                key=lambda job: (job.created_at, job.id),
                reverse=True,
            )
        )
        successful_job_ids = tuple(job.id for job in latest_successes)
        records = tuple(
            session.scalars(
                select(CostUsageRecord)
                .where(
                    CostUsageRecord.tenant_id == tenant.id,
                    CostUsageRecord.job_id.in_(successful_job_ids),
                )
                .order_by(CostUsageRecord.id)
            )
        )
        activity_events = _activity_events(session, tenant.id, ACTIVITY_LOOKBACK_HOURS)
        request_events = _activity_events(
            session, tenant.id, activity_hours, limit=1000
        )
        failed_attempts = _failed_provider_attempts(session, tenant.id, activity_hours)
        provider_attempts = _provider_attempt_events(session, tenant.id)
        view = dashboard_view(
            tenant,
            profiles,
            bindings,
            jobs,
            records,
            activity_events,
            circuit_scopes_by_profile=_provider_circuit_scopes_by_profile(
                database, tenant.id, profiles
            ),
            request_events=request_events,
            request_period_hours=activity_hours,
            provider_attempts=provider_attempts,
            failed_attempts=failed_attempts,
            catalog_model_ids_by_profile=catalog_model_ids_by_profile,
        )
        session.expunge_all()
    return render_template(
        "admin/dashboard.html",
        view=view,
        logout_form=LoginForm(),
        activity_period_options=ACTIVITY_LOOKBACK_OPTIONS,
    )


@admin_bp.get("/provider-status")
@login_required
def dashboard_provider_status():
    """Return the live provider profile status fragment for HTMX polling."""
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
        catalog_model_ids_by_profile = _catalog_model_ids_by_profile(session, profiles)
        activity_events = _latest_provider_activity_events(session, tenant.id)
        provider_attempts = _provider_attempt_events(session, tenant.id)
        circuit_scopes_by_profile = _provider_circuit_scopes_by_profile(
            database, tenant.id, profiles
        )
        view = dashboard_view(
            tenant,
            profiles,
            activity_events=activity_events,
            provider_attempts=provider_attempts,
            circuit_scopes_by_profile=circuit_scopes_by_profile,
            include_activity_board=False,
            catalog_model_ids_by_profile=catalog_model_ids_by_profile,
        )
        session.expunge_all()
    return render_template("admin/_provider_status.html", view=view)


@admin_bp.get("/activity")
@login_required
def dashboard_activity():
    """Return the live activity fragment for HTMX polling."""
    activity_hours = _activity_period_hours()
    database = _database()
    with database.sessions() as session:
        tenant = session.get(Tenant, g.admin.tenant_id)
        activity_events = _activity_events(session, tenant.id, ACTIVITY_LOOKBACK_HOURS)
        request_events = _activity_events(
            session, tenant.id, activity_hours, limit=1000
        )
        failed_attempts = _failed_provider_attempts(session, tenant.id, activity_hours)
        activity = activity_board(
            activity_events,
            custom_model_id=tenant.custom_model_id,
            request_events=request_events,
            request_period_hours=activity_hours,
            failed_attempts=failed_attempts,
        )
        session.expunge_all()
    return render_template(
        "admin/_activity.html",
        activity=activity,
        activity_period_options=ACTIVITY_LOOKBACK_OPTIONS,
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
        ("deepseek", "DeepSeek"),
    ]
    form.default_model.choices = [("", "Nach dem Abruf der Modellliste auswählen")]
    provider = request.args.get("provider", "")
    if provider in {"azure", "openai", "openrouter", "deepseek"}:
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
    _set_profile_secret_display(form, profile)
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
        form.api_key.data = ""
        flash("Angaben zum Konto fehlen noch.", "error")
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
        form.api_key.data = ""
        flash(
            "Das Konto konnte nicht angelegt werden. Prüfe den Namen und die Angaben.",
            "error",
        )
        return (
            render_template(
                "admin/settings/profile_form.html",
                form=form,
                profile=None,
                logout_form=LoginForm(),
            ),
            400,
        )
    flash(
        "Konto gespeichert. Es ist noch nicht in der Standardkaskade; "
        "nach dem Laden der Modellliste kann es direkt per Modellkennung "
        "verwendet oder der Kaskade hinzugefügt werden.",
        "info",
    )
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
        _set_profile_secret_display(form, profile)
        flash("Die Angaben zum Konto sind ungültig.", "error")
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
                (
                    form.api_key.data
                    if form.api_key.data and form.api_key.data != SAVED_SECRET_MASK
                    else None
                ),
                g.admin.username,
            )
    except (LookupError, ValueError, IntegrityError):
        _set_profile_secret_display(form, profile)
        flash(
            "Das Konto konnte nicht gespeichert werden. Prüfe den Namen und die Angaben.",
            "error",
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
    flash("Konto aktualisiert.", "info")
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
            "Das Konto kann nicht entfernt werden, solange noch Abrechnungsdaten "
            "damit verknüpft sind.",
            "error",
        )
        return redirect(url_for("admin.settings_connection"))
    flash("Konto entfernt.", "info")
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
    except ValueError:
        flash(
            "Das Konto konnte nicht aktiviert werden. Prüfe den Schlüssel, das "
            "Standardmodell und die Anbietereinstellungen.",
            "error",
        )
        return redirect(url_for("admin.settings_connection"))
    flash("Konto zur Standardkaskade hinzugefügt.", "info")
    return redirect(url_for("admin.settings_connection"))


@admin_bp.post("/settings/connection/deactivate")
@login_required
def deactivate_connection():
    """Clear the default route while preserving direct catalog-model routing."""
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
        "Die Standardkaskade ist leer. Konten bleiben gespeichert und können "
        "weiterhin direkt über ihre Modellkennung angesprochen werden.",
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
    flash(
        "Konto aus der Standardkaskade entfernt. Es bleibt gespeichert und "
        "über seine Modellkennung direkt nutzbar.",
        "info",
    )
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
    flash("Reihenfolge der Konten aktualisiert.", "info")
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
            flash(
                "Für den Abruf der Modellliste fehlt ein gespeicherter Schlüssel.",
                "error",
            )
            return redirect(url_for("admin.settings_connection"))
        provider = profile.provider
        inference_secret_ciphertext = profile.inference_secret_ciphertext
        secret = database.secret_cipher.decrypt(inference_secret_ciphertext)
        settings = dict(profile.settings)
        profile.catalog_generation += 1
        catalog_generation = profile.catalog_generation
    try:
        entries, pricing = refresh_provider_catalog_with_pricing(
            provider, settings, secret
        )
        error = None
    except CatalogRefreshError as exc:
        entries = []
        pricing = {}
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
                "Die Kontodaten wurden geändert, während die Modellliste abgerufen "
                "wurde. Bitte rufe die Modellliste erneut ab.",
                "warning",
            )
            return redirect(url_for("admin.settings_connection"))
        was_routed = profile.route_priority is not None
        replace_catalog_entries(session, profile, entries, error, pricing)
        if error is None and provider == "azure":
            deployments = azure_deployments_from_catalog(entries)
            profile.settings = {**profile.settings, "model_deployments": deployments}
            if profile.default_model not in deployments:
                profile.default_model = (
                    None if was_routed else next(iter(deployments), None)
                )
        elif error is None and provider in {"openai", "openrouter", "deepseek"}:
            selectable = selectable_catalog_models(provider, entries)
            model_ids = [model_id for model_id, _ in selectable]
            if profile.default_model not in model_ids:
                profile.default_model = (
                    None if was_routed or not model_ids else model_ids[0]
                )
    flash(
        (
            f"Modellliste aktualisiert: "
            f"{len(selectable_catalog_models(provider, entries))} Modelle verfügbar."
            if error is None
            else "Die Modellliste konnte nicht abgerufen werden. Prüfe den Schlüssel und versuche es erneut."
        ),
        "info" if error is None else "error",
    )
    return redirect(url_for("admin.settings_connection"))


@admin_bp.get("/settings/costs")
@login_required
def settings_costs():
    """Show bound billing scopes and cost records."""
    return render_template("admin/settings/costs.html", **_costs_context())


@admin_bp.post("/settings/costs/deepseek-balance/<profile_id>")
@login_required
def fetch_deepseek_profile_balance(profile_id: str):
    """Render an explicit live balance lookup without storing billing history."""
    database = _database()
    with database.sessions() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.id == profile_id,
                ProviderProfile.tenant_id == g.admin.tenant_id,
                ProviderProfile.provider == "deepseek",
                ProviderProfile.deleted_at.is_(None),
            )
        )
        if profile is None:
            return "Not Found", 404
        encrypted_key = profile.inference_secret_ciphertext
        if encrypted_key is None:
            return "Provider account is incomplete", 400
        inference_key = database.secret_cipher.decrypt(encrypted_key)
    try:
        balance = fetch_deepseek_balance(inference_key)
    except DeepSeekBalanceError as exc:
        return (
            render_template(
                "admin/settings/costs.html",
                **_costs_context(
                    deepseek_balance_profile_id=profile_id,
                    deepseek_balance_error=str(exc),
                ),
            ),
            502,
        )
    return render_template(
        "admin/settings/costs.html",
        **_costs_context(
            deepseek_balance_profile_id=profile_id,
            deepseek_balance=balance,
        ),
    )


@admin_bp.post("/settings/costs/azure-scope")
@login_required
def save_azure_cost_scopes():
    """Bind tenant-confirmed Azure billing and usage scopes to one profile."""
    form = AzureScopeForm()
    if not form.validate_on_submit():
        flash("Die Azure-Angaben sind ungültig oder unvollständig.", "error")
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
        flash(f"Die Azure-Ressourcen konnten nicht zugeordnet werden: {exc}", "error")
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
    flash("Die Azure-Ressourcen wurden zugeordnet.", "info")
    return redirect(url_for("admin.settings_costs"))


@admin_bp.post("/settings/costs/billing")
@login_required
def save_billing():
    """Store new billing credentials without replacing an existing masked key."""
    form = BillingCredentialsForm()
    if not form.validate_on_submit():
        flash("Die Angaben zur Abrechnung sind ungültig.", "error")
        return redirect(url_for("admin.settings_costs"))
    database = _database()
    secret = form.billing_secret.data
    if secret == SAVED_SECRET_MASK:
        secret = ""
    try:
        with database.sessions.begin() as session:
            profile = session.scalar(
                select(ProviderProfile).where(
                    ProviderProfile.tenant_id == g.admin.tenant_id,
                    ProviderProfile.id == (form.profile_id.data or ""),
                    ProviderProfile.deleted_at.is_(None),
                )
            )
            if profile is None:
                raise LookupError("Provider account was not found")
            if secret:
                save_billing_secret(
                    session,
                    database.secret_cipher,
                    g.admin.tenant_id,
                    profile.id,
                    secret,
                    g.admin.username,
                )
            elif not profile.billing_secret_ciphertext:
                flash("Der Schlüssel für die Abrechnung fehlt.", "error")
                return redirect(url_for("admin.settings_costs"))
    except (LookupError, ValueError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.settings_costs"))
    flash(
        (
            "Schlüssel für die Abrechnung gespeichert."
            if secret
            else "Gespeicherten Schlüssel für die Abrechnung beibehalten."
        ),
        "info",
    )
    return redirect(url_for("admin.settings_costs"))


@admin_bp.post("/settings/costs/openai-projects")
@login_required
def load_openai_projects():
    """Load OpenAI projects server-side using the stored Admin API key."""
    form = OpenAIProjectLookupForm()
    if not form.validate_on_submit():
        flash(
            "OpenAI-Projektliste konnte nicht geladen werden: ungültige Anfrage.",
            "error",
        )
        return redirect(url_for("admin.settings_costs"))
    database = _database()
    with database.sessions() as session:
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == g.admin.tenant_id,
                ProviderProfile.id == (form.profile_id.data or ""),
                ProviderProfile.provider == "openai",
                ProviderProfile.deleted_at.is_(None),
            )
        )
        if profile is None:
            flash("Das OpenAI-Konto wurde nicht gefunden.", "error")
            return redirect(url_for("admin.settings_costs"))
        if not profile.billing_secret_ciphertext:
            flash(
                "Speichere zuerst den OpenAI-Administratorschlüssel für die Abrechnung.",
                "error",
            )
            return redirect(url_for("admin.settings_costs"))
        try:
            api_key = database.secret_cipher.decrypt(profile.billing_secret_ciphertext)
        except (InvalidTag, ValueError, UnicodeDecodeError):
            flash(
                "Der gespeicherte OpenAI-Abrechnungsschlüssel ist ungültig. Speichere ihn erneut.",
                "error",
            )
            return redirect(url_for("admin.settings_costs"))
        profile_id = profile.id
    organization_id = form.organization_id.data or ""
    try:
        projects = list_openai_projects(api_key, organization_id)
    except OpenAIProjectLookupError as exc:
        flash(str(exc), "error")
        return (
            render_template(
                "admin/settings/costs.html",
                **_costs_context(
                    openai_scope_form_profile_id=profile_id,
                    openai_organization_id=organization_id,
                ),
            ),
            400,
        )
    if not projects:
        flash("Die OpenAI-Organisation enthält keine aktiven Projekte.", "error")
        return (
            render_template(
                "admin/settings/costs.html",
                **_costs_context(
                    openai_scope_form_profile_id=profile_id,
                    openai_organization_id=organization_id,
                ),
            ),
            400,
        )
    return render_template(
        "admin/settings/costs.html",
        **_costs_context(
            openai_projects={profile_id: projects},
            openai_scope_form_profile_id=profile_id,
            openai_organization_id=form.organization_id.data or "",
        ),
    )


@admin_bp.post("/settings/costs/openai-scope")
@login_required
def save_openai_cost_scope():
    """Bind an OpenAI organization and project from the tenant admin page."""
    form = OpenAIScopeForm()
    if not form.validate_on_submit():
        flash("Die OpenAI-Angaben sind ungültig oder unvollständig.", "error")
        return redirect(url_for("admin.settings_costs"))
    try:
        with _database().sessions.begin() as session:
            bind_provider_cost_scope(
                session,
                g.admin.tenant_id,
                form.profile_id.data or "",
                "openai",
                {
                    "organization": form.organization_id.data or "",
                    "project": form.project_id.data or "",
                },
                g.admin.username,
            )
    except (LookupError, ValueError, IntegrityError) as exc:
        flash(
            f"Die OpenAI-Abrechnungsdaten konnten nicht gespeichert werden: {exc}",
            "error",
        )
        return redirect(url_for("admin.settings_costs"))
    flash("Die OpenAI-Abrechnungsdaten wurden gespeichert.", "info")
    return redirect(url_for("admin.settings_costs"))


@admin_bp.post("/settings/costs/openrouter-scope")
@login_required
def save_openrouter_cost_scope():
    """Bind an OpenRouter workspace from the tenant admin page."""
    form = OpenRouterScopeForm()
    if not form.validate_on_submit():
        flash("Die OpenRouter-Angaben sind ungültig oder unvollständig.", "error")
        return redirect(url_for("admin.settings_costs"))
    try:
        with _database().sessions.begin() as session:
            bind_provider_cost_scope(
                session,
                g.admin.tenant_id,
                form.profile_id.data or "",
                "openrouter",
                {"workspace": form.workspace_id.data or ""},
                g.admin.username,
            )
    except (LookupError, ValueError, IntegrityError) as exc:
        flash(
            f"Die OpenRouter-Abrechnungsdaten konnten nicht gespeichert werden: {exc}",
            "error",
        )
        return redirect(url_for("admin.settings_costs"))
    flash("Der OpenRouter-Workspace wurde gespeichert.", "info")
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


def _set_profile_secret_display(
    form: ProviderProfileForm, profile: ProviderProfile
) -> None:
    """Render an existing inference secret only as a non-secret sentinel."""
    form.api_key.data = SAVED_SECRET_MASK if profile.inference_secret_ciphertext else ""


def _set_profile_form_choices(
    form: ProviderProfileForm, provider: str | None = None
) -> None:
    """Keep the submitted provider fixed on edit and render only supported choices."""
    providers = [
        ("azure", "Azure"),
        ("openai", "OpenAI"),
        ("openrouter", "OpenRouter"),
        ("deepseek", "DeepSeek"),
    ]
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
        resume_streams=profile.settings.get("resume_streams") is True,
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
        return {
            "base_url": validate_azure_base_url(form.base_url.data),
            "resume_streams": bool(form.resume_streams.data),
        }
    if provider == "openai":
        return {
            "organization": form.organization.data or "",
            "project": form.project.data or "",
        }
    if provider in {"openrouter", "deepseek"}:
        return {}
    raise ValueError("Unsupported provider")


def _connection_context() -> dict[str, object]:
    database = _database()
    provider_names = ("azure", "openai", "openrouter", "deepseek")
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
    openai_projects: dict[str, list[tuple[str, str]]] | None = None,
    openai_scope_form_profile_id: str | None = None,
    openai_organization_id: str = "",
    deepseek_balance_profile_id: str | None = None,
    deepseek_balance: object | None = None,
    deepseek_balance_error: str | None = None,
) -> dict[str, object]:
    database = _database()
    openai_projects = openai_projects or {}
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
                SAVED_SECRET_MASK
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
        openai_organization_ids = {}
        for profile, binding, node in bindings:
            if (
                binding.provider == "openai"
                and binding.purpose == "billing"
                and node.parent_node_id is not None
            ):
                organization_node = session.get(ProviderScopeNode, node.parent_node_id)
                if organization_node is not None:
                    openai_organization_ids[profile.id] = (
                        organization_node.canonical_scope_id
                    )
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
        scope_bound_profile_ids = frozenset(
            binding.profile_id for _profile, binding, _node in bindings
        )
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
            "billing_forms": {
                profile.id: BillingCredentialsForm(
                    profile_id=profile.id,
                    billing_secret=(
                        SAVED_SECRET_MASK if billing_key_masks[profile.id] else ""
                    ),
                )
                for profile in profiles
                if profile.provider in {"openai", "openrouter"}
            },
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
            "openai_scope_forms": {
                profile.id: OpenAIScopeForm(
                    profile_id=profile.id,
                    organization_id=(
                        openai_organization_id
                        if profile.id == openai_scope_form_profile_id
                        else str(profile.settings.get("organization", ""))
                    ),
                    project_id=str(profile.settings.get("project", "")),
                )
                for profile in profiles
                if profile.provider == "openai"
                and profile.id not in scope_bound_profile_ids
            },
            "openrouter_scope_forms": {
                profile.id: OpenRouterScopeForm(profile_id=profile.id)
                for profile in profiles
                if profile.provider == "openrouter"
                and profile.id not in scope_bound_profile_ids
            },
            "scope_bound_profile_ids": scope_bound_profile_ids,
            "openai_organization_ids": openai_organization_ids,
            "openai_projects": openai_projects,
            "deepseek_balance_profile_id": deepseek_balance_profile_id,
            "deepseek_balance": deepseek_balance,
            "deepseek_balance_error": deepseek_balance_error,
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
