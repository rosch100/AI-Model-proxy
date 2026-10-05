"""Flask blueprint and request routing for the proxy service.

This module defines the application blueprint, configures logging, and
forwards incoming HTTP requests to the configured backend implementation.
"""

import uuid

from flask import Blueprint, current_app, g, jsonify, request

from .auth import current_tenant, is_tenant_auth_mode, require_auth
from .azure.adapter import AzureAdapter
from .codex.adapter import CodexAdapter
from .codex.settings import codex_model_payload
from .common.logging import console, log_inbound_model, log_request
from .common.recording import (
    increment_last_recording,
    init_last_recording,
    record_payload,
)
from .exceptions import ConfigurationError, ServiceConfigurationError
from .providers.model_ids import tenant_catalog_model_ids
from .providers.routing import forward_tenant_route, routed_profiles
from .tenants import (
    AUTH_MODE_TENANT,
    DatabaseTenantRoutingSnapshot,
)

blueprint = Blueprint("blueprint", __name__)


@blueprint.before_app_request
def assign_proxy_request_id() -> None:
    """Assign a server-generated correlation ID to every incoming request."""
    g.proxy_request_id = uuid.uuid4().hex


@blueprint.after_app_request
def expose_proxy_request_id(response):
    """Return the correlation ID for matching a client report to server logs."""
    request_id = getattr(g, "proxy_request_id", None)
    if request_id is not None:
        response.headers["X-Proxy-Request-ID"] = request_id
    return response


ALL_METHODS = [
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "OPTIONS",
    "HEAD",
    "TRACE",
]


# ── Health check ────────────────────────────────────────────────────────────


@blueprint.route("/health", methods=["GET"])
def health():
    """Return a simple health check payload."""
    return jsonify({"status": "ok"})


# ── Proxy catch-all ─────────────────────────────────────────────────────────


def _provider_for_path(path: str) -> tuple[str, str]:
    """Return provider name and provider-local path for an incoming path."""
    clean_path = path.strip("/")
    if clean_path == "azure" or clean_path.startswith("azure/"):
        return "azure", clean_path.removeprefix("azure").strip("/")
    if clean_path == "codex" or clean_path.startswith("codex/"):
        return "codex", clean_path.removeprefix("codex").strip("/")
    return "azure", clean_path


def _ensure_provider_enabled(provider: str) -> None:
    flag = "ENABLE_AZURE" if provider == "azure" else "ENABLE_CODEX"
    if not current_app.config.get(flag, False):
        label = "Azure" if provider == "azure" else "Codex"
        raise ServiceConfigurationError(f"{label} provider is disabled.")


def _ensure_provider_allowed_for_auth(provider: str) -> None:
    """Reject Codex when the caller authenticated with a tenant API key."""
    if provider == "codex" and is_tenant_auth_mode():
        raise ServiceConfigurationError(
            "Codex is not available in AUTH_MODE=tenant. "
            "Tenant API keys may only use configured tenant providers."
        )


def _is_explicit_azure_path(path: str) -> bool:
    """Return whether a request explicitly targets the Azure path prefix."""
    clean_path = path.strip("/")
    return clean_path == "azure" or clean_path.startswith("azure/")


def _azure_model_ids() -> list[str]:
    """Return Cursor-facing Azure model ids for the authenticated principal."""
    tenant = current_tenant()
    if isinstance(tenant, DatabaseTenantRoutingSnapshot):
        profiles = routed_profiles(
            tenant, azure_only=_is_explicit_azure_path(request.path)
        )
        return list(
            tenant_catalog_model_ids(
                profiles, azure_only=_is_explicit_azure_path(request.path)
            )
        )
    if tenant is not None:
        return list(tenant.azure_model_deployments)
    if current_app.config.get("AUTH_MODE") == AUTH_MODE_TENANT:
        raise ServiceConfigurationError(
            "AUTH_MODE=tenant requires an authenticated tenant; "
            "refusing to expose global Azure model deployments."
        )
    return list(current_app.config["AZURE_MODEL_DEPLOYMENTS"])


@blueprint.route("/", defaults={"path": ""}, methods=ALL_METHODS)
@blueprint.route("/<path:path>", methods=ALL_METHODS)
@require_auth
def catch_all(path: str):
    """Forward any request path to the selected backend.

    Logs the incoming request and forwards it to the selected backend
    implementation, returning the backend's response. If forwarding fails,
    returns a 502 JSON error payload.
    """
    # Logging / recording must never crash the actual request
    try:
        if current_app.config.get("LOG_CONTEXT"):
            log_request(request)
        log_inbound_model(request)
        init_last_recording()
        increment_last_recording()
        record_payload(request.get_json(silent=True), "downstream_request")
    except Exception:  # noqa: BLE001 - see comment above
        console.print_exception()
        console.print("[yellow]Logging failed but continuing with request[/yellow]")

    provider, provider_path = _provider_for_path(path)
    _ensure_provider_allowed_for_auth(provider)
    if provider == "codex":
        _ensure_provider_enabled(provider)
        return CodexAdapter().forward(request, provider_path)
    tenant = current_tenant()
    if isinstance(tenant, DatabaseTenantRoutingSnapshot):
        return forward_tenant_route(
            request, tenant, azure_only=_is_explicit_azure_path(path)
        )
    _ensure_provider_enabled("azure")
    return AzureAdapter().forward(request)


# ── Model list ──────────────────────────────────────────────────────────────


@blueprint.route("/models", methods=["GET"])
@blueprint.route("/v1/models", methods=["GET"])
@blueprint.route("/azure/models", methods=["GET"])
@blueprint.route("/azure/v1/models", methods=["GET"])
@blueprint.route("/codex/models", methods=["GET"])
@blueprint.route("/codex/v1/models", methods=["GET"])
@require_auth
def models():
    """Return a list of available models."""
    provider, _ = _provider_for_path(request.path)
    _ensure_provider_allowed_for_auth(provider)
    if provider == "codex":
        _ensure_provider_enabled(provider)
        return jsonify(codex_model_payload())
    tenant = current_tenant()
    if not isinstance(tenant, DatabaseTenantRoutingSnapshot):
        _ensure_provider_enabled(provider)
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "openai",
                }
                for model in _azure_model_ids()
            ],
        }
    )


@blueprint.route("/codex/ready", methods=["GET"])
@require_auth
def codex_ready():
    """Return Codex provider readiness."""
    _ensure_provider_enabled("codex")
    _ensure_provider_allowed_for_auth("codex")
    return CodexAdapter().ready()


@blueprint.errorhandler(ConfigurationError)
def configuration_error(e: ConfigurationError):
    """Return a 400 JSON error payload for ValueError."""
    return e.get_response_content(), 400
