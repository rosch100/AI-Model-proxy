"""The app module, containing the app factory function."""

from flask import Flask
from rich.traceback import install as install_rich_traceback
from werkzeug.middleware.proxy_fix import ProxyFix

from . import commands
from .admin.security import admin_secret_bytes, configure_admin_security
from .admin.views import register_admin
from .admin.webauthn_service import webauthn_config_from_app
from .blueprint import blueprint
from .exceptions import ServiceConfigurationError
from .migrations import register_migration_commands
from .persistence.database import Database
from .tenant_commands import register_tenant_commands
from .tenants import (
    AUTH_MODE_SINGLE,
    AUTH_MODE_TENANT,
    TENANT_CONFIG_DATABASE,
    TENANT_CONFIG_ENVIRONMENT,
    parse_auth_mode,
    validate_tenant_startup,
)

_TENANT_CONFIG_SOURCES = frozenset({TENANT_CONFIG_ENVIRONMENT, TENANT_CONFIG_DATABASE})

_PLACEHOLDER_SERVICE_API_KEYS = frozenset({"change-me", "choose-a-local-secret"})


def create_app(config_object="app.settings"):
    """Create application factory, as explained here: http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split(".")[0])
    app.config.from_object(config_object)
    _configure_request_trust(app)
    _normalize_auth_config(app)
    _validate_auth_config(app)
    if (
        app.config["AUTH_MODE"] == AUTH_MODE_TENANT
        and app.config["TENANT_CONFIG_SOURCE"] == TENANT_CONFIG_DATABASE
    ):
        _validate_database_admin_config(app)
        database = Database.from_config(app.config)
        app.extensions["database"] = database
        _register_database_admin(app)
    if "AZURE_MODEL_DEPLOYMENTS" in app.config:
        app.config["AZURE_MODEL_DEPLOYMENTS"] = dict(
            app.config["AZURE_MODEL_DEPLOYMENTS"]
        )
    if "CODEX_MODEL_REWRITES" in app.config:
        app.config["CODEX_MODEL_REWRITES"] = dict(app.config["CODEX_MODEL_REWRITES"])
    if "CODEX_SUPPORTED_MODELS" in app.config:
        app.config["CODEX_SUPPORTED_MODELS"] = tuple(
            app.config["CODEX_SUPPORTED_MODELS"]
        )

    configure_logging(app)
    register_commands(app)
    register_blueprints(app)
    return app


def _configure_request_trust(app: Flask) -> None:
    """Apply host validation and trust forwarded headers only when configured."""
    trusted_hosts = app.config.get("TRUSTED_HOSTS")
    if trusted_hosts is not None:
        if not isinstance(trusted_hosts, (list, tuple)) or any(
            not isinstance(host, str) for host in trusted_hosts
        ):
            raise ServiceConfigurationError(
                "TRUSTED_HOSTS must be a list of host names."
            )
        trusted_hosts = tuple(host.strip() for host in trusted_hosts if host.strip())
        if not trusted_hosts:
            raise ServiceConfigurationError("TRUSTED_HOSTS cannot be empty.")
        app.config["TRUSTED_HOSTS"] = trusted_hosts

    if not app.config.get("TRUST_PROXY_HEADERS", False):
        return
    if trusted_hosts is None:
        raise ServiceConfigurationError(
            "TRUSTED_HOSTS must be configured when TRUST_PROXY_HEADERS is enabled."
        )

    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=1,
        x_proto=1,
        x_host=1,
        x_port=0,
        x_prefix=0,
    )


def _normalize_auth_config(app: Flask) -> None:
    """Normalize auth-related config values loaded from the settings object."""
    auth_mode = parse_auth_mode(app.config.get("AUTH_MODE", AUTH_MODE_SINGLE))
    app.config["AUTH_MODE"] = auth_mode
    source = app.config.get("TENANT_CONFIG_SOURCE", TENANT_CONFIG_ENVIRONMENT)
    if source not in _TENANT_CONFIG_SOURCES:
        raise ServiceConfigurationError(
            "TENANT_CONFIG_SOURCE must be either 'environment' or 'database'."
        )
    app.config["TENANT_CONFIG_SOURCE"] = source
    tenants = app.config.get("TENANTS") or ()
    if not isinstance(tenants, tuple):
        app.config["TENANTS"] = tuple(tenants)
    else:
        app.config["TENANTS"] = tenants


def _validate_auth_config(app: Flask) -> None:
    """Validate AUTH_MODE and the credentials required for that mode."""
    auth_mode = app.config["AUTH_MODE"]
    tenants = app.config.get("TENANTS") or ()
    service_api_key = app.config.get("SERVICE_API_KEY")
    config_source = app.config["TENANT_CONFIG_SOURCE"]

    if auth_mode == AUTH_MODE_TENANT and config_source == TENANT_CONFIG_DATABASE:
        if tenants or app.config.get("TENANTS_CONFIGURED", False):
            raise ServiceConfigurationError(
                "TENANTS cannot be set when TENANT_CONFIG_SOURCE=database."
            )
        if app.config.get("ENABLE_CODEX", False):
            raise ServiceConfigurationError(
                "AUTH_MODE=tenant cannot enable Codex. "
                "Codex still uses a shared auth.json login; set ENABLE_CODEX=false."
            )
        return

    validate_tenant_startup(
        auth_mode=auth_mode,
        tenants=tenants,
    )

    if auth_mode == AUTH_MODE_SINGLE:
        _validate_service_api_key(service_api_key)
        return

    if app.config.get("ENABLE_CODEX", False):
        raise ServiceConfigurationError(
            "AUTH_MODE=tenant cannot enable Codex. "
            "Codex still uses a shared auth.json login; set ENABLE_CODEX=false."
        )


def _validate_service_api_key(service_api_key: object) -> None:
    """Reject missing or example service keys before starting the proxy."""
    if (
        not isinstance(service_api_key, str)
        or not service_api_key.strip()
        or service_api_key.strip().casefold() in _PLACEHOLDER_SERVICE_API_KEYS
    ):
        raise ServiceConfigurationError(
            "SERVICE_API_KEY must be set to a non-empty secret, not an example value."
        )


def _validate_database_admin_config(app: Flask) -> None:
    """Require admin session and WebAuthn settings in database tenant mode."""
    try:
        admin_secret_bytes(app.config.get("ADMIN_SESSION_SECRET"))
        webauthn_config_from_app(app)
    except ValueError as exc:
        raise ServiceConfigurationError(str(exc)) from exc


def _register_database_admin(app: Flask) -> None:
    """Bind CSRF/cookie policy and register the admin blueprint."""
    configure_admin_security(app)
    register_admin(app)


def register_blueprints(app):
    """Register Flask blueprints."""
    app.register_blueprint(blueprint)
    return None


def register_commands(app):
    """Register Click commands."""
    app.cli.add_command(commands.test)
    app.cli.add_command(commands.lint)
    register_migration_commands(app)
    register_tenant_commands(app)


def configure_logging(app):
    """Configure logging."""
    install_rich_traceback()
