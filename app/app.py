"""The app module, containing the app factory function."""

from flask import Flask
from rich.traceback import install as install_rich_traceback

from . import commands
from .blueprint import blueprint
from .exceptions import ServiceConfigurationError
from .tenants import AUTH_MODE_SINGLE, parse_auth_mode, validate_tenant_startup

_PLACEHOLDER_SERVICE_API_KEYS = frozenset({"change-me", "choose-a-local-secret"})


def create_app(config_object="app.settings"):
    """Create application factory, as explained here: http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split(".")[0])
    app.config.from_object(config_object)
    _normalize_auth_config(app)
    _validate_auth_config(app)
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


def _normalize_auth_config(app: Flask) -> None:
    """Normalize auth-related config values loaded from the settings object."""
    auth_mode = parse_auth_mode(app.config.get("AUTH_MODE", AUTH_MODE_SINGLE))
    app.config["AUTH_MODE"] = auth_mode
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


def register_blueprints(app):
    """Register Flask blueprints."""
    app.register_blueprint(blueprint)
    return None


def register_commands(app):
    """Register Click commands."""
    app.cli.add_command(commands.test)
    app.cli.add_command(commands.lint)


def configure_logging(app):
    """Configure logging."""
    install_rich_traceback()
