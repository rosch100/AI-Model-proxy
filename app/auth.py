"""Authentication module."""

from functools import wraps
from hmac import compare_digest

from flask import Response, current_app, g, request

from .exceptions import ServiceConfigurationError
from .persistence.database import Database
from .tenants import (
    AUTH_MODE_SINGLE,
    AUTH_MODE_TENANT,
    TENANT_CONFIG_DATABASE,
    DatabaseTenantSnapshot,
    TenantConfig,
    resolve_tenant_for_api_key,
)

_GENERIC_UNAUTHORIZED = (
    "Authentication with the proxy service failed.\n\n"
    "Provide a valid Bearer token in:\n"
    "\tCursor Settings > Models > API Keys > OpenAI API Key\n\n"
    "The value must match a configured proxy API key.\n"
    "If modifying the .env file, restart the service for the changes to apply."
)


def _unauthorized_response() -> Response:
    """Return the same generic HTTP 401 for any failed authentication."""
    return Response(_GENERIC_UNAUTHORIZED, status=401, mimetype="text/plain")


def _bearer_token() -> str | None:
    authorization = request.authorization
    if authorization is None or not authorization.token:
        return None
    token = authorization.token
    if not isinstance(token, str) or not token.strip():
        return None
    return token


def authenticate_request() -> TenantConfig | DatabaseTenantSnapshot | None:
    """Authenticate the request and return the tenant when in tenant mode.

    In single mode returns None after validating SERVICE_API_KEY.
    Raises AuthenticationError on missing or invalid credentials.
    """
    token = _bearer_token()
    if token is None:
        raise AuthenticationError()

    auth_mode = current_app.config.get("AUTH_MODE", AUTH_MODE_SINGLE)
    if auth_mode == AUTH_MODE_TENANT:
        if current_app.config.get("TENANT_CONFIG_SOURCE") == TENANT_CONFIG_DATABASE:
            database = current_app.extensions.get("database")
            if not isinstance(database, Database):
                raise ServiceConfigurationError(
                    "Database tenant mode has no configured database."
                )
            tenant = database.get_proxy_snapshot_by_api_key(token)
        else:
            tenants = current_app.config.get("TENANTS") or ()
            tenant = resolve_tenant_for_api_key(token, tenants)
        if tenant is None:
            raise AuthenticationError()
        return tenant

    service_api_key = current_app.config.get("SERVICE_API_KEY")
    if not isinstance(service_api_key, str) or not compare_digest(
        token.encode("utf-8"), service_api_key.encode("utf-8")
    ):
        raise AuthenticationError()
    return None


class AuthenticationError(Exception):
    """Raised when the request bearer token is missing or invalid."""


def require_auth(func):
    """Require authentication for the given route."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function return Unauthorized if the token is invalid."""
        try:
            tenant = authenticate_request()
        except AuthenticationError:
            return _unauthorized_response()

        g.tenant = tenant
        g.auth_mode = current_app.config.get("AUTH_MODE", AUTH_MODE_SINGLE)
        return func(*args, **kwargs)

    return wrapper


def current_tenant() -> TenantConfig | DatabaseTenantSnapshot | None:
    """Return the authenticated tenant for the current request, if any."""
    return getattr(g, "tenant", None)


def is_tenant_auth_mode() -> bool:
    """Return True when the request was authenticated in tenant mode."""
    return getattr(g, "auth_mode", AUTH_MODE_SINGLE) == AUTH_MODE_TENANT
