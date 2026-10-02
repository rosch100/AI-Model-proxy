"""Application configuration."""

from environs import Env

from .codex.settings import (
    parse_codex_model_rewrites,
    parse_codex_supported_models,
)
from .models import parse_model_deployments
from .reasoning_display import (
    DEFAULT_REASONING_DISPLAY_MODE,
    parse_reasoning_display_mode,
)
from .tenants import parse_auth_mode, parse_tenants

env = Env()
env.read_env()

ENV = env.str("FLASK_ENV", default="production")
DEBUG = ENV == "development"
TRUSTED_HOSTS = env.list("TRUSTED_HOSTS", default=None)
TRUST_PROXY_HEADERS = env.bool("TRUST_PROXY_HEADERS", default=False)
RECORD_TRAFFIC = env.bool("RECORD_TRAFFIC", False)
LOG_CONTEXT = env.bool("LOG_CONTEXT", False)
LOG_COMPLETION = env.bool("LOG_COMPLETION", False)
REASONING_DISPLAY_MODE = parse_reasoning_display_mode(
    env.str("REASONING_DISPLAY_MODE", DEFAULT_REASONING_DISPLAY_MODE)
)

AUTH_MODE = parse_auth_mode(env.str("AUTH_MODE", default="single"))
TENANT_CONFIG_SOURCE = env.str("TENANT_CONFIG_SOURCE", default="environment")
DATABASE_URL = env.str("DATABASE_URL", default=None)
PROVIDER_ENCRYPTION_KEY = env.str("PROVIDER_ENCRYPTION_KEY", default=None)
ADMIN_SESSION_SECRET = env.str("ADMIN_SESSION_SECRET", default=None)
WEBAUTHN_RP_ID = env.str("WEBAUTHN_RP_ID", default=None)
WEBAUTHN_RP_NAME = env.str("WEBAUTHN_RP_NAME", default=None)
WEBAUTHN_ORIGINS = env.str("WEBAUTHN_ORIGINS", default=None)
SERVICE_API_KEY = env.str("SERVICE_API_KEY", default=None)
_raw_tenants = env.str("TENANTS", default="")
TENANTS_CONFIGURED = bool(_raw_tenants.strip())
TENANTS = () if TENANT_CONFIG_SOURCE == "database" else parse_tenants(_raw_tenants)

ENABLE_AZURE = env.bool("ENABLE_AZURE", True)
ENABLE_CODEX = env.bool("ENABLE_CODEX", False)

AZURE_BASE_URL = env.str("AZURE_BASE_URL", "change_me").rstrip("/")
AZURE_API_KEY = env.str("AZURE_API_KEY", "change_me")

AZURE_SUMMARY_LEVEL = env.str("AZURE_SUMMARY_LEVEL", default="") or "detailed"
AZURE_VERBOSITY_LEVEL = env.str("AZURE_VERBOSITY_LEVEL", default="") or "medium"
AZURE_TRUNCATION = env.str("AZURE_TRUNCATION", default="") or "disabled"
raw_model_deployments = env.str("AZURE_MODEL_DEPLOYMENTS", default="")
AZURE_MODEL_DEPLOYMENTS = parse_model_deployments(raw_model_deployments)

AZURE_RESPONSES_API_URL = f"{AZURE_BASE_URL}/openai/v1/responses"

CODEX_AUTH_PATH = env.path("CODEX_AUTH_PATH", "~/.codex/auth.json")
CODEX_RESPONSES_URL = env.str(
    "CODEX_RESPONSES_URL", "https://chatgpt.com/backend-api/codex/responses"
)
CODEX_SUPPORTED_MODELS = parse_codex_supported_models(
    env.str("CODEX_SUPPORTED_MODELS", default="")
)
CODEX_MODEL_REWRITES = parse_codex_model_rewrites(
    env.str("CODEX_MODEL_REWRITES", default=""),
    supported_models=CODEX_SUPPORTED_MODELS,
)
CODEX_ORIGINATOR = env.str("CODEX_ORIGINATOR", "codex_cli_rs")
CODEX_USER_AGENT = env.str(
    "CODEX_USER_AGENT", "codex_cli_rs/0.130.0 cursor-provider-proxy/0.1"
)
CODEX_DISCOVERY_MODE = env.bool("CODEX_DISCOVERY_MODE", False)
CODEX_TOKEN_REFRESH_SKEW_SECONDS = env.int("CODEX_TOKEN_REFRESH_SKEW_SECONDS", 300)
CODEX_REQUEST_TIMEOUT_SECONDS = env.float("CODEX_REQUEST_TIMEOUT_SECONDS", 600)
