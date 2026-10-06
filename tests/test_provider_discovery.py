"""Tenant-scoped provider and model discovery API contracts."""

from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from app.persistence.admin_ops import create_provider_profile, replace_catalog_entries
from app.persistence.models import ProviderCatalogEntry, ProviderProfile, Tenant
from app.providers.catalog import (
    _complete_openrouter_pricing,
    refresh_provider_catalog_with_pricing,
)
from app.tenants import hash_api_key


def _create_profile(
    database, tenant_id, provider, name, entries, pricing=None, settings=None
):
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            tenant_id,
            provider,
            name or "Temporary name",
            settings or {},
            entries[0][0],
            f"secret-{tenant_id}-{name}",
            "ada",
        )
        replace_catalog_entries(session, profile, entries, None, pricing)
        if name is None:
            profile.display_name = None
        return profile.id


@pytest.fixture
def discovery_app(admin_app):
    """Seed ready profiles for two database tenants and enable DB auth."""
    admin_app.config.update(AUTH_MODE="tenant", TENANT_CONFIG_SOURCE="database")
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        session.add(
            Tenant(
                id="beta",
                api_key_hash=hash_api_key("beta-key"),
                custom_model_id="cursor-beta",
            )
        )
    _create_profile(
        database,
        "acme",
        "openrouter",
        "Research/Team",
        [("anthropic/claude-3.7", None), ("openai/gpt-4.1", None)],
        {
            "anthropic/claude-3.7": {
                "input_per_1m_tokens": "1.25",
                "output_per_1m_tokens": "2.5",
                "cache_per_1m_tokens": "0.25",
                "currency": "USD",
                "source": "https://openrouter.ai/api/v1/models",
            }
        },
    )
    _create_profile(
        database, "acme", "openrouter", "Production", [("openai/gpt-4.1", None)]
    )
    _create_profile(database, "acme", "openai", None, [("gpt-4.1", None)])
    _create_profile(database, "beta", "deepseek", "Beta", [("deepseek-chat", None)])
    with database.sessions.begin() as session:
        profiles = session.query(ProviderProfile).filter_by(tenant_id="acme").all()
        for profile in profiles:
            profile.catalog_refreshed_at = datetime(2026, 10, 5, tzinfo=timezone.utc)
    return admin_app


def _auth(key="cursor-key"):
    return {"Authorization": f"Bearer {key}"}


def test_provider_list_projects_only_snapshot_profiles_without_sensitive_data(
    discovery_app,
):
    """List safe provider and model metadata from the tenant snapshot."""
    response = discovery_app.test_client().get("/v1/providers", headers=_auth())

    assert response.status_code == 200
    assert response.json == {
        "object": "list",
        "data": [
            {
                "provider": "openrouter",
                "name": "Research/Team",
                "models": [
                    {
                        "id": "anthropic/claude-3.7",
                        "qualified_id": "openrouter:Research%2FTeam/anthropic/claude-3.7",
                        "pricing": {
                            "input_per_1m_tokens": "1.25",
                            "output_per_1m_tokens": "2.5",
                            "cache_per_1m_tokens": "0.25",
                            "currency": "USD",
                            "source": "https://openrouter.ai/api/v1/models",
                            "updated_at": "2026-10-05T00:00:00+00:00",
                        },
                    },
                    {
                        "id": "openai/gpt-4.1",
                        "qualified_id": "openrouter:Research%2FTeam/openai/gpt-4.1",
                    },
                ],
            },
            {
                "provider": "openrouter",
                "name": "Production",
                "models": [
                    {
                        "id": "openai/gpt-4.1",
                        "qualified_id": "openrouter:Production/openai/gpt-4.1",
                    }
                ],
            },
            {"provider": "openai", "name": None, "models": [{"id": "gpt-4.1"}]},
        ],
    }
    serialized = response.get_data(as_text=True)
    assert "acme" not in serialized
    assert "cursor-key" not in serialized
    assert "secret-" not in serialized


def test_provider_detail_resolves_url_encoded_slash_profile_name(discovery_app):
    """Resolve named profiles through URL-encoded display names."""
    profile_name = quote("Research/Team", safe="")
    response = discovery_app.test_client().get(
        f"/v1/providers/openrouter/models?profile={profile_name}", headers=_auth()
    )

    assert response.status_code == 200
    assert response.json == {
        "provider": "openrouter",
        "name": "Research/Team",
        "models": [
            {
                "id": "anthropic/claude-3.7",
                "qualified_id": "openrouter:Research%2FTeam/anthropic/claude-3.7",
                "pricing": {
                    "input_per_1m_tokens": "1.25",
                    "output_per_1m_tokens": "2.5",
                    "cache_per_1m_tokens": "0.25",
                    "currency": "USD",
                    "source": "https://openrouter.ai/api/v1/models",
                    "updated_at": "2026-10-05T00:00:00+00:00",
                },
            },
            {
                "id": "openai/gpt-4.1",
                "qualified_id": "openrouter:Research%2FTeam/openai/gpt-4.1",
            },
        ],
    }


def test_provider_detail_rejects_unnamed_profile_and_missing_or_empty_name(
    discovery_app,
):
    """Reject absent names and keep unnamed profiles non-addressable."""
    client = discovery_app.test_client()
    assert (
        client.get("/v1/providers/openai/models?profile=", headers=_auth()).status_code
        == 400
    )
    assert client.get("/v1/providers/openai/models", headers=_auth()).status_code == 400
    assert (
        client.get(
            "/v1/providers/openai/models?profile=anything", headers=_auth()
        ).status_code
        == 404
    )


@pytest.mark.parametrize(
    "path",
    [
        "/v1/providers/unknown/models?profile=Production",
        "/v1/providers/openrouter/models?profile=Missing",
    ],
)
def test_provider_detail_returns_404_for_unknown_provider_or_profile(
    discovery_app, path
):
    """Hide nonexistent provider profiles behind a tenant-scoped 404."""
    assert discovery_app.test_client().get(path, headers=_auth()).status_code == 404


def test_discovery_hides_profiles_when_provider_switch_is_disabled(discovery_app):
    """Keep discovery aligned with profile routing eligibility."""
    _create_profile(
        discovery_app.extensions["database"],
        "acme",
        "azure",
        "Azure",
        [("gpt-4.1", "deployment-gpt-4.1")],
        settings={"base_url": "https://resource.openai.azure.com"},
    )
    discovery_app.config["ENABLE_AZURE"] = False
    client = discovery_app.test_client()

    response = client.get("/v1/providers", headers=_auth())
    assert response.status_code == 200
    assert all(item["provider"] != "azure" for item in response.json["data"])
    assert (
        client.get(
            "/v1/providers/azure/models?profile=Azure", headers=_auth()
        ).status_code
        == 404
    )


@pytest.mark.parametrize(
    "path",
    [
        "/v1/providers",
        "/v1/providers/openrouter/models?profile=Research%2FTeam",
    ],
)
def test_discovery_requires_authentication_and_database_tenant_mode(
    discovery_app, path
):
    """Require bearer auth and database-backed tenant mode."""
    client = discovery_app.test_client()
    assert client.get(path).status_code == 401

    discovery_app.config.update(AUTH_MODE="single", TENANT_CONFIG_SOURCE="environment")
    assert client.get(path, headers=_auth("test-service-api-key")).status_code == 400


def test_discovery_is_tenant_isolated_and_models_contract_is_unchanged(discovery_app):
    """Keep tenant isolation and the OpenAI-compatible model list contract."""
    client = discovery_app.test_client()
    beta = client.get("/v1/providers", headers=_auth("beta-key"))
    assert beta.status_code == 200
    assert beta.json["data"] == [
        {
            "provider": "deepseek",
            "name": "Beta",
            "models": [
                {
                    "id": "deepseek-chat",
                    "qualified_id": "deepseek:Beta/deepseek-chat",
                }
            ],
        }
    ]

    models = client.get("/v1/models", headers=_auth())
    assert models.status_code == 200
    assert models.json == {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1686935002,
                "owned_by": "openai",
            }
            for model_id in [
                "anthropic/claude-3.7",
                "openai/gpt-4.1",
                "gpt-4.1",
                "Research%2FTeam/anthropic/claude-3.7",
                "openrouter:Research%2FTeam/anthropic/claude-3.7",
                "Research%2FTeam/openai/gpt-4.1",
                "openrouter:Research%2FTeam/openai/gpt-4.1",
                "Production/openai/gpt-4.1",
                "openrouter:Production/openai/gpt-4.1",
            ]
        ],
    }


def test_catalog_price_payload_requires_complete_openrouter_rates(requests_mock):
    """Normalize complete published rates and omit incomplete price sets."""
    requests_mock.get(
        "https://openrouter.ai/api/v1/models",
        json={
            "data": [
                {
                    "id": "valid",
                    "pricing": {
                        "prompt": "0.00000125",
                        "completion": "0.0000025",
                        "input_cache_read": "0.00000025",
                        "input_cache_write": "0.000004",
                    },
                },
                {
                    "id": "incomplete",
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000002",
                        "input_cache_write": "0.000003",
                    },
                },
            ]
        },
    )

    entries, pricing = refresh_provider_catalog_with_pricing(
        "openrouter", {}, "sk-or-secret"
    )
    assert entries == [("valid", None), ("incomplete", None)]
    assert pricing == {
        "valid": {
            "input_per_1m_tokens": "1.25",
            "output_per_1m_tokens": "2.5",
            "cache_per_1m_tokens": "0.25",
            "currency": "USD",
            "source": "https://openrouter.ai/api/v1/models",
        }
    }


def test_openrouter_catalog_parses_decimal_json_without_float_rounding(requests_mock):
    """Preserve decimal precision when normalizing provider prices."""
    requests_mock.get(
        "https://openrouter.ai/api/v1/models",
        text=(
            '{"data":[{"id":"precise","pricing":{"prompt":0.000000123456789123,'
            '"completion":0.000000000000001,"input_cache_read":0.0000005}},'
            '{"id":"too-precise","pricing":{"prompt":1e-70,'
            '"completion":1e-70,"input_cache_read":1e-70}}]}'
        ),
    )

    entries, pricing = refresh_provider_catalog_with_pricing("openrouter", {}, "secret")
    assert entries == [("precise", None), ("too-precise", None)]
    assert set(pricing) == {"precise"}
    assert pricing["precise"]["input_per_1m_tokens"] == "0.123456789123"
    assert pricing["precise"]["output_per_1m_tokens"] == "0.000000001"
    assert pricing["precise"]["cache_per_1m_tokens"] == "0.5"


@pytest.mark.parametrize("key", ["prompt", "completion", "input_cache_read"])
@pytest.mark.parametrize(
    "rate",
    [
        "1e1000000000",
        "1e-1000000000",
        "1e58",
        "1e-69",
        "1" * 59,
        "1" * 64 + "e-7",
    ],
)
def test_openrouter_pricing_rejects_oversized_rates_before_formatting(key, rate):
    """Reject the whole price set before any potentially large allocation."""
    rates = {"prompt": "0.000001", "completion": "0", "input_cache_read": "0"}
    rates[key] = rate

    with patch("app.providers.catalog._decimal_string") as formatter:
        assert _complete_openrouter_pricing(rates) is None
        formatter.assert_not_called()


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        ("1e57", "1" + "0" * 63),
        ("1e-68", "0." + "0" * 61 + "1"),
        ("1" * 58, "1" * 58 + "0" * 6),
        ("1" * 63 + "e-7", "1" * 62 + ".1"),
        ("0.0000012500", "1.25"),
        ("0e1000000000", "0"),
        ("-0e-1000000000", "0"),
    ],
)
def test_openrouter_pricing_accepts_bounded_rates_and_zero(rate, expected):
    """Keep exact boundary values, decimal normalization, and compact zeros."""
    result = _complete_openrouter_pricing(
        {"prompt": Decimal(rate), "completion": "0", "input_cache_read": "0"}
    )

    assert result is not None
    assert result["input_per_1m_tokens"] == expected


@pytest.mark.parametrize("provider", ["azure", "openai", "deepseek"])
def test_non_openrouter_catalog_refresh_has_no_pricing(provider):
    """Leave pricing empty for providers without a supported price source."""
    with patch("app.providers.catalog.refresh_provider_catalog", return_value=[]):
        assert refresh_provider_catalog_with_pricing(provider, {}, "secret") == (
            [],
            {},
        )


@pytest.mark.parametrize("provider", ["azure", "openai", "deepseek"])
def test_non_openrouter_profile_does_not_persist_catalog_pricing(admin_app, provider):
    """Ignore catalog pricing unless the profile has a pricing source."""
    database = admin_app.extensions["database"]
    settings = (
        {"base_url": "https://resource.openai.azure.com"} if provider == "azure" else {}
    )
    with database.sessions.begin() as session:
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "acme",
            provider,
            f"{provider}-pricing-test",
            settings,
            "model-1",
            f"secret-{provider}",
            "ada",
        )
        replace_catalog_entries(
            session,
            profile,
            [("model-1", None)],
            None,
            {
                "model-1": {
                    "input_per_1m_tokens": "1",
                    "output_per_1m_tokens": "2",
                    "cache_per_1m_tokens": "3",
                    "currency": "USD",
                    "source": "unsupported-source",
                }
            },
        )
        profile_id = profile.id

    with database.sessions() as session:
        entry = (
            session.query(ProviderCatalogEntry)
            .filter_by(profile_id=profile_id, model_id="model-1")
            .one()
        )

    assert entry.input_price_per_1m_tokens is None
    assert entry.output_price_per_1m_tokens is None
    assert entry.cache_price_per_1m_tokens is None
    assert entry.pricing_currency is None
    assert entry.pricing_source is None


def test_provider_catalog_entry_has_nullable_pricing_storage():
    """Keep optional pricing metadata nullable in the catalog schema."""
    columns = ProviderCatalogEntry.__table__.columns
    for name in (
        "input_price_per_1m_tokens",
        "output_price_per_1m_tokens",
        "cache_price_per_1m_tokens",
        "pricing_currency",
        "pricing_source",
    ):
        assert columns[name].nullable


def test_pricing_migration_adds_only_nullable_columns(monkeypatch):
    """Compile PostgreSQL DDL for each optional pricing column."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://user:password@localhost/proxy"
    )
    output = StringIO()
    config = Config(str(project_root / "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", str(project_root / "migrations"))
    migration = (
        ScriptDirectory.from_config(config)
        .get_revision("20261012_catalog_pricing")
        .module
    )
    migration_context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(migration_context):
        migration.upgrade()

    sql = output.getvalue()
    for column in (
        "input_price_per_1m_tokens",
        "output_price_per_1m_tokens",
        "cache_price_per_1m_tokens",
        "pricing_currency",
        "pricing_source",
    ):
        assert f"ADD COLUMN {column} " in sql
        assert f"{column} VARCHAR" in sql
        assert "NOT NULL" not in sql
