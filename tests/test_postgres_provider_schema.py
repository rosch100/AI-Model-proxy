"""PostgreSQL-only tests for the provider-account schema migration."""

from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.persistence.models import ProviderProfile, Tenant
from app.persistence.provider_circuit_breaker import ProviderCircuitBreakerStore
from app.persistence.repositories import import_tenants
from app.persistence.secrets import SecretCipher
from app.tenants import TenantConfig
from tests.postgres_test_utils import postgres_urls_available

pytestmark = [
    pytest.mark.postgresql,
    pytest.mark.skipif(
        not postgres_urls_available(),
        reason=(
            "PostgreSQL integration tests require both "
            "TEST_DATABASE_ADMIN_URL and TEST_DATABASE_RUNTIME_URL"
        ),
    ),
]


def test_multiple_provider_profiles_and_casefolded_active_names_are_enforced(
    postgres_test_databases,
):
    """Same-provider accounts coexist, but active names are case-insensitively unique."""
    databases = postgres_test_databases
    databases.upgrade("head")
    with databases.runtime_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (id, api_key_hash, custom_model_id) "
                "VALUES ('tenant-a', 'digest-a', 'cursor-a')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO provider_profiles "
                "(id, tenant_id, provider, settings, display_name, display_name_key, history_generation) "
                "VALUES ('profile-a', 'tenant-a', 'azure', '{}', 'Production', 'production', 0), "
                "('profile-b', 'tenant-a', 'azure', '{}', 'Staging', 'staging', 0)"
            )
        )

    with databases.runtime_engine.connect() as connection:
        transaction = connection.begin()
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO provider_profiles "
                    "(id, tenant_id, provider, settings, display_name, display_name_key, history_generation) "
                    "VALUES ('profile-c', 'tenant-a', 'azure', :settings, 'pRODUCTION', 'production', 0)"
                ),
                {"settings": json.dumps({})},
            )
        transaction.rollback()


def test_duplicate_node_preflight_reports_collisions_without_schema_changes(
    postgres_test_databases,
):
    """Migration reports duplicated node bindings and leaves the baseline untouched."""
    databases = postgres_test_databases
    databases.upgrade("20261002_passkeys")
    with databases.admin_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (id, api_key_hash, custom_model_id) "
                "VALUES ('tenant-a', 'digest-a', 'cursor-a')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO provider_profiles "
                "(id, tenant_id, provider, settings) "
                "VALUES ('profile-a', 'tenant-a', 'azure', '{}')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO provider_scope_nodes "
                "(id, tenant_id, provider, scope_type, canonical_scope_id) "
                "VALUES ('node-a', 'tenant-a', 'azure', 'resource_group', 'rg-a')"
            )
        )
        connection.execute(
            text(
                "DROP TRIGGER trg_validate_provider_scope_binding ON provider_scope_bindings"
            )
        )
        connection.execute(
            text(
                "INSERT INTO provider_scope_bindings "
                "(id, tenant_id, provider, profile_id, purpose, node_id) "
                "VALUES ('binding-a', 'tenant-a', 'azure', 'profile-a', 'billing', 'node-a'), "
                "('binding-b', 'tenant-a', 'azure', 'profile-a', 'usage', 'node-a')"
            )
        )

    with pytest.raises(RuntimeError) as failure:
        databases.upgrade("head")
    assert "node-a" in str(failure.value)
    assert "binding-a" in str(failure.value)
    assert "binding-b" in str(failure.value)

    inspector = inspect(databases.admin_engine)
    assert "display_name" not in {
        column["name"] for column in inspector.get_columns("provider_profiles")
    }
    constraints = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("provider_scope_bindings")
    }
    assert "uq_binding_tenant_purpose" in constraints
    with databases.admin_engine.connect() as connection:
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM provider_scope_bindings WHERE node_id = 'node-a'"
                )
            ).scalar_one()
            == 2
        )


def test_environment_import_is_idempotent_for_active_profile_with_azure_peers(
    postgres_test_databases,
):
    """Runtime import is idempotent and leaves the peer account unchanged."""
    databases = postgres_test_databases
    databases.upgrade("head")
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    cipher = SecretCipher.from_key(key)
    source = TenantConfig(
        id="tenant-import",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://production.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "production-gpt54"},
        azure_default_model="gpt-5.4",
    )

    with Session(databases.runtime_engine) as session, session.begin():
        tenant = Tenant(
            id=source.id,
            api_key_hash=source.api_key_hash,
            custom_model_id="cursor-import-model",
        )
        active_profile = ProviderProfile(
            id="azure-active",
            tenant_id=source.id,
            provider="azure",
            display_name="Production",
            settings={
                "base_url": source.azure_base_url,
                "model_deployments": dict(source.azure_model_deployments),
            },
            default_model=source.azure_default_model,
            inference_secret_ciphertext=cipher.encrypt(source.azure_api_key),
        )
        peer_profile = ProviderProfile(
            id="azure-peer",
            tenant_id=source.id,
            provider="azure",
            display_name="Staging",
            settings={
                "base_url": "https://staging.openai.azure.com",
                "model_deployments": {"gpt-5.5": "staging-gpt55"},
            },
            default_model="gpt-5.5",
            inference_secret_ciphertext=cipher.encrypt("peer-secret"),
            billing_secret_ciphertext=cipher.encrypt("peer-billing-secret"),
            catalog_error="peer catalog state",
        )
        session.add(tenant)
        session.flush()
        session.add_all((active_profile, peer_profile))
        session.flush()
        active_profile.route_priority = 1
        session.flush()
        peer_before = session.execute(
            text(
                "SELECT id, tenant_id, provider, display_name, deleted_at, "
                "history_generation, settings, inference_secret_ciphertext, "
                "billing_secret_ciphertext, default_model, catalog_refreshed_at, "
                "catalog_error, created_at FROM provider_profiles "
                "WHERE id = 'azure-peer'"
            )
        ).one()

        assert import_tenants((source,), session, cipher) == 0
        assert import_tenants((source,), session, cipher) == 0

        peer_after = session.execute(
            text(
                "SELECT id, tenant_id, provider, display_name, deleted_at, "
                "history_generation, settings, inference_secret_ciphertext, "
                "billing_secret_ciphertext, default_model, catalog_refreshed_at, "
                "catalog_error, created_at FROM provider_profiles "
                "WHERE id = 'azure-peer'"
            )
        ).one()
        assert peer_after == peer_before
        assert active_profile.route_priority == 1
        assert peer_profile.route_priority is None
        assert active_profile.display_name == "Production"
        assert cipher.decrypt(peer_profile.inference_secret_ciphertext) == "peer-secret"


@pytest.mark.parametrize(
    ("azure_profile_ids", "primary_profile_id"),
    [
        (("azure-one", "azure-two"), None),
        ((), "openai-active"),
    ],
    ids=("ambiguous-azure-candidates", "no-active-azure-profile"),
)
def test_environment_import_fails_closed_without_unique_active_azure_profile(
    postgres_test_databases, azure_profile_ids, primary_profile_id
):
    """Missing and ambiguous Azure selection never creates or mutates a profile."""
    databases = postgres_test_databases
    databases.upgrade("head")
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    cipher = SecretCipher.from_key(key)
    source = TenantConfig(
        id="tenant-import",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://production.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "production-gpt54"},
        azure_default_model="gpt-5.4",
    )

    with Session(databases.runtime_engine) as session, session.begin():
        tenant = Tenant(
            id=source.id,
            api_key_hash=source.api_key_hash,
            custom_model_id="cursor-import-model",
        )
        session.add(tenant)
        session.flush()
        profiles = [
            ProviderProfile(
                id=profile_id,
                tenant_id=source.id,
                provider="azure",
                display_name=profile_id,
                settings={
                    "base_url": source.azure_base_url,
                    "model_deployments": dict(source.azure_model_deployments),
                },
                default_model=source.azure_default_model,
                inference_secret_ciphertext=cipher.encrypt(source.azure_api_key),
            )
            for profile_id in azure_profile_ids
        ]
        if primary_profile_id == "openai-active":
            profiles.append(
                ProviderProfile(
                    id=primary_profile_id,
                    tenant_id=source.id,
                    provider="openai",
                    display_name="OpenAI",
                    settings={"base_url": "https://api.openai.com"},
                    default_model="gpt-5.5",
                    inference_secret_ciphertext=cipher.encrypt("openai-secret"),
                )
            )
        session.add_all(profiles)
        session.flush()
        if azure_profile_ids:
            active_profile_ids = set(azure_profile_ids)
        elif primary_profile_id is not None:
            active_profile_ids = {primary_profile_id}
        else:
            active_profile_ids = set()
        for priority, profile in enumerate(
            (profile for profile in profiles if profile.id in active_profile_ids),
            start=1,
        ):
            profile.route_priority = priority
        session.flush()
        profile_columns = tuple(ProviderProfile.__table__.columns)
        profile_rows_before = session.execute(
            select(*profile_columns)
            .where(ProviderProfile.tenant_id == source.id)
            .order_by(ProviderProfile.id)
        ).all()

        with pytest.raises(ValueError, match="ambiguous active Azure profile"):
            import_tenants((source,), session, cipher)

        profile_rows_after = session.execute(
            select(*profile_columns)
            .where(ProviderProfile.tenant_id == source.id)
            .order_by(ProviderProfile.id)
        ).all()
        assert profile_rows_after == profile_rows_before
        assert [
            profile.id for profile in profiles if profile.route_priority is not None
        ] == ([primary_profile_id] if primary_profile_id is not None else [])


def _seed_legacy_profile_and_cost_data(connection) -> None:
    """Seed a valid Azure billing root and its resource-scoped usage child."""
    connection.execute(
        text(
            "INSERT INTO tenants (id, api_key_hash, custom_model_id) "
            "VALUES ('tenant-a', 'digest-a', 'cursor-a')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO provider_profiles "
            "(id, tenant_id, provider, settings, inference_secret_ciphertext, "
            "billing_secret_ciphertext, default_model, catalog_refreshed_at, "
            "catalog_error, created_at) "
            "VALUES ('profile-a', 'tenant-a', 'azure', CAST(:settings AS json), "
            "'ciphertext-inference', NULL, 'gpt-5.4', "
            "'2026-09-29T12:00:00Z', 'catalog warning', '2026-09-01T12:00:00Z')"
        ),
        {
            "settings": json.dumps(
                {
                    "base_url": "https://resource.openai.azure.com",
                    "model_deployments": {"gpt-5.4": "deployment-a"},
                }
            )
        },
    )
    connection.execute(
        text("UPDATE tenants SET active_profile_id = 'profile-a' WHERE id = 'tenant-a'")
    )
    connection.execute(
        text(
            "INSERT INTO provider_catalog_entries "
            "(id, profile_id, model_id, deployment_id, source) "
            "VALUES (71, 'profile-a', 'gpt-5.4', 'deployment-a', 'provider')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO provider_scope_nodes "
            "(id, tenant_id, provider, scope_type, canonical_scope_id, parent_node_id) "
            "VALUES ('billing-node', 'tenant-a', 'azure', 'resource_group', "
            "'/subscriptions/sub-safe/resourceGroups/rg-safe', NULL), "
            "('usage-node', 'tenant-a', 'azure', 'cognitive_resource', "
            "'/subscriptions/sub-safe/resourceGroups/rg-safe/providers/Microsoft.CognitiveServices/accounts/resource', "
            "'billing-node')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO provider_scope_bindings "
            "(id, tenant_id, provider, profile_id, purpose, node_id, "
            "parent_binding_id, created_at) "
            "VALUES ('binding-a', 'tenant-a', 'azure', 'profile-a', 'billing', "
            "'billing-node', NULL, '2026-09-02T12:00:00Z')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO provider_scope_bindings "
            "(id, tenant_id, provider, profile_id, purpose, node_id, "
            "parent_binding_id, created_at) "
            "VALUES ('binding-b', 'tenant-a', 'azure', 'profile-a', 'usage', "
            "'usage-node', 'binding-a', '2026-09-03T12:00:00Z')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO cost_refresh_jobs "
            "(id, tenant_id, provider, binding_id, period_start, period_end, source_api, "
            "operation_key, status, retry_at, limitation, created_at, completed_at) "
            "VALUES ('job-a', 'tenant-a', 'azure', 'binding-a', "
            "'2026-09-01T00:00:00Z', '2026-10-01T00:00:00Z', 'azure.costs', "
            "'operation-a', 'success', NULL, 'complete window', "
            "'2026-10-01T12:00:00Z', '2026-10-01T12:05:00Z')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO cost_refresh_events "
            "(job_id, previous_status, new_status, reason) "
            "VALUES ('job-a', NULL, 'success', 'completed')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO cost_usage_records "
            "(id, job_id, tenant_id, provider, binding_id, kind, metric, value, unit, "
            "currency, bucket_start, bucket_end, source, granularity, dimensions, "
            "price_source, price_version, usage_source, usage_bucket_start, "
            "formula_parameters, model_key, region_key, deployment_key) "
            "VALUES (93, 'job-a', 'tenant-a', 'azure', 'binding-a', 'estimate', 'cost', "
            "1.25, 'currency', 'USD', '2026-09-01T00:00:00Z', '2026-10-01T00:00:00Z', "
            "'azure.costs', 'window', '{\"resource_group\": \"rg-safe\"}', "
            "'price-list', 'v42', 'azure.usage', '2026-09-30T00:00:00Z', "
            "'{\"multiplier\": 1}', 'gpt-5.4', 'eastus', 'deployment-a')"
        )
    )


def _legacy_snapshot(connection) -> dict[str, tuple]:
    """Read every persisted column that existed before the profile migration."""
    tenant_columns = {
        column["name"] for column in inspect(connection).get_columns("tenants")
    }
    primary_profile = (
        "active_profile_id"
        if "active_profile_id" in tenant_columns
        else "(SELECT p.id FROM provider_profiles p WHERE p.tenant_id = tenants.id "
        "AND p.route_priority IS NOT NULL ORDER BY p.route_priority LIMIT 1)"
    )
    queries = {
        "tenant": f"SELECT id, api_key_hash, custom_model_id, {primary_profile}, created_at "
        "FROM tenants ORDER BY tenants.id",
        "profile": "SELECT id, tenant_id, provider, settings, "
        "inference_secret_ciphertext, billing_secret_ciphertext, default_model, "
        "catalog_refreshed_at, catalog_error, created_at FROM provider_profiles ORDER BY id",
        "catalog": "SELECT id, profile_id, model_id, deployment_id, source "
        "FROM provider_catalog_entries ORDER BY id",
        "scope": "SELECT id, tenant_id, provider, scope_type, canonical_scope_id, "
        "parent_node_id, created_at FROM provider_scope_nodes ORDER BY id",
        "binding": "SELECT id, tenant_id, provider, profile_id, purpose, node_id, "
        "parent_binding_id, created_at FROM provider_scope_bindings ORDER BY id",
        "job": "SELECT id, tenant_id, provider, binding_id, period_start, period_end, "
        "source_api, operation_key, status, retry_at, limitation, created_at, completed_at "
        "FROM cost_refresh_jobs ORDER BY id",
        "event": "SELECT id, job_id, previous_status, new_status, reason, created_at "
        "FROM cost_refresh_events ORDER BY id",
        "cost": "SELECT id, job_id, tenant_id, provider, binding_id, kind, metric, value, "
        "unit, currency, bucket_start, bucket_end, source, granularity, dimensions, "
        "price_source, price_version, usage_source, usage_bucket_start, formula_parameters, "
        "model_key, region_key, deployment_key FROM cost_usage_records ORDER BY id",
    }
    return {
        label: tuple(tuple(row) for row in connection.execute(text(query)).all())
        for label, query in queries.items()
    }


def test_migration_preserves_profile_catalog_bindings_jobs_and_costs_round_trip(
    postgres_test_databases,
):
    """Existing active profile and all attributed provider data survive upgrade/rollback."""
    databases = postgres_test_databases
    databases.upgrade("20261002_passkeys")
    with databases.admin_engine.begin() as connection:
        _seed_legacy_profile_and_cost_data(connection)
        before = _legacy_snapshot(connection)

    databases.upgrade("head")
    with databases.admin_engine.connect() as connection:
        after_upgrade = _legacy_snapshot(connection)
        profile = connection.execute(
            text(
                "SELECT display_name, deleted_at, history_generation "
                "FROM provider_profiles WHERE id = 'profile-a'"
            )
        ).one()
    assert after_upgrade == before
    assert tuple(profile) == ("Azure", None, 0)

    databases.downgrade("20261002_passkeys")
    with databases.admin_engine.connect() as connection:
        after_downgrade = _legacy_snapshot(connection)
    assert after_downgrade == before

    databases.upgrade("head")
    with databases.admin_engine.connect() as connection:
        after_round_trip = _legacy_snapshot(connection)
    assert after_round_trip == before


def test_provider_circuit_probe_lease_is_exclusive_across_postgres_sessions(
    postgres_test_databases,
):
    """Concurrent workers cannot both claim the same due provider scope."""
    databases = postgres_test_databases
    databases.upgrade("head")
    with databases.runtime_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (id, api_key_hash, custom_model_id) "
                "VALUES ('breaker-tenant', 'breaker-digest', 'breaker-cursor')"
            )
        )

    now = datetime.now(timezone.utc)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    sessions = sessionmaker(bind=databases.runtime_engine, expire_on_commit=False)
    store = ProviderCircuitBreakerStore(sessions, cipher)
    scope = store.scope("breaker-tenant", "openai", "profile", "profile-a")
    store.open_quota(scope, now=now - timedelta(hours=1))

    barrier = Barrier(2)

    def acquire_after_barrier():
        barrier.wait(timeout=10)
        return store.acquire((scope,), now=now)

    with ThreadPoolExecutor(max_workers=2) as pool:
        permits = tuple(pool.map(lambda _worker: acquire_after_barrier(), range(2)))

    assert sum(permit.allowed for permit in permits) == 1
    assert sum(bool(permit.leases) for permit in permits) == 1
    assert sum(not permit.allowed for permit in permits) == 1
