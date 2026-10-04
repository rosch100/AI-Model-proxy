"""Tests for database-backed persistence primitives."""

import base64
import hashlib
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from cryptography.exceptions import InvalidTag
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.persistence.database import Database
from app.persistence.models import (
    Base,
    InferenceActivityEvent,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeNode,
    Tenant,
)
from app.persistence.repositories import TenantRepository, import_tenants
from app.persistence.secrets import SecretCipher
from app.tenants import TenantConfig


def test_database_requires_postgresql_url_and_encryption_key():
    """Database mode rejects missing settings before creating an engine."""
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Database.from_config({"PROVIDER_ENCRYPTION_KEY": "unused"})

    with pytest.raises(ValueError, match="PROVIDER_ENCRYPTION_KEY"):
        Database.from_config({"DATABASE_URL": "postgresql+psycopg://db"})


def test_database_builds_psycopg_engine_without_connecting():
    """Database construction configures psycopg without opening a connection."""
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")

    database = Database.from_config(
        {
            "DATABASE_URL": "postgresql+psycopg://user:password@localhost/proxy",
            "PROVIDER_ENCRYPTION_KEY": key,
        }
    )

    assert database.engine.url.drivername == "postgresql+psycopg"
    assert (
        database.secret_cipher.decrypt(database.secret_cipher.encrypt("secret"))
        == "secret"
    )
    database.engine.dispose()


def test_database_rejects_sqlite_and_legacy_postgresql_drivers():
    """Only PostgreSQL through psycopg is supported for tenant persistence."""
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    for url in ("sqlite:///local.db", "postgresql://user:password@localhost/proxy"):
        with pytest.raises(ValueError, match=r"postgresql\+psycopg"):
            Database.from_config({"DATABASE_URL": url, "PROVIDER_ENCRYPTION_KEY": key})


def test_postgresql_downgrade_removes_all_migration_functions(monkeypatch):
    """Offline downgrade drops every trigger function before removing tables."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://user:password@localhost/proxy"
    )
    output = StringIO()
    config = Config(str(project_root / "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", str(project_root / "migrations"))

    command.downgrade(config, "20261001_initial:base", sql=True)

    generated_sql = output.getvalue()
    assert (
        "DROP FUNCTION IF EXISTS prevent_unsuccessful_job_with_cost_records()"
        in generated_sql
    )
    assert "DROP FUNCTION IF EXISTS require_successful_cost_job()" in generated_sql
    assert "DROP FUNCTION IF EXISTS validate_provider_scope_binding()" in generated_sql


def test_openrouter_workspace_migration_updates_postgresql_binding_validator(
    monkeypatch,
):
    """Upgrade migration accepts workspace scopes in PostgreSQL validator."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://user:password@localhost/proxy"
    )
    output = StringIO()
    config = Config(str(project_root / "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", str(project_root / "migrations"))

    command.upgrade(config, "20261002_provider_accounts_usage:head", sql=True)

    generated_sql = output.getvalue()
    assert (
        "CREATE OR REPLACE FUNCTION validate_provider_scope_binding()" in generated_sql
    )
    assert (
        "IF scope_kind <> 'workspace' OR parent_node IS NOT NULL"
        " OR NEW.parent_binding_id IS NOT NULL THEN"
    ) in generated_sql
    assert "NEW.created_at = OLD.created_at" in generated_sql
    assert "AND NOT EXISTS" in generated_sql
    assert "JOIN cost_refresh_jobs j ON j.binding_id = b.id" in generated_sql


def test_deepseek_migration_changes_only_profile_and_activity_provider_checks(
    monkeypatch,
):
    """Upgrade SQL enables DeepSeek inference but leaves billing-scope checks alone."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://user:password@localhost/proxy"
    )
    output = StringIO()
    config = Config(str(project_root / "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", str(project_root / "migrations"))

    script = ScriptDirectory.from_config(config)
    migration = script.get_revision("20261007_deepseek_provider").module
    migration_context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(migration_context):
        migration.upgrade()

    generated_sql = output.getvalue()
    assert "ck_profile_provider" in generated_sql
    assert "ck_inference_activity_provider" in generated_sql
    assert "'deepseek'" in generated_sql
    assert "ck_scope_provider" not in generated_sql
    assert "provider_scope_nodes" not in generated_sql


def test_azure_billing_secret_cleanup_migration_is_explicit(monkeypatch):
    """The Azure identity migration permanently clears obsolete tenant secrets."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://user:password@localhost/proxy"
    )
    output = StringIO()
    config = Config(str(project_root / "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", str(project_root / "migrations"))

    script = ScriptDirectory.from_config(config)
    migration = script.get_revision("20261004_clear_azure_billing").module
    migration_context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(migration_context):
        migration.upgrade()

    generated_sql = output.getvalue()
    assert (
        "UPDATE provider_profiles SET billing_secret_ciphertext = NULL "
        "WHERE provider = 'azure' AND billing_secret_ciphertext IS NOT NULL"
    ) in generated_sql


def test_deepseek_profiles_and_activity_are_accepted_but_unknown_provider_is_rejected():
    """The inference allowlists include DeepSeek and reject unregistered vendors."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            Tenant(
                id="acme",
                api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
                custom_model_id="cursor-acme-model",
            )
        )
        session.flush()
        session.add(
            ProviderProfile(
                id="deepseek-profile",
                tenant_id="acme",
                provider="deepseek",
                settings={},
            )
        )
        session.add(
            InferenceActivityEvent(
                tenant_id="acme",
                provider="deepseek",
                inbound_model="cursor-acme-model",
            )
        )
        session.commit()
        session.add(
            ProviderProfile(
                id="unsupported-profile",
                tenant_id="acme",
                provider="unsupported",
                settings={},
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()
    engine.dispose()


def test_provider_profile_activity_and_scope_provider_constraints_are_separate():
    """Only inference profile and activity checks accept DeepSeek."""
    profile_provider_check = next(
        constraint
        for constraint in ProviderProfile.__table__.constraints
        if constraint.name == "ck_profile_provider"
    )
    scope_provider_check = next(
        constraint
        for constraint in ProviderScopeNode.__table__.constraints
        if constraint.name == "ck_scope_provider"
    )
    activity_provider_check = next(
        constraint
        for constraint in InferenceActivityEvent.__table__.constraints
        if constraint.name == "ck_inference_activity_provider"
    )
    assert "'deepseek'" in str(profile_provider_check.sqltext)
    assert "'deepseek'" not in str(scope_provider_check.sqltext)
    assert "'deepseek'" in str(activity_provider_check.sqltext)


def test_persistence_schema_creates_on_sqlite_for_portable_constraint_checks():
    """Persistence metadata is consistent for unit-level schema checks."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    engine.dispose()


def test_persistence_schema_defines_tenant_and_provider_tables():
    """The persistence metadata exposes core tenant routing entities."""
    assert {
        "tenants",
        "admin_accounts",
        "provider_profiles",
        "provider_scope_nodes",
        "provider_scope_bindings",
        "cost_refresh_jobs",
        "cost_refresh_events",
        "cost_usage_records",
        "audit_events",
        "admin_passkeys",
        "admin_webauthn_challenges",
    }.issubset(Base.metadata.tables)


def test_tenant_repository_resolves_only_the_api_key_digest():
    """Cursor bearer credentials resolve a tenant without storing plaintext."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        tenant = Tenant(
            id="acme",
            api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
            custom_model_id="tenant-acme-model",
        )
        session.add(tenant)
        session.commit()

        resolved = TenantRepository(session).get_by_api_key("cursor-key")

        assert resolved is not None
        assert resolved.id == "acme"
        assert resolved.api_key_hash != "cursor-key"
        assert TenantRepository(session).get_by_api_key("wrong-key") is None
    engine.dispose()


def test_tenant_import_satisfies_immediate_tenant_profile_foreign_keys():
    """Tenant and active profile insertion respects immediately checked FKs."""
    engine = create_engine("sqlite:///:memory:")
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    source = TenantConfig(
        id="acme",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
        azure_default_model="gpt-5.4",
    )

    with Session(engine) as session, session.begin():
        assert import_tenants((source,), session, cipher) == 1

    engine.dispose()


def test_tenant_import_is_idempotent_and_encrypts_provider_credentials():
    """Repeated identical imports create a single tenant and encrypted profile."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    cipher = SecretCipher.from_key(key)
    source = TenantConfig(
        id="acme",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
        azure_default_model="gpt-5.4",
    )

    with Session(engine) as session:
        assert import_tenants((source,), session, cipher) == 1
        session.commit()
        assert import_tenants((source,), session, cipher) == 0
        session.commit()

        tenant = session.get(Tenant, "acme")
        profile = session.scalar(
            select(ProviderProfile).where(ProviderProfile.tenant_id == "acme")
        )
        assert tenant.custom_model_id.startswith("cursor-")
        assert profile.route_priority == 1
        assert profile.default_model == "gpt-5.4"
        assert cipher.decrypt(profile.inference_secret_ciphertext) == "azure-secret"
        assert profile.inference_secret_ciphertext != "azure-secret"
    engine.dispose()


def test_tenant_import_rejects_missing_azure_default_model():
    """A database-backed active profile must identify its default Azure model."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    source = TenantConfig(
        id="acme",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
    )

    with Session(engine) as session, pytest.raises(
        ValueError, match="azure_default_model"
    ):
        with session.begin():
            import_tenants((source,), session, cipher)
    engine.dispose()


def test_tenant_import_rejects_a_changed_provider_secret():
    """Idempotent imports reject credentials that differ from the stored profile."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    original = TenantConfig(
        id="acme",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="original-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
        azure_default_model="gpt-5.4",
    )
    changed = TenantConfig(
        id="acme",
        api_key_hash=original.api_key_hash,
        azure_base_url=original.azure_base_url,
        azure_api_key="rotated-secret",
        azure_model_deployments=dict(original.azure_model_deployments),
        azure_default_model="gpt-5.4",
    )

    with Session(engine) as session:
        with session.begin():
            assert import_tenants((original,), session, cipher) == 1
        with pytest.raises(ValueError, match="different provider secret"):
            with session.begin():
                import_tenants((changed,), session, cipher)
    engine.dispose()


def test_tenant_import_conflicts_roll_back_without_partial_rows():
    """A conflicting tenant import leaves no unrelated staged tenant behind."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    cipher = SecretCipher.from_key(key)
    first = TenantConfig(
        id="acme",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
        azure_default_model="gpt-5.4",
    )
    conflicting = TenantConfig(
        id="acme",
        api_key_hash=hashlib.sha256(b"different-key").hexdigest(),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
        azure_default_model="gpt-5.4",
    )

    with Session(engine) as session:
        with session.begin():
            import_tenants((first,), session, cipher)
        with pytest.raises(ValueError, match="different API key hash"):
            with session.begin():
                import_tenants(
                    (
                        TenantConfig(
                            id="beta",
                            api_key_hash=hashlib.sha256(b"beta-key").hexdigest(),
                            azure_base_url="https://beta.openai.azure.com",
                            azure_api_key="beta-secret",
                            azure_model_deployments={"gpt-5.4": "beta-gpt54"},
                            azure_default_model="gpt-5.4",
                        ),
                        conflicting,
                    ),
                    session,
                    cipher,
                )
        assert session.get(Tenant, "beta") is None
    engine.dispose()


def test_environment_import_uses_the_active_azure_profile_without_mutating_peers():
    """An idempotent import targets the active Azure account by profile ID."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    source = TenantConfig(
        id="acme",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
        azure_default_model="gpt-5.4",
    )
    with Session(engine) as session, session.begin():
        tenant = Tenant(
            id="acme",
            api_key_hash=source.api_key_hash,
            custom_model_id="cursor-acme-model",
        )
        session.add(tenant)
        session.flush()
        active_profile = ProviderProfile(
            id="azure-active",
            tenant_id="acme",
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
            tenant_id="acme",
            provider="azure",
            display_name="Staging",
            settings={"base_url": "https://staging.openai.azure.com"},
            default_model="gpt-5.5",
            inference_secret_ciphertext=cipher.encrypt("peer-secret"),
        )
        session.add_all((active_profile, peer_profile))
        session.flush()
        active_profile.route_priority = 1
        session.flush()

        assert import_tenants((source,), session, cipher) == 0
        assert active_profile.route_priority == 1
        assert peer_profile.route_priority is None
        assert active_profile.display_name == "Production"
        assert peer_profile.settings == {"base_url": "https://staging.openai.azure.com"}
        assert peer_profile.default_model == "gpt-5.5"
        assert cipher.decrypt(peer_profile.inference_secret_ciphertext) == "peer-secret"
    engine.dispose()


def test_environment_import_rejects_ambiguous_active_azure_profile():
    """Environment import fails closed if it cannot identify one active Azure account."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    source = TenantConfig(
        id="acme",
        api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
        azure_base_url="https://acme.openai.azure.com",
        azure_api_key="azure-secret",
        azure_model_deployments={"gpt-5.4": "acme-gpt54"},
        azure_default_model="gpt-5.4",
    )
    with Session(engine) as session:
        with session.begin():
            tenant = Tenant(
                id="acme",
                api_key_hash=source.api_key_hash,
                custom_model_id="cursor-acme-model",
            )
            session.add(tenant)
            session.flush()
            session.add_all(
                ProviderProfile(
                    id=profile_id,
                    tenant_id="acme",
                    provider="azure",
                    display_name=profile_id,
                    settings={
                        "base_url": source.azure_base_url,
                        "model_deployments": dict(source.azure_model_deployments),
                    },
                    default_model=source.azure_default_model,
                    inference_secret_ciphertext=cipher.encrypt(source.azure_api_key),
                )
                for profile_id in ("azure-one", "azure-two")
            )
        with pytest.raises(ValueError, match="ambiguous active Azure profile"):
            with session.begin():
                import_tenants((source,), session, cipher)
    engine.dispose()


def test_proxy_snapshot_freezes_active_profile_identity_and_generation():
    """A proxy snapshot retains the profile selected by its initial lookup."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    with Session(engine) as session, session.begin():
        session.add_all(
            (
                Tenant(
                    id="acme",
                    api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
                    custom_model_id="cursor-acme",
                ),
                ProviderProfile(
                    id="profile-a",
                    route_priority=1,
                    tenant_id="acme",
                    provider="azure",
                    display_name="Production",
                    settings={
                        "base_url": "https://production.openai.azure.com",
                        "model_deployments": {"gpt-5.4": "production-deployment"},
                    },
                    default_model="gpt-5.4",
                    history_generation=3,
                    inference_secret_ciphertext=cipher.encrypt("production-key"),
                ),
                ProviderProfile(
                    id="profile-b",
                    tenant_id="acme",
                    provider="azure",
                    display_name="Staging",
                    settings={"base_url": "https://staging.example"},
                    default_model="gpt-5.5",
                    history_generation=0,
                    inference_secret_ciphertext=cipher.encrypt("staging-key"),
                ),
            )
        )

        session.flush()
        session.add(
            ProviderCatalogEntry(
                profile_id="profile-a",
                model_id="gpt-5.4",
                deployment_id="production-deployment",
                source="provider",
            )
        )

    with Session(engine) as session:
        routing_snapshot = TenantRepository(session).get_proxy_snapshot_by_api_key(
            "cursor-key", cipher
        )
        assert routing_snapshot is not None
        assert len(routing_snapshot.profiles) == 1
        snapshot = routing_snapshot.profiles[0]
        assert snapshot.profile_id == "profile-a"
        assert snapshot.profile_name == "Production"
        assert snapshot.history_generation == 3
        assert snapshot.profile_deleted is False

    with Session(engine) as session, session.begin():
        session.get(ProviderProfile, "profile-a").route_priority = None
        session.flush()
        session.get(ProviderProfile, "profile-b").route_priority = 1

    assert snapshot.profile_id == "profile-a"
    assert snapshot.profile_name == "Production"
    assert snapshot.inference_secret == "production-key"
    engine.dispose()


def test_proxy_snapshot_distinguishes_missing_profile_and_deleted_tombstone():
    """No active profile and a deleted profile remain explicit configurations."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    with Session(engine) as session, session.begin():
        session.add_all(
            (
                Tenant(
                    id="without-profile",
                    api_key_hash=hashlib.sha256(b"no-profile").hexdigest(),
                    custom_model_id="cursor-no-profile",
                ),
                Tenant(
                    id="deleted-profile",
                    api_key_hash=hashlib.sha256(b"deleted-profile").hexdigest(),
                    custom_model_id="cursor-deleted",
                ),
                ProviderProfile(
                    id="profile-deleted",
                    route_priority=1,
                    tenant_id="deleted-profile",
                    provider="azure",
                    display_name=None,
                    deleted_at=datetime.now(timezone.utc),
                    history_generation=4,
                    settings={},
                    default_model=None,
                    inference_secret_ciphertext=None,
                ),
            )
        )

    with Session(engine) as session:
        no_profile = TenantRepository(session).get_proxy_snapshot_by_api_key(
            "no-profile", cipher
        )
        deleted = TenantRepository(session).get_proxy_snapshot_by_api_key(
            "deleted-profile", cipher
        )

    assert no_profile is not None
    assert no_profile.profiles == ()
    assert deleted is not None
    assert deleted.profiles == ()
    with Session(engine) as session:
        tombstone = session.get(ProviderProfile, "profile-deleted")
        assert tombstone.deleted_at is not None
        assert tombstone.history_generation == 4
    engine.dispose()


def test_provider_inference_secret_is_stored_as_ciphertext():
    """Credential persistence has no plaintext secret column."""
    columns = ProviderProfile.__table__.columns

    assert "inference_secret_ciphertext" in columns
    assert "inference_secret" not in columns


def test_secret_cipher_round_trips_with_a_fresh_nonce():
    """Encrypted secrets can be read without reusing nonces."""
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    cipher = SecretCipher.from_key(key)

    first = cipher.encrypt("provider-secret")
    second = cipher.encrypt("provider-secret")

    assert first != second
    assert cipher.decrypt(first) == "provider-secret"
    assert cipher.decrypt(second) == "provider-secret"


def test_secret_cipher_rejects_malformed_key():
    """Invalid key encodings fail at configuration time."""
    with pytest.raises(ValueError, match="32 bytes"):
        SecretCipher.from_key("not-a-valid-key")


def test_secret_cipher_propagates_authentication_failure():
    """Ciphertext tampering is detected rather than returning altered data."""
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    cipher = SecretCipher.from_key(key)
    envelope = bytearray(base64.urlsafe_b64decode(cipher.encrypt("provider-secret")))
    envelope[-1] ^= 1
    tampered = base64.urlsafe_b64encode(envelope).decode("ascii")

    with pytest.raises(InvalidTag):
        cipher.decrypt(tampered)
