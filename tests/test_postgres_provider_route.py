"""PostgreSQL routing migration and serialized writer integration contracts."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from app.persistence.admin_ops import (
    activate_provider_profile,
    create_provider_profile,
    replace_catalog_entries,
)
from app.persistence.models import ProviderProfile, Tenant
from app.persistence.secrets import SecretCipher
from app.tenants import hash_api_key
from tests.postgres_test_utils import postgres_urls_available

pytestmark = [
    pytest.mark.postgresql,
    pytest.mark.skipif(
        not postgres_urls_available(),
        reason="PostgreSQL integration tests require both TEST_DATABASE_ADMIN_URL and TEST_DATABASE_RUNTIME_URL",
    ),
]


def test_route_migration_transfers_only_existing_nondeleted_primary_profiles(
    postgres_test_databases,
):
    """The migration preserves profile data and copies only valid legacy pointers."""
    databases = postgres_test_databases
    databases.upgrade("20261004_clear_azure_billing")
    with databases.admin_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (id, api_key_hash, custom_model_id) VALUES "
                "('routed', 'digest-routed', 'cursor-routed'), "
                "('inactive', 'digest-inactive', 'cursor-inactive'), "
                "('deleted', 'digest-deleted', 'cursor-deleted')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO provider_profiles (id, tenant_id, provider, settings, display_name, deleted_at) VALUES "
                "('primary', 'routed', 'openai', '{}', 'Primary', NULL), "
                "('peer', 'routed', 'openai', '{}', 'Peer', NULL), "
                "('inactive', 'inactive', 'openai', '{}', 'Inactive', NULL), "
                "('tombstone', 'deleted', 'openai', '{}', 'Retired', now())"
            )
        )
        connection.execute(
            text("UPDATE tenants SET active_profile_id = 'primary' WHERE id = 'routed'")
        )
        connection.execute(
            text(
                "UPDATE tenants SET active_profile_id = 'tombstone' WHERE id = 'deleted'"
            )
        )
        before = tuple(
            connection.execute(
                text(
                    "SELECT id, tenant_id, provider, settings, display_name, deleted_at "
                    "FROM provider_profiles ORDER BY id"
                )
            )
        )
    databases.upgrade("head")
    with databases.admin_engine.connect() as connection:
        assert "active_profile_id" not in {
            column["name"] for column in inspect(connection).get_columns("tenants")
        }
        assert dict(
            connection.execute(
                text("SELECT id, route_priority FROM provider_profiles")
            ).all()
        ) == {
            "primary": 1,
            "peer": None,
            "inactive": None,
            "tombstone": None,
        }
        after = tuple(
            connection.execute(
                text(
                    "SELECT id, tenant_id, provider, settings, display_name, deleted_at "
                    "FROM provider_profiles ORDER BY id"
                )
            )
        )
        assert after == before
    databases.downgrade("20261004_clear_azure_billing")
    with databases.admin_engine.connect() as connection:
        assert dict(
            connection.execute(text("SELECT id, active_profile_id FROM tenants")).all()
        ) == {
            "routed": "primary",
            "inactive": None,
            "deleted": None,
        }


def test_route_downgrade_fails_before_schema_changes_when_multiple_profiles_are_active(
    postgres_test_databases,
):
    """A lossy downgrade fails transactionally before changing schema or data."""
    databases = postgres_test_databases
    databases.upgrade("head")
    with databases.runtime_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (id, api_key_hash, custom_model_id) VALUES ('tenant', 'digest', 'cursor')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO provider_profiles "
                "(id, tenant_id, provider, settings, display_name, route_priority) VALUES "
                "('one', 'tenant', 'openai', '{}', 'One', 1), ('two', 'tenant', 'openai', '{}', 'Two', 2)"
            )
        )
    with pytest.raises(DBAPIError, match="multiple active profiles would be lost"):
        databases.downgrade("20261004_clear_azure_billing")
    with databases.admin_engine.connect() as connection:
        assert "active_profile_id" not in {
            column["name"] for column in inspect(connection).get_columns("tenants")
        }
        assert list(
            connection.execute(
                text(
                    "SELECT route_priority FROM provider_profiles ORDER BY route_priority"
                )
            ).scalars()
        ) == [1, 2]
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == "20261005_provider_route"
        )


def test_postgres_route_unique_constraint_allows_nulls_and_rejects_collisions(
    postgres_test_databases,
):
    """Enforce unique positive priorities while permitting inactive rows."""
    databases = postgres_test_databases
    databases.upgrade("head")
    with databases.runtime_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (id, api_key_hash, custom_model_id) VALUES ('tenant', 'digest', 'cursor')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO provider_profiles "
                "(id, tenant_id, provider, settings, display_name, route_priority) VALUES "
                "('one', 'tenant', 'openai', '{}', 'One', 1), ('two', 'tenant', 'openai', '{}', 'Two', NULL), "
                "('three', 'tenant', 'openai', '{}', 'Three', NULL)"
            )
        )
    with databases.runtime_engine.connect() as connection:
        transaction = connection.begin()
        with pytest.raises(IntegrityError):
            connection.execute(
                text("UPDATE provider_profiles SET route_priority = 1 WHERE id = 'two'")
            )
        transaction.rollback()


def test_concurrent_activations_use_tenant_lock_and_append_after_previous_commit(
    postgres_test_databases,
):
    """The second writer reads its route only after the first writer commits."""
    databases = postgres_test_databases
    databases.upgrade("head")
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii"))
    with Session(databases.runtime_engine) as session, session.begin():
        session.add(
            Tenant(
                id="tenant",
                api_key_hash=hash_api_key("tenant"),
                custom_model_id="cursor",
            )
        )
        session.flush()
        ids = []
        for name in ("One", "Two"):
            profile = create_provider_profile(
                session,
                cipher,
                "tenant",
                "openai",
                name,
                {},
                "gpt-5.4",
                "secret",
                "ada",
            )
            replace_catalog_entries(session, profile, [("gpt-5.4", None)], None)
            ids.append(profile.id)
    attempting = Event()

    def activate_second():
        with Session(databases.runtime_engine) as session, session.begin():
            session.execute(text("SET LOCAL lock_timeout = '10s'"))
            tenant = session.get(Tenant, "tenant")
            attempting.set()
            activate_provider_profile(session, tenant, ids[1], "ada")

    with ThreadPoolExecutor(max_workers=1) as executor, Session(
        databases.runtime_engine
    ) as first:
        first.begin()
        tenant = first.scalar(
            select(Tenant).where(Tenant.id == "tenant").with_for_update()
        )
        activate_provider_profile(first, tenant, ids[0], "ada")
        second = executor.submit(activate_second)
        assert attempting.wait(timeout=10)
        first.commit()
        second.result(timeout=10)
    with Session(databases.runtime_engine) as session:
        profiles = list(
            session.scalars(
                select(ProviderProfile).order_by(ProviderProfile.route_priority)
            )
        )
        assert [(p.id, p.route_priority) for p in profiles] == [
            (ids[0], 1),
            (ids[1], 2),
        ]
