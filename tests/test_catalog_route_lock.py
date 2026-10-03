"""PostgreSQL lock ordering for catalog refresh finalization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from flask import g
from sqlalchemy import event, select
from sqlalchemy.orm import sessionmaker

from app.admin.views import refresh_catalog
from app.persistence.admin_ops import (
    activate_provider_profile,
    create_provider_profile,
    replace_catalog_entries,
)
from app.persistence.database import Database
from app.persistence.models import ProviderCatalogEntry, ProviderProfile, Tenant
from app.tenants import hash_api_key
from tests.postgres_test_utils import postgres_urls_available

pytestmark = [
    pytest.mark.postgresql,
    pytest.mark.skipif(
        not postgres_urls_available(),
        reason="PostgreSQL integration tests require both test database URLs",
    ),
]


@pytest.mark.parametrize("catalog_status", [200, 503])
def test_catalog_finalization_locks_tenant_before_profile(
    admin_app, postgres_test_databases, requests_mock, catalog_status
) -> None:
    """Finalization follows route writers' Tenant -> Profile order after HTTP."""
    databases = postgres_test_databases
    databases.upgrade("head")
    database = Database(
        engine=databases.runtime_engine,
        sessions=sessionmaker(bind=databases.runtime_engine, expire_on_commit=False),
        secret_cipher=admin_app.extensions["database"].secret_cipher,
    )
    admin_app.extensions["database"] = database
    with database.sessions.begin() as session:
        session.add(
            Tenant(
                id="catalog-lock-tenant",
                api_key_hash=hash_api_key("catalog-lock-key"),
                custom_model_id="cursor-catalog-lock",
            )
        )
        session.flush()
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            "catalog-lock-tenant",
            "openai",
            "Catalog account",
            {},
            "model-a",
            "inference-secret",
            "ada",
        )
        replace_catalog_entries(session, profile, [("model-a", None)], None)
        profile_id = profile.id

    finalizing = False
    final_locks: list[str] = []
    final_transactions = []

    def record_lock(connection, _cursor, statement, _parameters, _context, _many):
        if finalizing and "FOR UPDATE" in statement.upper():
            final_locks.append(statement)
            final_transactions.append(connection.get_transaction())

    def catalog_response(_request, _context):
        nonlocal finalizing
        assert database.engine.pool.checkedout() == 0, "HTTP must run outside DB work"
        with database.sessions.begin() as session:
            profile = session.get(ProviderProfile, profile_id)
            assert profile.catalog_generation == 1
            tenant = session.get(Tenant, "catalog-lock-tenant")
            activate_provider_profile(session, tenant, profile_id, "ada")
        finalizing = True
        return {"data": [{"id": "model-b"}]}

    requests_mock.get(
        "https://api.openai.com/v1/models",
        status_code=catalog_status,
        json=catalog_response,
    )
    event.listen(database.engine, "before_cursor_execute", record_lock)
    try:
        with admin_app.test_request_context(
            f"/admin/settings/connection/{profile_id}/catalog", method="POST"
        ):
            g.admin = SimpleNamespace(tenant_id="catalog-lock-tenant", username="ada")
            response = refresh_catalog.__wrapped__(profile_id)
    finally:
        event.remove(database.engine, "before_cursor_execute", record_lock)

    assert response.status_code == 302
    assert len(final_locks) >= 2
    assert "FROM tenants" in final_locks[0], final_locks
    assert "FROM provider_profiles" in final_locks[1], final_locks
    assert final_transactions[0] is not None
    assert all(
        transaction is final_transactions[0] for transaction in final_transactions
    )
    with database.sessions() as session:
        profile = session.get(ProviderProfile, profile_id)
        models = list(
            session.scalars(
                select(ProviderCatalogEntry.model_id).where(
                    ProviderCatalogEntry.profile_id == profile_id
                )
            )
        )
        if catalog_status == 200:
            assert models == ["model-b"]
            assert profile.route_priority is None
            assert profile.default_model is None
        else:
            assert models == ["model-a"]
            assert profile.route_priority == 1
            assert profile.default_model == "model-a"
            assert profile.catalog_error == "OpenAI catalog HTTP 503"
