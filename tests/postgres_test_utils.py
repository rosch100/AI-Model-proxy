"""Isolated PostgreSQL schemas for migration and runtime-role integration tests."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, text

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def postgres_urls_available() -> bool:
    """Return whether both explicitly configured PostgreSQL test roles exist."""
    return bool(
        os.environ.get("TEST_DATABASE_ADMIN_URL")
        and os.environ.get("TEST_DATABASE_RUNTIME_URL")
    )


def _engine_with_schema(database_url: str, schema: str) -> Engine:
    engine = create_engine(database_url, pool_pre_ping=True)

    def set_search_path(dbapi_connection, _connection_record) -> None:
        # SQLAlchemy rolls back on connect/pool return. Persist the session setting
        # outside a transaction, or the runtime role silently loses its test schema.
        previous_autocommit = dbapi_connection.autocommit
        dbapi_connection.autocommit = True
        try:
            with dbapi_connection.cursor() as cursor:
                cursor.execute(f'SET search_path TO "{schema}"')
        finally:
            dbapi_connection.autocommit = previous_autocommit

    event.listen(engine, "connect", set_search_path)
    return engine


@dataclass
class PostgresTestDatabases:
    """Admin/runtime engines isolated to one disposable PostgreSQL schema."""

    schema: str
    admin_engine: Engine
    runtime_engine: Engine

    def upgrade(self, revision: str) -> None:
        """Run an explicit Alembic upgrade with the isolated admin connection."""
        self._run_migration(command.upgrade, revision)
        self.grant_runtime_access()

    def downgrade(self, revision: str) -> None:
        """Run an explicit Alembic downgrade with the isolated admin connection."""
        self._run_migration(command.downgrade, revision)
        self.grant_runtime_access()

    def grant_runtime_access(self) -> None:
        """Grant runtime DML on the fixture's schema without changing ownership."""
        with self.runtime_engine.connect() as connection:
            runtime_role = connection.execute(text("SELECT current_user")).scalar_one()
        admin_quote = self.admin_engine.dialect.identifier_preparer.quote
        runtime_role_sql = admin_quote(runtime_role)
        with self.admin_engine.begin() as connection:
            connection.execute(
                text(f'GRANT USAGE ON SCHEMA "{self.schema}" TO {runtime_role_sql}')
            )
            connection.execute(
                text(
                    f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                    f'IN SCHEMA "{self.schema}" TO {runtime_role_sql}'
                )
            )
            connection.execute(
                text(
                    f"GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES "
                    f'IN SCHEMA "{self.schema}" TO {runtime_role_sql}'
                )
            )

    def _run_migration(
        self, migration_command: Callable[[Config, str], None], revision: str
    ) -> None:
        config = Config(os.path.join(PROJECT_ROOT, "alembic.ini"))
        config.set_main_option(
            "script_location", os.path.join(PROJECT_ROOT, "migrations")
        )
        with self.admin_engine.begin() as connection:
            config.attributes["connection"] = connection
            migration_command(config, revision)


@pytest.fixture
def postgres_test_databases() -> Iterator[PostgresTestDatabases]:
    """Create and always remove an isolated PostgreSQL schema for one test."""
    admin_url = os.environ.get("TEST_DATABASE_ADMIN_URL")
    runtime_url = os.environ.get("TEST_DATABASE_RUNTIME_URL")
    if not admin_url or not runtime_url:
        pytest.skip(
            "PostgreSQL integration tests require both "
            "TEST_DATABASE_ADMIN_URL and TEST_DATABASE_RUNTIME_URL"
        )

    schema = f"provider_task1_{uuid4().hex}"
    admin_engine = _engine_with_schema(admin_url, schema)
    runtime_engine = _engine_with_schema(runtime_url, schema)
    databases = PostgresTestDatabases(schema, admin_engine, runtime_engine)
    try:
        with admin_engine.begin() as connection:
            if connection.dialect.name != "postgresql":
                pytest.fail("PostgreSQL integration URLs must use PostgreSQL")
            admin_database = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        with runtime_engine.connect() as connection:
            if connection.dialect.name != "postgresql":
                pytest.fail("PostgreSQL integration URLs must use PostgreSQL")
            runtime_database = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()
            runtime_role = connection.execute(text("SELECT current_user")).scalar_one()
        if admin_database != runtime_database:
            pytest.fail(
                "PostgreSQL admin and runtime URLs must target the same database"
            )
        role_sql = admin_engine.dialect.identifier_preparer.quote(runtime_role)
        with admin_engine.begin() as connection:
            connection.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO {role_sql}'))
        yield databases
    finally:
        runtime_engine.dispose()
        admin_engine.dispose()
        cleanup_engine = create_engine(admin_url, pool_pre_ping=True)
        try:
            with cleanup_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            cleanup_engine.dispose()
