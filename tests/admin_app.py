"""Shared SQLite admin application for tenant UI tests."""

from __future__ import annotations

import base64
import hashlib
import logging

import pytest
from flask import Flask
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import create_app
from app.admin.passwords import hash_admin_password
from app.admin.security import configure_admin_security
from app.admin.views import register_admin
from app.persistence.database import Database
from app.persistence.models import AdminAccount, Base, Tenant
from app.persistence.secrets import SecretCipher

ENCRYPTION_KEY = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")


def _enable_sqlite_foreign_keys(engine) -> None:
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )


def build_admin_database():
    """Create an in-memory tenant schema for administrator tests."""
    engine = create_engine("sqlite:///:memory:")
    _enable_sqlite_foreign_keys(engine)
    Base.metadata.create_all(engine)
    return Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(ENCRYPTION_KEY),
    )


def seed_admin(database: Database, password: str = "correct-horse-battery") -> None:
    """Insert one tenant and administrator used by the UI tests."""
    with database.sessions.begin() as session:
        session.add(
            Tenant(
                id="acme",
                api_key_hash=hashlib.sha256(b"cursor-key").hexdigest(),
                custom_model_id="cursor-acme-model",
            )
        )
        session.flush()
        session.add(
            AdminAccount(
                tenant_id="acme",
                username="ada",
                password_hash=hash_admin_password(password),
            )
        )


@pytest.fixture
def admin_app() -> Flask:
    """Flask app with an attached SQLite tenant database and admin UI."""
    app = create_app("tests.settings")
    app.logger.setLevel(logging.CRITICAL)
    database = build_admin_database()
    seed_admin(database)
    app.extensions["database"] = database
    configure_admin_security(app)
    register_admin(app)
    ctx = app.test_request_context()
    ctx.push()
    yield app
    ctx.pop()
    database.engine.dispose()
