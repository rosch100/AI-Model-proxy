"""SQLAlchemy engine and session lifecycle for PostgreSQL persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.tenants import DatabaseTenantRoutingSnapshot

from .repositories import TenantRepository
from .secrets import SecretCipher


@dataclass(frozen=True)
class Database:
    """Own an engine, its session factory, and provider-secret cipher."""

    engine: Engine
    sessions: sessionmaker[Session]
    secret_cipher: SecretCipher

    def get_proxy_snapshot_by_api_key(
        self, api_key: str
    ) -> DatabaseTenantRoutingSnapshot | None:
        """Resolve persisted proxy identity and provider settings for one request."""
        with self.sessions() as session:
            return TenantRepository(session).get_proxy_snapshot_by_api_key(
                api_key, self.secret_cipher
            )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Database:
        """Validate tenant database settings and construct a PostgreSQL engine."""
        database_url = config.get("DATABASE_URL")
        encryption_key = config.get("PROVIDER_ENCRYPTION_KEY")
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("DATABASE_URL is required for database tenant mode")
        if not isinstance(encryption_key, str) or not encryption_key:
            raise ValueError(
                "PROVIDER_ENCRYPTION_KEY is required for database tenant mode"
            )
        if not database_url.startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must use the postgresql+psycopg driver")

        secret_cipher = SecretCipher.from_key(encryption_key)
        engine = create_engine(database_url, pool_pre_ping=True)
        return cls(
            engine=engine,
            sessions=sessionmaker(bind=engine, expire_on_commit=False),
            secret_cipher=secret_cipher,
        )
