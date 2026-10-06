"""PostgreSQL coordination tests for the batch worker."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.batch_worker import _fenced_update, claim_batch
from app.persistence.database import Database
from app.persistence.models import BatchJob, ProviderProfile, Tenant
from app.persistence.secrets import SecretCipher

pytestmark = pytest.mark.postgresql


def test_concurrent_batch_claims_are_exclusive(postgres_test_databases):
    """Ensure skip-locked leases let one worker claim a queued job."""
    postgres_test_databases.upgrade("head")
    database = _seed_postgres_batch(postgres_test_databases)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(
                executor.map(
                    lambda owner: claim_batch(database, owner),
                    ("worker-one", "worker-two"),
                )
            )
        assert sum(claim is not None for claim in claims) == 1
    finally:
        database.engine.dispose()


def test_expired_postgres_lease_fences_stale_worker_updates(postgres_test_databases):
    """Advance the fence before an expired worker can write after reclaim."""
    postgres_test_databases.upgrade("head")
    database = _seed_postgres_batch(postgres_test_databases)
    try:
        with database.sessions.begin() as session:
            job = session.get(BatchJob, "batch-concurrency-job")
            job.status = "submitted"
            job.provider_batch_id = "provider-batch-id"
        start = datetime.now(timezone.utc)
        first = claim_batch(database, "worker-one", now=start)
        second = claim_batch(database, "worker-two", now=start + timedelta(seconds=120))
        assert first is not None and second is not None
        assert second[1] > first[1]
        assert not _fenced_update(
            database,
            "batch-concurrency-job",
            "worker-one",
            first[1],
            provider_input_file_id="stale-file",
        )
    finally:
        database.engine.dispose()


def _seed_postgres_batch(postgres_test_databases) -> Database:
    engine = postgres_test_databases.runtime_engine
    cipher = SecretCipher.from_key("a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=")
    database = Database(
        engine, sessionmaker(bind=engine, expire_on_commit=False), cipher
    )
    tenant_id = "batch_concurrency_tenant"
    with postgres_test_databases.admin_engine.begin() as connection:
        with Session(bind=connection) as session:
            session.add(
                Tenant(
                    id=tenant_id,
                    api_key_hash=hashlib.sha256(b"batch-key").hexdigest(),
                    custom_model_id="cursor-batch-concurrency",
                )
            )
            session.flush()
            session.add(
                ProviderProfile(
                    id="batch-concurrency-profile",
                    tenant_id=tenant_id,
                    provider="openai",
                    display_name="Batch concurrency",
                    settings={},
                    default_model="gpt-5.4",
                    inference_secret_ciphertext=cipher.encrypt("sk-secret"),
                )
            )
            session.flush()
            now = datetime.now(timezone.utc)
            session.add(
                BatchJob(
                    id="batch-concurrency-job",
                    tenant_id=tenant_id,
                    profile_id="batch-concurrency-profile",
                    model="gpt-5.4",
                    idempotency_key_hash=hashlib.sha256(b"idempotency").hexdigest(),
                    payload_digest=hashlib.sha256(b"payload").hexdigest(),
                    request_json={"model": "gpt-5.4", "requests": []},
                    status="queued",
                    fencing_token=0,
                    created_at=now,
                    updated_at=now,
                    expires_at=now + timedelta(days=7),
                )
            )
    return database
