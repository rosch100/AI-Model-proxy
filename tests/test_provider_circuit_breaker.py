"""Persistent quota circuit breaker state and half-open lease behavior."""

import base64
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.persistence.database import Database
from app.persistence.models import Base, ProviderCircuitState, Tenant
from app.persistence.provider_circuit_breaker import (
    BreakerScope,
    ProviderCircuitBreakerStore,
)
from app.persistence.secrets import SecretCipher
from app.providers.circuit_breaker import retry_after_header


@pytest.fixture
def breaker_store():
    """Provide a persistent-store test fixture with isolated in-memory state."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    database = Database(
        engine, sessionmaker(bind=engine, expire_on_commit=False), cipher
    )
    with database.sessions.begin() as session:
        session.add(
            Tenant(
                id="tenant-a",
                api_key_hash="a" * 64,
                custom_model_id="cursor-tenant-a",
            )
        )
    yield ProviderCircuitBreakerStore(database.sessions, cipher)
    engine.dispose()


def scope(cipher, scope_type="profile", scope_id="profile-a"):
    """Build a scope fixture with its real domain-separated HMAC fingerprint."""
    return BreakerScope(
        tenant_id="tenant-a",
        provider="openai",
        scope_type=scope_type,
        fingerprint=cipher.scope_fingerprint(
            "tenant-a", "openai", scope_type, scope_id
        ),
    )


def test_scope_fingerprint_is_stable_domain_separated_and_one_way():
    """Keep fingerprints stable while separating tenant and raw-ID inputs."""
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    first = cipher.scope_fingerprint("tenant-a", "openai", "profile", "profile-a")

    assert first == cipher.scope_fingerprint(
        "tenant-a", "openai", "profile", "profile-a"
    )
    assert first != cipher.scope_fingerprint(
        "tenant-a", "openai", "profile", "profile-b"
    )
    assert first != cipher.scope_fingerprint(
        "tenant-b", "openai", "profile", "profile-a"
    )
    assert first != cipher.scope_fingerprint(
        "tenant-a", "azure", "profile", "profile-a"
    )
    assert first != cipher.scope_fingerprint(
        "tenant-a", "openai", "organization", "profile-a"
    )
    assert len(first) == 64
    assert "profile-a" not in first


def test_quota_open_uses_hourly_exponential_backoff_and_retry_after_floor(
    breaker_store,
):
    """Increase cooldown exponentially and never let Retry-After shorten it."""
    now = datetime(2026, 10, 4, 10, tzinfo=timezone.utc)
    key = breaker_store.scope("tenant-a", "openai", "profile", "profile-a")
    opened = []
    for retry_after in (None, None, 3 * 60 * 60, None, None, None, None):
        opened.append(
            breaker_store.open_quota(key, now=now, retry_after_seconds=retry_after)
        )

    assert [int((item.probe_at - now).total_seconds() / 3600) for item in opened] == [
        1,
        2,
        4,
        8,
        16,
        24,
        24,
    ]
    assert all(item.failure_category == "quota_exhausted" for item in opened)


def test_later_retry_after_extends_quota_backoff(breaker_store):
    """A valid provider retry floor later than backoff controls the probe time."""
    now = datetime(2026, 10, 4, 10, tzinfo=timezone.utc)
    key = breaker_store.scope("tenant-a", "openai", "profile", "profile-a")

    breaker_store.open_quota(key, now=now)
    breaker_store.open_quota(key, now=now, retry_after_seconds=6 * 60 * 60)

    snapshot = breaker_store.snapshots("tenant-a", (key,))[0]
    assert snapshot.probe_at == now + timedelta(hours=6)
    assert snapshot.failure_count == 2


def test_retry_after_header_rounds_up_to_a_safe_delta_seconds():
    """Never advertise a retry earlier than its absolute readiness time."""
    now = datetime(2026, 10, 4, 10, tzinfo=timezone.utc)

    assert retry_after_header(now + timedelta(seconds=1.1), now=now) == "2"
    assert retry_after_header(now - timedelta(seconds=1), now=now) == "1"


def test_due_breaker_has_one_half_open_lease_and_expired_lease_can_be_claimed(
    breaker_store,
):
    """Lease a due scope once, then permit a new lease after expiry."""
    now = datetime(2026, 10, 4, 10, tzinfo=timezone.utc)
    key = breaker_store.scope("tenant-a", "openai", "profile", "profile-a")
    breaker_store.open_quota(key, now=now - timedelta(hours=1))

    first = breaker_store.acquire((key,), now=now)
    concurrent = breaker_store.acquire((key,), now=now + timedelta(seconds=1))
    expired = breaker_store.acquire((key,), now=now + timedelta(seconds=31))

    assert first.allowed and first.leases[0].token
    assert not concurrent.allowed
    assert concurrent.retry_at == first.leases[0].lease_until
    assert concurrent.lease_blocked
    assert expired.allowed and expired.leases[0].token != first.leases[0].token


def test_probe_success_clears_state_and_stale_token_cannot_clear_new_lease(
    breaker_store,
):
    """Close only the current lease; stale tokens cannot clear newer probes."""
    now = datetime(2026, 10, 4, 10, tzinfo=timezone.utc)
    key = breaker_store.scope("tenant-a", "openai", "profile", "profile-a")
    breaker_store.open_quota(key, now=now - timedelta(hours=1))
    first = breaker_store.acquire((key,), now=now)
    second = breaker_store.acquire((key,), now=now + timedelta(seconds=31))

    breaker_store.resolve_probe(
        first, category="success", now=now + timedelta(seconds=32)
    )
    with breaker_store._sessions() as session:
        assert session.scalar(select(ProviderCircuitState)) is not None

    breaker_store.resolve_probe(
        second, category="success", now=now + timedelta(seconds=33)
    )
    with breaker_store._sessions() as session:
        assert session.scalar(select(ProviderCircuitState)) is None


def test_nonquota_probe_failure_clears_quota_breaker(breaker_store):
    """A recovered quota scope is cleared when its genuine probe is nonquota."""
    now = datetime(2026, 10, 4, 10, tzinfo=timezone.utc)
    key = breaker_store.scope("tenant-a", "openai", "profile", "profile-a")
    breaker_store.open_quota(key, now=now - timedelta(hours=1))
    permit = breaker_store.acquire((key,), now=now)

    breaker_store.resolve_probe(permit, category="transient", now=now)

    with breaker_store._sessions() as session:
        assert session.scalar(select(ProviderCircuitState)) is None


def test_only_due_blockers_are_leased_and_future_blocker_controls_retry_at(
    breaker_store,
):
    """Block all scopes until the earliest due retry time without leasing early."""
    now = datetime(2026, 10, 4, 10, tzinfo=timezone.utc)
    profile = scope(breaker_store._cipher)
    organization = scope(breaker_store._cipher, "organization", "org-1")
    breaker_store.open_quota(profile, now=now - timedelta(hours=1))
    breaker_store.open_quota(
        organization,
        now=now - timedelta(hours=1),
        retry_after_seconds=3 * 60 * 60,
    )

    permit = breaker_store.acquire((profile, organization), now=now)

    assert not permit.allowed
    assert permit.retry_at == now + timedelta(hours=2)
