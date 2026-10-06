"""PostgreSQL integration tests for atomic provider budget reservations."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import sessionmaker

from app.persistence.models import (
    ProviderBudgetLease,
    ProviderBudgetLeaseAllocation,
    ProviderBudgetScopeState,
    ProviderBudgetWindow,
    ProviderConcurrencyLease,
    ProviderConcurrencyState,
    Tenant,
)
from app.persistence.provider_scheduler import (
    BudgetMetric,
    BudgetPolicy,
    BudgetScope,
    BudgetScopeKind,
    ProviderBudgetScheduler,
    ProviderConcurrencyCandidate,
    ProviderConcurrencyController,
    ProviderSchedulerPolicyConflict,
    ReservationRequest,
    SharedBudgetIdentity,
)
from app.persistence.secrets import SecretCipher
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


def _request(tenant_id, scopes, *, limit=1, estimated_tokens=None):
    """Build one explicitly configured request-budget reservation."""
    return ReservationRequest(
        tenant_id,
        tuple(
            BudgetPolicy(scope, BudgetMetric.REQUESTS, limit, 60) for scope in scopes
        ),
        estimated_tokens=estimated_tokens,
    )


def _scheduler(databases):
    """Create a scheduler using the test runtime role and test-only key."""
    sessions = sessionmaker(bind=databases.runtime_engine, expire_on_commit=False)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    return ProviderBudgetScheduler(sessions, cipher)


def _create_tenant(databases, tenant_id):
    """Insert the tenant required by persisted reservation foreign keys."""
    with databases.runtime_engine.begin() as connection:
        connection.execute(
            Tenant.__table__.insert().values(
                id=tenant_id,
                api_key_hash=f"digest-{tenant_id}",
                custom_model_id=f"cursor-{tenant_id}",
            )
        )


def test_postgres_aimd_persists_success_increase_and_transient_decrease(
    postgres_test_databases,
):
    """AIMD concurrency limits survive workers and update by completed outcomes."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"aimd-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    controller = ProviderConcurrencyController(
        sessionmaker(bind=databases.runtime_engine, expire_on_commit=False),
        scheduler._cipher,
        initial_limit=2,
    )
    now = datetime.now(timezone.utc)

    first = controller.acquire(tenant_id, "openai", "profile-aimd", now=now)
    assert first.allowed and first.lease is not None
    assert controller.settle(first.lease, outcome="success", now=now)
    with databases.runtime_engine.connect() as connection:
        assert (
            connection.execute(
                select(ProviderConcurrencyState.concurrency_limit).where(
                    ProviderConcurrencyState.tenant_id == tenant_id
                )
            ).scalar_one()
            == 3
        )

    second = controller.acquire(
        tenant_id, "openai", "profile-aimd", now=now + timedelta(seconds=1)
    )
    assert second.allowed and second.lease is not None
    assert controller.settle(
        second.lease, outcome="transient_failure", now=now + timedelta(seconds=2)
    )
    with databases.runtime_engine.connect() as connection:
        assert (
            connection.execute(
                select(ProviderConcurrencyState.concurrency_limit).where(
                    ProviderConcurrencyState.tenant_id == tenant_id
                )
            ).scalar_one()
            == 1
        )


def test_postgres_aimd_group_selection_uses_weighted_least_loaded_capacity(
    postgres_test_databases,
):
    """The PostgreSQL group scoring selects the less-loaded profile."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"aimd-group-{uuid4()}"
    _create_tenant(databases, tenant_id)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    controller = ProviderConcurrencyController(
        sessionmaker(bind=databases.runtime_engine, expire_on_commit=False), cipher
    )
    candidates = tuple(
        ProviderConcurrencyCandidate(
            provider="openai",
            profile_id=profile_id,
            target_model="model-a",
            profile_initial_limit=4,
            profile_minimum_limit=1,
            profile_maximum_limit=4,
            model_initial_limit=4,
            model_minimum_limit=1,
            model_maximum_limit=4,
            weight=1.0,
        )
        for profile_id in ("profile-a", "profile-b")
    )

    occupied = controller.acquire(
        tenant_id,
        "openai",
        "profile-a",
        target_model="model-a",
        profile_initial_limit=4,
        profile_minimum_limit=1,
        profile_maximum_limit=4,
        model_initial_limit=4,
        model_minimum_limit=1,
        model_maximum_limit=4,
    )
    selected = controller.acquire_best(tenant_id, candidates)

    assert occupied.allowed and occupied.lease is not None
    assert selected.allowed and selected.lease is not None
    assert selected.profile_id == "profile-b"
    assert controller.settle(occupied.lease, outcome="released")
    assert controller.settle(selected.lease, outcome="released")


def test_postgres_aimd_profile_only_candidate_acquires_single_scope(
    postgres_test_databases,
):
    """Acquire without a target model leases only the profile scope."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"aimd-profile-only-{uuid4()}"
    _create_tenant(databases, tenant_id)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    controller = ProviderConcurrencyController(
        sessionmaker(bind=databases.runtime_engine, expire_on_commit=False), cipher
    )

    decision = controller.acquire(tenant_id, "openai", "profile-only")

    assert decision.allowed and decision.lease is not None
    assert decision.lease.additional_permits == ()
    assert controller.settle(decision.lease, outcome="released")


def test_postgres_aimd_parallel_group_selection_assigns_distinct_available_profiles(
    postgres_test_databases,
):
    """Concurrent PostgreSQL selectors receive distinct available profiles."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"aimd-group-lock-{uuid4()}"
    _create_tenant(databases, tenant_id)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    controllers = tuple(
        ProviderConcurrencyController(
            sessionmaker(bind=databases.runtime_engine, expire_on_commit=False), cipher
        )
        for _ in range(2)
    )
    candidates = tuple(
        ProviderConcurrencyCandidate(
            provider="openai",
            profile_id=profile_id,
            target_model="model-a",
            profile_initial_limit=1,
            profile_minimum_limit=1,
            profile_maximum_limit=1,
            model_initial_limit=1,
            model_minimum_limit=1,
            model_maximum_limit=1,
        )
        for profile_id in ("profile-a", "profile-b")
    )
    barrier = Barrier(2)

    def acquire(controller):
        barrier.wait(timeout=10)
        return controller.acquire_best(tenant_id, candidates)

    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = tuple(pool.map(acquire, controllers))

    assert all(
        decision.allowed and decision.lease is not None for decision in decisions
    )
    assert {decision.profile_id for decision in decisions} == {"profile-a", "profile-b"}


def test_postgres_budget_headroom_includes_the_pending_reservation(
    postgres_test_databases,
):
    """Budget headroom reflects pending reservations until they're released."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-headroom-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-headroom")
    request = ReservationRequest(
        tenant_id,
        (BudgetPolicy(scope, BudgetMetric.REQUESTS, 4, 60),),
    )

    assert scheduler.headroom(request) == 1.0
    reserved = scheduler.reserve(request)
    assert reserved.allowed and reserved.lease is not None
    assert scheduler.headroom(request) == 0.5
    assert scheduler.release(reserved.lease)
    assert scheduler.headroom(request) == 1.0


def test_postgres_aimd_atomically_reserves_profile_and_model_scopes(
    postgres_test_databases,
):
    """A full model scope leaves no partial profile lease and other models can proceed."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"aimd-dual-{uuid4()}"
    _create_tenant(databases, tenant_id)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    controller = ProviderConcurrencyController(
        sessionmaker(bind=databases.runtime_engine, expire_on_commit=False),
        cipher,
    )
    options = {
        "profile_initial_limit": 2,
        "profile_minimum_limit": 1,
        "profile_maximum_limit": 2,
        "model_initial_limit": 1,
        "model_minimum_limit": 1,
        "model_maximum_limit": 4,
    }

    first = controller.acquire(
        tenant_id, "openai", "profile-dual", target_model="model-a", **options
    )
    denied_same_model = controller.acquire(
        tenant_id, "openai", "profile-dual", target_model="model-a", **options
    )
    allowed_other_model = controller.acquire(
        tenant_id, "openai", "profile-dual", target_model="model-b", **options
    )
    denied_profile = controller.acquire(
        tenant_id, "openai", "profile-dual", target_model="model-c", **options
    )

    assert first.allowed and first.lease is not None
    assert not denied_same_model.allowed
    assert allowed_other_model.allowed and allowed_other_model.lease is not None
    assert not denied_profile.allowed
    with databases.runtime_engine.connect() as connection:
        active_by_fingerprint = dict(
            connection.execute(
                select(
                    ProviderConcurrencyState.scope_fingerprint,
                    func.count(ProviderConcurrencyLease.id),
                )
                .join(
                    ProviderConcurrencyLease,
                    ProviderConcurrencyLease.state_id == ProviderConcurrencyState.id,
                )
                .where(
                    ProviderConcurrencyState.tenant_id == tenant_id,
                    ProviderConcurrencyLease.status == "active",
                )
                .group_by(ProviderConcurrencyState.scope_fingerprint)
            ).all()
        )
    assert len(active_by_fingerprint) == 4
    assert sorted(active_by_fingerprint.values()) == [1, 1, 1, 1]
    assert controller.settle(first.lease, outcome="success")
    assert controller.settle(allowed_other_model.lease, outcome="released")


def test_postgres_aimd_cleanup_expires_stale_active_permits(postgres_test_databases):
    """Periodic bounded cleanup marks expired active permits terminal."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"aimd-cleanup-{uuid4()}"
    _create_tenant(databases, tenant_id)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    controller = ProviderConcurrencyController(
        sessionmaker(bind=databases.runtime_engine, expire_on_commit=False),
        cipher,
        initial_limit=1,
        lease_ttl=timedelta(seconds=5),
    )
    start = datetime.now(timezone.utc)
    decision = controller.acquire(tenant_id, "openai", "profile-cleanup", now=start)
    assert decision.allowed and decision.lease is not None

    assert controller.cleanup(now=start + timedelta(seconds=6), batch_size=1) == 1
    with databases.runtime_engine.connect() as connection:
        assert (
            connection.execute(
                select(ProviderConcurrencyLease.status).where(
                    ProviderConcurrencyLease.id == decision.lease.id
                )
            ).scalar_one()
            == "expired"
        )


def test_postgres_aimd_row_lock_prevents_cross_worker_overclaim(
    postgres_test_databases,
):
    """Concurrent workers serialize one profile's persistent capacity claim."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"aimd-lock-{uuid4()}"
    _create_tenant(databases, tenant_id)
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    controllers = tuple(
        ProviderConcurrencyController(
            sessionmaker(bind=databases.runtime_engine, expire_on_commit=False),
            cipher,
            initial_limit=1,
        )
        for _ in range(2)
    )
    barrier = Barrier(2)

    def acquire(controller):
        barrier.wait(timeout=10)
        return controller.acquire(tenant_id, "openai", "profile-lock")

    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = tuple(pool.map(acquire, controllers))

    assert sum(decision.allowed for decision in decisions) == 1
    assert sum(decision.lease is not None for decision in decisions) == 1


def test_postgres_reservation_is_all_or_nothing_across_hierarchical_budgets(
    postgres_test_databases,
):
    """An exhausted shared scope does not consume another scope's available slot."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    profile_a = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-a")
    profile_b = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-b")
    organization = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "org-shared",
        shared_identity=SharedBudgetIdentity("canonical:org-shared"),
    )
    another_organization = BudgetScope(
        "openai", BudgetScopeKind.ORGANIZATION, "org-other"
    )

    first = scheduler.reserve(_request(tenant_id, (profile_a, organization)))
    denied = scheduler.reserve(_request(tenant_id, (profile_b, organization)))
    after_denial = scheduler.reserve(
        _request(tenant_id, (profile_b, another_organization))
    )

    assert first.allowed and first.lease is not None
    assert not denied.allowed
    assert after_denial.allowed and after_denial.lease is not None


def test_postgres_shared_scope_budget_is_global_across_tenants(postgres_test_databases):
    """A provider organization limit is shared across tenants using its credentials."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_a = f"scheduler-{uuid4()}"
    tenant_b = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_a)
    _create_tenant(databases, tenant_b)
    scheduler = _scheduler(databases)
    scope = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "local-org-label",
        shared_identity=SharedBudgetIdentity("provider-organization:verified-123"),
    )

    first = scheduler.reserve(_request(tenant_a, (scope,)))
    second = scheduler.reserve(_request(tenant_b, (scope,)))

    assert first.allowed
    assert not second.allowed


def test_postgres_concurrent_claims_respect_single_shared_request_slot(
    postgres_test_databases,
):
    """Concurrent sessions cannot reserve more than the configured window budget."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scope = BudgetScope("openai", BudgetScopeKind.ORGANIZATION, "org-concurrent")
    barrier = Barrier(2)

    def reserve_once():
        scheduler = _scheduler(databases)
        barrier.wait(timeout=10)
        return scheduler.reserve(_request(tenant_id, (scope,)))

    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = tuple(pool.map(lambda _index: reserve_once(), range(2)))

    assert sum(decision.allowed for decision in decisions) == 1


def test_postgres_lease_settlement_is_token_safe_and_charges_actual_tokens(
    postgres_test_databases,
):
    """Only the matching lease token can release or settle request/token claims."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-settlement")
    request = ReservationRequest(
        tenant_id,
        (
            BudgetPolicy(scope, BudgetMetric.REQUESTS, 3, 60),
            BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 60),
        ),
        estimated_tokens=80,
    )
    decision = scheduler.reserve(request)
    assert decision.lease is not None

    forged_lease = replace(decision.lease, token="not-the-lease-token")
    assert scheduler.complete(forged_lease, actual_tokens=70) is False
    assert scheduler.release(forged_lease) is False
    assert scheduler.complete(decision.lease, actual_tokens=70) is True
    assert scheduler.release(decision.lease) is False

    next_request = ReservationRequest(
        tenant_id,
        (
            BudgetPolicy(scope, BudgetMetric.REQUESTS, 3, 60),
            BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 60),
        ),
        estimated_tokens=30,
    )
    next_decision = scheduler.reserve(next_request)
    assert next_decision.allowed
    assert next_decision.lease is not None
    assert scheduler.complete(next_decision.lease, actual_tokens=30)


def test_postgres_policy_window_change_is_rejected_across_buckets(
    postgres_test_databases,
):
    """The policy registry prevents parallel windows for one shared metric."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-policy-conflict")

    scheduler.reserve(_request(tenant_id, (scope,)))
    changed_policy = ReservationRequest(
        tenant_id,
        (BudgetPolicy(scope, BudgetMetric.REQUESTS, 10, 300),),
    )

    with pytest.raises(
        ProviderSchedulerPolicyConflict, match="window policy conflicts"
    ):
        scheduler.reserve(changed_policy)


def test_postgres_failed_attempt_counts_request_and_releases_token_reservation(
    postgres_test_databases,
):
    """A failed upstream attempt consumes its request slot, not its token estimate."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-failed-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-failed")
    request = ReservationRequest(
        tenant_id,
        (
            BudgetPolicy(scope, BudgetMetric.REQUESTS, 1, 60),
            BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 60),
        ),
        estimated_tokens=80,
    )
    fixed_window_now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    decision = scheduler.reserve(request, now=fixed_window_now)
    assert decision.allowed and decision.lease is not None

    assert scheduler.failed(decision.lease)
    with databases.runtime_engine.connect() as connection:
        status = connection.execute(
            select(ProviderBudgetLease.status).where(
                ProviderBudgetLease.id == decision.lease.id
            )
        ).scalar_one()
    assert status == "failed"

    next_decision = scheduler.reserve(request)

    assert not next_decision.allowed
    assert next_decision.retry_at is not None
    token_only = ReservationRequest(
        tenant_id,
        (BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 60),),
        estimated_tokens=100,
    )
    assert scheduler.reserve(token_only).allowed


def test_postgres_actual_usage_overrun_blocks_following_reservations(
    postgres_test_databases,
):
    """Completion charges actual usage and denies further over-limit reservations."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-overrun")
    request = ReservationRequest(
        tenant_id,
        (BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 60),),
        estimated_tokens=80,
    )
    decision = scheduler.reserve(request)
    assert decision.lease is not None
    assert scheduler.complete(decision.lease, actual_tokens=120)

    denied = scheduler.reserve(
        ReservationRequest(
            tenant_id,
            (BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 60),),
            estimated_tokens=1,
        )
    )

    assert not denied.allowed


def test_postgres_reversed_policy_lists_share_window_lock_order(
    postgres_test_databases,
):
    """Concurrent reverse-order policy lists contend without acquiring locks differently."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope_a = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "local-label-a",
        shared_identity=SharedBudgetIdentity("canonical:scope-order-a"),
    )
    scope_b = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "local-label-b",
        shared_identity=SharedBudgetIdentity("canonical:scope-order-b"),
    )
    barrier = Barrier(2)

    def reserve(scopes):
        policies = tuple(
            BudgetPolicy(scope, BudgetMetric.REQUESTS, 5, 60) for scope in scopes
        )
        barrier.wait(timeout=10)
        return scheduler.reserve(ReservationRequest(tenant_id, policies))

    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = tuple(
            pool.map(
                lambda scopes: reserve(scopes), ((scope_a, scope_b), (scope_b, scope_a))
            )
        )

    assert all(decision.allowed for decision in decisions)
    assert all(decision.lease is not None for decision in decisions)


def test_postgres_expired_window_is_not_charged_by_stale_completion(
    postgres_test_databases,
):
    """A stream completing after its fixed window cannot charge a newer window."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-stale-window")
    request = ReservationRequest(
        tenant_id,
        (
            BudgetPolicy(scope, BudgetMetric.REQUESTS, 10, 60),
            BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 60),
        ),
        estimated_tokens=80,
    )
    decision = scheduler.reserve(request)
    assert decision.lease is not None

    with databases.runtime_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE provider_budget_windows "
                "SET window_start = window_start - INTERVAL '60 seconds' "
                "WHERE id IN (SELECT window_id FROM provider_budget_lease_allocations "
                "WHERE lease_id = :lease_id)"
            ),
            {"lease_id": decision.lease.id},
        )

    assert scheduler.cleanup(now=datetime.now(timezone.utc), batch_size=10) == 4
    with databases.runtime_engine.connect() as connection:
        allocation_count = (
            connection.execute(
                select(ProviderBudgetLeaseAllocation.id).where(
                    ProviderBudgetLeaseAllocation.lease_id == decision.lease.id
                )
            )
            .scalars()
            .all()
        )
        window_count = (
            connection.execute(
                select(ProviderBudgetWindow.id).where(
                    ProviderBudgetWindow.provider == "openai",
                    ProviderBudgetWindow.scope_fingerprint
                    == scheduler._fingerprint(tenant_id, scope),
                )
            )
            .scalars()
            .all()
        )
        lease_status = connection.execute(
            select(ProviderBudgetLease.status).where(
                ProviderBudgetLease.id == decision.lease.id
            )
        ).scalar_one()

    assert allocation_count == []
    assert window_count == []
    assert lease_status == "expired"
    assert scheduler.complete(decision.lease, actual_tokens=75) is False


def test_postgres_window_cleanup_is_bounded_and_preserves_active_windows(
    postgres_test_databases,
):
    """Cleanup frees only expired windows and retains active allocation ownership."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-cleanup")
    request = ReservationRequest(
        tenant_id,
        (
            BudgetPolicy(scope, BudgetMetric.REQUESTS, 10, 60),
            BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 300),
        ),
        estimated_tokens=80,
    )
    decision = scheduler.reserve(request)
    assert decision.lease is not None

    with databases.runtime_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE provider_budget_windows "
                "SET window_start = window_start - "
                "CASE WHEN window_seconds = 60 THEN INTERVAL '120 seconds' "
                "ELSE INTERVAL '600 seconds' END "
                "WHERE id IN (SELECT window_id FROM provider_budget_lease_allocations "
                "WHERE lease_id = :lease_id)"
            ),
            {"lease_id": decision.lease.id},
        )

    assert scheduler.cleanup(now=datetime.now(timezone.utc), batch_size=1) == 1
    with databases.runtime_engine.connect() as connection:
        status = connection.execute(
            select(ProviderBudgetLease.status).where(
                ProviderBudgetLease.id == decision.lease.id
            )
        ).scalar_one()
        allocations = (
            connection.execute(
                select(ProviderBudgetLeaseAllocation.id).where(
                    ProviderBudgetLeaseAllocation.lease_id == decision.lease.id
                )
            )
            .scalars()
            .all()
        )
        retained_window_count = (
            connection.execute(
                select(ProviderBudgetWindow.id).where(
                    ProviderBudgetWindow.id.in_(
                        select(ProviderBudgetLeaseAllocation.window_id).where(
                            ProviderBudgetLeaseAllocation.lease_id == decision.lease.id
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
    assert status == "active"
    assert len(allocations) == 1
    assert len(retained_window_count) == 1
    assert scheduler.complete(decision.lease, actual_tokens=30)


def test_postgres_window_cleanup_reclaims_only_expired_leases_in_batches(
    postgres_test_databases,
):
    """Expired windows are reclaimed in bounded batches without touching live buckets."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scopes = tuple(
        BudgetScope("openai", BudgetScopeKind.PROFILE, f"profile-batch-{index}")
        for index in range(3)
    )
    decisions = tuple(
        scheduler.reserve(_request(tenant_id, (scope,))) for scope in scopes
    )
    assert all(decision.lease is not None for decision in decisions)
    with databases.runtime_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE provider_budget_windows "
                "SET window_start = window_start - INTERVAL '120 seconds'"
            )
        )

    assert scheduler.cleanup(now=datetime.now(timezone.utc), batch_size=1) == 2
    with databases.runtime_engine.connect() as connection:
        remaining_window_ids = (
            connection.execute(select(ProviderBudgetWindow.id)).scalars().all()
        )
        statuses = (
            connection.execute(
                select(ProviderBudgetLease.status).order_by(ProviderBudgetLease.id)
            )
            .scalars()
            .all()
        )
    assert len(remaining_window_ids) == 2
    assert statuses.count("expired") == 1
    assert statuses.count("active") == 2
    assert scheduler.cleanup(now=datetime.now(timezone.utc), batch_size=1) == 2
    with databases.runtime_engine.connect() as connection:
        assert connection.execute(select(ProviderBudgetWindow.id)).scalars().all() == [
            remaining_window_ids[-1]
        ]
        statuses = (
            connection.execute(
                select(ProviderBudgetLease.status).order_by(ProviderBudgetLease.id)
            )
            .scalars()
            .all()
        )
    assert statuses.count("expired") == 2
    assert statuses.count("active") == 1
    assert scheduler.cleanup(now=datetime.now(timezone.utc), batch_size=1) == 2
    with databases.runtime_engine.connect() as connection:
        assert connection.execute(select(ProviderBudgetWindow.id)).scalars().all() == []
        statuses = (
            connection.execute(
                select(ProviderBudgetLease.status).order_by(ProviderBudgetLease.id)
            )
            .scalars()
            .all()
        )
    assert statuses == ["expired", "expired", "expired"]
    assert all(
        scheduler.complete(decision.lease, actual_tokens=0) is False
        for decision in decisions
        if decision.lease is not None
    )


def test_postgres_window_cleanup_does_not_remove_active_allocations(
    postgres_test_databases,
):
    """Current windows and their lease allocations are never removed by cleanup."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-active-cleanup")
    decision = scheduler.reserve(_request(tenant_id, (scope,)))
    assert decision.lease is not None

    assert scheduler.cleanup(now=datetime.now(timezone.utc), batch_size=10) == 0
    with databases.runtime_engine.connect() as connection:
        assert connection.execute(
            select(ProviderBudgetLeaseAllocation.id).where(
                ProviderBudgetLeaseAllocation.lease_id == decision.lease.id
            )
        ).scalar_one()
        assert (
            connection.execute(
                select(ProviderBudgetLease.status).where(
                    ProviderBudgetLease.id == decision.lease.id
                )
            ).scalar_one()
            == "active"
        )


def test_postgres_cleanup_removes_expired_cooldown_but_keeps_active_cooldown(
    postgres_test_databases,
):
    """Scope-state cleanup removes expired cooldowns only, bounded to the batch."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    expired_scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-expired")
    active_scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-active")
    now = datetime.now(timezone.utc)
    scheduler.apply_cooldown(expired_scope, 1, tenant_id=tenant_id, now=now)
    scheduler.apply_cooldown(active_scope, 60, tenant_id=tenant_id, now=now)

    with databases.runtime_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE provider_budget_scope_states "
                "SET cooldown_until = :expired "
                "WHERE scope_fingerprint = :fingerprint"
            ),
            {
                "expired": now - timedelta(seconds=1),
                "fingerprint": scheduler._fingerprint(tenant_id, expired_scope),
            },
        )

    scheduler.cleanup(now=now, batch_size=1)
    with databases.runtime_engine.connect() as connection:
        remaining_fingerprints = (
            connection.execute(select(ProviderBudgetScopeState.scope_fingerprint))
            .scalars()
            .all()
        )

    assert remaining_fingerprints == [scheduler._fingerprint(tenant_id, active_scope)]


def test_postgres_retry_after_cooldown_blocks_until_retry_time(
    postgres_test_databases,
):
    """A provider Retry-After is stored at the precise budget scope."""
    databases = postgres_test_databases
    databases.upgrade("head")
    tenant_id = f"scheduler-{uuid4()}"
    _create_tenant(databases, tenant_id)
    scheduler = _scheduler(databases)
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-cooldown")
    now = datetime.now(timezone.utc)

    retry_at = scheduler.apply_cooldown(scope, 17, tenant_id=tenant_id, now=now)
    decision = scheduler.reserve(
        _request(tenant_id, (scope,)),
        now=now + timedelta(seconds=16),
    )

    assert retry_at == now + timedelta(seconds=17)
    assert not decision.allowed
    assert decision.retry_at == retry_at
