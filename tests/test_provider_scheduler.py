"""Provider-neutral validation for persistent scheduler budgets."""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.persistence import provider_scheduler as provider_scheduler_module
from app.persistence.models import (
    Base,
    ProviderBudgetLease,
    ProviderBudgetLeaseAllocation,
    ProviderBudgetWindow,
    ProviderConcurrencyState,
    Tenant,
)
from app.persistence.provider_scheduler import (
    BudgetLease,
    BudgetMetric,
    BudgetPolicy,
    BudgetScope,
    BudgetScopeKind,
    ProviderBudgetScheduler,
    ProviderSchedulerStoreError,
    ReservationRequest,
    SharedBudgetIdentity,
    _adjust_aimd_limit,
    _capacity_score,
    _validate_aimd_bounds,
    _WindowClaim,
)
from app.persistence.secrets import SecretCipher
from app.providers.scheduler_config import parse_scheduler_limits


def test_existing_budget_window_adopts_changed_limit():
    """An active fixed window enforces an updated policy limit without 503."""
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-a")
    policy = BudgetPolicy(scope, BudgetMetric.REQUESTS, 20, 60)
    row = Mock(limit_units=10)
    session = Mock()
    session.scalar.return_value = row

    result = ProviderBudgetScheduler._lock_windows(
        session,
        "openai",
        (
            _WindowClaim(
                policy, "fingerprint", datetime(2026, 10, 6, tzinfo=timezone.utc), 1
            ),
        ),
    )

    assert result == [row]
    assert row.limit_units == 20


def test_scheduler_policy_can_resolve_scope_id_from_runtime_context():
    """A policy can omit a profile ID unavailable before its first save."""
    policies = parse_scheduler_limits(
        '[{"scope_kind":"profile","metric":"requests",'
        '"limit":12,"window_seconds":60}]'
    )

    assert policies == [
        {
            "scope_kind": "profile",
            "metric": "requests",
            "limit": 12,
            "window_seconds": 60,
        }
    ]


@pytest.mark.parametrize(
    ("initial", "minimum", "maximum"),
    [(4, 1, 64), (1, 1, 1), (64, 1, 64), (12, 8, 16)],
)
def test_aimd_scope_bounds_accept_valid_limits(initial, minimum, maximum):
    """A profile or target-model scope can configure inclusive AIMD bounds."""
    assert _validate_aimd_bounds(initial, minimum, maximum) == (
        initial,
        minimum,
        maximum,
    )


@pytest.mark.parametrize(
    ("initial", "minimum", "maximum"),
    [(4, 0, 4), (4, 5, 8), (9, 1, 8), (True, 1, 8), (4, 1, 65)],
)
def test_aimd_scope_bounds_reject_invalid_limits(initial, minimum, maximum):
    """Invalid or inconsistent concurrency bounds cannot reach persistence."""
    with pytest.raises(ValueError):
        _validate_aimd_bounds(initial, minimum, maximum)


@pytest.mark.parametrize(
    (
        "profile_active",
        "profile_limit",
        "model_active",
        "model_limit",
        "weight",
        "budget_headroom",
        "expected",
    ),
    [
        (0, 4, 0, 4, 1.0, 0.5, 0.25),
        (2, 4, 1, 4, 1.0, 1.0, 0.5),
        (0, 4, 2, 4, 2.0, 1.0, 0.25),
        (1, 4, 0, 4, 2.0, 1.0, 0.125),
    ],
)
def test_capacity_score_accounts_for_profile_model_weight_and_headroom(
    profile_active,
    profile_limit,
    model_active,
    model_limit,
    weight,
    budget_headroom,
    expected,
):
    """Capacity score combines active ratios, weight, and budget headroom."""
    assert (
        _capacity_score(
            profile_active,
            profile_limit,
            model_active,
            model_limit,
            weight,
            budget_headroom=budget_headroom,
            headroom_weight=0.5,
        )
        == expected
    )


def test_aimd_limit_increases_additively_and_decreases_multiplicatively():
    """AIMD advances one slot on success and halves transiently on overload."""
    assert _adjust_aimd_limit(4, outcome="success") == 5
    assert _adjust_aimd_limit(4, outcome="transient_failure") == 2
    assert _adjust_aimd_limit(1, outcome="transient_failure") == 1
    assert _adjust_aimd_limit(64, outcome="success") == 64
    assert _adjust_aimd_limit(4, outcome="terminal_failure") == 4


@pytest.mark.parametrize(
    ("current", "outcome", "minimum", "maximum", "expected"),
    [
        (4, "success", 2, 5, 5),
        (4, "success", 2, 4, 4),
        (3, "transient_failure", 3, 8, 3),
        (7, "transient_failure", 2, 8, 3),
    ],
)
def test_aimd_limit_respects_configured_scope_bounds(
    current, outcome, minimum, maximum, expected
):
    """Each profile/model AIMD learner stays within its configured bounds."""
    assert (
        _adjust_aimd_limit(
            current,
            outcome=outcome,
            minimum=minimum,
            maximum=maximum,
        )
        == expected
    )


def test_aimd_limit_rejects_current_value_outside_configured_bounds():
    """Persisted bounds cannot silently legitimize an out-of-range capacity."""
    with pytest.raises(ValueError, match="supported range"):
        _adjust_aimd_limit(4, outcome="success", minimum=5, maximum=8)


def test_aimd_release_keeps_limit_unchanged():
    """Releasing a permit without upstream I/O must not change capacity."""
    assert _adjust_aimd_limit(4, outcome="released") == 4


def test_aimd_state_schema_persists_configured_minimum_and_maximum():
    """Persisted adaptive states retain their distinct configured bounds."""
    assert ProviderConcurrencyState.__table__.columns["minimum_limit"].nullable is False
    assert ProviderConcurrencyState.__table__.columns["maximum_limit"].nullable is False
    assert any(
        constraint.name == "ck_concurrency_bounds"
        for constraint in ProviderConcurrencyState.__table__.constraints
    )


def test_new_persistent_migration_revisions_fit_alembic_version_column():
    """New Alembic revision IDs do not exceed the default version column."""
    migration_dir = Path(__file__).resolve().parents[1] / "migrations" / "versions"
    for migration_name in (
        "20261013_provider_budget_scheduler.py",
        "20261014_batch_jobs.py",
        "20261015_provider_concurrency_aimd.py",
    ):
        source = (migration_dir / migration_name).read_text(encoding="utf-8")
        revision = re.search(r'^revision: str = "([^"]+)"$', source, re.MULTILINE)
        assert revision is not None
        assert len(revision.group(1)) <= 32


def test_scopes_and_policies_represent_profile_provider_model_org_project_windows():
    """Multiple explicit limits cover each supported identity and time window."""
    scopes = tuple(
        BudgetScope("openai", kind, f"scope-{kind.value}") for kind in BudgetScopeKind
    )
    policies = tuple(
        BudgetPolicy(scope, BudgetMetric.REQUESTS, 12, 60) for scope in scopes
    ) + (
        BudgetPolicy(scopes[2], BudgetMetric.TOKENS, 120_000, 60),
        BudgetPolicy(scopes[3], BudgetMetric.TOKENS, 2_000_000, 86_400),
    )

    request = ReservationRequest("tenant-a", policies, estimated_tokens=1_500)

    assert set(request.scopes) == set(scopes)
    assert len(request.policies) == 7
    assert {policy.window_seconds for policy in request.policies} == {60, 86_400}


def test_policy_rejects_unconfigured_or_invalid_capacity_values():
    """Only explicit positive integer limits and windows are accepted."""
    scope = BudgetScope("openai", BudgetScopeKind.MODEL, "gpt-model")

    for limit, window_seconds in ((0, 60), (-1, 60), (10, 0), (10, -60)):
        with pytest.raises(ValueError):
            BudgetPolicy(scope, BudgetMetric.REQUESTS, limit, window_seconds)

    with pytest.raises(ValueError):
        BudgetPolicy(scope, BudgetMetric.REQUESTS, 1.5, 60)


def test_scheduler_policy_rejects_values_that_exceed_postgresql_integer_range():
    """Accepted limits and windows must fit the persisted PostgreSQL INTEGER."""
    too_large = 2**31
    policies = (
        {
            "scope_kind": "profile",
            "metric": "requests",
            "limit": too_large,
            "window_seconds": 60,
        },
        {
            "scope_kind": "profile",
            "metric": "requests",
            "limit": 10,
            "window_seconds": too_large,
        },
    )
    for policy in policies:
        with pytest.raises(ValueError, match="32-bit PostgreSQL INTEGER"):
            parse_scheduler_limits([policy])

    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-large")
    with pytest.raises(ValueError, match="32-bit PostgreSQL INTEGER"):
        BudgetPolicy(scope, BudgetMetric.TOKENS, too_large, 60)
    with pytest.raises(ValueError, match="32-bit PostgreSQL INTEGER"):
        BudgetPolicy(scope, BudgetMetric.REQUESTS, 10, too_large)


def test_token_policy_requires_nonnegative_integer_estimate():
    """Token limits cannot be bypassed with unknown or malformed estimates."""
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-a")
    policy = BudgetPolicy(scope, BudgetMetric.TOKENS, 100, 60)

    for estimate in (None, -1, 1.5):
        with pytest.raises(ValueError, match="estimated_tokens"):
            ReservationRequest("tenant-a", (policy,), estimated_tokens=estimate)


def test_request_rejects_mixed_provider_scopes_and_duplicate_windows():
    """A reservation maps to one provider candidate and one policy per window key."""
    openai = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-a")
    azure = BudgetScope("azure", BudgetScopeKind.MODEL, "gpt-model")
    with pytest.raises(ValueError, match="provider"):
        ReservationRequest(
            "tenant-a",
            (
                BudgetPolicy(openai, BudgetMetric.REQUESTS, 10, 60),
                BudgetPolicy(azure, BudgetMetric.REQUESTS, 10, 60),
            ),
        )

    duplicate = BudgetPolicy(openai, BudgetMetric.REQUESTS, 10, 60)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        ReservationRequest(
            "tenant-a",
            (duplicate, duplicate),
        )


def test_provider_scope_fingerprints_share_provider_identity_across_tenants():
    """Identical provider/project scopes coordinate globally without storing IDs."""
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())

    first = cipher.provider_budget_fingerprint("openai", "project", "project-a")
    second = cipher.provider_budget_fingerprint("openai", "project", "project-a")
    other = cipher.provider_budget_fingerprint("openai", "project", "project-b")

    assert first == second
    assert first != other
    assert "project-a" not in first


def test_unvalidated_organization_names_are_tenant_scoped_by_default():
    """Locally supplied organization labels never create accidental global budgets."""
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    scheduler = ProviderBudgetScheduler(sessionmaker(), cipher)
    first = BudgetScope("openai", BudgetScopeKind.ORGANIZATION, "Acme")
    second = BudgetScope("openai", BudgetScopeKind.ORGANIZATION, "Acme")

    assert scheduler._fingerprint("tenant-a", first) != scheduler._fingerprint(
        "tenant-b", second
    )


def test_explicit_shared_identity_coordinates_canonical_organization_across_tenants():
    """Only explicit trusted canonical identities opt organization budgets into sharing."""
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    scheduler = ProviderBudgetScheduler(sessionmaker(), cipher)
    first = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "local display label",
        shared_identity=SharedBudgetIdentity("provider-org:org-123"),
    )
    second = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "different local label",
        shared_identity=SharedBudgetIdentity("provider-org:org-123"),
    )

    assert scheduler._fingerprint("tenant-a", first) == scheduler._fingerprint(
        "tenant-b", second
    )
    assert "org-123" not in scheduler._fingerprint("tenant-a", first)


def test_non_shared_scope_cannot_set_shared_identity():
    """Tenant-local provider/profile/model keys cannot opt into global sharing."""
    with pytest.raises(ValueError, match="organization or project"):
        BudgetScope(
            "openai",
            BudgetScopeKind.PROFILE,
            "profile-local",
            shared_identity=SharedBudgetIdentity("untrusted"),
        )


def test_request_rejects_duplicate_policies_for_one_shared_canonical_scope():
    """Different labels cannot produce duplicate allocations for one shared key."""
    first = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "local-label-a",
        shared_identity=SharedBudgetIdentity("canonical:org-7"),
    )
    second = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "local-label-b",
        shared_identity=SharedBudgetIdentity("canonical:org-7"),
    )

    with pytest.raises(ValueError, match="Duplicate"):
        ReservationRequest(
            "tenant-a",
            (
                BudgetPolicy(first, BudgetMetric.REQUESTS, 10, 60),
                BudgetPolicy(second, BudgetMetric.REQUESTS, 10, 60),
            ),
        )


def test_shared_scope_fingerprints_cover_each_policy_label():
    """Distinct local labels for one shared scope both resolve to one fingerprint."""
    first = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "local-label-a",
        shared_identity=SharedBudgetIdentity("canonical:org-9"),
    )
    second = BudgetScope(
        "openai",
        BudgetScopeKind.ORGANIZATION,
        "local-label-b",
        shared_identity=SharedBudgetIdentity("canonical:org-9"),
    )
    cipher = SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode())
    scheduler = ProviderBudgetScheduler(sessionmaker(), cipher)

    fingerprints = scheduler._fingerprints("tenant-a", (first, second))

    assert fingerprints[first] == fingerprints[second]


def test_shared_scope_is_deduplicated_across_request_metrics():
    """One canonical cross-tenant scope is locked once for its request metrics."""
    scopes = (
        BudgetScope(
            "openai",
            BudgetScopeKind.ORGANIZATION,
            "local-label-a",
            shared_identity=SharedBudgetIdentity("canonical:org-9"),
        ),
        BudgetScope(
            "openai",
            BudgetScopeKind.ORGANIZATION,
            "local-label-b",
            shared_identity=SharedBudgetIdentity("canonical:org-9"),
        ),
    )
    request = ReservationRequest(
        "tenant-a",
        (
            BudgetPolicy(scopes[0], BudgetMetric.REQUESTS, 10, 60),
            BudgetPolicy(scopes[1], BudgetMetric.TOKENS, 100, 60),
        ),
        estimated_tokens=10,
    )

    assert len(request.scopes) == 1


def test_budget_policy_rejects_multiple_windows_for_same_scope_and_metric():
    """A request cannot bypass one configured metric policy with parallel buckets."""
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-a")
    with pytest.raises(ValueError, match="window_seconds"):
        ReservationRequest(
            "tenant-a",
            (
                BudgetPolicy(scope, BudgetMetric.REQUESTS, 10, 60),
                BudgetPolicy(scope, BudgetMetric.REQUESTS, 10, 300),
            ),
        )


def test_cleanup_rejects_unbounded_or_nonpositive_batch_sizes():
    """Scheduler retention is always explicit and batch limited."""
    scheduler = ProviderBudgetScheduler(sessionmaker())
    for batch_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            scheduler.cleanup(batch_size=batch_size)


def test_failed_settlement_charges_request_and_releases_token_estimate(monkeypatch):
    """Failed upstream calls consume request quota without charging estimated tokens."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            Tenant.__table__,
            ProviderBudgetLease.__table__,
            ProviderBudgetWindow.__table__,
            ProviderBudgetLeaseAllocation.__table__,
        ],
    )
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = "tenant-failed-settlement"
    lease_id = "lease-failed-settlement"
    window_start = datetime.fromtimestamp(0, timezone.utc)
    window_seconds = 2_000_000_000
    request_window = ProviderBudgetWindow(
        provider="openai",
        scope_kind="profile",
        scope_fingerprint="request-fingerprint",
        metric="requests",
        window_seconds=window_seconds,
        window_start=window_start,
        limit_units=1,
        used_units=0,
        reserved_units=1,
    )
    token_window = ProviderBudgetWindow(
        provider="openai",
        scope_kind="profile",
        scope_fingerprint="token-fingerprint",
        metric="tokens",
        window_seconds=window_seconds,
        window_start=window_start,
        limit_units=100,
        used_units=0,
        reserved_units=80,
    )
    with sessions.begin() as session:
        session.add(
            Tenant(
                id=tenant_id,
                api_key_hash="tenant-hash",
                custom_model_id="tenant-model",
            )
        )
        session.add_all([request_window, token_window])
        session.flush()
        session.add(
            ProviderBudgetLease(
                id=lease_id,
                tenant_id=tenant_id,
                provider="openai",
                lease_token="lease-token",
                status="active",
                created_at=datetime.now(timezone.utc),
            )
        )
        session.flush()
        session.add_all(
            [
                ProviderBudgetLeaseAllocation(
                    lease_id=lease_id,
                    window_id=request_window.id,
                    reserved_units=1,
                ),
                ProviderBudgetLeaseAllocation(
                    lease_id=lease_id,
                    window_id=token_window.id,
                    reserved_units=80,
                ),
            ]
        )

    monkeypatch.setattr(
        provider_scheduler_module, "_require_postgresql", lambda _: None
    )
    scheduler = ProviderBudgetScheduler(sessions)
    lease = BudgetLease(lease_id, tenant_id, "openai", "lease-token")

    assert scheduler.failed(lease)
    with sessions() as session:
        assert session.get(ProviderBudgetLease, lease_id).status == "failed"
        settled_windows = session.scalars(
            select(ProviderBudgetWindow).order_by(ProviderBudgetWindow.metric)
        ).all()

    assert [
        (window.metric, window.used_units, window.reserved_units)
        for window in settled_windows
    ] == [("requests", 1, 0), ("tokens", 0, 0)]
    engine.dispose()


def test_sqlite_is_rejected_instead_of_used_as_a_scheduler_fallback():
    """Non-PostgreSQL persistence is rejected before any state is created."""
    engine = create_engine("sqlite:///:memory:")
    scheduler = ProviderBudgetScheduler(sessionmaker(bind=engine))
    scope = BudgetScope("openai", BudgetScopeKind.PROFILE, "profile-a")
    request = ReservationRequest(
        "tenant-a", (BudgetPolicy(scope, BudgetMetric.REQUESTS, 5, 60),)
    )

    with pytest.raises(ProviderSchedulerStoreError, match="PostgreSQL"):
        scheduler.reserve(request)

    engine.dispose()
