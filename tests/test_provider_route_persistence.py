"""Contracts for tenant-local ordered provider routing persistence."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.admin.view_models import dashboard_view
from app.persistence import admin_ops
from app.persistence.database import Database
from app.persistence.models import (
    AuditEvent,
    Base,
    ProviderCatalogEntry,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from app.persistence.repositories import TenantRepository
from app.persistence.secrets import SecretCipher
from app.tenants import hash_api_key


@pytest.fixture
def route_database():
    """Real portable schema, transactions, catalogs, and encrypted credentials."""
    engine = create_engine("sqlite:///:memory:")
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    database = Database(
        engine,
        sessionmaker(bind=engine, expire_on_commit=False),
        SecretCipher.from_key(base64.urlsafe_b64encode(b"k" * 32).decode("ascii")),
    )
    with database.sessions.begin() as session:
        session.add_all(
            Tenant(
                id=name,
                api_key_hash=hash_api_key(name),
                custom_model_id=f"cursor-{name}",
            )
            for name in ("acme", "other")
        )
    yield database
    engine.dispose()


def create_ready_profile(
    session, database, name, *, provider="openai", tenant_id="acme"
):
    """Create a catalog-verified but inactive account through public mutations."""
    settings = (
        {"base_url": "https://resource.openai.azure.com"} if provider == "azure" else {}
    )
    profile = admin_ops.create_provider_profile(
        session,
        database.secret_cipher,
        tenant_id,
        provider,
        name,
        settings,
        "gpt-5.4",
        f"secret-{name}",
        "ada",
    )
    entries = [("gpt-5.4", "deployment-a" if provider == "azure" else None)]
    admin_ops.replace_catalog_entries(session, profile, entries, None)
    if provider == "azure":
        profile.settings = {
            **profile.settings,
            "model_deployments": {"gpt-5.4": "deployment-a"},
        }
    return profile


def routed_profiles(session, tenant_id="acme"):
    """Read the persisted route, not derived UI state."""
    return list(
        session.scalars(
            select(ProviderProfile)
            .where(
                ProviderProfile.tenant_id == tenant_id,
                ProviderProfile.route_priority.is_not(None),
            )
            .order_by(ProviderProfile.route_priority)
        )
    )


def test_route_schema_replaces_tenant_pointer_with_nullable_positive_priority():
    """The route has exactly one tenant-local ordering source of truth."""
    assert "active_profile_id" not in Tenant.__table__.columns
    column = ProviderProfile.__table__.columns["route_priority"]
    assert column.nullable
    assert column.type.python_type is int


def test_tenant_schema_has_typed_routing_defaults():
    """Tenant routing strategy and policies have safe persisted defaults."""
    expected_types = {
        "routing_strategy": str,
        "routing_load_balancing_method": str,
        "routing_cost_policy": str,
        "routing_headroom_weight": float,
        "routing_max_retry_wait_seconds": int,
        "routing_tie_breaker": str,
    }
    for name, python_type in expected_types.items():
        column = Tenant.__table__.columns[name]
        assert column.type.python_type is python_type
        assert column.nullable is False
        assert column.server_default is not None


@pytest.mark.parametrize("invalid_priority", [0, -1])
def test_route_priority_rejects_nonpositive_values(route_database, invalid_priority):
    """Database constraints reject zero and negative route positions."""
    with route_database.sessions.begin() as session:
        profile = create_ready_profile(session, route_database, "Invalid")
        profile.route_priority = invalid_priority
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()


def test_route_priority_allows_equal_groups_within_a_tenant(route_database):
    """Repeated positive priorities are valid grouping keys within one tenant."""
    constraints = [
        constraint
        for constraint in ProviderProfile.__table__.constraints
        if getattr(constraint, "name", None) == "uq_profile_tenant_route_priority"
    ]
    assert constraints == []

    with route_database.sessions.begin() as session:
        one = create_ready_profile(session, route_database, "One")
        two = create_ready_profile(session, route_database, "Two")
        peer = create_ready_profile(session, route_database, "Peer", tenant_id="other")
        one.route_priority = two.route_priority = peer.route_priority = 1
        session.flush()
        assert [profile.route_priority for profile in (one, two, peer)] == [1, 1, 1]


def test_activation_appends_and_reactivation_preserves_position_and_audits(
    route_database,
):
    """Activation appends once and records its stable route position."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        one = create_ready_profile(session, route_database, "One")
        two = create_ready_profile(
            session, route_database, "Two", provider="openrouter"
        )
        assert one.route_priority is None
        admin_ops.activate_provider_profile(session, tenant, one.id, "ada")
        admin_ops.activate_provider_profile(session, tenant, two.id, "ada")
        admin_ops.activate_provider_profile(session, tenant, one.id, "ada")
        assert [(p.id, p.route_priority) for p in routed_profiles(session)] == [
            (one.id, 1),
            (two.id, 2),
        ]
        events = list(
            session.scalars(
                select(AuditEvent).where(AuditEvent.action == "profile.activate")
            )
        )
        assert len(events) == 3
        assert events[-1].details["route_priority"] == 1
        assert "secret-" not in repr([e.details for e in events])


def test_catalog_refresh_prunes_stale_model_concurrency_and_audits(route_database):
    """Retain valid model limits and audit limits removed with catalog entries."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profile = create_ready_profile(session, route_database, "Catalog limits")
        admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        profile.settings = {
            "routing": {
                "model_concurrency": {
                    "gpt-5.4": {"initial": 3, "min": 1, "max": 8},
                    "removed-model": {"initial": 2, "min": 1, "max": 4},
                }
            }
        }

        admin_ops.replace_catalog_entries(session, profile, [("gpt-5.4", None)], None)

        assert profile.settings["routing"]["model_concurrency"] == {
            "gpt-5.4": {"initial": 3, "min": 1, "max": 8}
        }
        assert profile.route_priority == 1
        event = session.scalar(
            select(AuditEvent).where(
                AuditEvent.action == "profile.model_routing_pruned"
            )
        )
        assert event is not None
        assert event.actor_id == "system:catalog"
        assert event.details["model_ids"] == ["removed-model"]


def test_equal_priority_groups_can_be_set_and_reordered_without_splitting(
    route_database,
):
    """Admin route changes retain group membership and compact whole groups."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profiles = [
            create_ready_profile(session, route_database, name)
            for name in ("One", "Two", "Three")
        ]
        for profile in profiles:
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        admin_ops.set_provider_profile_priority(
            session, tenant, profiles[1].id, 1, "ada"
        )
        assert [profile.route_priority for profile in profiles] == [1, 1, 2]

        admin_ops.reorder_provider_profile(
            session, tenant, profiles[0].id, "down", "ada"
        )

        assert [profile.route_priority for profile in profiles] == [2, 2, 1]
        admin_ops.set_provider_profile_priority(
            session, tenant, profiles[2].id, None, "ada"
        )
        assert [profile.route_priority for profile in profiles] == [1, 1, None]


def test_reorder_uses_collision_free_contiguous_swap_and_audit(route_database):
    """Moves retain unique contiguous priorities and an audit trail."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profiles = [
            create_ready_profile(session, route_database, name)
            for name in ("One", "Two", "Three")
        ]
        for profile in profiles:
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        admin_ops.reorder_provider_profile(session, tenant, profiles[2].id, "up", "ada")
        assert [(p.id, p.route_priority) for p in routed_profiles(session)] == [
            (profiles[0].id, 1),
            (profiles[2].id, 2),
            (profiles[1].id, 3),
        ]
        admin_ops.reorder_provider_profile(
            session, tenant, profiles[0].id, "down", "ada"
        )
        assert [p.id for p in routed_profiles(session)] == [
            profiles[2].id,
            profiles[0].id,
            profiles[1].id,
        ]
        assert (
            session.scalar(
                select(AuditEvent).where(AuditEvent.action == "profile.reorder")
            )
            is not None
        )


@pytest.mark.parametrize("direction", ["up", "down"])
def test_reorder_at_route_boundary_is_stable(route_database, direction):
    """The first and last route positions cannot move beyond a boundary."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profile = create_ready_profile(session, route_database, "Only")
        admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        admin_ops.reorder_provider_profile(
            session, tenant, profile.id, direction, "ada"
        )
        assert profile.route_priority == 1


def test_route_mutations_reject_foreign_missing_inactive_and_invalid_direction(
    route_database,
):
    """Tenant authorization and direction validation precede mutations."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        foreign = create_ready_profile(
            session, route_database, "Foreign", tenant_id="other"
        )
        inactive = create_ready_profile(session, route_database, "Inactive")
        with pytest.raises(LookupError):
            admin_ops.activate_provider_profile(session, tenant, foreign.id, "ada")
        with pytest.raises(LookupError):
            admin_ops.deactivate_provider_profile(session, tenant, "missing", "ada")
        with pytest.raises(LookupError):
            admin_ops.deactivate_provider_profile(session, tenant, foreign.id, "ada")
        with pytest.raises(ValueError, match="inactive"):
            admin_ops.reorder_provider_profile(
                session, tenant, inactive.id, "up", "ada"
            )
        with pytest.raises(ValueError, match="direction"):
            admin_ops.reorder_provider_profile(
                session, tenant, inactive.id, "sideways", "ada"
            )


def test_deactivation_compacts_route_without_erasing_credentials(route_database):
    """Routing removal preserves saved account credentials."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profiles = [
            create_ready_profile(session, route_database, name)
            for name in ("One", "Two", "Three")
        ]
        for profile in profiles:
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        admin_ops.deactivate_provider_profile(session, tenant, profiles[1].id, "ada")
        assert profiles[1].route_priority is None
        assert (
            route_database.secret_cipher.decrypt(
                profiles[1].inference_secret_ciphertext
            )
            == "secret-Two"
        )
        assert [(p.id, p.route_priority) for p in routed_profiles(session)] == [
            (profiles[0].id, 1),
            (profiles[2].id, 2),
        ]
        assert (
            session.scalar(
                select(AuditEvent).where(AuditEvent.action == "profile.deactivate")
            )
            is not None
        )


def test_delete_compacts_only_affected_route(route_database):
    """Soft deletion clears credentials and removes only one route member."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        one = create_ready_profile(session, route_database, "One")
        two = create_ready_profile(session, route_database, "Two")
        for profile in (one, two):
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        admin_ops.delete_provider_profile(session, tenant.id, one.id, "ada")
        assert one.route_priority is None
        assert one.deleted_at is not None
        assert one.inference_secret_ciphertext is None
        assert two.route_priority == 1


def test_endpoint_change_deactivates_only_changed_profile(route_database):
    """A new Azure endpoint invalidates its catalog and existing route."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        one = create_ready_profile(session, route_database, "Azure", provider="azure")
        two = create_ready_profile(session, route_database, "OpenAI")
        for profile in (one, two):
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        admin_ops.update_provider_profile(
            session,
            route_database.secret_cipher,
            tenant.id,
            one.id,
            one.display_name,
            {"base_url": "https://new-resource.openai.azure.com"},
            "gpt-5.4",
            None,
            "ada",
        )
        assert one.route_priority is None
        assert one.default_model is None
        assert two.route_priority == 1
        assert not list(
            session.scalars(
                select(ProviderCatalogEntry).where(
                    ProviderCatalogEntry.profile_id == one.id
                )
            )
        )


def test_refresh_without_target_deactivates_without_inventing_default_model(
    route_database,
):
    """Disappearing targets deactivate rather than replacing the chosen model."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        one = create_ready_profile(session, route_database, "One")
        two = create_ready_profile(session, route_database, "Two")
        for profile in (one, two):
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        admin_ops.replace_catalog_entries(session, one, [("gpt-5.5", None)], None)
        assert one.route_priority is None
        assert one.default_model == "gpt-5.4"
        assert two.route_priority == 1
        audit = session.scalar(
            select(AuditEvent).where(AuditEvent.action == "profile.deactivate")
        )
        assert audit.details["reason"] == "default_model_unavailable"


def test_failed_catalog_refresh_preserves_valid_route_and_catalog(route_database):
    """Transient catalog errors retain the last successful target validation."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profile = create_ready_profile(session, route_database, "One")
        admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        admin_ops.replace_catalog_entries(session, profile, [], "provider unavailable")
        assert profile.route_priority == 1
        assert (
            session.scalar(
                select(ProviderCatalogEntry.model_id).where(
                    ProviderCatalogEntry.profile_id == profile.id
                )
            )
            == "gpt-5.4"
        )


def test_routing_snapshot_parses_tenant_and_profile_routing_settings(route_database):
    """The snapshot exposes validated typed tenant and profile routing policies."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        tenant.routing_strategy = "load_balanced"
        tenant.routing_cost_policy = "cost_tiers"
        profile = create_ready_profile(session, route_database, "Weighted")
        profile.route_priority = 1
        profile.settings = {
            **profile.settings,
            "routing": {
                "cost_tier": 2,
                "load_balancing_weight": 1.5,
                "profile_concurrency": {"initial": 6, "min": 2, "max": 20},
                "model_concurrency": {"gpt-5.4": {"initial": 3, "min": 1, "max": 8}},
            },
        }

    snapshot = route_database.get_proxy_snapshot_by_api_key("acme")

    assert snapshot.routing_settings.strategy == "load_balanced"
    assert snapshot.routing_settings.cost_policy == "cost_tiers"
    assert snapshot.profiles[0].routing_settings.cost_tier == 2
    assert snapshot.profiles[0].routing_settings.load_balancing_weight == 1.5
    assert snapshot.profiles[0].routing_settings.profile_concurrency.maximum == 20
    assert (
        snapshot.profiles[0].routing_settings.model_concurrency["gpt-5.4"].initial == 3
    )


def test_routing_snapshot_uses_one_statement_ordered_catalog_and_immutable_profile_dtos(
    route_database,
):
    """One query freezes route identity and configuration without leaking secrets."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        one = create_ready_profile(session, route_database, "One", provider="azure")
        two = create_ready_profile(session, route_database, "Two")
        create_ready_profile(session, route_database, "Inactive")
        for profile in (two, one):
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        session.add(
            ProviderCatalogEntry(
                profile_id=two.id, model_id="gpt-5.5", source="provider"
            )
        )
        ids = [two.id, one.id]
    statements = []

    def record(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(route_database.engine, "before_cursor_execute", record)
    try:
        snapshot = route_database.get_proxy_snapshot_by_api_key("acme")
    finally:
        event.remove(route_database.engine, "before_cursor_execute", record)
    assert len(statements) == 1
    assert snapshot.id == "acme"
    assert isinstance(snapshot.profiles, tuple)
    assert [p.profile_id for p in snapshot.profiles] == ids
    assert [p.profile_name for p in snapshot.available_profiles] == [
        "Two",
        "One",
        "Inactive",
    ]
    assert snapshot.profiles[1].azure_model_deployments == {"gpt-5.4": "deployment-a"}
    assert snapshot.profiles[1].catalog_model_ids == ("gpt-5.4",)
    assert snapshot.profiles[1].default_model == "gpt-5.4"
    assert "secret-One" not in repr(snapshot)
    assert "secret-Two" not in repr(snapshot)
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        admin_ops.reorder_provider_profile(session, tenant, ids[0], "down", "ada")
        session.get(ProviderProfile, ids[0]).settings = {"project": "changed"}
    assert [p.profile_id for p in snapshot.profiles] == ids
    assert snapshot.profiles[0].provider_settings == {}


@pytest.mark.parametrize(
    "invalid",
    ["catalog", "default", "secret", "settings", "deployment", "routing", "deleted"],
)
def test_snapshot_excludes_invalid_routed_profiles_without_default_fallback(
    route_database, invalid
):
    """Invalid settings, catalogs, secrets, or tombstones never become route DTOs."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        bad = create_ready_profile(session, route_database, "Bad", provider="azure")
        good = create_ready_profile(session, route_database, "Good")
        for profile in (bad, good):
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        if invalid == "catalog":
            session.get(
                ProviderCatalogEntry,
                session.scalar(
                    select(ProviderCatalogEntry.id).where(
                        ProviderCatalogEntry.profile_id == bad.id
                    )
                ),
            ).model_id = "unsupported"
        elif invalid == "default":
            bad.default_model = "unavailable"
        elif invalid == "secret":
            bad.inference_secret_ciphertext = None
        elif invalid == "settings":
            bad.settings = {"base_url": "http://invalid.example"}
        elif invalid == "deployment":
            bad.settings = {**bad.settings, "model_deployments": {"gpt-5.4": "stale"}}
        elif invalid == "routing":
            bad.settings = {**bad.settings, "routing": {"load_balancing_weight": 0}}
        else:
            bad.deleted_at = datetime.now(timezone.utc)
        good_id = good.id
    snapshot = route_database.get_proxy_snapshot_by_api_key("acme")
    assert [p.profile_id for p in snapshot.profiles] == [good_id]
    assert snapshot.profiles[0].default_model == "gpt-5.4"


def test_snapshot_empty_route_is_authenticated_and_unknown_key_is_not(route_database):
    """A known tenant with an empty route is distinct from failed authentication."""
    snapshot = route_database.get_proxy_snapshot_by_api_key("acme")
    assert snapshot.id == "acme"
    assert snapshot.profiles == ()
    assert route_database.get_proxy_snapshot_by_api_key("unknown") is None


def test_admin_dashboard_separates_ready_profiles_from_default_route(route_database):
    """An account can be ready for direct model use without joining the route."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        one = create_ready_profile(session, route_database, "One")
        two = create_ready_profile(
            session, route_database, "Two", provider="openrouter"
        )
        direct = create_ready_profile(
            session, route_database, "Direct", provider="deepseek"
        )
        for profile in (two, one):
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        resolved, primary = TenantRepository(session).get_admin_snapshot("acme")
        assert resolved.id == tenant.id
        assert primary.id == two.id
        dashboard = dashboard_view(
            tenant,
            (one, two, direct),
            catalog_model_ids_by_profile={
                profile.id: (profile.default_model,) for profile in (one, two, direct)
            },
        )
        assert {p.provider for p in dashboard.providers if p.is_routed} == {
            "openai",
            "openrouter",
        }
        direct_status = next(
            status for status in dashboard.providers if status.profile_name == "Direct"
        )
        assert direct_status.is_active
        assert not direct_status.is_routed


def test_route_mutations_hold_tenant_lock_and_flush_null_phase(route_database):
    """Every mutation locks the tenant; swaps expose no partial route outside the transaction."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        one = create_ready_profile(session, route_database, "One")
        two = create_ready_profile(session, route_database, "Two")
        locks = []
        phases = []

        def capture_statement(state):
            statement = state.statement
            if getattr(statement, "_for_update_arg", None) is not None:
                locks.append(statement.column_descriptions[0]["entity"])

        def capture_flush(_session, _context):
            phases.append((one.route_priority, two.route_priority))

        event.listen(session, "do_orm_execute", capture_statement)
        event.listen(session, "after_flush", capture_flush)
        admin_ops.activate_provider_profile(session, tenant, one.id, "ada")
        admin_ops.activate_provider_profile(session, tenant, two.id, "ada")
        phases.clear()
        admin_ops.reorder_provider_profile(session, tenant, one.id, "down", "ada")
        assert (None, None) in phases
        assert (2, 1) in phases
        admin_ops.deactivate_provider_profile(session, tenant, one.id, "ada")
        assert locks.count(Tenant) == 4
        event.remove(session, "do_orm_execute", capture_statement)
        event.remove(session, "after_flush", capture_flush)


def test_route_mutation_rollback_restores_priorities_and_audit(route_database):
    """The NULL phase and audit are rolled back with the whole route mutation."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        one = create_ready_profile(session, route_database, "One")
        two = create_ready_profile(session, route_database, "Two")
        for profile in (one, two):
            admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        ids = [one.id, two.id]
    with pytest.raises(RuntimeError, match="abort"):
        with route_database.sessions.begin() as session:
            admin_ops.reorder_provider_profile(
                session, session.get(Tenant, "acme"), ids[1], "up", "ada"
            )
            raise RuntimeError("abort transaction")
    with route_database.sessions() as session:
        assert [(p.id, p.route_priority) for p in routed_profiles(session)] == [
            (ids[0], 1),
            (ids[1], 2),
        ]
        assert (
            session.scalar(
                select(AuditEvent).where(AuditEvent.action == "profile.reorder")
            )
            is None
        )


def test_legacy_upsert_rejects_ambiguous_provider_and_validates_routed_target(
    route_database,
):
    """The legacy save path cannot overwrite a routed model or an arbitrary peer."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profile = create_ready_profile(session, route_database, "One")
        admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        with pytest.raises(ValueError, match="not in its available catalog"):
            admin_ops.upsert_provider_profile(
                session,
                route_database.secret_cipher,
                "acme",
                "openai",
                {},
                "unavailable",
                None,
                "ada",
            )
        assert profile.default_model == "gpt-5.4"
        create_ready_profile(session, route_database, "Two")
        with pytest.raises(ValueError, match="ambiguous"):
            admin_ops.upsert_provider_profile(
                session,
                route_database.secret_cipher,
                "acme",
                "openai",
                {},
                "gpt-5.4",
                None,
                "ada",
            )


def test_bound_profile_deactivation_preserves_billing_ownership_and_refuses_delete(
    route_database,
):
    """Route changes never remove immutable scope bindings or billing credentials."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profile = create_ready_profile(
            session, route_database, "Bound", provider="openrouter"
        )
        profile.billing_secret_ciphertext = route_database.secret_cipher.encrypt(
            "billing-secret"
        )
        node = ProviderScopeNode(
            id="workspace",
            tenant_id=tenant.id,
            provider="openrouter",
            scope_type="workspace",
            canonical_scope_id="workspace-id",
        )
        session.add(node)
        session.flush()
        binding = ProviderScopeBinding(
            id="binding",
            tenant_id=tenant.id,
            provider="openrouter",
            profile_id=profile.id,
            purpose="billing",
            node_id=node.id,
        )
        session.add(binding)
        admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        with pytest.raises(ValueError, match="scopes are bound"):
            admin_ops.delete_provider_profile(session, tenant.id, profile.id, "ada")
        assert profile.route_priority == 1
        admin_ops.deactivate_provider_profile(session, tenant, profile.id, "ada")
        assert profile.route_priority is None
        assert session.get(ProviderScopeBinding, binding.id).node_id == node.id
        assert session.get(ProviderScopeBinding, binding.id).profile_id == profile.id
        assert (
            route_database.secret_cipher.decrypt(profile.billing_secret_ciphertext)
            == "billing-secret"
        )
        assert profile.deleted_at is None


def test_snapshot_refreshes_cached_orm_configuration_in_its_single_read(route_database):
    """An existing identity map must not mix old secrets/settings with new route data."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profile = create_ready_profile(session, route_database, "One")
        admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        profile_id = profile.id
    with route_database.sessions() as session:
        cached = session.get(ProviderProfile, profile_id)
        assert cached.settings == {}
        with route_database.sessions.begin() as writer:
            writer.get(ProviderProfile, profile_id).settings = {
                "project": "updated-project"
            }
        snapshot = TenantRepository(session).get_proxy_snapshot_by_api_key(
            "acme", route_database.secret_cipher
        )
        assert snapshot.profiles[0].provider_settings == {"project": "updated-project"}


def test_snapshot_excludes_undecryptable_route_credentials(route_database):
    """A corrupt provider credential cannot be included in an authenticated route."""
    with route_database.sessions.begin() as session:
        tenant = session.get(Tenant, "acme")
        profile = create_ready_profile(session, route_database, "One")
        admin_ops.activate_provider_profile(session, tenant, profile.id, "ada")
        profile.inference_secret_ciphertext = "not-an-encrypted-envelope"
    assert route_database.get_proxy_snapshot_by_api_key("acme").profiles == ()


def test_route_migration_follows_current_head_and_emits_lossless_transition():
    """Offline SQL includes pointer transfer and a loss-preventing downgrade guard."""
    root = Path(__file__).resolve().parents[1]
    script = ScriptDirectory.from_config(Config(str(root / "alembic.ini")))
    migration = script.get_revision("20261005_provider_route").module
    assert migration.down_revision == "20261004_clear_azure_billing"
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        migration.upgrade()
    sql = output.getvalue()
    assert "route_priority" in sql
    assert "active_profile_id" in sql
    assert "DROP CONSTRAINT fk_tenant_active_profile_same_tenant" in sql
    assert "DROP COLUMN active_profile_id" in sql
    assert "deleted_at IS NULL" in sql
    output.seek(0)
    output.truncate()
    with Operations.context(context):
        migration.downgrade()
    sql = output.getvalue()
    assert "RAISE EXCEPTION" in sql
    assert "HAVING count(*) > 1" in sql
    assert sql.index("RAISE EXCEPTION") < sql.index("ADD COLUMN active_profile_id")
