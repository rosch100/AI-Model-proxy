"""SQLAlchemy schema for tenant-scoped proxy administration."""

from __future__ import annotations

import unicodedata
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    UniqueConstraint,
    column,
    event,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def provider_profile_name_key(display_name: str) -> str:
    """Return the normalized, case-insensitive key used for profile names."""
    key = unicodedata.normalize("NFKC", display_name).casefold()
    if len(key) > 384:
        raise ValueError("Normalized account name must be at most 384 characters")
    return key


class Base(DeclarativeBase):
    """Base metadata for tenant persistence."""


class Tenant(Base):
    """Proxy identity and stable Cursor-facing model identity."""

    __tablename__ = "tenants"
    __table_args__ = (
        UniqueConstraint("api_key_hash", name="uq_tenants_api_key_hash"),
        UniqueConstraint("custom_model_id", name="uq_tenants_custom_model_id"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    api_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    custom_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AdminAccount(Base):
    """Tenant-local administrator login identity."""

    __tablename__ = "admin_accounts"
    __table_args__ = (
        UniqueConstraint("tenant_id", "username", name="uq_admin_tenant_username"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    username: Mapped[str] = mapped_column(String(254), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    webauthn_user_handle: Mapped[bytes | None] = mapped_column(
        LargeBinary(64), unique=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ProviderProfile(Base):
    """Provider-specific connection settings and encrypted credentials."""

    __tablename__ = "provider_profiles"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "provider", "id", name="uq_profile_tenant_provider_id"
        ),
        UniqueConstraint("tenant_id", "id", name="uq_profile_tenant_id"),
        UniqueConstraint(
            "tenant_id", "route_priority", name="uq_profile_tenant_route_priority"
        ),
        CheckConstraint(
            "route_priority IS NULL OR route_priority > 0",
            name="ck_profile_route_priority_positive",
        ),
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_profile_provider",
        ),
        CheckConstraint(
            "history_generation >= 0", name="ck_profile_history_generation"
        ),
        Index(
            "uq_profile_active_name",
            "tenant_id",
            "provider",
            column("display_name_key"),
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    display_name_key: Mapped[str | None] = mapped_column(String(384), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    history_generation: Mapped[int] = mapped_column(
        nullable=False, default=0, server_default="0"
    )
    catalog_generation: Mapped[int] = mapped_column(
        nullable=False, default=0, server_default="0"
    )
    settings: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    inference_secret_ciphertext: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    billing_secret_ciphertext: Mapped[str | None] = mapped_column(String, nullable=True)
    default_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    route_priority: Mapped[int | None] = mapped_column(Integer, nullable=True)
    catalog_refreshed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    catalog_error: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


@event.listens_for(ProviderProfile, "before_insert")
@event.listens_for(ProviderProfile, "before_update")
def _ensure_profile_name_key(_mapper, _connection, profile: ProviderProfile) -> None:
    profile.display_name_key = (
        provider_profile_name_key(profile.display_name)
        if profile.display_name is not None
        else None
    )


class ProviderCatalogEntry(Base):
    """Provider catalog models and optional published per-million prices."""

    __tablename__ = "provider_catalog_entries"
    __table_args__ = (
        UniqueConstraint("profile_id", "model_id", name="uq_catalog_profile_model"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("provider_profiles.id", ondelete="CASCADE"), nullable=False
    )
    model_id: Mapped[str] = mapped_column(String(256), nullable=False)
    deployment_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    input_price_per_1m_tokens: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    output_price_per_1m_tokens: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    cache_price_per_1m_tokens: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    pricing_currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    pricing_source: Mapped[str | None] = mapped_column(String(256), nullable=True)


class ProviderScopeNode(Base):
    """Globally exclusive canonical provider scope and optional parent node."""

    __tablename__ = "provider_scope_nodes"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "scope_type",
            "canonical_scope_id",
            name="uq_scope_global_identity",
        ),
        UniqueConstraint(
            "tenant_id", "provider", "id", name="uq_scope_tenant_provider_id"
        ),
        ForeignKeyConstraint(
            ["tenant_id", "provider", "parent_node_id"],
            [
                "provider_scope_nodes.tenant_id",
                "provider_scope_nodes.provider",
                "provider_scope_nodes.id",
            ],
            name="fk_scope_parent_same_tenant_provider",
        ),
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter')",
            name="ck_scope_provider",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(32), nullable=False)
    canonical_scope_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    parent_node_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ProviderScopeBinding(Base):
    """Immutable purpose-specific binding from a profile to a canonical scope."""

    __tablename__ = "provider_scope_bindings"
    __table_args__ = (
        UniqueConstraint(
            "profile_id", "provider", "purpose", name="uq_binding_profile_purpose"
        ),
        UniqueConstraint("node_id", name="uq_binding_node_id"),
        UniqueConstraint(
            "tenant_id", "provider", "id", name="uq_binding_tenant_provider_id"
        ),
        ForeignKeyConstraint(
            ["tenant_id", "provider", "profile_id"],
            [
                "provider_profiles.tenant_id",
                "provider_profiles.provider",
                "provider_profiles.id",
            ],
            name="fk_binding_profile_same_tenant_provider",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "provider", "node_id"],
            [
                "provider_scope_nodes.tenant_id",
                "provider_scope_nodes.provider",
                "provider_scope_nodes.id",
            ],
            name="fk_binding_node_same_tenant_provider",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "provider", "parent_binding_id"],
            [
                "provider_scope_bindings.tenant_id",
                "provider_scope_bindings.provider",
                "provider_scope_bindings.id",
            ],
            name="fk_binding_parent_same_tenant_provider",
        ),
        CheckConstraint("purpose IN ('billing', 'usage')", name="ck_binding_purpose"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    profile_id: Mapped[str] = mapped_column(String(36), nullable=False)
    purpose: Mapped[str] = mapped_column(String(16), nullable=False)
    node_id: Mapped[str] = mapped_column(String(36), nullable=False)
    parent_binding_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CostRefreshJob(Base):
    """One explicit synchronous provider billing/usage query attempt."""

    __tablename__ = "cost_refresh_jobs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "operation_key", name="uq_refresh_tenant_operation"
        ),
        UniqueConstraint(
            "id",
            "tenant_id",
            "provider",
            "binding_id",
            name="uq_refresh_job_tenant_provider_binding",
        ),
        CheckConstraint(
            "status IN ('running', 'retry_wait', 'success', 'unavailable', 'failed')",
            name="ck_refresh_status",
        ),
        CheckConstraint(
            "(status = 'retry_wait' AND retry_at IS NOT NULL) OR "
            "(status != 'retry_wait' AND retry_at IS NULL)",
            name="ck_refresh_retry_at",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "provider", "binding_id"],
            [
                "provider_scope_bindings.tenant_id",
                "provider_scope_bindings.provider",
                "provider_scope_bindings.id",
            ],
            name="fk_refresh_binding_tenant_provider",
        ),
        Index("ix_refresh_job_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    binding_id: Mapped[str] = mapped_column(String(36), nullable=False)
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source_api: Mapped[str] = mapped_column(String(512), nullable=False)
    operation_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    limitation: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class CostRefreshEvent(Base):
    """Append-only history of refresh state transitions."""

    __tablename__ = "cost_refresh_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("cost_refresh_jobs.id", ondelete="RESTRICT"), nullable=False
    )
    previous_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    new_status: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CostUsageRecord(Base):
    """Typed, scope-verified cost, usage, or price-derived estimate bucket."""

    __tablename__ = "cost_usage_records"
    __table_args__ = (
        CheckConstraint("kind IN ('actual', 'usage', 'estimate')", name="ck_cost_kind"),
        CheckConstraint(
            "(kind IN ('actual', 'estimate') AND currency IS NOT NULL) OR "
            "(kind = 'usage' AND currency IS NULL)",
            name="ck_cost_currency_by_kind",
        ),
        CheckConstraint(
            "kind != 'estimate' OR (price_source IS NOT NULL AND price_version IS NOT NULL "
            "AND usage_source IS NOT NULL AND usage_bucket_start IS NOT NULL "
            "AND formula_parameters IS NOT NULL AND model_key IS NOT NULL "
            "AND region_key IS NOT NULL AND deployment_key IS NOT NULL)",
            name="ck_estimate_provenance",
        ),
        ForeignKeyConstraint(
            ["job_id", "tenant_id", "provider", "binding_id"],
            [
                "cost_refresh_jobs.id",
                "cost_refresh_jobs.tenant_id",
                "cost_refresh_jobs.provider",
                "cost_refresh_jobs.binding_id",
            ],
            name="fk_cost_job_tenant_provider_binding",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    binding_id: Mapped[str] = mapped_column(String(36), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    metric: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(28, 10), nullable=False)
    unit: Mapped[str] = mapped_column(String(64), nullable=False)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    bucket_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    bucket_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    granularity: Mapped[str] = mapped_column(String(64), nullable=False)
    dimensions: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    price_source: Mapped[str | None] = mapped_column(String(512), nullable=True)
    price_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    usage_source: Mapped[str | None] = mapped_column(String(512), nullable=True)
    usage_bucket_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    formula_parameters: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True
    )
    model_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    region_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    deployment_key: Mapped[str | None] = mapped_column(String(256), nullable=True)


class AdminSession(Base):
    """Server-revocable opaque admin session token digest."""

    __tablename__ = "admin_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("admin_accounts.id"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    enrollment_only: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )


class AdminPasskey(Base):
    """Registered WebAuthn credential for an administrator account."""

    __tablename__ = "admin_passkeys"
    __table_args__ = (
        UniqueConstraint("credential_id", name="uq_admin_passkey_credential"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("admin_accounts.id", ondelete="CASCADE"), nullable=False
    )
    credential_id: Mapped[bytes] = mapped_column(LargeBinary(1024), nullable=False)
    public_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    sign_count: Mapped[int] = mapped_column(nullable=False, default=0)
    user_handle: Mapped[bytes] = mapped_column(LargeBinary(64), nullable=False)
    transports: Mapped[list[object] | None] = mapped_column(JSON, nullable=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    aaguid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    backed_up: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AdminWebAuthnChallenge(Base):
    """One-time WebAuthn ceremony challenge shared across workers."""

    __tablename__ = "admin_webauthn_challenges"
    __table_args__ = (
        CheckConstraint(
            "purpose IN ('registration', 'authentication')",
            name="ck_webauthn_purpose",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    account_id: Mapped[int | None] = mapped_column(
        ForeignKey("admin_accounts.id", ondelete="CASCADE"), nullable=True
    )
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    challenge: Mapped[bytes] = mapped_column(LargeBinary(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    consumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class LoginRateLimit(Base):
    """Persistent worker-shared login failure window."""

    __tablename__ = "login_rate_limits"

    subject_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    failures: Mapped[int] = mapped_column(nullable=False, default=0)
    window_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AuditEvent(Base):
    """Append-only secret-free record of tenant and operator actions."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    target: Mapped[str] = mapped_column(String(256), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    details: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict
    )


class ProviderAttemptEvent(Base):
    """One pending or completed upstream attempt for a provider profile."""

    __tablename__ = "provider_attempt_events"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_provider_attempt_provider",
        ),
        CheckConstraint(
            "outcome IN ('pending', 'success', 'failure', 'aborted')",
            name="ck_provider_attempt_outcome",
        ),
        Index(
            "ix_provider_attempt_tenant_profile_time",
            "tenant_id",
            "profile_id",
            "occurred_at",
        ),
        Index("ix_provider_attempt_occurred_at", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("provider_profiles.id", ondelete="CASCADE"), nullable=False
    )
    inbound_model: Mapped[str] = mapped_column(String(2048), nullable=False)
    routed_model: Mapped[str] = mapped_column(String(256), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_details: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ProviderBudgetScopeState(Base):
    """Persistent cooldown for one provider-neutral scheduling scope."""

    __tablename__ = "provider_budget_scope_states"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "scope_kind",
            "scope_fingerprint",
            name="uq_budget_scope_identity",
        ),
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_budget_scope_provider",
        ),
        CheckConstraint(
            "scope_kind IN ('provider', 'profile', 'model', 'organization', 'project')",
            name="ck_budget_scope_kind",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    scope_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    cooldown_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ProviderBudgetPolicy(Base):
    """One immutable fixed-window duration per shared budget scope and metric."""

    __tablename__ = "provider_budget_policies"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "scope_kind",
            "scope_fingerprint",
            "metric",
            name="uq_budget_policy_identity",
        ),
        CheckConstraint("window_seconds > 0", name="ck_budget_policy_window_seconds"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    scope_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    metric: Mapped[str] = mapped_column(String(16), nullable=False)
    window_seconds: Mapped[int] = mapped_column(Integer, nullable=False)


class ProviderBudgetWindow(Base):
    """Fixed-window request/token counters shared by all app workers."""

    __tablename__ = "provider_budget_windows"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "scope_kind",
            "scope_fingerprint",
            "metric",
            "window_seconds",
            "window_start",
            name="uq_budget_window_identity",
        ),
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_budget_window_provider",
        ),
        CheckConstraint(
            "scope_kind IN ('provider', 'profile', 'model', 'organization', 'project')",
            name="ck_budget_window_scope",
        ),
        CheckConstraint(
            "metric IN ('requests', 'tokens')", name="ck_budget_window_metric"
        ),
        CheckConstraint("window_seconds > 0", name="ck_budget_window_seconds"),
        CheckConstraint("limit_units > 0", name="ck_budget_window_limit"),
        CheckConstraint(
            "used_units >= 0 AND reserved_units >= 0", name="ck_budget_window_units"
        ),
        Index("ix_budget_window_expiry", "window_start"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    scope_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    metric: Mapped[str] = mapped_column(String(16), nullable=False)
    window_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    limit_units: Mapped[int] = mapped_column(Integer, nullable=False)
    used_units: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reserved_units: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ProviderBudgetLease(Base):
    """Opaque token-bound reservation lifecycle shared across workers."""

    __tablename__ = "provider_budget_leases"
    __table_args__ = (
        UniqueConstraint("lease_token", name="uq_budget_lease_token"),
        CheckConstraint(
            "status IN ('active', 'completed', 'failed', 'released', 'expired')",
            name="ck_budget_lease_status",
        ),
        Index("ix_budget_lease_tenant_status", "tenant_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    lease_token: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ProviderBudgetLeaseAllocation(Base):
    """Budget units held by a lease against one fixed-window row."""

    __tablename__ = "provider_budget_lease_allocations"
    __table_args__ = (
        UniqueConstraint("lease_id", "window_id", name="uq_budget_lease_window"),
        CheckConstraint("reserved_units >= 0", name="ck_budget_allocation_units"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    lease_id: Mapped[str] = mapped_column(
        ForeignKey("provider_budget_leases.id", ondelete="CASCADE"), nullable=False
    )
    window_id: Mapped[int] = mapped_column(
        ForeignKey("provider_budget_windows.id", ondelete="RESTRICT"), nullable=False
    )
    reserved_units: Mapped[int] = mapped_column(Integer, nullable=False)


class ProviderConcurrencyState(Base):
    """Persistent AIMD concurrency bound for one tenant provider profile."""

    __tablename__ = "provider_concurrency_states"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "provider", "scope_fingerprint", name="uq_concurrency_scope"
        ),
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_concurrency_provider",
        ),
        CheckConstraint(
            "concurrency_limit BETWEEN 1 AND 64", name="ck_concurrency_limit"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    scope_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    concurrency_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ProviderConcurrencyLease(Base):
    """Durable expiring permit held by one in-flight provider request or stream."""

    __tablename__ = "provider_concurrency_leases"
    __table_args__ = (
        UniqueConstraint("lease_token", name="uq_concurrency_lease_token"),
        CheckConstraint(
            "status IN ('active', 'completed', 'released', 'expired')",
            name="ck_concurrency_lease_status",
        ),
        Index("ix_concurrency_lease_state_expiry", "state_id", "status", "expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    state_id: Mapped[int] = mapped_column(
        ForeignKey("provider_concurrency_states.id", ondelete="CASCADE"),
        nullable=False,
    )
    lease_token: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class BatchJob(Base):
    """Tenant-owned OpenAI batch request, state, and bounded result payload."""

    __tablename__ = "batch_jobs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "idempotency_key_hash", name="uq_batch_tenant_idempotency"
        ),
        ForeignKeyConstraint(
            ["tenant_id", "profile_id"],
            ["provider_profiles.tenant_id", "provider_profiles.id"],
            name="fk_batch_profile_same_tenant",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "status IN ('queued', 'uploading', 'submitting', 'retry_submit', "
            "'submitted', 'polling', 'completed', 'failed', 'unknown_upload', "
            "'unknown_submit', 'expired')",
            name="ck_batch_status",
        ),
        CheckConstraint(
            "idempotency_key_hash IS NULL OR length(idempotency_key_hash) = 64",
            name="ck_batch_key_hash",
        ),
        CheckConstraint("length(payload_digest) = 64", name="ck_batch_payload_digest"),
        CheckConstraint("fencing_token >= 0", name="ck_batch_fencing_token"),
        CheckConstraint(
            "upload_attempts BETWEEN 0 AND 5", name="ck_batch_upload_attempts"
        ),
        CheckConstraint(
            "submit_attempts BETWEEN 0 AND 5", name="ck_batch_submit_attempts"
        ),
        CheckConstraint(
            "(worker_owner IS NULL) = (lease_expires_at IS NULL)",
            name="ck_batch_lease_pair",
        ),
        Index(
            "ix_batch_jobs_queue",
            "status",
            "poll_after",
            "created_at",
            postgresql_where=text(
                "status IN ('queued', 'uploading', 'retry_submit', 'submitted', 'polling')"
            ),
            sqlite_where=text(
                "status IN ('queued', 'uploading', 'retry_submit', 'submitted', 'polling')"
            ),
        ),
        Index("ix_batch_jobs_lease", "lease_expires_at"),
        Index("ix_batch_jobs_retention", "expires_at"),
    )

    id: Mapped[str] = mapped_column(String(38), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    profile_id: Mapped[str] = mapped_column(String(36), nullable=False)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    idempotency_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    request_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    provider_batch_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    provider_input_file_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )
    provider_output_file_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )
    provider_error_file_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )
    results_json: Mapped[list[dict[str, object]] | None] = mapped_column(
        JSON, nullable=True
    )
    error_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    worker_owner: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    fencing_token: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    upload_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    submit_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    poll_after: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ProviderCircuitState(Base):
    """Persistent quota breaker state with an optional exclusive probe lease."""

    __tablename__ = "provider_circuit_states"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "scope_fingerprint",
            name="uq_provider_circuit_scope",
        ),
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_provider_circuit_provider",
        ),
        CheckConstraint(
            "scope_type IN ('profile', 'organization', 'project')",
            name="ck_provider_circuit_scope_type",
        ),
        CheckConstraint(
            "failure_category = 'quota_exhausted'",
            name="ck_provider_circuit_failure_category",
        ),
        CheckConstraint("failure_count >= 1", name="ck_provider_circuit_failure_count"),
        CheckConstraint(
            "(lease_token IS NULL) = (lease_until IS NULL)",
            name="ck_provider_circuit_lease_pair",
        ),
        Index(
            "ix_provider_circuit_tenant_probe",
            "tenant_id",
            "probe_at",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    failure_category: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False)
    probe_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lease_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class InferenceActivityEvent(Base):
    """One completed proxy inference, for live tenant activity on the overview."""

    __tablename__ = "inference_activity_events"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter', 'deepseek')",
            name="ck_inference_activity_provider",
        ),
        CheckConstraint(
            "(input_tokens IS NULL) = (output_tokens IS NULL) "
            "AND (input_tokens IS NULL) = (total_tokens IS NULL)",
            name="ck_inference_activity_usage_complete",
        ),
        Index(
            "ix_inference_activity_tenant_occurred",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_inference_activity_tenant_provider_model",
            "tenant_id",
            "provider",
            "inbound_model",
            "occurred_at",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("provider_profiles.id", ondelete="SET NULL"), nullable=True
    )
    inbound_model: Mapped[str] = mapped_column(String(2048), nullable=False)
    routed_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
