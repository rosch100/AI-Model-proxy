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
            "provider IN ('azure', 'openai', 'openrouter')",
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
    """Provider model IDs and optional Azure deployment mappings."""

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


class InferenceActivityEvent(Base):
    """One completed proxy inference, for live tenant activity on the overview."""

    __tablename__ = "inference_activity_events"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('azure', 'openai', 'openrouter')",
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
    inbound_model: Mapped[str] = mapped_column(String(256), nullable=False)
    routed_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
