"""Persist cost-refresh jobs after provider I/O has completed."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence.models import (
    AuditEvent,
    CostRefreshEvent,
    CostRefreshJob,
    CostUsageRecord,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
)
from app.persistence.secrets import SecretCipher
from app.providers.costs import (
    CostBucket,
    CostRefreshError,
    fetch_azure_costs,
    fetch_openai_costs,
    fetch_openrouter_costs,
)


def load_billing_binding(session: Session, tenant_id: str, profile_id: str) -> tuple[
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    ProviderScopeNode | None,
]:
    """Return one account's billing scope and optional Azure usage resource."""
    profile = session.scalar(
        select(ProviderProfile).where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.id == profile_id,
            ProviderProfile.deleted_at.is_(None),
        )
    )
    if profile is None:
        raise LookupError("Provider account was not found")
    binding = session.scalar(
        select(ProviderScopeBinding).where(
            ProviderScopeBinding.tenant_id == tenant_id,
            ProviderScopeBinding.profile_id == profile.id,
            ProviderScopeBinding.provider == profile.provider,
            ProviderScopeBinding.purpose == "billing",
        )
    )
    if binding is None:
        raise LookupError(f"No billing scope is bound to {profile.display_name}")
    node = session.get(ProviderScopeNode, binding.node_id)
    if node is None:
        raise LookupError(f"No {profile.provider} billing scope node is bound")
    usage_node = None
    if profile.provider == "azure":
        usage_binding = session.scalar(
            select(ProviderScopeBinding).where(
                ProviderScopeBinding.tenant_id == tenant_id,
                ProviderScopeBinding.profile_id == profile.id,
                ProviderScopeBinding.provider == profile.provider,
                ProviderScopeBinding.purpose == "usage",
            )
        )
        if usage_binding is None:
            raise LookupError(
                f"No Azure usage resource is bound to {profile.display_name}"
            )
        usage_node = session.get(ProviderScopeNode, usage_binding.node_id)
        if usage_node is None:
            raise LookupError(
                f"No Azure usage resource is bound to {profile.display_name}"
            )
    return profile, binding, node, usage_node


def collect_provider_costs(
    cipher: SecretCipher,
    profile: ProviderProfile,
    canonical_scope_id: str,
    start: datetime,
    end: datetime,
    *,
    usage_scope_id: str | None = None,
) -> list[CostBucket]:
    """Call the provider billing API using decrypted credentials."""
    if not profile.billing_secret_ciphertext:
        raise CostRefreshError("Billing credentials are missing.")
    secret = cipher.decrypt(profile.billing_secret_ciphertext)
    if profile.provider == "openai":
        return fetch_openai_costs(secret, canonical_scope_id, start, end)
    if profile.provider == "openrouter":
        return fetch_openrouter_costs(secret, start, end)
    if profile.provider == "azure":
        if usage_scope_id is None:
            raise CostRefreshError("Azure usage resource is missing.")
        return fetch_azure_costs(secret, canonical_scope_id, usage_scope_id, start, end)
    raise CostRefreshError(f"Unsupported provider {profile.provider!r}")


def persist_cost_refresh(
    session: Session,
    tenant_id: str,
    actor_id: str,
    profile: ProviderProfile,
    binding: ProviderScopeBinding,
    start: datetime,
    end: datetime,
    buckets: list[CostBucket] | None,
    error: CostRefreshError | None,
) -> CostRefreshJob:
    """Write job, events, records, and audit in one unit of work."""
    now = datetime.now(timezone.utc)
    status = "success" if error is None else error.status
    job = CostRefreshJob(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider=profile.provider,
        binding_id=binding.id,
        period_start=start,
        period_end=end,
        source_api=profile.provider,
        operation_key=f"{profile.id}:{binding.id}:{now.isoformat()}",
        status=status,
        retry_at=None,
        limitation=str(error) if error is not None else None,
        completed_at=now,
    )
    session.add(job)
    session.flush()
    session.add(
        CostRefreshEvent(
            job_id=job.id,
            previous_status=None,
            new_status=status,
            reason=str(error) if error is not None else "refresh completed",
        )
    )
    if buckets:
        for bucket in buckets:
            session.add(
                CostUsageRecord(
                    job_id=job.id,
                    tenant_id=tenant_id,
                    provider=profile.provider,
                    binding_id=binding.id,
                    kind=bucket.kind,
                    metric=bucket.metric,
                    value=bucket.value,
                    unit=bucket.unit,
                    currency=bucket.currency,
                    bucket_start=bucket.bucket_start,
                    bucket_end=bucket.bucket_end,
                    source=bucket.source,
                    granularity=bucket.granularity,
                    dimensions=bucket.dimensions,
                )
            )
    session.add(
        AuditEvent(
            tenant_id=tenant_id,
            actor_id=actor_id,
            target=f"costs:{profile.id}",
            action="costs.refresh",
            outcome=status,
            details={
                "provider": profile.provider,
                "profile_id": profile.id,
                "job_id": job.id,
            },
        )
    )
    return job
