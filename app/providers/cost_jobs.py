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


def load_billing_binding(
    session: Session, tenant_id: str, provider: str
) -> tuple[ProviderProfile, ProviderScopeBinding, ProviderScopeNode]:
    """Return the exclusive billing binding for a tenant provider."""
    profile = session.scalar(
        select(ProviderProfile).where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.provider == provider,
        )
    )
    if profile is None:
        raise LookupError(f"No {provider} profile is configured")
    binding = session.scalar(
        select(ProviderScopeBinding).where(
            ProviderScopeBinding.tenant_id == tenant_id,
            ProviderScopeBinding.provider == provider,
            ProviderScopeBinding.purpose == "billing",
        )
    )
    if binding is None:
        raise LookupError(f"No {provider} billing scope is bound")
    node = session.get(ProviderScopeNode, binding.node_id)
    if node is None:
        raise LookupError(f"No {provider} billing scope node is bound")
    return profile, binding, node


def collect_provider_costs(
    cipher: SecretCipher,
    profile: ProviderProfile,
    canonical_scope_id: str,
    start: datetime,
    end: datetime,
) -> list[CostBucket]:
    """Call the provider billing API using decrypted credentials."""
    secret = None
    if profile.billing_secret_ciphertext:
        secret = cipher.decrypt(profile.billing_secret_ciphertext)
    elif profile.inference_secret_ciphertext:
        secret = cipher.decrypt(profile.inference_secret_ciphertext)
    if not secret:
        raise CostRefreshError("Billing credentials are missing.")
    if profile.provider == "openai":
        return fetch_openai_costs(secret, canonical_scope_id, start, end)
    if profile.provider == "openrouter":
        return fetch_openrouter_costs(secret, start, end)
    if profile.provider == "azure":
        return fetch_azure_costs(secret, canonical_scope_id, start, end)
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
        operation_key=f"{profile.provider}:{now.isoformat()}",
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
            target=f"costs:{profile.provider}",
            action="costs.refresh",
            outcome=status,
            details={"provider": profile.provider, "job_id": job.id},
        )
    )
    return job
