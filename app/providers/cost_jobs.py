"""Coordinate cost-refresh reservations and persist their terminal results."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import requests
from cryptography.exceptions import InvalidTag
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

REFRESH_STALE_AFTER = timedelta(seconds=180)


def fail_stale_running_jobs(
    session: Session, binding_id: str, now: datetime | None = None
) -> bool:
    """Fail expired jobs and return whether a refresh is still active."""
    now = now or datetime.now(timezone.utc)
    running_jobs = list(
        session.scalars(
            select(CostRefreshJob)
            .where(
                CostRefreshJob.binding_id == binding_id,
                CostRefreshJob.status == "running",
            )
            .with_for_update()
        )
    )
    active = False
    for job in running_jobs:
        created_at = job.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if now - created_at < REFRESH_STALE_AFTER:
            active = True
            continue
        job.status = "failed"
        job.completed_at = now
        job.limitation = "Refresh process did not complete within its time limit."
        session.add(
            CostRefreshEvent(
                job_id=job.id,
                previous_status="running",
                new_status="failed",
                reason=job.limitation,
            )
        )
    session.flush()
    return active


def start_cost_refresh(
    session: Session,
    tenant_id: str,
    provider: str,
    start: datetime,
    end: datetime,
) -> tuple[ProviderProfile, ProviderScopeBinding, ProviderScopeNode, CostRefreshJob]:
    """Reserve a refresh attempt and lock its binding before provider I/O."""
    profile = session.scalar(
        select(ProviderProfile).where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.provider == provider,
        )
    )
    if profile is None:
        raise LookupError(f"No {provider} profile is configured")
    binding = session.scalar(
        select(ProviderScopeBinding)
        .where(
            ProviderScopeBinding.tenant_id == tenant_id,
            ProviderScopeBinding.provider == provider,
            ProviderScopeBinding.purpose == "billing",
        )
        .with_for_update()
    )
    if binding is None:
        raise LookupError(f"No {provider} billing scope is bound")
    node = session.get(ProviderScopeNode, binding.node_id)
    if node is None:
        raise LookupError(f"No {provider} billing scope node is bound")
    if fail_stale_running_jobs(session, binding.id):
        raise LookupError(f"A {provider} cost refresh is already running")

    job = CostRefreshJob(
        id=str(uuid4()),
        tenant_id=tenant_id,
        provider=provider,
        binding_id=binding.id,
        period_start=start,
        period_end=end,
        source_api=provider,
        operation_key=f"{provider}:{uuid4()}",
        status="running",
        retry_at=None,
        limitation=None,
        completed_at=None,
    )
    session.add(job)
    session.flush()
    session.add(
        CostRefreshEvent(
            job_id=job.id,
            previous_status=None,
            new_status="running",
            reason="refresh started",
        )
    )
    return profile, binding, node, job


def collect_provider_costs(
    cipher: SecretCipher,
    profile: ProviderProfile,
    canonical_scope_id: str,
    start: datetime,
    end: datetime,
    usage_scope_id: str | None = None,
) -> list[CostBucket]:
    """Call provider billing APIs using stored credentials or host identity."""
    if profile.provider == "azure":
        if not usage_scope_id:
            raise CostRefreshError(
                "Azure usage scope is missing.", status="unavailable"
            )
        try:
            return fetch_azure_costs(canonical_scope_id, usage_scope_id, start, end)
        except requests.RequestException as exc:
            raise CostRefreshError("Azure billing request failed.") from exc
    if profile.billing_secret_ciphertext is None:
        raise CostRefreshError("Billing credentials are missing.")
    try:
        secret = cipher.decrypt(profile.billing_secret_ciphertext)
    except (InvalidTag, ValueError, UnicodeDecodeError) as exc:
        raise CostRefreshError("Billing credentials could not be decrypted.") from exc
    try:
        if profile.provider == "openai":
            return fetch_openai_costs(secret, canonical_scope_id, start, end)
        if profile.provider == "openrouter":
            return fetch_openrouter_costs(secret, canonical_scope_id, start, end)
    except requests.RequestException as exc:
        raise CostRefreshError(
            f"{profile.provider.capitalize()} billing request failed."
        ) from exc
    raise CostRefreshError(f"Unsupported provider {profile.provider!r}")


def persist_cost_refresh(
    session: Session,
    tenant_id: str,
    actor_id: str,
    profile: ProviderProfile,
    binding: ProviderScopeBinding,
    job: CostRefreshJob,
    buckets: list[CostBucket] | None,
    error: CostRefreshError | None,
) -> CostRefreshJob | None:
    """Complete a still-running job; discard results after stale-job recovery."""
    binding = session.scalar(
        select(ProviderScopeBinding)
        .where(ProviderScopeBinding.id == binding.id)
        .with_for_update()
    )
    if binding is None:
        return None
    job = session.scalar(
        select(CostRefreshJob)
        .where(CostRefreshJob.id == job.id, CostRefreshJob.status == "running")
        .with_for_update()
    )
    if job is None:
        return None

    now = datetime.now(timezone.utc)
    status = "success" if error is None else error.status
    job.status = status
    job.limitation = str(error) if error is not None else None
    job.completed_at = now
    session.flush()
    session.add(
        CostRefreshEvent(
            job_id=job.id,
            previous_status="running",
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
