"""Validation and database submission for OpenAI batch requests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.persistence.database import Database
from app.persistence.models import (
    BatchJob,
    ProviderCatalogEntry,
    ProviderProfile,
    Tenant,
)
from app.providers.catalog import selectable_catalog_models

MAX_BATCH_REQUESTS = 1000
MAX_BATCH_JSONL_BYTES = 4 * 1024 * 1024
BATCH_RETENTION = timedelta(days=7)
_IDEMPOTENCY_KEY_PATTERN = re.compile(r"[A-Za-z0-9._~:-]{1,128}\Z")
_EXPECTED_RECORD_KEYS = frozenset({"custom_id", "method", "url", "body"})
TERMINAL_BATCH_STATES = frozenset(
    {"completed", "failed", "expired", "unknown_upload", "unknown_submit"}
)


class BatchValidationError(ValueError):
    """Raised for an invalid batch submission before any database write."""


class BatchIdempotencyConflictError(Exception):
    """Raised when a retained idempotency key names a different request."""


class BatchJobStillActiveError(Exception):
    """Raised when expiry passes while a valid worker lease still owns the job."""


class BatchQueueFullError(Exception):
    """Raised when the tenant has reached its configured active batch capacity."""


@dataclass(frozen=True)
class BatchSubmission:
    """Normalized batch payload and canonical provider JSONL bytes."""

    model: str
    request_json: dict[str, object]
    jsonl: bytes
    payload_digest: str
    idempotency_key_hash: str


@dataclass(frozen=True)
class BatchSubmissionResult:
    """Persisted job and whether it was returned by idempotent replay."""

    job: BatchJob
    replayed: bool


def validate_idempotency_key(value: object) -> str:
    """Validate a bounded, header-safe idempotency key and return its digest."""
    if not isinstance(value, str) or _IDEMPOTENCY_KEY_PATTERN.fullmatch(value) is None:
        raise BatchValidationError(
            "A safe Idempotency-Key of 1 to 128 characters is required."
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_batch_payload(
    payload: object, idempotency_key_hash: str
) -> BatchSubmission:
    """Validate records and produce canonical JSONL and request digests."""
    if not isinstance(payload, dict) or set(payload) != {"model", "requests"}:
        raise BatchValidationError("The batch body must contain model and requests.")
    model = payload.get("model")
    records = payload.get("requests")
    if not isinstance(model, str) or not model.strip():
        raise BatchValidationError("model must be a non-empty configured model.")
    if not isinstance(records, list) or not records:
        raise BatchValidationError("requests must contain at least one record.")
    if len(records) > MAX_BATCH_REQUESTS:
        raise BatchValidationError("A batch may contain at most 1000 requests.")

    normalized_records: list[dict[str, object]] = []
    custom_ids: set[str] = set()
    jsonl_lines: list[bytes] = []
    for record in records:
        normalized = _normalize_record(record, model, custom_ids)
        try:
            line = json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise BatchValidationError(
                "Each request body must contain valid JSON values."
            ) from exc
        jsonl_lines.append(line + b"\n")
        normalized_records.append(normalized)

    jsonl = b"".join(jsonl_lines)
    if len(jsonl) > MAX_BATCH_JSONL_BYTES:
        raise BatchValidationError("The canonical JSONL input exceeds 4 MiB.")
    request_json: dict[str, object] = {"model": model, "requests": normalized_records}
    canonical_payload = json.dumps(
        request_json,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return BatchSubmission(
        model=model,
        request_json=request_json,
        jsonl=jsonl,
        payload_digest=hashlib.sha256(canonical_payload).hexdigest(),
        idempotency_key_hash=idempotency_key_hash,
    )


def _normalize_record(
    record: object, job_model: str, custom_ids: set[str]
) -> dict[str, object]:
    if not isinstance(record, dict) or set(record) != _EXPECTED_RECORD_KEYS:
        raise BatchValidationError(
            "Every request must contain custom_id, method, url, and body only."
        )
    custom_id = record.get("custom_id")
    if (
        not isinstance(custom_id, str)
        or not custom_id.strip()
        or len(custom_id) > 64
        or custom_id in custom_ids
    ):
        raise BatchValidationError(
            "custom_id values must be unique, non-empty strings of at most 64 characters."
        )
    if record.get("method") != "POST" or record.get("url") != "/v1/chat/completions":
        raise BatchValidationError("Only POST /v1/chat/completions is allowed.")
    body = record.get("body")
    if not isinstance(body, dict):
        raise BatchValidationError("Each request body must be a JSON object.")
    if body.get("model") != job_model:
        raise BatchValidationError(
            "Every request body model must match the batch model."
        )
    custom_ids.add(custom_id)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def submit_batch(
    database: Database,
    tenant_id: str,
    submission: BatchSubmission,
    *,
    max_queued_jobs: int,
    now: datetime | None = None,
) -> BatchSubmissionResult:
    """Persist a new job or return a matching tenant-local idempotent replay."""
    if (
        isinstance(max_queued_jobs, bool)
        or not isinstance(max_queued_jobs, int)
        or max_queued_jobs <= 0
    ):
        raise ValueError("max_queued_jobs must be a positive integer")
    created_at = now or datetime.now(timezone.utc)
    try:
        with database.sessions.begin() as session:
            tenant = session.get(Tenant, tenant_id, with_for_update=True)
            if tenant is None:
                raise BatchValidationError(
                    "Authenticated batch tenant no longer exists."
                )
            existing = _existing_job(
                session, tenant_id, submission.idempotency_key_hash
            )
            if existing is not None:
                if _utc(existing.expires_at) <= created_at:
                    if existing.status not in TERMINAL_BATCH_STATES:
                        if (
                            existing.lease_expires_at is not None
                            and _utc(existing.lease_expires_at) > created_at
                        ):
                            raise BatchJobStillActiveError
                        if existing.status == "submitting":
                            existing.status = "unknown_submit"
                            existing.error_json = {
                                "type": "unknown_submit",
                                "message": (
                                    "Submission outcome is ambiguous after retention expiry."
                                ),
                            }
                        elif existing.status == "uploading":
                            existing.status = "unknown_upload"
                            existing.error_json = {
                                "type": "unknown_upload",
                                "message": (
                                    "Upload outcome is ambiguous and will not be retried. "
                                    "No provider file ID is available for explicit cleanup; "
                                    "provider-managed retention may expire the file."
                                ),
                            }
                        else:
                            existing.status = "expired"
                            existing.error_json = {
                                "type": "expired",
                                "message": "Batch retention expired before completion.",
                            }
                        existing.completed_at = created_at
                        existing.worker_owner = None
                        existing.lease_expires_at = None
                        existing.fencing_token += 1
                    existing.idempotency_key_hash = None
                else:
                    return _replay_or_conflict(existing, submission)
            active_count = session.scalar(
                select(func.count(BatchJob.id)).where(
                    BatchJob.tenant_id == tenant_id,
                    BatchJob.status.in_(
                        (
                            "queued",
                            "uploading",
                            "submitting",
                            "retry_submit",
                            "submitted",
                            "polling",
                        )
                    ),
                    BatchJob.expires_at > created_at,
                )
            )
            if active_count >= max_queued_jobs:
                raise BatchQueueFullError
            profile_id = _find_openai_profile(
                session, database, tenant_id, submission.model
            )
            job = BatchJob(
                id=f"batch_{uuid4().hex}",
                tenant_id=tenant_id,
                profile_id=profile_id,
                model=submission.model,
                idempotency_key_hash=submission.idempotency_key_hash,
                payload_digest=submission.payload_digest,
                request_json=submission.request_json,
                status="queued",
                fencing_token=0,
                created_at=created_at,
                updated_at=created_at,
                expires_at=created_at + BATCH_RETENTION,
            )
            session.add(job)
            session.flush()
            return BatchSubmissionResult(job=job, replayed=False)
    except IntegrityError:
        with database.sessions() as session:
            existing = _existing_job(
                session, tenant_id, submission.idempotency_key_hash
            )
            if existing is None:
                raise
            return _replay_or_conflict(existing, submission)


def _utc(value: datetime) -> datetime:
    """Normalize database timestamps before comparing retention deadlines."""
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def _existing_job(
    session: Session, tenant_id: str, idempotency_key_hash: str
) -> BatchJob | None:
    return session.scalar(
        select(BatchJob)
        .where(
            BatchJob.tenant_id == tenant_id,
            BatchJob.idempotency_key_hash == idempotency_key_hash,
        )
        .with_for_update()
    )


def _replay_or_conflict(
    job: BatchJob, submission: BatchSubmission
) -> BatchSubmissionResult:
    if job.payload_digest != submission.payload_digest:
        raise BatchIdempotencyConflictError
    return BatchSubmissionResult(job=job, replayed=True)


def _find_openai_profile(
    session: Session, database: Database, tenant_id: str, model: str
) -> str:
    profiles = session.scalars(
        select(ProviderProfile)
        .where(
            ProviderProfile.tenant_id == tenant_id,
            ProviderProfile.provider == "openai",
            ProviderProfile.deleted_at.is_(None),
            ProviderProfile.inference_secret_ciphertext.is_not(None),
        )
        .order_by(
            ProviderProfile.route_priority.is_(None),
            ProviderProfile.route_priority,
            ProviderProfile.id,
        )
    )
    for profile in profiles:
        if not isinstance(profile.settings, dict):
            continue
        try:
            secret = database.secret_cipher.decrypt(profile.inference_secret_ciphertext)
        except (InvalidTag, ValueError):
            continue
        if not secret.strip():
            continue
        entries = session.scalars(
            select(ProviderCatalogEntry).where(
                ProviderCatalogEntry.profile_id == profile.id
            )
        )
        models = selectable_catalog_models(
            "openai", [(entry.model_id, entry.deployment_id) for entry in entries]
        )
        if any(model_id == model for model_id, _deployment_id in models):
            return profile.id
    raise BatchValidationError(
        "model must be present in a ready OpenAI provider catalog for this tenant."
    )
