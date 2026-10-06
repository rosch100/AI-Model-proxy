"""Resumable, lease-coordinated OpenAI Batch API worker."""

from __future__ import annotations

import json
import logging
import math
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from uuid import uuid4

import click
import requests
from flask import Flask, current_app
from sqlalchemy import delete, or_, select, text, update
from sqlalchemy.exc import SQLAlchemyError

from app.batch_config import MAX_BATCH_SUBMIT_ATTEMPTS, MAX_BATCH_UPLOAD_ATTEMPTS
from app.batch_service import BATCH_RETENTION, MAX_BATCH_JSONL_BYTES
from app.common.retry_after import RETRY_AFTER_MAX_SECONDS
from app.persistence.database import Database
from app.persistence.models import BatchJob, ProviderProfile
from app.persistence.provider_scheduler import (
    ProviderBudgetScheduler,
    ProviderConcurrencyController,
)
from app.providers.error_classification import classify_upstream_error
from app.tenants import AUTH_MODE_TENANT, TENANT_CONFIG_DATABASE

OPENAI_BATCH_BASE_URL = "https://api.openai.com/v1"
HTTP_TIMEOUT = (10.0, 30.0)
LEASE_SECONDS = 90
LEASE_HEARTBEAT_INTERVAL_SECONDS = LEASE_SECONDS / 3
POLL_INTERVAL_SECONDS = 10
MAX_POLL_INTERVAL_SECONDS = 60
RETENTION_CLEANUP_INTERVAL_SECONDS = 300
CLEANUP_LIMIT = 50
CLEANUP_RETRY_DELAY_SECONDS = 300


class BatchWorkerError(RuntimeError):
    """Raised when worker configuration or persistence is unavailable."""


class BatchResultError(BatchWorkerError):
    """Raised for an oversized or structurally invalid provider result file."""


class BatchProviderHTTPError(BatchWorkerError):
    """One explicit non-success HTTP response from OpenAI."""

    def __init__(
        self,
        status_code: int,
        *,
        retryable: bool,
        retry_after_seconds: float | None = None,
    ):
        """Retain safe classification without exposing provider response prose."""
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"OpenAI batch API returned HTTP {status_code}.")


class OpenAIBatchClient:
    """Small finite-timeout OpenAI files and batches API client."""

    def __init__(self, api_key: str, settings: dict[str, object], http=requests):
        """Create a client using an encrypted-profile secret and HTTP transport."""
        self._http = http
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": str(settings.get("organization") or ""),
            "OpenAI-Project": str(settings.get("project") or ""),
        }
        if not self._headers["OpenAI-Organization"]:
            self._headers.pop("OpenAI-Organization")
        if not self._headers["OpenAI-Project"]:
            self._headers.pop("OpenAI-Project")

    def upload(self, jsonl: bytes) -> str:
        """Upload canonical JSONL and return its provider file identifier."""
        response = self._http.post(
            f"{OPENAI_BATCH_BASE_URL}/files",
            headers=self._headers,
            data={"purpose": "batch"},
            files={"file": ("batch.jsonl", jsonl, "application/jsonl")},
            timeout=HTTP_TIMEOUT,
        )
        payload = _provider_json(response)
        file_id = payload.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise BatchWorkerError("OpenAI file upload response omitted its ID.")
        return file_id

    def submit(self, input_file_id: str, model: str) -> str:
        """Create one OpenAI batch for an uploaded JSONL file."""
        response = self._http.post(
            f"{OPENAI_BATCH_BASE_URL}/batches",
            headers={**self._headers, "Content-Type": "application/json"},
            json={
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"model": model},
            },
            timeout=HTTP_TIMEOUT,
        )
        payload = _provider_json(response)
        batch_id = payload.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise BatchWorkerError("OpenAI batch response omitted its ID.")
        return batch_id

    def retrieve(self, batch_id: str) -> dict[str, object]:
        """Retrieve the latest remote state for a submitted batch."""
        response = self._http.get(
            f"{OPENAI_BATCH_BASE_URL}/batches/{batch_id}",
            headers=self._headers,
            timeout=HTTP_TIMEOUT,
        )
        payload = _provider_json(response)
        return payload

    def download(
        self, file_id: str, *, max_bytes: int = MAX_BATCH_JSONL_BYTES
    ) -> bytes:
        """Download one provider result file within its assigned byte budget."""
        response = self._http.get(
            f"{OPENAI_BATCH_BASE_URL}/files/{file_id}/content",
            headers=self._headers,
            timeout=HTTP_TIMEOUT,
            stream=True,
        )
        try:
            if response.status_code >= 400:
                raise _provider_http_error(response)
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise BatchResultError(
                        "OpenAI batch result files exceed the 4 MiB limit."
                    )
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            response.close()

    def delete_file(self, file_id: str) -> None:
        """Delete one known provider file, treating an already-missing file as done."""
        response = self._http.delete(
            f"{OPENAI_BATCH_BASE_URL}/files/{file_id}",
            headers=self._headers,
            timeout=HTTP_TIMEOUT,
        )
        if response.status_code >= 400 and response.status_code != 404:
            raise BatchWorkerError(
                f"OpenAI file deletion returned HTTP {response.status_code}."
            )


def _provider_http_error(response) -> BatchProviderHTTPError:
    """Classify one provider status and retain only safe retry metadata."""
    classification = classify_upstream_error(
        "openai", response.status_code, None, response.headers, None
    )
    return BatchProviderHTTPError(
        response.status_code,
        retryable=classification.category == "transient",
        retry_after_seconds=classification.retry_after_seconds,
    )


def _provider_json(response) -> dict[str, object]:
    if response.status_code >= 400:
        raise _provider_http_error(response)
    try:
        payload = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError) as exc:
        raise BatchWorkerError("OpenAI batch API returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise BatchWorkerError("OpenAI batch API response must be an object.")
    return payload


def _unknown_upload_message() -> str:
    return (
        "Upload outcome is ambiguous and will not be retried. No provider file ID "
        "is available for explicit cleanup; provider-managed retention may expire "
        "the file, which this worker cannot verify."
    )


def _retry_batch_poll(
    database: Database,
    job_id: str,
    owner: str,
    fence: int,
    *,
    retry_after_seconds: float | None,
) -> bool:
    delay = (
        retry_after_seconds
        if retry_after_seconds is not None
        else POLL_INTERVAL_SECONDS
    )
    delay = min(max(delay, 1.0), RETRY_AFTER_MAX_SECONDS)
    return _fenced_update(
        database,
        job_id,
        owner,
        fence,
        status="polling",
        poll_after=datetime.now(timezone.utc) + timedelta(seconds=math.ceil(delay)),
        worker_owner=None,
        lease_expires_at=None,
    )


def _finish_batch_failure(
    database: Database,
    job_id: str,
    owner: str,
    fence: int,
    *,
    status: str,
    error_type: str,
    message: str,
    status_code: int | None = None,
) -> bool:
    now = datetime.now(timezone.utc)
    error: dict[str, object] = {"type": error_type, "message": message}
    if status_code is not None:
        error["status_code"] = status_code
    return _fenced_update(
        database,
        job_id,
        owner,
        fence,
        status=status,
        error_json=error,
        completed_at=now,
        expires_at=now + BATCH_RETENTION,
        worker_owner=None,
        lease_expires_at=None,
    )


def _request_jsonl(request_json: dict[str, object]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in request_json["requests"]
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_output_jsonl(content: bytes) -> list[dict[str, object]]:
    """Parse bounded provider JSONL records into OpenAI response/error objects."""
    if len(content) > MAX_BATCH_JSONL_BYTES:
        raise BatchResultError("OpenAI batch result file exceeds the 4 MiB limit.")
    results: list[dict[str, object]] = []
    for line in content.splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BatchResultError(
                "OpenAI batch result file contains invalid JSONL."
            ) from exc
        if not isinstance(record, dict) or not isinstance(record.get("custom_id"), str):
            raise BatchResultError("OpenAI batch result record is invalid.")
        result: dict[str, object] = {"custom_id": record["custom_id"]}
        if isinstance(record.get("response"), dict):
            result["response"] = record["response"]
        elif isinstance(record.get("error"), dict):
            result["error"] = record["error"]
        else:
            result["error"] = {"message": "Provider result omitted response and error."}
        results.append(result)
    return results


def _normalize_results(
    output_content: bytes | None,
    error_content: bytes | None,
    request_json: dict[str, object],
) -> list[dict[str, object]]:
    """Associate provider output with exactly one result for every input ID."""
    records = []
    if output_content is not None:
        records.extend(parse_output_jsonl(output_content))
    if error_content is not None:
        records.extend(parse_output_jsonl(error_content))
    expected_ids = {record["custom_id"] for record in request_json["requests"]}
    by_id: dict[str, dict[str, object]] = {}
    for record in records:
        custom_id = record["custom_id"]
        if custom_id not in expected_ids or custom_id in by_id:
            raise BatchResultError(
                "OpenAI batch results contain unknown or duplicate custom_id values."
            )
        by_id[custom_id] = record
    for custom_id in expected_ids - by_id.keys():
        by_id[custom_id] = {
            "custom_id": custom_id,
            "error": {"message": "Provider omitted a result for this request."},
        }
    normalized = [by_id[custom_id] for custom_id in sorted(by_id)]
    serialized = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(serialized) > MAX_BATCH_JSONL_BYTES:
        raise BatchResultError("OpenAI batch results exceed the 4 MiB response limit.")
    return normalized


def claim_batch(
    database: Database, owner: str, *, now: datetime | None = None
) -> tuple[str, int] | None:
    """Atomically claim one eligible row with PostgreSQL skip-locked fencing."""
    current = now or datetime.now(timezone.utc)
    lease_end = current + timedelta(seconds=LEASE_SECONDS)
    with database.sessions.begin() as session:
        query = (
            select(BatchJob)
            .where(
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
                BatchJob.expires_at > current,
                or_(BatchJob.poll_after.is_(None), BatchJob.poll_after <= current),
                or_(
                    BatchJob.lease_expires_at.is_(None),
                    BatchJob.lease_expires_at <= current,
                ),
            )
            .order_by(BatchJob.created_at, BatchJob.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        job = session.scalar(query)
        if job is None:
            return None
        if job.status in {"submitting", "uploading"}:
            ambiguous_submit = job.status == "submitting"
            job.status = "unknown_submit" if ambiguous_submit else "unknown_upload"
            job.error_json = {
                "type": "unknown_submit" if ambiguous_submit else "unknown_upload",
                "message": (
                    "Submission outcome is ambiguous; it will not be retried."
                    if ambiguous_submit
                    else _unknown_upload_message()
                ),
            }
            job.completed_at = current
            job.expires_at = current + timedelta(days=7)
            job.worker_owner = None
            job.lease_expires_at = None
            return None
        job.worker_owner = owner
        job.lease_expires_at = lease_end
        job.fencing_token += 1
        if job.status == "queued":
            job.status = "uploading"
        elif job.status == "submitted":
            job.status = "polling"
        return job.id, job.fencing_token


def _load_job_context(
    database: Database, job_id: str
) -> tuple[BatchJob, str, dict[str, object]]:
    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        if job is None:
            raise BatchWorkerError("Claimed batch job disappeared.")
        profile = session.scalar(
            select(ProviderProfile).where(
                ProviderProfile.tenant_id == job.tenant_id,
                ProviderProfile.id == job.profile_id,
                ProviderProfile.provider == "openai",
                ProviderProfile.deleted_at.is_(None),
            )
        )
        if profile is None or not profile.inference_secret_ciphertext:
            raise BatchWorkerError("Batch OpenAI profile is no longer available.")
        secret = database.secret_cipher.decrypt(profile.inference_secret_ciphertext)
        if not isinstance(profile.settings, dict):
            raise BatchWorkerError("Batch OpenAI profile settings are invalid.")
        return job, secret, dict(profile.settings)


def _fenced_update(
    database: Database, job_id: str, owner: str, fence: int, **values
) -> bool:
    now = datetime.now(timezone.utc)
    values["updated_at"] = now
    with database.sessions.begin() as session:
        updated = session.execute(
            update(BatchJob)
            .where(
                BatchJob.id == job_id,
                BatchJob.worker_owner == owner,
                BatchJob.fencing_token == fence,
                BatchJob.lease_expires_at > now,
            )
            .values(**values)
        )
        return updated.rowcount == 1


def _renew_batch_lease(
    database: Database,
    job_id: str,
    owner: str,
    fence: int,
    *,
    now: datetime | None = None,
) -> bool:
    """Extend only a still-live lease owned by this worker and fencing token."""
    moment = now or datetime.now(timezone.utc)
    with database.sessions.begin() as session:
        renewed = session.execute(
            update(BatchJob)
            .where(
                BatchJob.id == job_id,
                BatchJob.worker_owner == owner,
                BatchJob.fencing_token == fence,
                BatchJob.lease_expires_at > moment,
            )
            .values(
                lease_expires_at=moment + timedelta(seconds=LEASE_SECONDS),
                updated_at=moment,
            )
        )
        return renewed.rowcount == 1


@contextmanager
def _lease_heartbeat(database: Database, job_id: str, owner: str, fence: int):
    """Renew a live claim while synchronous provider calls or streams are blocked."""
    stopping = Event()
    lease_lost = Event()

    def heartbeat() -> None:
        while not stopping.wait(LEASE_HEARTBEAT_INTERVAL_SECONDS):
            try:
                if not _renew_batch_lease(database, job_id, owner, fence):
                    lease_lost.set()
                    return
            except SQLAlchemyError:
                lease_lost.set()
                return

    thread = Thread(target=heartbeat, name=f"batch-lease-{job_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopping.set()
        thread.join(timeout=LEASE_HEARTBEAT_INTERVAL_SECONDS * 2)
        if lease_lost.is_set():
            current_app.logger.warning(
                "Batch worker lost lease while provider request was active: job=%s",
                job_id,
            )


def process_batch_claim(
    database: Database,
    job_id: str,
    owner: str,
    fence: int,
    *,
    http=requests,
    now: datetime | None = None,
) -> None:
    """Perform one provider operation outside DB locks then fence persistence."""
    current = now or datetime.now(timezone.utc)
    job, secret, settings = _load_job_context(database, job_id)
    client = OpenAIBatchClient(secret, settings, http=http)
    if job.status == "retry_submit":
        _submit_claim(database, job_id, owner, fence, http=http)
        return
    if job.status in {"queued", "uploading"}:
        if job.upload_attempts >= MAX_BATCH_UPLOAD_ATTEMPTS:
            _finish_batch_failure(
                database,
                job_id,
                owner,
                fence,
                status="failed",
                error_type="provider_http_error",
                message="OpenAI rejected the batch file upload too many times.",
                status_code=429,
            )
            return
        attempt_started = _fenced_update(
            database,
            job_id,
            owner,
            fence,
            status="uploading",
            upload_attempts=job.upload_attempts + 1,
        )
        if not attempt_started:
            return
        try:
            input_id = client.upload(_request_jsonl(job.request_json))
        except BatchProviderHTTPError as exc:
            if exc.status_code == 429 and exc.retryable:
                if job.upload_attempts + 1 >= MAX_BATCH_UPLOAD_ATTEMPTS:
                    _finish_batch_failure(
                        database,
                        job_id,
                        owner,
                        fence,
                        status="failed",
                        error_type="provider_http_error",
                        message="OpenAI rejected the batch file upload too many times.",
                        status_code=429,
                    )
                    return
                retry_after = (
                    exc.retry_after_seconds
                    if exc.retry_after_seconds is not None
                    else POLL_INTERVAL_SECONDS
                )
                delay = min(max(retry_after, 1.0), RETRY_AFTER_MAX_SECONDS)
                _fenced_update(
                    database,
                    job_id,
                    owner,
                    fence,
                    status="queued",
                    poll_after=datetime.now(timezone.utc)
                    + timedelta(seconds=math.ceil(delay)),
                    worker_owner=None,
                    lease_expires_at=None,
                    error_json=None,
                )
            elif not exc.retryable:
                _finish_batch_failure(
                    database,
                    job_id,
                    owner,
                    fence,
                    status="failed",
                    error_type="provider_http_error",
                    message=str(exc),
                    status_code=exc.status_code,
                )
            else:
                _finish_batch_failure(
                    database,
                    job_id,
                    owner,
                    fence,
                    status="unknown_upload",
                    error_type="unknown_upload",
                    message=_unknown_upload_message(),
                    status_code=exc.status_code,
                )
            return
        except (requests.RequestException, BatchWorkerError):
            _finish_batch_failure(
                database,
                job_id,
                owner,
                fence,
                status="unknown_upload",
                error_type="unknown_upload",
                message=_unknown_upload_message(),
            )
            return
        if _fenced_update(
            database,
            job_id,
            owner,
            fence,
            provider_input_file_id=input_id,
            status="submitting",
            poll_after=current,
        ):
            _submit_claim(database, job_id, owner, fence, http=http)
        return
    if job.status in {"submitted", "polling"}:
        if not job.provider_batch_id:
            _fenced_update(
                database,
                job_id,
                owner,
                fence,
                status="failed",
                error_json={
                    "type": "missing_provider_batch_id",
                    "message": "Provider batch ID is missing.",
                },
                completed_at=current,
                expires_at=current + timedelta(days=7),
                worker_owner=None,
                lease_expires_at=None,
            )
            return
        try:
            provider_batch = client.retrieve(job.provider_batch_id)
        except BatchProviderHTTPError as exc:
            if exc.retryable:
                _retry_batch_poll(
                    database,
                    job_id,
                    owner,
                    fence,
                    retry_after_seconds=exc.retry_after_seconds,
                )
            else:
                _finish_batch_failure(
                    database,
                    job_id,
                    owner,
                    fence,
                    status="failed",
                    error_type="provider_http_error",
                    message=str(exc),
                    status_code=exc.status_code,
                )
            return
        except requests.RequestException:
            _retry_batch_poll(
                database,
                job_id,
                owner,
                fence,
                retry_after_seconds=None,
            )
            return
        provider_status = provider_batch.get("status")
        if provider_status in {"completed", "failed", "expired", "cancelled"}:
            output_id = provider_batch.get("output_file_id")
            error_id = provider_batch.get("error_file_id")
            output_id = output_id if isinstance(output_id, str) else None
            error_id = error_id if isinstance(error_id, str) else None
            if not _fenced_update(
                database,
                job_id,
                owner,
                fence,
                provider_output_file_id=output_id,
                provider_error_file_id=error_id,
            ):
                return
            try:
                output_content = client.download(output_id) if output_id else None
                error_content = (
                    client.download(
                        error_id,
                        max_bytes=MAX_BATCH_JSONL_BYTES
                        - (len(output_content) if output_content else 0),
                    )
                    if error_id
                    else None
                )
                results = _normalize_results(
                    output_content, error_content, job.request_json
                )
            except BatchProviderHTTPError as exc:
                if exc.retryable:
                    _retry_batch_poll(
                        database,
                        job_id,
                        owner,
                        fence,
                        retry_after_seconds=exc.retry_after_seconds,
                    )
                else:
                    _finish_batch_failure(
                        database,
                        job_id,
                        owner,
                        fence,
                        status="failed",
                        error_type="provider_http_error",
                        message=str(exc),
                        status_code=exc.status_code,
                    )
                return
            except requests.RequestException:
                _retry_batch_poll(
                    database,
                    job_id,
                    owner,
                    fence,
                    retry_after_seconds=None,
                )
                return
            except BatchResultError:
                _fenced_update(
                    database,
                    job_id,
                    owner,
                    fence,
                    status="failed",
                    error_json={
                        "type": "invalid_provider_results",
                        "message": "OpenAI batch results were invalid or exceeded the response limit.",
                    },
                    completed_at=current,
                    expires_at=current + timedelta(days=7),
                    worker_owner=None,
                    lease_expires_at=None,
                )
                return
            status = (
                "completed"
                if provider_status == "completed"
                else "expired" if provider_status == "expired" else "failed"
            )
            _fenced_update(
                database,
                job_id,
                owner,
                fence,
                status=status,
                provider_output_file_id=output_id,
                provider_error_file_id=error_id,
                results_json=results,
                error_json=(
                    {"type": str(provider_status)} if status == "failed" else None
                ),
                completed_at=current,
                expires_at=current + timedelta(days=7),
                worker_owner=None,
                lease_expires_at=None,
            )
            return
        _fenced_update(
            database,
            job_id,
            owner,
            fence,
            status="polling",
            poll_after=current + timedelta(seconds=POLL_INTERVAL_SECONDS),
            worker_owner=None,
            lease_expires_at=None,
        )


def _submit_claim(
    database: Database, job_id: str, owner: str, fence: int, http=requests
) -> None:
    """Submit an uploaded file exactly once; ambiguous attempts become terminal."""
    job, secret, settings = _load_job_context(database, job_id)
    if (
        job.status not in {"submitting", "retry_submit"}
        or not job.provider_input_file_id
    ):
        return
    if job.submit_attempts >= MAX_BATCH_SUBMIT_ATTEMPTS:
        _finish_batch_failure(
            database,
            job_id,
            owner,
            fence,
            status="failed",
            error_type="provider_http_error",
            message="OpenAI rejected the batch submission too many times.",
            status_code=429,
        )
        return
    client = OpenAIBatchClient(secret, settings, http=http)
    attempt_started = _fenced_update(
        database,
        job_id,
        owner,
        fence,
        status="submitting",
        submit_attempts=job.submit_attempts + 1,
    )
    if not attempt_started:
        return
    try:
        provider_batch_id = client.submit(job.provider_input_file_id, job.model)
    except BatchProviderHTTPError as exc:
        if exc.retryable and exc.status_code == 429:
            if job.submit_attempts + 1 >= MAX_BATCH_SUBMIT_ATTEMPTS:
                _finish_batch_failure(
                    database,
                    job_id,
                    owner,
                    fence,
                    status="failed",
                    error_type="provider_http_error",
                    message="OpenAI rejected the batch submission too many times.",
                    status_code=429,
                )
                return
            retry_after = (
                exc.retry_after_seconds
                if exc.retry_after_seconds is not None
                else POLL_INTERVAL_SECONDS
            )
            delay = min(max(retry_after, 1.0), RETRY_AFTER_MAX_SECONDS)
            _fenced_update(
                database,
                job_id,
                owner,
                fence,
                status="retry_submit",
                poll_after=datetime.now(timezone.utc)
                + timedelta(seconds=math.ceil(delay)),
                worker_owner=None,
                lease_expires_at=None,
                error_json=None,
            )
        elif exc.status_code < 500:
            _finish_batch_failure(
                database,
                job_id,
                owner,
                fence,
                status="failed",
                error_type="provider_http_error",
                message=str(exc),
                status_code=exc.status_code,
            )
        else:
            _finish_batch_failure(
                database,
                job_id,
                owner,
                fence,
                status="unknown_submit",
                error_type="unknown_submit",
                message="OpenAI returned a server error; submission acceptance is ambiguous.",
                status_code=exc.status_code,
            )
        return
    except (requests.RequestException, BatchWorkerError):
        _finish_batch_failure(
            database,
            job_id,
            owner,
            fence,
            status="unknown_submit",
            error_type="unknown_submit",
            message="OpenAI submission outcome is ambiguous; it will not be retried.",
        )
        return
    _fenced_update(
        database,
        job_id,
        owner,
        fence,
        provider_batch_id=provider_batch_id,
        status="submitted",
        poll_after=datetime.now(timezone.utc),
        worker_owner=None,
        lease_expires_at=None,
    )


def cleanup_expired_batches(
    database: Database, *, http=requests, limit: int = CLEANUP_LIMIT
) -> int:
    """Claim a bounded expiry set, delete remote files, then remove local payloads."""
    now = datetime.now(timezone.utc)
    owner = str(uuid4())
    lease_end = now + timedelta(seconds=LEASE_SECONDS)
    with database.sessions.begin() as session:
        jobs = list(
            session.scalars(
                select(BatchJob)
                .where(
                    BatchJob.expires_at <= now,
                    or_(
                        BatchJob.lease_expires_at.is_(None),
                        BatchJob.lease_expires_at <= now,
                    ),
                )
                .order_by(BatchJob.expires_at, BatchJob.id)
                .with_for_update(skip_locked=True)
                .limit(limit)
            )
        )
        targets = []
        for job in jobs:
            if job.status == "submitting":
                job.status = "unknown_submit"
                job.error_json = {
                    "type": "unknown_submit",
                    "message": "Submission outcome is ambiguous.",
                }
                job.completed_at = now
                job.expires_at = now + timedelta(days=7)
                continue
            if job.status not in {
                "completed",
                "failed",
                "expired",
                "unknown_upload",
                "unknown_submit",
            }:
                job.status = "expired"
            job.worker_owner = owner
            job.lease_expires_at = lease_end
            job.fencing_token += 1
            targets.append(
                (
                    job.id,
                    job.fencing_token,
                    job.tenant_id,
                    job.profile_id,
                    job.provider_input_file_id,
                    job.provider_output_file_id,
                    job.provider_error_file_id,
                )
            )

    deleted = 0
    for job_id, fence, tenant_id, profile_id, input_id, output_id, error_id in targets:
        file_ids = tuple(dict.fromkeys((input_id, output_id, error_id)))
        try:
            if file_ids:
                with database.sessions() as session:
                    profile = session.scalar(
                        select(ProviderProfile).where(
                            ProviderProfile.tenant_id == tenant_id,
                            ProviderProfile.id == profile_id,
                            ProviderProfile.provider == "openai",
                        )
                    )
                    if profile is None or not profile.inference_secret_ciphertext:
                        raise BatchWorkerError(
                            "OpenAI profile is unavailable for expired file cleanup."
                        )
                    secret = database.secret_cipher.decrypt(
                        profile.inference_secret_ciphertext
                    )
                    if not isinstance(profile.settings, dict):
                        raise BatchWorkerError("OpenAI profile settings are invalid.")
                    settings = dict(profile.settings)
                remote = OpenAIBatchClient(secret, settings, http=http)
                for file_id in file_ids:
                    if file_id:
                        remote.delete_file(file_id)

            with database.sessions.begin() as session:
                removed = session.execute(
                    delete(BatchJob).where(
                        BatchJob.id == job_id,
                        BatchJob.worker_owner == owner,
                        BatchJob.fencing_token == fence,
                        BatchJob.expires_at <= now,
                    )
                )
                deleted += removed.rowcount
        except Exception as exc:  # noqa: BLE001 - isolate each retained batch target
            try:
                released = _fenced_update(
                    database,
                    job_id,
                    owner,
                    fence,
                    worker_owner=None,
                    lease_expires_at=None,
                    expires_at=now + timedelta(seconds=CLEANUP_RETRY_DELAY_SECONDS),
                )
            except (
                Exception
            ) as update_error:  # noqa: BLE001 - preserve target isolation
                logging.getLogger(__name__).warning(
                    "Batch file retention cleanup deferral failed "
                    "(cleanup=%s persistence=%s)",
                    type(exc).__name__,
                    type(update_error).__name__,
                )
                continue
            if released:
                logging.getLogger(__name__).warning(
                    "Batch file retention cleanup deferred (%s)", type(exc).__name__
                )
            else:
                logging.getLogger(__name__).warning(
                    "Batch file retention cleanup lost ownership (%s)",
                    type(exc).__name__,
                )
    return deleted


def run_batch_worker(
    app: Flask, *, http=requests, poll_interval: float = POLL_INTERVAL_SECONDS
) -> None:
    """Poll PostgreSQL indefinitely with bounded sleeps and a process owner UUID."""
    if (
        app.config.get("AUTH_MODE") != AUTH_MODE_TENANT
        or app.config.get("TENANT_CONFIG_SOURCE") != TENANT_CONFIG_DATABASE
    ):
        raise BatchWorkerError("batch-worker requires database tenant mode.")
    database = app.extensions.get("database")
    if not isinstance(database, Database):
        raise BatchWorkerError("batch-worker has no configured PostgreSQL database.")
    if database.engine.dialect.name != "postgresql":
        raise BatchWorkerError(
            "batch-worker requires PostgreSQL for shared queue coordination."
        )
    try:
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise BatchWorkerError("batch-worker cannot connect to PostgreSQL.") from exc
    owner = str(uuid4())
    backoff = poll_interval
    batch_cleanup_at = time.monotonic()
    scheduler_cleanup_at = batch_cleanup_at + RETENTION_CLEANUP_INTERVAL_SECONDS / 2
    while True:
        try:
            monotonic_now = time.monotonic()
            if monotonic_now >= batch_cleanup_at:
                batch_cleanup_at = monotonic_now + RETENTION_CLEANUP_INTERVAL_SECONDS
                try:
                    cleanup_expired_batches(database, http=http, limit=CLEANUP_LIMIT)
                except (BatchWorkerError, requests.RequestException) as exc:
                    app.logger.warning(
                        "Batch file retention cleanup failed (%s)", type(exc).__name__
                    )
            if monotonic_now >= scheduler_cleanup_at:
                ProviderBudgetScheduler(
                    database.sessions, database.secret_cipher
                ).cleanup(batch_size=CLEANUP_LIMIT)
                ProviderConcurrencyController(
                    database.sessions, database.secret_cipher
                ).cleanup(batch_size=CLEANUP_LIMIT)
                scheduler_cleanup_at = (
                    monotonic_now + RETENTION_CLEANUP_INTERVAL_SECONDS
                )

            claim = claim_batch(database, owner)
            if claim is None:
                time.sleep(poll_interval)
                backoff = poll_interval
                continue
            job_id, fence = claim
            with _lease_heartbeat(database, job_id, owner, fence):
                process_batch_claim(database, job_id, owner, fence, http=http)
            backoff = poll_interval
        except SQLAlchemyError as exc:
            raise BatchWorkerError(
                "batch-worker lost PostgreSQL connectivity."
            ) from exc
        except (
            Exception
        ) as exc:  # noqa: B902 - keep the worker alive after a job-level failure.
            app.logger.error("Batch worker iteration failed (%s)", type(exc).__name__)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_POLL_INTERVAL_SECONDS)


@click.command("batch-worker")
@click.option(
    "--poll-interval", type=click.FloatRange(min=0.1), default=POLL_INTERVAL_SECONDS
)
def batch_worker_command(poll_interval: float) -> None:
    """Run the PostgreSQL-backed OpenAI batch worker."""
    app = current_app._get_current_object()
    if app.config.get("BATCH_WORKER_ENABLED") is not True:
        raise click.ClickException("batch-worker requires BATCH_WORKER_ENABLED=true.")
    try:
        run_batch_worker(app, poll_interval=poll_interval)
    except BatchWorkerError as exc:
        raise click.ClickException(str(exc)) from exc


def register_batch_worker_command(app: Flask) -> None:
    """Register the separate batch worker command."""
    app.cli.add_command(batch_worker_command)
