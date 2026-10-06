"""Tests for the tenant-scoped OpenAI batch pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests
from sqlalchemy import select

from app.batch_service import (
    MAX_BATCH_JSONL_BYTES,
    BatchValidationError,
    normalize_batch_payload,
    validate_idempotency_key,
)
from app.batch_worker import (
    BatchProviderHTTPError,
    BatchWorkerError,
    OpenAIBatchClient,
    _fenced_update,
    _renew_batch_lease,
    claim_batch,
    cleanup_expired_batches,
    parse_output_jsonl,
    process_batch_claim,
    run_batch_worker,
)
from app.persistence.admin_ops import delete_provider_profile
from app.persistence.models import (
    BatchJob,
    ProviderCatalogEntry,
    ProviderProfile,
    Tenant,
)
from app.persistence.provider_scheduler import ProviderBudgetScheduler


def _prepare_batch_app(admin_app):
    """Enable database tenant auth and seed one ready OpenAI catalog profile."""
    admin_app.config.update(
        AUTH_MODE="tenant",
        TENANT_CONFIG_SOURCE="database",
        BATCH_WORKER_ENABLED=True,
    )
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        profile = ProviderProfile(
            id="openai-batch-profile",
            tenant_id="acme",
            provider="openai",
            display_name="OpenAI batch",
            settings={"organization": "org-test", "project": "proj_test"},
            default_model="gpt-5.4",
            inference_secret_ciphertext=database.secret_cipher.encrypt("sk-secret"),
        )
        session.add(profile)
        session.flush()
        session.add(
            ProviderCatalogEntry(
                profile_id=profile.id,
                model_id="gpt-5.4",
                source="provider",
            )
        )
        session.add(
            Tenant(
                id="beta",
                api_key_hash=hashlib.sha256(b"beta-key").hexdigest(),
                custom_model_id="cursor-beta-model",
            )
        )
    return admin_app.test_client(), database


def _headers(key="cursor-key", idempotency_key="batch-key-1"):
    return {
        "Authorization": f"Bearer {key}",
        "Idempotency-Key": idempotency_key,
    }


def _payload(prompt="hello"):
    return {
        "model": "gpt-5.4",
        "requests": [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        ],
    }


def _create_job(client):
    response = client.post("/v1/batches", json=_payload(), headers=_headers())
    assert response.status_code == 202
    return response.json["id"]


def test_batch_submit_is_unavailable_without_an_enabled_worker(admin_app):
    """Do not accept queued work when no configured worker can process it."""
    client, database = _prepare_batch_app(admin_app)
    admin_app.config["BATCH_WORKER_ENABLED"] = False

    response = client.post("/v1/batches", json=_payload(), headers=_headers())

    assert response.status_code == 503
    assert response.json["error"]["type"] == "batch_worker_unavailable"
    assert response.headers["Retry-After"]
    with database.sessions() as session:
        assert session.scalar(select(BatchJob.id)) is None


def test_submit_persists_canonical_jsonl_and_replays_idempotently(admin_app):
    """A same-key normalized replay returns the original persisted queued job."""
    client, database = _prepare_batch_app(admin_app)

    first = client.post("/v1/batches", json=_payload(), headers=_headers())
    replay = client.post("/v1/batches", json=_payload(), headers=_headers())

    assert first.status_code == replay.status_code == 202
    assert first.json["object"] == "batch"
    assert len(first.json["id"]) == 38
    assert first.json["id"].startswith("batch_")
    assert BatchJob.__table__.c.id.type.length == 38
    assert first.json["status"] == "validating"
    assert first.json["id"] == replay.json["id"]
    with database.sessions() as session:
        job = session.scalar(select(BatchJob))
        assert job is not None
        assert job.request_json == _payload()
        assert job.idempotency_key_hash == hashlib.sha256(b"batch-key-1").hexdigest()
        assert "sk-secret" not in json.dumps(job.request_json)


def test_expired_idempotency_key_allows_new_job_without_deleting_remote_owner(
    admin_app,
):
    """Expiry releases the key but retains remote-file ownership for cleanup."""
    client, database = _prepare_batch_app(admin_app)
    old_id = _create_job(client)
    with database.sessions.begin() as session:
        old = session.get(BatchJob, old_id)
        old.status = "completed"
        old.provider_input_file_id = "file-still-owned"
        old.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    replacement = client.post("/v1/batches", json=_payload("new"), headers=_headers())

    assert replacement.status_code == 202
    assert replacement.json["id"] != old_id
    with database.sessions() as session:
        old = session.get(BatchJob, old_id)
        assert old is not None
        assert old.provider_input_file_id == "file-still-owned"
        assert old.idempotency_key_hash is None


def test_expired_ambiguous_job_is_tombstoned_without_losing_remote_file_id(admin_app):
    """Expiry terminalizes stale submits while preserving remote cleanup ownership."""
    client, database = _prepare_batch_app(admin_app)
    old_id = _create_job(client)
    with database.sessions.begin() as session:
        old = session.get(BatchJob, old_id)
        old.status = "submitting"
        old.provider_input_file_id = "file-known-before-submit"
        old.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    replacement = client.post("/v1/batches", json=_payload("new"), headers=_headers())

    assert replacement.status_code == 202
    with database.sessions() as session:
        old = session.get(BatchJob, old_id)
        assert old.status == "unknown_submit"
        assert old.idempotency_key_hash is None
        assert old.provider_input_file_id == "file-known-before-submit"
        assert old.error_json["type"] == "unknown_submit"


def test_expired_job_with_live_worker_lease_does_not_replay_or_release_key(admin_app):
    """An unexpired worker fence prevents releasing an expired active job key."""
    client, database = _prepare_batch_app(admin_app)
    old_id = _create_job(client)
    with database.sessions.begin() as session:
        old = session.get(BatchJob, old_id)
        old.status = "submitting"
        old.worker_owner = "worker-active"
        old.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
        old.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        key_hash = old.idempotency_key_hash

    response = client.post("/v1/batches", json=_payload(), headers=_headers())

    assert response.status_code == 503
    assert response.json["error"]["type"] == "batch_job_still_active"
    with database.sessions() as session:
        old = session.get(BatchJob, old_id)
        assert old.idempotency_key_hash == key_hash
        assert old.status == "submitting"


def test_batch_queue_limit_is_enforced_per_tenant_after_idempotent_replay(admin_app):
    """A tenant cannot grow an unbounded active queue, but matching replay works."""
    client, database = _prepare_batch_app(admin_app)
    admin_app.config["BATCH_MAX_QUEUED_JOBS"] = 1

    first = client.post("/v1/batches", json=_payload(), headers=_headers())
    replay = client.post("/v1/batches", json=_payload(), headers=_headers())
    rejected = client.post(
        "/v1/batches",
        json=_payload("second"),
        headers=_headers(idempotency_key="batch-key-2"),
    )

    assert first.status_code == replay.status_code == 202
    assert first.json["id"] == replay.json["id"]
    assert rejected.status_code == 429
    assert rejected.json["error"]["type"] == "batch_queue_full"
    assert rejected.headers["Retry-After"]
    with database.sessions() as session:
        assert len(session.scalars(select(BatchJob)).all()) == 1


def test_expired_active_job_does_not_consume_tenant_queue_capacity(admin_app):
    """Jobs past retention stop counting even before background cleanup runs."""
    client, database = _prepare_batch_app(admin_app)
    admin_app.config["BATCH_MAX_QUEUED_JOBS"] = 1
    first_id = _create_job(client)
    with database.sessions.begin() as session:
        session.get(BatchJob, first_id).expires_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)

    replacement = client.post(
        "/v1/batches",
        json=_payload("replacement"),
        headers=_headers(idempotency_key="batch-key-2"),
    )

    assert replacement.status_code == 202
    assert replacement.json["id"] != first_id


def test_retry_submit_counts_toward_tenant_queue_limit(admin_app):
    """A scheduled submission retry must continue consuming tenant capacity."""
    client, database = _prepare_batch_app(admin_app)
    admin_app.config["BATCH_MAX_QUEUED_JOBS"] = 1
    _create_job(client)
    with database.sessions.begin() as session:
        job = session.scalar(select(BatchJob))
        job.status = "retry_submit"

    rejected = client.post(
        "/v1/batches",
        json=_payload("second"),
        headers=_headers(idempotency_key="batch-key-2"),
    )

    assert rejected.status_code == 429
    assert rejected.json["error"]["type"] == "batch_queue_full"


def test_submit_rejects_reused_idempotency_key_with_different_payload(admin_app):
    """A tenant cannot reuse an active idempotency key for changed input."""
    client, _database = _prepare_batch_app(admin_app)
    assert (
        client.post("/v1/batches", json=_payload(), headers=_headers()).status_code
        == 202
    )

    conflict = client.post("/v1/batches", json=_payload("changed"), headers=_headers())

    assert conflict.status_code == 409
    assert "batch-key-1" not in conflict.get_data(as_text=True)
    assert "changed" not in conflict.get_data(as_text=True)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"model": "", "requests": []},
        {"model": "gpt-5.4", "requests": []},
        {
            "model": "gpt-5.4",
            "requests": [
                {
                    "custom_id": "a",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "gpt-5.5"},
                }
            ],
        },
        {
            "model": "gpt-5.4",
            "requests": [
                {
                    "custom_id": "a",
                    "method": "GET",
                    "url": "/v1/chat/completions",
                    "body": {"model": "gpt-5.4"},
                }
            ],
        },
    ],
)
def test_submit_rejects_invalid_batch_records_before_persistence(admin_app, payload):
    """Invalid shape, empty requests, mismatched model, and methods are rejected."""
    client, database = _prepare_batch_app(admin_app)

    response = client.post("/v1/batches", json=payload, headers=_headers())

    assert response.status_code == 400
    with database.sessions() as session:
        assert session.scalar(select(BatchJob.id)) is None


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_permanent_batch_poll_http_errors_fail_without_reclaim_loop(
    admin_app, requests_mock, status_code
):
    """Permanent provider retrieval responses terminate the job immediately."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "submitted"
        job.provider_batch_id = "batch-retrieval-error"
    requests_mock.get(
        "https://api.openai.com/v1/batches/batch-retrieval-error",
        status_code=status_code,
        json={"error": {"message": "sensitive provider response"}},
    )

    claim = claim_batch(database, "worker-permanent-poll-error")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-permanent-poll-error", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "failed"
        assert job.error_json["type"] == "provider_http_error"
        assert job.error_json["status_code"] == status_code
        assert "sensitive provider response" not in str(job.error_json)
        assert job.worker_owner is None
        assert job.lease_expires_at is None
    assert claim_batch(database, "worker-no-loop") is None


def test_transient_batch_poll_error_respects_retry_after(
    admin_app, requests_mock, monkeypatch
):
    """Transient poll failures wait Retry-After from the response time."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "submitted"
        job.provider_batch_id = "batch-transient-poll-error"
    response_time = datetime.now(timezone.utc)
    requests_mock.get(
        "https://api.openai.com/v1/batches/batch-transient-poll-error",
        status_code=429,
        headers={"Retry-After": "45"},
        json={"error": {"message": "retry later"}},
    )
    original_retrieve = OpenAIBatchClient.retrieve

    def slow_retrieve(batch_client, batch_id):
        try:
            return original_retrieve(batch_client, batch_id)
        except BatchProviderHTTPError:
            monkeypatch.setattr(
                "app.batch_worker.datetime",
                type(
                    "DelayedDateTime",
                    (datetime,),
                    {
                        "now": classmethod(
                            lambda cls, tz=None: response_time + timedelta(seconds=60)
                        )
                    },
                ),
            )
            raise

    monkeypatch.setattr(OpenAIBatchClient, "retrieve", slow_retrieve)
    claim = claim_batch(database, "worker-transient-poll-error")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-transient-poll-error", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "polling"
        poll_after = job.poll_after.replace(tzinfo=timezone.utc)
        assert poll_after >= response_time + timedelta(seconds=104)
        assert job.worker_owner is None
    assert claim_batch(database, "worker-too-early") is None


def test_batch_status_and_results_are_tenant_scoped(admin_app):
    """Another database-authenticated tenant receives 404 for the job and results."""
    client, _database = _prepare_batch_app(admin_app)
    submitted = client.post("/v1/batches", json=_payload(), headers=_headers())
    job_id = submitted.json["id"]

    status = client.get(f"/v1/batches/{job_id}", headers=_headers("beta-key"))
    results = client.get(f"/v1/batches/{job_id}/results", headers=_headers("beta-key"))

    assert submitted.status_code == 202
    assert status.status_code == results.status_code == 404


@pytest.mark.parametrize("status", ["unknown_upload", "unknown_submit"])
def test_unknown_terminal_batch_results_are_available(admin_app, status):
    """Ambiguous terminal jobs return their empty result list, not not-ready."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = status
        job.results_json = None

    response = client.get(f"/v1/batches/{job_id}/results", headers=_headers())

    assert response.status_code == 200
    assert response.json == {"object": "list", "data": []}


def test_single_and_environment_tenant_modes_are_explicitly_rejected(admin_app):
    """Batch routes require database-backed tenant authentication, not shared auth."""
    client, _database = _prepare_batch_app(admin_app)
    admin_app.config.update(AUTH_MODE="single", TENANT_CONFIG_SOURCE="environment")

    response = client.post(
        "/v1/batches", json=_payload(), headers=_headers("test-service-api-key")
    )

    assert response.status_code == 400
    assert "database tenant mode" in response.get_data(as_text=True).lower()


def test_batch_idempotency_key_is_required_and_safely_limited(admin_app):
    """The API requires a bounded safe idempotency key and persists no plaintext key."""
    client, database = _prepare_batch_app(admin_app)

    missing = client.post(
        "/v1/batches", json=_payload(), headers={"Authorization": "Bearer cursor-key"}
    )
    invalid = client.post(
        "/v1/batches",
        json=_payload(),
        headers={**_headers(), "Idempotency-Key": "unsafe key"},
    )

    assert missing.status_code == invalid.status_code == 400
    with database.sessions() as session:
        assert session.scalar(select(BatchJob.id)) is None


def test_batch_results_are_associated_and_sorted_by_custom_id(admin_app):
    """Stored results are exposed in deterministic custom_id order."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "completed"
        job.completed_at = job.created_at
        job.results_json = [
            {"custom_id": "z", "response": {"status_code": 200, "body": {}}},
            {"custom_id": "a", "error": {"message": "failed"}},
        ]

    result = client.get(f"/v1/batches/{job_id}/results", headers=_headers())

    assert result.status_code == 200
    assert [item["custom_id"] for item in result.json["data"]] == ["a", "z"]


def test_batch_request_body_limit_returns_json_413_without_global_limit(admin_app):
    """Reject oversized batch HTTP bodies before JSON parsing on this route."""
    client, _database = _prepare_batch_app(admin_app)
    body = json.dumps(
        {"model": "gpt-5.4", "requests": [], "padding": "x" * (5 * 1024 * 1024)}
    ).encode()

    response = client.post(
        "/v1/batches",
        data=body,
        content_type="application/json",
        headers=_headers(),
    )

    assert response.status_code == 413
    assert response.is_json
    assert response.json["error"]["type"] == "request_too_large"
    assert admin_app.config.get("MAX_CONTENT_LENGTH") is None


def test_input_jsonl_is_bounded_before_persistence(admin_app):
    """The aggregate canonical UTF-8 JSONL cap is checked before a row is created."""
    client, database = _prepare_batch_app(admin_app)
    payload = _payload("x" * (4 * 1024 * 1024))

    response = client.post("/v1/batches", json=payload, headers=_headers())

    assert response.status_code == 400
    with database.sessions() as session:
        assert session.scalar(select(BatchJob.id)) is None


def test_batch_validation_enforces_unique_ids_count_and_utf8_jsonl_size():
    """Validation limits count, uniqueness, key syntax, and encoded JSONL bytes."""
    key_digest = validate_idempotency_key("batch:valid-key")
    with pytest.raises(BatchValidationError, match="unique"):
        normalize_batch_payload(
            {"model": "gpt-5.4", "requests": [_payload()["requests"][0]] * 2},
            key_digest,
        )
    oversized = _payload("ä" * (max_jsonl_bytes := 2 * 1024 * 1024))
    with pytest.raises(BatchValidationError, match="4 MiB"):
        normalize_batch_payload(oversized, key_digest)
    too_many = {
        "model": "gpt-5.4",
        "requests": [
            {
                "custom_id": str(index),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-5.4"},
            }
            for index in range(1001)
        ],
    }
    with pytest.raises(BatchValidationError, match="1000"):
        normalize_batch_payload(too_many, key_digest)
    with pytest.raises(BatchValidationError, match="Idempotency-Key"):
        validate_idempotency_key("unsafe key")
    assert max_jsonl_bytes > 0


def test_worker_uploads_and_submits_once_with_openai_scope_headers(
    admin_app, requests_mock
):
    """A claimed job uploads canonical JSONL and submits the fixed endpoint/window."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    requests_mock.post("https://api.openai.com/v1/files", json={"id": "file-input"})
    requests_mock.post("https://api.openai.com/v1/batches", json={"id": "batch-1"})

    claim = claim_batch(database, "worker-1")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-1", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "submitted"
        assert job.provider_input_file_id == "file-input"
        assert job.provider_batch_id == "batch-1"
    upload, submit = requests_mock.request_history
    assert upload.headers["Authorization"] == "Bearer sk-secret"
    assert upload.headers["OpenAI-Organization"] == "org-test"
    assert upload.headers["OpenAI-Project"] == "proj_test"
    assert b'"custom_id":"request-1"' in upload.body
    assert submit.json() == {
        "input_file_id": "file-input",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"model": "gpt-5.4"},
    }


def test_worker_maps_output_and_error_jsonl_by_custom_id(admin_app, requests_mock):
    """Terminal provider output and error files are persisted as sorted records."""
    client, database = _prepare_batch_app(admin_app)
    payload = _payload()
    payload["requests"].append(
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-5.4", "messages": []},
        }
    )
    submitted = client.post("/v1/batches", json=payload, headers=_headers())
    job_id = submitted.json["id"]
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "submitted"
        job.provider_batch_id = "batch-1"
        job.provider_input_file_id = "file-input"
    requests_mock.get(
        "https://api.openai.com/v1/batches/batch-1",
        json={
            "status": "completed",
            "output_file_id": "file-output",
            "error_file_id": "file-error",
        },
    )
    requests_mock.get(
        "https://api.openai.com/v1/files/file-output/content",
        text='{"custom_id":"request-1","response":{"status_code":200,"body":{"ok":true}}}\n',
    )
    requests_mock.get(
        "https://api.openai.com/v1/files/file-error/content",
        text='{"custom_id":"request-2","error":{"code":"failed"}}\n',
    )

    claim = claim_batch(database, "worker-1")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-1", claim[1])

    response = client.get(f"/v1/batches/{job_id}/results", headers=_headers())
    assert response.status_code == 200
    assert [result["custom_id"] for result in response.json["data"]] == [
        "request-1",
        "request-2",
    ]
    assert response.json["data"][1]["error"] == {"code": "failed"}


def test_expired_provider_batch_is_persisted_as_expired(admin_app, requests_mock):
    """Preserve the provider's expired state separately from failure."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "submitted"
        job.provider_batch_id = "batch-expired"
    requests_mock.get(
        "https://api.openai.com/v1/batches/batch-expired",
        json={"status": "expired"},
    )

    claim = claim_batch(database, "worker-expired")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-expired", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "expired"
        assert job.expires_at > job.completed_at


def test_invalid_provider_results_fail_and_retain_file_ids(admin_app, requests_mock):
    """Malformed result files terminate safely after persisting their cleanup IDs."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "submitted"
        job.provider_batch_id = "batch-invalid"
    requests_mock.get(
        "https://api.openai.com/v1/batches/batch-invalid",
        json={"status": "completed", "output_file_id": "file-invalid"},
    )
    requests_mock.get(
        "https://api.openai.com/v1/files/file-invalid/content",
        text="not-json\\n",
    )

    claim = claim_batch(database, "worker-invalid")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-invalid", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "failed"
        assert job.provider_output_file_id == "file-invalid"
        assert job.error_json["type"] == "invalid_provider_results"


def test_combined_provider_results_are_limited_to_input_cap(admin_app, requests_mock):
    """The combined output/error download cannot exceed the bounded JSONL cap."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "submitted"
        job.provider_batch_id = "batch-large"
    requests_mock.get(
        "https://api.openai.com/v1/batches/batch-large",
        json={
            "status": "completed",
            "output_file_id": "file-large-output",
            "error_file_id": "file-large-error",
        },
    )
    requests_mock.get(
        "https://api.openai.com/v1/files/file-large-output/content",
        content=b"x" * (MAX_BATCH_JSONL_BYTES - 1),
    )
    requests_mock.get(
        "https://api.openai.com/v1/files/file-large-error/content",
        content=b"x" * 2,
    )

    claim = claim_batch(database, "worker-large")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-large", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "failed"
        assert job.provider_output_file_id == "file-large-output"
        assert job.provider_error_file_id == "file-large-error"


def test_worker_submits_due_retry_once_and_respects_retry_after(
    admin_app, monkeypatch, requests_mock, mocker
):
    """The worker must not immediately resubmit a retry after a 429."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "retry_submit"
        job.provider_input_file_id = "file-input"
        job.poll_after = datetime.now(timezone.utc) - timedelta(seconds=1)
    requests_mock.post(
        "https://api.openai.com/v1/batches",
        status_code=429,
        headers={"Retry-After": "30"},
        json={"error": {"code": "rate_limit_exceeded"}},
    )
    monkeypatch.setattr(database.engine.dialect, "name", "postgresql")
    monkeypatch.setattr("app.batch_worker.time.monotonic", lambda: 0.0)
    mocker.patch("app.batch_worker.cleanup_expired_batches")
    mocker.patch.object(ProviderBudgetScheduler, "cleanup")
    mocker.patch("app.batch_worker.ProviderConcurrencyController.cleanup")
    mocker.patch("app.batch_worker._lease_heartbeat", return_value=nullcontext())
    claim_job = claim_batch

    class StopWorker(BaseException):
        pass

    def claim_once(*args, **kwargs):
        claim = claim_job(*args, **kwargs)
        if claim is None:
            raise StopWorker
        return claim

    monkeypatch.setattr("app.batch_worker.claim_batch", claim_once)
    with admin_app.app_context(), pytest.raises(StopWorker):
        run_batch_worker(admin_app)

    assert len(requests_mock.request_history) == 1
    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "retry_submit"
        poll_after = job.poll_after.replace(tzinfo=timezone.utc)
        assert poll_after > datetime.now(timezone.utc)


@pytest.mark.parametrize(
    ("status_code", "expected_status", "expected_error"),
    [(400, "failed", "provider_http_error"), (500, "unknown_submit", "unknown_submit")],
)
def test_batch_submit_http_errors_are_terminal_or_ambiguous(
    admin_app, requests_mock, status_code, expected_status, expected_error
):
    """Deterministic provider 4xx fails; 5xx remains acceptance-ambiguous."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    requests_mock.post("https://api.openai.com/v1/files", json={"id": "file-input"})
    requests_mock.post(
        "https://api.openai.com/v1/batches",
        status_code=status_code,
        json={"error": {"message": "must not be persisted"}},
    )

    claim = claim_batch(database, "worker-http")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-http", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == expected_status
        assert job.error_json["type"] == expected_error
        assert job.error_json["status_code"] == status_code
        assert "must not be persisted" not in str(job.error_json)


def test_batch_status_uses_openai_public_lifecycle_values(admin_app):
    """Internal worker statuses map to OpenAI's public Batch API states."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "retry_submit"

    response = client.get(f"/v1/batches/{job_id}", headers=_headers())

    assert response.status_code == 200
    assert response.json["status"] == "validating"


def test_rate_limited_batch_upload_retries_after_retry_after(admin_app, requests_mock):
    """An explicit upload 429 is retried after provider guidance."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    requests_mock.post(
        "https://api.openai.com/v1/files",
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "3"},
                "json": {"error": {"code": "rate_limit_exceeded"}},
            },
            {"status_code": 200, "json": {"id": "file-input"}},
        ],
    )
    requests_mock.post("https://api.openai.com/v1/batches", json={"id": "batch-1"})

    first_claim = claim_batch(database, "worker-1")
    assert first_claim is not None
    process_batch_claim(database, job_id, "worker-1", first_claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "queued"
        assert job.upload_attempts == 1
        assert job.provider_input_file_id is None
        assert job.poll_after is not None
        retry_at = job.poll_after.replace(tzinfo=timezone.utc)
    assert claim_batch(database, "worker-too-early") is None

    retry_claim = claim_batch(database, "worker-2", now=retry_at)
    assert retry_claim is not None
    process_batch_claim(database, job_id, "worker-2", retry_claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "submitted"
        assert job.provider_input_file_id == "file-input"
        assert job.provider_batch_id == "batch-1"
    assert [request.url for request in requests_mock.request_history] == [
        "https://api.openai.com/v1/files",
        "https://api.openai.com/v1/files",
        "https://api.openai.com/v1/batches",
    ]


def test_ambiguous_upload_is_retained_as_unknown_without_false_file_cleanup_claim(
    admin_app, requests_mock
):
    """Unknown upload outcomes retain no invented ID and disclose provider expiry."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    requests_mock.post("https://api.openai.com/v1/files", exc=requests.Timeout)

    claim = claim_batch(database, "worker-upload-timeout")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-upload-timeout", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "unknown_upload"
        assert job.provider_input_file_id is None
        assert "provider-managed" in job.error_json["message"]
    assert not any(
        request.method == "DELETE" for request in requests_mock.request_history
    )


def test_batch_download_closes_response_after_success_failure_and_oversize():
    """Streaming result responses close on every exit path."""

    class Response:
        def __init__(self, status_code, chunks):
            self.status_code = status_code
            self.headers = {}
            self.chunks = chunks
            self.closed = False

        def iter_content(self, chunk_size):
            yield from self.chunks

        def close(self):
            self.closed = True

    class HTTP:
        response = None

        def get(self, *_args, **_kwargs):
            return self.response

    http = HTTP()
    client = OpenAIBatchClient("key", {}, http=http)
    for status_code, chunks, max_bytes, raises in (
        (200, [b"ok"], 2, False),
        (500, [], 2, True),
        (200, [b"abc"], 2, True),
    ):
        http.response = Response(status_code, chunks)
        if raises:
            with pytest.raises(BatchWorkerError):
                client.download("file-id", max_bytes=max_bytes)
        else:
            assert client.download("file-id", max_bytes=max_bytes) == b"ok"
        assert http.response.closed


def test_ambiguous_batch_post_timeout_never_retries_submission(
    admin_app, requests_mock
):
    """An uncertain batch POST becomes unknown_submit and is never blindly repeated."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    requests_mock.post("https://api.openai.com/v1/files", json={"id": "file-input"})
    requests_mock.post("https://api.openai.com/v1/batches", exc=requests.Timeout)

    claim = claim_batch(database, "worker-1")
    assert claim is not None
    process_batch_claim(database, job_id, "worker-1", claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "unknown_submit"
        assert job.provider_batch_id is None
        assert job.error_json["type"] == "unknown_submit"
    assert claim_batch(database, "worker-2") is None
    assert len(requests_mock.request_history) == 2


def test_rate_limited_batch_submit_is_retried_after_retry_after(
    admin_app, requests_mock
):
    """Retry an explicitly rejected batch submission after provider guidance."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    requests_mock.post("https://api.openai.com/v1/files", json={"id": "file-input"})
    requests_mock.post(
        "https://api.openai.com/v1/batches",
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "3"},
                "json": {"error": {"code": "rate_limit_exceeded"}},
            },
            {"status_code": 200, "json": {"id": "batch-1"}},
        ],
    )

    first_claim = claim_batch(database, "worker-1")
    assert first_claim is not None
    process_batch_claim(database, job_id, "worker-1", first_claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "retry_submit"
        assert job.provider_input_file_id == "file-input"
        assert job.poll_after is not None
        retry_at = job.poll_after.replace(tzinfo=timezone.utc)
    assert claim_batch(database, "worker-too-early") is None

    retry_claim = claim_batch(database, "worker-2", now=retry_at)
    assert retry_claim is not None
    process_batch_claim(database, job_id, "worker-2", retry_claim[1])

    with database.sessions() as session:
        job = session.get(BatchJob, job_id)
        assert job.status == "submitted"
        assert job.provider_batch_id == "batch-1"
    assert len(requests_mock.request_history) == 3


def test_worker_lease_renews_without_allowing_overlapping_claims(admin_app):
    """A renewed lease blocks another worker while the provider request runs."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    start = datetime.now(timezone.utc)
    claim = claim_batch(database, "worker-long-request", now=start)
    assert claim is not None

    renewed = _renew_batch_lease(
        database,
        job_id,
        "worker-long-request",
        claim[1],
        now=start + timedelta(seconds=60),
    )

    assert renewed
    with database.sessions() as session:
        lease_expiry = session.get(BatchJob, job_id).lease_expires_at
    assert lease_expiry.replace(tzinfo=timezone.utc) > start + timedelta(seconds=100)
    assert (
        claim_batch(database, "worker-overlap", now=start + timedelta(seconds=100))
        is None
    )
    assert not _renew_batch_lease(
        database,
        job_id,
        "worker-long-request",
        claim[1],
        now=start + timedelta(seconds=200),
    )


def test_stale_worker_fence_cannot_overwrite_reclaimed_job(admin_app):
    """A new lease increments the fence and rejects a prior worker update."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "submitted"
        job.provider_batch_id = "batch-1"
    start = datetime.now(timezone.utc)
    first = claim_batch(database, "worker-1", now=start)
    assert first is not None
    second = claim_batch(database, "worker-2", now=start + timedelta(seconds=120))
    assert second is not None and second[1] > first[1]

    updated = _fenced_update(
        database,
        job_id,
        "worker-1",
        first[1],
        provider_input_file_id="stale-file",
    )

    assert updated is False
    with database.sessions() as session:
        assert session.get(BatchJob, job_id).provider_input_file_id is None


def test_supervisor_accepts_disabled_worker_clean_exit_without_restart_loop():
    """Supervisor treats the disabled wrapper's immediate exit as clean."""
    config = Path(__file__).resolve().parents[1] / "supervisord" / "batch-worker.conf"
    settings = config.read_text(encoding="utf-8")

    assert "startsecs=0" in settings
    assert "autorestart=unexpected" in settings


def test_supervisor_worker_wrapper_is_disabled_unless_explicitly_enabled():
    """The separate supervised process exits cleanly unless its flag is true."""
    script = Path(__file__).resolve().parents[1] / "supervisord" / "batch-worker.sh"
    environment = {**os.environ, "BATCH_WORKER_ENABLED": "false"}

    result = subprocess.run(
        ["/bin/sh", str(script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_supervisor_worker_wrapper_requires_database_tenant_mode():
    """An enabled worker wrapper rejects process configurations outside DB tenants."""
    script = Path(__file__).resolve().parents[1] / "supervisord" / "batch-worker.sh"
    environment = {
        **os.environ,
        "BATCH_WORKER_ENABLED": "true",
        "AUTH_MODE": "tenant",
        "TENANT_CONFIG_SOURCE": "environment",
    }

    result = subprocess.run(
        ["/bin/sh", str(script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "AUTH_MODE=tenant" in result.stderr
    assert "TENANT_CONFIG_SOURCE=database" in result.stderr


def test_batch_worker_cli_requires_explicit_enable_flag(admin_app):
    """The registered worker command fails clearly while its runtime flag is off."""
    admin_app.config.update(
        AUTH_MODE="tenant",
        TENANT_CONFIG_SOURCE="database",
        BATCH_WORKER_ENABLED=False,
    )

    result = admin_app.test_cli_runner().invoke(args=["batch-worker"])

    assert result.exit_code != 0
    assert "BATCH_WORKER_ENABLED=true" in result.output


def test_rate_limited_batch_submission_attempts_are_bounded(admin_app, requests_mock):
    """Repeated explicit 429 rejections stop after five total POST attempts."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    requests_mock.post("https://api.openai.com/v1/files", json={"id": "file-input"})
    requests_mock.post(
        "https://api.openai.com/v1/batches",
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "1"},
                "json": {"error": {"code": "rate_limit_exceeded"}},
            }
            for _ in range(5)
        ],
    )

    retry_at = datetime.now(timezone.utc)
    for attempt in range(5):
        claim = claim_batch(database, f"worker-{attempt}", now=retry_at)
        assert claim is not None
        process_batch_claim(database, job_id, f"worker-{attempt}", claim[1])
        with database.sessions() as session:
            job = session.get(BatchJob, job_id)
            assert job.submit_attempts == attempt + 1
            if attempt < 4:
                assert job.status == "retry_submit"
                retry_at = job.poll_after.replace(tzinfo=timezone.utc)
            else:
                assert job.status == "failed"
                assert job.error_json["type"] == "provider_http_error"
                assert job.error_json["status_code"] == 429

    assert len(requests_mock.request_history) == 6
    assert claim_batch(database, "worker-after-limit") is None


def test_rate_limited_batch_upload_attempts_are_bounded(admin_app, requests_mock):
    """Repeated explicit 429 upload rejections fail after five POST attempts."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    requests_mock.post(
        "https://api.openai.com/v1/files",
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "1"},
                "json": {"error": {"code": "rate_limit_exceeded"}},
            }
            for _ in range(5)
        ],
    )

    retry_at = datetime.now(timezone.utc)
    for attempt in range(5):
        claim = claim_batch(database, f"upload-worker-{attempt}", now=retry_at)
        assert claim is not None
        process_batch_claim(database, job_id, f"upload-worker-{attempt}", claim[1])
        with database.sessions() as session:
            job = session.get(BatchJob, job_id)
            assert job.upload_attempts == attempt + 1
            if attempt < 4:
                assert job.status == "queued"
                retry_at = job.poll_after.replace(tzinfo=timezone.utc)
            else:
                assert job.status == "failed"
                assert job.error_json["type"] == "provider_http_error"
                assert job.error_json["status_code"] == 429

    assert len(requests_mock.request_history) == 5
    assert claim_batch(database, "upload-worker-after-limit") is None


def test_worker_runs_bounded_retention_cleanup_while_queue_is_busy(
    admin_app, monkeypatch, mocker
):
    """Periodic retention work runs on cadence even when a job is claimable."""
    _client, database = _prepare_batch_app(admin_app)
    monkeypatch.setattr(database.engine.dialect, "name", "postgresql")
    clock = iter((0.0, 200.0))
    monkeypatch.setattr("app.batch_worker.time.monotonic", lambda: next(clock))
    batch_cleanup = mocker.patch("app.batch_worker.cleanup_expired_batches")
    scheduler_cleanup = mocker.patch.object(ProviderBudgetScheduler, "cleanup")
    concurrency_cleanup = mocker.patch(
        "app.batch_worker.ProviderConcurrencyController.cleanup"
    )
    monkeypatch.setattr("app.batch_worker.claim_batch", lambda *_args: ("job", 1))
    mocker.patch("app.batch_worker._lease_heartbeat", return_value=nullcontext())
    mocker.patch("app.batch_worker._submit_claim")

    class StopWorker(BaseException):
        pass

    def stop(*_args, **_kwargs):
        raise StopWorker

    monkeypatch.setattr("app.batch_worker.process_batch_claim", stop)
    with admin_app.app_context(), pytest.raises(StopWorker):
        run_batch_worker(admin_app)

    batch_cleanup.assert_called_once_with(database, http=requests, limit=50)
    scheduler_cleanup.assert_called_once_with(batch_size=50)
    concurrency_cleanup.assert_called_once_with(batch_size=50)


def test_batch_file_cleanup_failure_does_not_starve_queue_claims(
    admin_app, monkeypatch, mocker
):
    """A transient cleanup failure must not block unrelated work claims."""
    _client, database = _prepare_batch_app(admin_app)
    monkeypatch.setattr(database.engine.dialect, "name", "postgresql")
    monkeypatch.setattr("app.batch_worker.time.monotonic", lambda: 0.0)
    monkeypatch.setattr("app.batch_worker.time.sleep", lambda _seconds: None)

    class StopWorker(BaseException):
        pass

    cleanup = mocker.patch(
        "app.batch_worker.cleanup_expired_batches",
        side_effect=BatchWorkerError("remote file delete failed"),
    )
    claim = mocker.patch("app.batch_worker.claim_batch", side_effect=StopWorker)
    with admin_app.app_context(), pytest.raises(StopWorker):
        run_batch_worker(admin_app)

    cleanup.assert_called_once_with(database, http=requests, limit=50)
    claim.assert_called_once_with(database, mocker.ANY)


def test_batch_worker_rejects_non_postgresql_queue_backend(admin_app):
    """The worker cannot silently run shared queue claims against SQLite."""
    _client, _database = _prepare_batch_app(admin_app)

    with pytest.raises(BatchWorkerError, match="requires PostgreSQL"):
        run_batch_worker(admin_app)


def test_profile_delete_is_blocked_while_batch_retention_needs_its_secret(admin_app):
    """Retained jobs keep their provider profile credential available for cleanup."""
    client, database = _prepare_batch_app(admin_app)
    _create_job(client)

    with database.sessions.begin() as session:
        with pytest.raises(ValueError, match="batch jobs are retained"):
            delete_provider_profile(
                session,
                "acme",
                "openai-batch-profile",
                "ada",
            )


def test_cleanup_deletes_remote_files_before_removing_expired_job(
    admin_app, requests_mock
):
    """Expiry cleanup deletes provider files, then removes bounded local payload."""
    client, database = _prepare_batch_app(admin_app)
    job_id = _create_job(client)
    with database.sessions.begin() as session:
        job = session.get(BatchJob, job_id)
        job.status = "completed"
        job.provider_input_file_id = "file-input"
        job.provider_output_file_id = "file-output"
        job.provider_error_file_id = "file-error"
        job.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    for file_id in ("file-input", "file-output", "file-error"):
        requests_mock.delete(
            f"https://api.openai.com/v1/files/{file_id}", json={"deleted": True}
        )

    deleted = cleanup_expired_batches(database)

    assert deleted == 1
    assert [request.method for request in requests_mock.request_history] == [
        "DELETE",
        "DELETE",
        "DELETE",
    ]
    with database.sessions() as session:
        assert session.get(BatchJob, job_id) is None


def test_output_jsonl_rejects_malformed_content_and_preserves_response_envelope():
    """Provider JSONL is parsed only as bounded custom_id response/error envelopes."""
    with pytest.raises(Exception, match="invalid JSONL"):
        parse_output_jsonl(b"not-json\n")
    parsed = parse_output_jsonl(
        b'{"custom_id":"r1","response":{"status_code":200,"body":{"ok":true}}}\n'
    )
    assert parsed == [
        {"custom_id": "r1", "response": {"status_code": 200, "body": {"ok": True}}}
    ]
