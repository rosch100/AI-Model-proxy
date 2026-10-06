"""Tenant-authenticated API routes for OpenAI batch jobs."""

from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, request
from sqlalchemy import select
from werkzeug.exceptions import RequestEntityTooLarge

from app.auth import current_tenant, require_auth
from app.batch_service import (
    BatchIdempotencyConflictError,
    BatchJobStillActiveError,
    BatchQueueFullError,
    BatchValidationError,
    normalize_batch_payload,
    submit_batch,
    validate_idempotency_key,
)
from app.persistence.database import Database
from app.persistence.models import BatchJob
from app.tenants import (
    AUTH_MODE_TENANT,
    TENANT_CONFIG_DATABASE,
    DatabaseTenantRoutingSnapshot,
)

batch_blueprint = Blueprint("batch_api", __name__)
MAX_BATCH_REQUEST_BODY_BYTES = 5 * 1024 * 1024
_PUBLIC_BATCH_STATUSES = {
    "queued": "validating",
    "uploading": "validating",
    "submitting": "validating",
    "retry_submit": "validating",
    "submitted": "in_progress",
    "polling": "in_progress",
    "completed": "completed",
    "expired": "expired",
    "failed": "failed",
    "unknown_upload": "failed",
    "unknown_submit": "failed",
}


@batch_blueprint.before_request
def _limit_batch_request_body() -> None:
    """Bound batch API bodies before Flask parses JSON, without a global cap."""
    request.max_content_length = MAX_BATCH_REQUEST_BODY_BYTES


@batch_blueprint.errorhandler(RequestEntityTooLarge)
def _batch_request_too_large(_error: RequestEntityTooLarge):
    return (
        jsonify(
            {
                "error": {
                    "message": "Batch request body exceeds the 5 MiB limit.",
                    "type": "request_too_large",
                }
            }
        ),
        413,
    )


def _database_tenant() -> tuple[Database, DatabaseTenantRoutingSnapshot]:
    """Require the database-backed tenant auth mode for this API."""
    if (
        current_app.config.get("AUTH_MODE") != AUTH_MODE_TENANT
        or current_app.config.get("TENANT_CONFIG_SOURCE") != TENANT_CONFIG_DATABASE
    ):
        abort(400, description="Batch API requires database tenant mode.")
    tenant = current_tenant()
    database = current_app.extensions.get("database")
    if not isinstance(database, Database) or not isinstance(
        tenant, DatabaseTenantRoutingSnapshot
    ):
        abort(400, description="Batch API requires database tenant mode.")
    return database, tenant


def _job_response(job: BatchJob) -> dict[str, object]:
    """Return safe OpenAI-compatible batch metadata without request contents."""
    return {
        "id": job.id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "status": _PUBLIC_BATCH_STATUSES[job.status],
        "created_at": int(job.created_at.timestamp()),
        "completed_at": (
            int(job.completed_at.timestamp()) if job.completed_at is not None else None
        ),
        "errors": job.error_json,
    }


@batch_blueprint.route("/v1/batches", methods=["POST"])
@require_auth
def create_batch():
    """Validate, idempotently persist, and enqueue one batch request."""
    database, tenant = _database_tenant()
    if current_app.config.get("BATCH_WORKER_ENABLED") is not True:
        response = jsonify(
            {
                "error": {
                    "message": "The batch worker is not enabled for this service.",
                    "type": "batch_worker_unavailable",
                }
            }
        )
        response.status_code = 503
        response.headers["Retry-After"] = "30"
        return response
    try:
        idempotency_key_hash = validate_idempotency_key(
            request.headers.get("Idempotency-Key")
        )
        payload = request.get_json(silent=True)
        submission = normalize_batch_payload(payload, idempotency_key_hash)
        result = submit_batch(
            database,
            tenant.id,
            submission,
            max_queued_jobs=current_app.config["BATCH_MAX_QUEUED_JOBS"],
        )
    except BatchValidationError:
        return (
            jsonify(
                {
                    "error": {
                        "message": "The batch request is invalid.",
                        "type": "invalid_request_error",
                    }
                }
            ),
            400,
        )
    except BatchQueueFullError:
        response = jsonify(
            {
                "error": {
                    "message": "The tenant batch queue is at capacity.",
                    "type": "batch_queue_full",
                }
            }
        )
        response.status_code = 429
        response.headers["Retry-After"] = "30"
        return response
    except BatchJobStillActiveError:
        response = jsonify(
            {
                "error": {
                    "message": "The expired batch is still owned by an active worker.",
                    "type": "batch_job_still_active",
                }
            }
        )
        response.status_code = 503
        response.headers["Retry-After"] = "30"
        return response
    except BatchIdempotencyConflictError:
        return (
            jsonify(
                {
                    "error": {
                        "message": "Idempotency-Key was already used with a different request.",
                        "type": "idempotency_conflict",
                    }
                }
            ),
            409,
        )
    return jsonify(_job_response(result.job)), 202


@batch_blueprint.route("/v1/batches/<job_id>", methods=["GET"])
@require_auth
def get_batch(job_id: str):
    """Return one batch only within the authenticated tenant."""
    database, tenant = _database_tenant()
    with database.sessions() as session:
        job = session.scalar(
            select(BatchJob).where(
                BatchJob.id == job_id,
                BatchJob.tenant_id == tenant.id,
            )
        )
        if job is None:
            abort(404)
        return jsonify(_job_response(job))


@batch_blueprint.route("/v1/batches/<job_id>/results", methods=["GET"])
@require_auth
def get_batch_results(job_id: str):
    """Return bounded terminal output records ordered by custom_id."""
    database, tenant = _database_tenant()
    with database.sessions() as session:
        job = session.scalar(
            select(BatchJob).where(
                BatchJob.id == job_id,
                BatchJob.tenant_id == tenant.id,
            )
        )
        if job is None:
            abort(404)
        if job.status not in {
            "completed",
            "failed",
            "expired",
            "unknown_upload",
            "unknown_submit",
        }:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Batch results are not ready.",
                            "type": "batch_not_ready",
                        }
                    }
                ),
                409,
            )
        results = job.results_json or []
        return jsonify(
            {
                "object": "list",
                "data": sorted(results, key=lambda result: str(result["custom_id"])),
            }
        )
