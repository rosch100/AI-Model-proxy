"""Persist tenant-isolated OpenAI batch jobs and worker leases.

Revision ID: 20261014_batch_jobs
Revises: 20261013_budget_sched
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261014_batch_jobs"
down_revision: str | None = "20261013_budget_sched"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create PostgreSQL-only tenant batch jobs."""
    _require_postgresql()
    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(38), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(128),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("model", sa.String(256), nullable=False),
        sa.Column("idempotency_key_hash", sa.String(64), nullable=True),
        sa.Column("payload_digest", sa.String(64), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("provider_batch_id", sa.String(256), nullable=True),
        sa.Column("provider_input_file_id", sa.String(256), nullable=True),
        sa.Column("provider_output_file_id", sa.String(256), nullable=True),
        sa.Column("provider_error_file_id", sa.String(256), nullable=True),
        sa.Column("results_json", sa.JSON(), nullable=True),
        sa.Column("error_json", sa.JSON(), nullable=True),
        sa.Column("worker_owner", sa.String(36), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fencing_token", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "upload_attempts", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "submit_attempts", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("poll_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "tenant_id", "idempotency_key_hash", name="uq_batch_tenant_idempotency"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "profile_id"],
            ["provider_profiles.tenant_id", "provider_profiles.id"],
            name="fk_batch_profile_same_tenant",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'uploading', 'submitting', 'retry_submit', "
            "'submitted', 'polling', 'completed', 'failed', 'unknown_upload', "
            "'unknown_submit', 'expired')",
            name="ck_batch_status",
        ),
        sa.CheckConstraint(
            "idempotency_key_hash IS NULL OR length(idempotency_key_hash) = 64",
            name="ck_batch_key_hash",
        ),
        sa.CheckConstraint("length(payload_digest) = 64", name="ck_batch_payload_digest"),
        sa.CheckConstraint("fencing_token >= 0", name="ck_batch_fencing_token"),
        sa.CheckConstraint(
            "upload_attempts BETWEEN 0 AND 5", name="ck_batch_upload_attempts"
        ),
        sa.CheckConstraint(
            "submit_attempts BETWEEN 0 AND 5", name="ck_batch_submit_attempts"
        ),
        sa.CheckConstraint(
            "(worker_owner IS NULL) = (lease_expires_at IS NULL)",
            name="ck_batch_lease_pair",
        ),
    )
    op.create_index(
        "ix_batch_jobs_queue",
        "batch_jobs",
        ["status", "poll_after", "created_at"],
        postgresql_where=sa.text(
            "status IN ('queued', 'uploading', 'retry_submit', 'submitted', 'polling')"
        ),
    )
    op.create_index("ix_batch_jobs_lease", "batch_jobs", ["lease_expires_at"])
    op.create_index("ix_batch_jobs_retention", "batch_jobs", ["expires_at"])


def downgrade() -> None:
    """Remove PostgreSQL batch state and indexes."""
    _require_postgresql()
    op.drop_index("ix_batch_jobs_retention", table_name="batch_jobs")
    op.drop_index("ix_batch_jobs_lease", table_name="batch_jobs")
    op.drop_index("ix_batch_jobs_queue", table_name="batch_jobs")
    op.drop_table("batch_jobs")


def _require_postgresql() -> None:
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
