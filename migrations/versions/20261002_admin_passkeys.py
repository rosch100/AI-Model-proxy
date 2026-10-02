"""Add admin passkey and WebAuthn challenge tables.

Revision ID: 20261002_passkeys
Revises: 20261001_initial
Create Date: 2026-10-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261002_passkeys"
down_revision: str | None = "20261001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add passkey enrollment columns and tables."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")

    op.add_column(
        "admin_accounts",
        sa.Column("webauthn_user_handle", sa.LargeBinary(64), nullable=True),
    )
    op.create_unique_constraint(
        "uq_admin_accounts_webauthn_user_handle",
        "admin_accounts",
        ["webauthn_user_handle"],
    )
    op.add_column(
        "admin_sessions",
        sa.Column(
            "enrollment_only",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_table(
        "admin_passkeys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("admin_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("credential_id", sa.LargeBinary(1024), nullable=False),
        sa.Column("public_key", sa.LargeBinary(), nullable=False),
        sa.Column("sign_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("user_handle", sa.LargeBinary(64), nullable=False),
        sa.Column("transports", sa.JSON(), nullable=True),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("aaguid", sa.String(64), nullable=True),
        sa.Column(
            "backed_up", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("credential_id", name="uq_admin_passkey_credential"),
    )
    op.create_table(
        "admin_webauthn_challenges",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("admin_accounts.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("purpose", sa.String(32), nullable=False),
        sa.Column("challenge", sa.LargeBinary(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "purpose IN ('registration', 'authentication')",
            name="ck_webauthn_purpose",
        ),
    )


def downgrade() -> None:
    """Remove passkey enrollment columns and tables."""
    if op.get_context().dialect.name != "postgresql":
        raise RuntimeError("Tenant-admin migrations require PostgreSQL")
    op.drop_table("admin_webauthn_challenges")
    op.drop_table("admin_passkeys")
    op.drop_column("admin_sessions", "enrollment_only")
    op.drop_constraint(
        "uq_admin_accounts_webauthn_user_handle", "admin_accounts", type_="unique"
    )
    op.drop_column("admin_accounts", "webauthn_user_handle")
