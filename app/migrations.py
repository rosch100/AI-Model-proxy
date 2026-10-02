"""Explicit Alembic commands exposed through the Flask CLI."""

from __future__ import annotations

import os
from pathlib import Path

import click
from alembic import command
from alembic.config import Config
from flask import Flask


@click.group("db")
def database_commands() -> None:
    """Manage the tenant persistence schema explicitly."""


@database_commands.command("upgrade")
@click.option("--revision", default="head", show_default=True)
def upgrade_database(revision: str) -> None:
    """Apply schema migrations to the configured PostgreSQL database."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise click.ClickException(
            "DATABASE_URL is required to run database migrations"
        )
    if not database_url.startswith("postgresql+psycopg://"):
        raise click.ClickException(
            "DATABASE_URL must use the postgresql+psycopg driver"
        )

    project_root = Path(__file__).resolve().parent.parent
    alembic_config = Config(str(project_root / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(project_root / "migrations"))
    command.upgrade(alembic_config, revision)


def register_migration_commands(app: Flask) -> None:
    """Register database migration commands without executing them at startup."""
    app.cli.add_command(database_commands)
