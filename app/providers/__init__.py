"""Dispatch proxy traffic to the active database-backed provider profile."""

from __future__ import annotations

from flask import Request, Response

from app.azure.adapter import AzureAdapter
from app.exceptions import ServiceConfigurationError
from app.providers.openai_compat import forward_openai_compatible
from app.tenants import DatabaseTenantSnapshot


def forward_database_snapshot(
    req: Request, snapshot: DatabaseTenantSnapshot
) -> Response:
    """Forward an authenticated proxy request using the active tenant profile."""
    if snapshot.provider == "azure":
        return AzureAdapter().forward(req)
    if snapshot.provider in {"openai", "openrouter"}:
        return forward_openai_compatible(req, snapshot)
    raise ServiceConfigurationError(
        "No active provider profile is configured for this tenant."
    )
