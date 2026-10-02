"""Provider billing refresh results fetched outside a database transaction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import requests


class CostRefreshError(Exception):
    """Raised when a provider billing API cannot be queried."""

    def __init__(self, message: str, status: str = "failed") -> None:
        """Store a user-visible message and a persisted job status."""
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class CostBucket:
    """One typed cost or usage bucket returned by a provider adapter."""

    kind: str
    metric: str
    value: Decimal
    unit: str
    currency: str | None
    bucket_start: datetime
    bucket_end: datetime
    source: str
    granularity: str
    dimensions: dict[str, object]


def fetch_openai_costs(
    api_key: str, organization: str, start: datetime, end: datetime
) -> list[CostBucket]:
    """Query OpenAI organization costs for the requested window."""
    response = requests.get(
        "https://api.openai.com/v1/organization/costs",
        headers={
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": organization,
        },
        params={
            "start_time": int(start.timestamp()),
            "end_time": int(end.timestamp()),
        },
        timeout=60,
    )
    if response.status_code == 429:
        raise CostRefreshError("OpenAI costs are rate limited.", status="unavailable")
    if response.status_code >= 400:
        raise CostRefreshError(f"OpenAI costs HTTP {response.status_code}")
    payload = response.json()
    buckets = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(buckets, list):
        raise CostRefreshError("OpenAI costs payload is invalid.")
    records: list[CostBucket] = []
    for item in buckets:
        if not isinstance(item, dict):
            continue
        amount = item.get("amount")
        value = amount.get("value") if isinstance(amount, dict) else item.get("cost")
        currency = amount.get("currency") if isinstance(amount, dict) else "USD"
        if value is None:
            continue
        records.append(
            CostBucket(
                kind="actual",
                metric="cost",
                value=Decimal(str(value)),
                unit="currency",
                currency=str(currency).upper()[:3] if currency else "USD",
                bucket_start=start,
                bucket_end=end,
                source="openai.organization.costs",
                granularity="window",
                dimensions={"organization": organization},
            )
        )
    return records


def fetch_openrouter_costs(
    api_key: str, start: datetime, end: datetime
) -> list[CostBucket]:
    """Query OpenRouter credit usage for the authenticated account."""
    response = requests.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    if response.status_code == 429:
        raise CostRefreshError(
            "OpenRouter credits are rate limited.", status="unavailable"
        )
    if response.status_code >= 400:
        raise CostRefreshError(f"OpenRouter credits HTTP {response.status_code}")
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise CostRefreshError("OpenRouter credits payload is invalid.")
    usage = data.get("total_usage")
    if usage is None:
        raise CostRefreshError("OpenRouter credits payload is missing total_usage.")
    return [
        CostBucket(
            kind="actual",
            metric="usage",
            value=Decimal(str(usage)),
            unit="credits",
            currency="USD",
            bucket_start=start,
            bucket_end=end,
            source="openrouter.credits",
            granularity="lifetime",
            dimensions={},
        )
    ]


def fetch_azure_costs(
    billing_secret: str, resource_group_id: str, start: datetime, end: datetime
) -> list[CostBucket]:
    """Query Azure Cost Management for the bound resource group scope."""
    token = _azure_arm_token(billing_secret)
    url = (
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2023-11-01"
    )
    body = {
        "type": "ActualCost",
        "timeframe": "Custom",
        "timePeriod": {
            "from": start.astimezone(timezone.utc).date().isoformat(),
            "to": end.astimezone(timezone.utc).date().isoformat(),
        },
        "dataset": {
            "granularity": "Daily",
            "aggregation": {
                "totalCost": {"name": "Cost", "function": "Sum"},
            },
        },
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=60,
    )
    if response.status_code == 429:
        raise CostRefreshError("Azure costs are rate limited.", status="unavailable")
    if response.status_code >= 400:
        raise CostRefreshError(f"Azure costs HTTP {response.status_code}")
    payload = response.json()
    properties = payload.get("properties") if isinstance(payload, dict) else None
    rows = properties.get("rows") if isinstance(properties, dict) else None
    if not isinstance(rows, list):
        raise CostRefreshError("Azure costs payload is invalid.")
    total = Decimal("0")
    for row in rows:
        if isinstance(row, list) and row:
            total += Decimal(str(row[0]))
    return [
        CostBucket(
            kind="actual",
            metric="cost",
            value=total,
            unit="currency",
            currency="USD",
            bucket_start=start,
            bucket_end=end,
            source="azure.costmanagement.query",
            granularity="window",
            dimensions={"scope": resource_group_id},
        )
    ]


def _azure_arm_token(billing_secret: str) -> str:
    try:
        payload = json.loads(billing_secret)
    except json.JSONDecodeError as exc:
        raise CostRefreshError(
            "Azure billing credentials must be a JSON service principal."
        ) from exc
    tenant_id = payload.get("tenant_id")
    client_id = payload.get("client_id")
    client_secret = payload.get("client_secret")
    if not all(
        isinstance(value, str) and value
        for value in (tenant_id, client_id, client_secret)
    ):
        raise CostRefreshError("Azure billing credentials are incomplete.")
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    response = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://management.azure.com/.default",
        },
        timeout=30,
    )
    if response.status_code >= 400:
        raise CostRefreshError("Azure billing authentication failed.")
    token = response.json().get("access_token")
    if not isinstance(token, str) or not token:
        raise CostRefreshError("Azure billing authentication returned no token.")
    return token
