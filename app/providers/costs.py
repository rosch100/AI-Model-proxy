"""Provider billing refresh results fetched outside a database transaction."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from time import monotonic
from uuid import UUID

import requests
from azure.core.exceptions import AzureError
from azure.identity import (
    CertificateCredential,
    ManagedIdentityCredential,
    WorkloadIdentityCredential,
)

OPENAI_MAX_PAGES = 5
OPENAI_REQUEST_BUDGET_SECONDS = 45
OPENAI_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
OPENAI_CONNECT_TIMEOUT_SECONDS = 3
OPENAI_READ_TIMEOUT_SECONDS = 3
PROVIDER_REQUEST_TIMEOUT = (3, 15)


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
    api_key: str, project_id: str, start: datetime, end: datetime
) -> list[CostBucket]:
    """Query project-scoped OpenAI costs and completion token usage."""
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = monotonic() + OPENAI_REQUEST_BUDGET_SECONDS
    params = {
        "start_time": int(start.timestamp()),
        "end_time": int(end.timestamp()),
        "bucket_width": "1d",
        "project_ids": [project_id],
    }
    costs_payload = _openai_paginated_payload(
        "https://api.openai.com/v1/organization/costs",
        headers,
        {**params, "group_by": ["project_id"], "limit": 180},
        "costs",
        deadline,
    )
    usage_payload = _openai_paginated_payload(
        "https://api.openai.com/v1/organization/usage/completions",
        headers,
        {**params, "group_by": ["project_id", "model"], "limit": 31},
        "usage",
        deadline,
    )

    records: list[CostBucket] = []
    for bucket in costs_payload:
        for result in _openai_project_results(bucket, project_id, "costs"):
            amount = result.get("amount")
            value = amount.get("value") if isinstance(amount, dict) else None
            currency = amount.get("currency") if isinstance(amount, dict) else None
            if value is None or not currency:
                raise CostRefreshError(
                    "OpenAI costs result is missing amount or currency."
                )
            currency_code = currency.upper() if isinstance(currency, str) else ""
            if currency_code != "USD":
                raise CostRefreshError("OpenAI costs currency code is unsupported.")
            decimal_value = _openai_decimal(value, "costs", "cost", "currency")
            bucket_start, bucket_end = _openai_bucket_period(bucket)
            records.append(
                CostBucket(
                    kind="actual",
                    metric="cost",
                    value=decimal_value,
                    unit="currency",
                    currency=currency_code,
                    bucket_start=bucket_start,
                    bucket_end=bucket_end,
                    source="openai.organization.costs",
                    granularity="day",
                    dimensions={"project_id": project_id},
                )
            )

    for bucket in usage_payload:
        for result in _openai_project_results(bucket, project_id, "usage"):
            bucket_start, bucket_end = _openai_bucket_period(bucket)
            dimensions = {"project_id": project_id}
            model = result.get("model")
            if isinstance(model, str):
                dimensions["model"] = model
            for metric in ("input_tokens", "output_tokens"):
                value = result.get(metric)
                if value is not None:
                    decimal_value = _openai_decimal(value, "usage", metric, "tokens")
                    records.append(
                        CostBucket(
                            kind="usage",
                            metric=metric,
                            value=decimal_value,
                            unit="tokens",
                            currency=None,
                            bucket_start=bucket_start,
                            bucket_end=bucket_end,
                            source="openai.organization.usage.completions",
                            granularity="day",
                            dimensions=dimensions,
                        )
                    )
    return records


def _validate_storage_decimal(
    value: Decimal, provider: str, data_type: str, metric: str
) -> None:
    _, digits, exponent = value.as_tuple()
    while digits and digits[-1] == 0:
        digits = digits[:-1]
        exponent += 1
    integer_digits = max(len(digits) + exponent, 0)
    fractional_digits = max(-exponent, 0)
    if integer_digits > 18 or fractional_digits > 10:
        raise CostRefreshError(
            f"{provider} {data_type} {metric} exceeds storage precision."
        )


def _openai_decimal(value: object, data_type: str, metric: str, unit: str) -> Decimal:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CostRefreshError(f"OpenAI {data_type} {metric} is invalid.") from exc
    if not decimal_value.is_finite() or decimal_value < 0:
        raise CostRefreshError(f"OpenAI {data_type} {metric} is invalid.")
    _validate_storage_decimal(decimal_value, "OpenAI", data_type, metric)
    if unit == "tokens" and decimal_value != decimal_value.to_integral_value():
        raise CostRefreshError(f"OpenAI {data_type} {metric} is invalid.")
    return decimal_value


def _openai_paginated_payload(
    url: str,
    headers: dict[str, str],
    params: dict[str, object],
    data_type: str,
    deadline: float,
) -> list[dict[str, object]]:
    buckets: list[dict[str, object]] = []
    page_cursor = None
    seen_cursors: set[str] = set()
    for _ in range(OPENAI_MAX_PAGES):
        page_params = {**params}
        if page_cursor is not None:
            page_params["page"] = page_cursor
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise CostRefreshError("OpenAI billing exceeded its request time budget.")
        response = requests.get(
            url,
            headers=headers,
            params=page_params,
            timeout=(
                min(OPENAI_CONNECT_TIMEOUT_SECONDS, remaining),
                min(OPENAI_READ_TIMEOUT_SECONDS, remaining),
            ),
            stream=True,
        )
        try:
            payload = _openai_response_payload(response, data_type, deadline)
        finally:
            response.close()
        page_buckets = payload.get("data")
        if not isinstance(page_buckets, list) or any(
            not isinstance(bucket, dict) or not isinstance(bucket.get("results"), list)
            for bucket in page_buckets
        ):
            raise CostRefreshError(f"OpenAI {data_type} payload is invalid.")
        buckets.extend(page_buckets)

        has_more = payload.get("has_more")
        if not isinstance(has_more, bool):
            raise CostRefreshError(f"OpenAI {data_type} pagination is invalid.")
        if not has_more:
            return buckets

        next_page = payload.get("next_page")
        if not isinstance(next_page, str) or not next_page:
            raise CostRefreshError(
                f"OpenAI {data_type} has more pages but no next page cursor."
            )
        if next_page in seen_cursors:
            raise CostRefreshError(f"OpenAI {data_type} pagination cursor repeated.")
        seen_cursors.add(next_page)
        page_cursor = next_page

    raise CostRefreshError(
        f"OpenAI {data_type} pagination exceeded the maximum page count."
    )


def _openai_response_payload(
    response, data_type: str, deadline: float
) -> dict[str, object]:
    if response.status_code == 429:
        raise CostRefreshError(
            f"OpenAI {data_type} is rate limited.", status="unavailable"
        )
    if response.status_code >= 400:
        raise CostRefreshError(f"OpenAI {data_type} HTTP {response.status_code}")
    chunks = []
    response_size = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if monotonic() > deadline:
            raise CostRefreshError("OpenAI billing exceeded its request time budget.")
        response_size += len(chunk)
        if response_size > OPENAI_MAX_RESPONSE_BYTES:
            raise CostRefreshError(f"OpenAI {data_type} response is too large.")
        chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CostRefreshError(f"OpenAI {data_type} payload is invalid.") from exc
    if not isinstance(payload, dict):
        raise CostRefreshError(f"OpenAI {data_type} payload is invalid.")
    return payload


def _openai_project_results(
    bucket: dict[str, object], project_id: str, data_type: str
) -> list[dict[str, object]]:
    results = bucket.get("results")
    if not isinstance(results, list) or any(
        not isinstance(result, dict) or result.get("project_id") != project_id
        for result in results
    ):
        raise CostRefreshError(
            f"OpenAI {data_type} response does not confirm the bound project.",
            status="unavailable",
        )
    return results


def _openai_bucket_period(bucket: dict[str, object]) -> tuple[datetime, datetime]:
    start_time = bucket.get("start_time")
    end_time = bucket.get("end_time")
    if not isinstance(start_time, int) or not isinstance(end_time, int):
        raise CostRefreshError("OpenAI billing bucket has no valid time range.")
    return (
        datetime.fromtimestamp(start_time, timezone.utc),
        datetime.fromtimestamp(end_time, timezone.utc),
    )


def fetch_openrouter_costs(
    api_key: str, workspace_id: str, start: datetime, end: datetime
) -> list[CostBucket]:
    """Query OpenRouter's daily activity for the bound workspace and period."""
    if not isinstance(workspace_id, str):
        raise CostRefreshError("OpenRouter workspace ID must be a UUID.")
    try:
        workspace_id = str(UUID(workspace_id))
    except ValueError as exc:
        raise CostRefreshError("OpenRouter workspace ID must be a UUID.") from exc
    if start.tzinfo is None or end.tzinfo is None:
        raise CostRefreshError(
            "OpenRouter activity period must use timezone-aware UTC days."
        )
    period_start = start.astimezone(timezone.utc)
    period_end = end.astimezone(timezone.utc)
    if (
        period_start.time() != datetime.min.time()
        or period_end.time() != datetime.min.time()
        or period_start >= period_end
    ):
        raise CostRefreshError("OpenRouter activity period must use full UTC days.")
    response = requests.get(
        "https://openrouter.ai/api/v1/activity",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"workspace_id": workspace_id},
        timeout=PROVIDER_REQUEST_TIMEOUT,
    )
    if response.status_code == 429:
        raise CostRefreshError(
            "OpenRouter activity is rate limited.", status="unavailable"
        )
    if response.status_code >= 400:
        raise CostRefreshError(f"OpenRouter activity HTTP {response.status_code}")

    payload = response.json()
    activity = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(activity, list):
        raise CostRefreshError("OpenRouter activity payload is invalid.")

    records: list[CostBucket] = []
    for item in activity:
        if not isinstance(item, dict):
            raise CostRefreshError("OpenRouter activity item is invalid.")
        activity_date = item.get("date")
        try:
            parsed_date = date.fromisoformat(activity_date)
        except (TypeError, ValueError) as exc:
            raise CostRefreshError("OpenRouter activity date is invalid.") from exc
        if parsed_date.isoformat() != activity_date:
            raise CostRefreshError("OpenRouter activity date is invalid.")
        bucket_start = datetime.combine(parsed_date, datetime.min.time(), timezone.utc)
        bucket_end = bucket_start + timedelta(days=1)
        if bucket_end <= period_start or bucket_start >= period_end:
            continue

        dimensions = {
            key: item[key]
            for key in ("model", "endpoint_id")
            if isinstance(item.get(key), str)
        }
        for metric, kind, unit, currency in (
            ("prompt_tokens", "usage", "tokens", None),
            ("completion_tokens", "usage", "tokens", None),
            ("reasoning_tokens", "usage", "tokens", None),
            ("usage", "actual", "currency", "USD"),
            ("byok_usage_inference", "actual", "currency", "USD"),
        ):
            value = item.get(metric)
            if value is None:
                raise CostRefreshError(f"OpenRouter activity item is missing {metric}.")
            try:
                decimal_value = Decimal(str(value))
            except (InvalidOperation, ValueError) as exc:
                raise CostRefreshError(
                    f"OpenRouter activity {metric} is invalid."
                ) from exc
            if not decimal_value.is_finite() or decimal_value < 0:
                raise CostRefreshError(f"OpenRouter activity {metric} is invalid.")
            _validate_storage_decimal(decimal_value, "OpenRouter", "activity", metric)
            if unit == "tokens" and decimal_value != decimal_value.to_integral_value():
                raise CostRefreshError(f"OpenRouter activity {metric} is invalid.")
            records.append(
                CostBucket(
                    kind=kind,
                    metric=(
                        "cost"
                        if metric == "usage"
                        else (
                            "byok_inference_cost"
                            if metric == "byok_usage_inference"
                            else metric
                        )
                    ),
                    value=decimal_value,
                    unit=unit,
                    currency=currency,
                    bucket_start=bucket_start,
                    bucket_end=bucket_end,
                    source="openrouter.activity",
                    granularity="day",
                    dimensions=dimensions,
                )
            )
    return records


def fetch_azure_costs(
    resource_group_id: str,
    resource_id: str,
    start: datetime,
    end: datetime,
) -> list[CostBucket]:
    """Query daily actual cost for exactly the bound Azure Cognitive Services resource."""
    if start.tzinfo is None or end.tzinfo is None:
        raise CostRefreshError("Azure billing period must use timezone-aware UTC days.")
    period_start = start.astimezone(timezone.utc)
    period_end = end.astimezone(timezone.utc)
    if (
        period_start.time() != datetime.min.time()
        or period_end.time() != datetime.min.time()
        or period_start >= period_end
    ):
        raise CostRefreshError("Azure billing period must use full UTC days.")
    resource_group_prefix = resource_group_id.rstrip("/").casefold() + "/"
    if not resource_id.casefold().startswith(resource_group_prefix):
        raise CostRefreshError(
            "Azure usage resource is outside the billing Resource Group."
        )
    token = _azure_arm_token()
    url = (
        f"https://management.azure.com{resource_group_id}"
        "/providers/Microsoft.CostManagement/query?api-version=2026-06-01"
    )
    body = {
        "type": "ActualCost",
        "timeframe": "Custom",
        "timePeriod": {
            "from": period_start.date().isoformat(),
            "to": (period_end.date() - timedelta(days=1)).isoformat(),
        },
        "dataset": {
            "granularity": "Daily",
            "aggregation": {"totalCost": {"name": "PreTaxCost", "function": "Sum"}},
            "filter": {
                "dimensions": {
                    "name": "ResourceId",
                    "operator": "In",
                    "values": [resource_id],
                }
            },
            "grouping": [
                {"type": "Dimension", "name": "ResourceId"},
                {"type": "Dimension", "name": "Currency"},
            ],
        },
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=PROVIDER_REQUEST_TIMEOUT,
    )
    if response.status_code == 204:
        return []
    if response.status_code == 429:
        raise CostRefreshError("Azure costs are rate limited.", status="unavailable")
    if response.status_code in {401, 403}:
        raise CostRefreshError(
            "Azure Cost Management access is unavailable for the bound scope.",
            status="unavailable",
        )
    if response.status_code >= 400:
        raise CostRefreshError(f"Azure costs HTTP {response.status_code}")
    try:
        payload = response.json()
    except (requests.JSONDecodeError, ValueError) as exc:
        raise CostRefreshError("Azure costs payload is invalid.") from exc
    properties = payload.get("properties") if isinstance(payload, dict) else None
    if isinstance(properties, dict) and properties.get("nextLink"):
        raise CostRefreshError(
            "Azure costs response is paginated and cannot be shown as a complete snapshot.",
            status="unavailable",
        )
    columns = properties.get("columns") if isinstance(properties, dict) else None
    rows = properties.get("rows") if isinstance(properties, dict) else None
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise CostRefreshError("Azure costs payload is invalid.")
    column_names = [
        column.get("name") if isinstance(column, dict) else None for column in columns
    ]
    if any(not isinstance(name, str) for name in column_names):
        raise CostRefreshError(
            "Azure costs column names are invalid.", status="unavailable"
        )
    required_columns = {"PreTaxCost", "ResourceId", "UsageDate", "Currency"}
    if len(set(column_names)) != len(column_names) or not required_columns.issubset(
        column_names
    ):
        raise CostRefreshError(
            "Azure costs response is missing required dimensions.",
            status="unavailable",
        )
    positions = {name: column_names.index(name) for name in required_columns}
    buckets: list[CostBucket] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(column_names):
            raise CostRefreshError("Azure costs row is invalid.")
        returned_resource_id = row[positions["ResourceId"]]
        if (
            not isinstance(returned_resource_id, str)
            or returned_resource_id.casefold() != resource_id.casefold()
        ):
            raise CostRefreshError(
                "Azure costs response does not confirm the bound resource.",
                status="unavailable",
            )
        currency = row[positions["Currency"]]
        if (
            not isinstance(currency, str)
            or re.fullmatch(r"[A-Za-z]{3}", currency) is None
        ):
            raise CostRefreshError("Azure costs currency is invalid.")
        raw_usage_date = row[positions["UsageDate"]]
        usage_date_text = str(raw_usage_date)
        if (
            isinstance(raw_usage_date, bool)
            or not usage_date_text.isascii()
            or re.fullmatch(r"[0-9]{8}", usage_date_text) is None
        ):
            raise CostRefreshError("Azure costs row contains invalid values.")
        try:
            value = Decimal(str(row[positions["PreTaxCost"]]))
            usage_date = datetime.strptime(usage_date_text, "%Y%m%d").replace(
                tzinfo=timezone.utc
            )
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise CostRefreshError("Azure costs row contains invalid values.") from exc
        if not value.is_finite():
            raise CostRefreshError("Azure costs row contains invalid values.")
        _validate_storage_decimal(value, "Azure", "costs", "PreTaxCost")
        bucket_end = usage_date + timedelta(days=1)
        if usage_date < period_start or bucket_end > period_end:
            raise CostRefreshError("Azure costs row is outside the requested period.")
        buckets.append(
            CostBucket(
                kind="actual",
                metric="cost",
                value=value,
                unit="currency",
                currency=currency.upper(),
                bucket_start=usage_date,
                bucket_end=bucket_end,
                source="azure.costmanagement.query",
                granularity="day",
                dimensions={"resource_id": resource_id},
            )
        )
    return buckets


def _azure_arm_token() -> str:
    """Acquire an ARM token from operator-managed host identity."""
    credential = None
    try:
        if os.getenv("AZURE_FEDERATED_TOKEN_FILE"):
            credential = WorkloadIdentityCredential()
        elif any(
            os.getenv(name)
            for name in (
                "IDENTITY_ENDPOINT",
                "MSI_ENDPOINT",
                "WEBSITE_HOSTNAME",
                "CONTAINER_APP_HOSTNAME",
            )
        ):
            credential = ManagedIdentityCredential(
                client_id=os.getenv("AZURE_CLIENT_ID")
            )
        elif all(
            os.getenv(name)
            for name in (
                "AZURE_TENANT_ID",
                "AZURE_CLIENT_ID",
                "AZURE_CLIENT_CERTIFICATE_PATH",
            )
        ):
            credential = CertificateCredential(
                tenant_id=os.environ["AZURE_TENANT_ID"],
                client_id=os.environ["AZURE_CLIENT_ID"],
                certificate_path=os.environ["AZURE_CLIENT_CERTIFICATE_PATH"],
                password=os.getenv("AZURE_CLIENT_CERTIFICATE_PASSWORD"),
            )
        else:
            # Azure VMs expose managed identity through IMDS without environment markers.
            credential = ManagedIdentityCredential(
                client_id=os.getenv("AZURE_CLIENT_ID")
            )
        return credential.get_token("https://management.azure.com/.default").token
    except (AzureError, OSError, ValueError) as exc:
        raise CostRefreshError("Azure host identity authentication failed.") from exc
    finally:
        if credential is not None:
            credential.close()
