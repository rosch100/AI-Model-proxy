"""Provider-neutral classification for structured upstream error signals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from app.common.retry_after import RETRY_AFTER_MAX_SECONDS

_OPENAI_QUOTA_SCOPES = {
    "organization_spend_limit_exceeded": "organization",
    "organization_usage_limit_exceeded": "organization",
    "project_spend_limit_exceeded": "project",
}
_OPENAI_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        *_OPENAI_QUOTA_SCOPES,
    }
)
_OPENAI_TRANSIENT_CODES = frozenset(
    {
        "rate_limit_exceeded",
        "requests_rate_limit_exceeded",
        "tokens_rate_limit_exceeded",
    }
)
_TERMINAL_CODES = frozenset(
    {
        "authentication_error",
        "invalid_api_key",
        "permission_error",
        "invalid_request_error",
        "model_not_found",
    }
)


@dataclass(frozen=True)
class UpstreamErrorClassification:
    """Safe classification metadata; never contains raw provider error text."""

    category: str
    scope_type: str | None = None
    scope_id: str | None = None
    retry_after_seconds: float | None = None


def classify_upstream_error(
    provider: str,
    status: int | None,
    error: Mapping[str, Any] | None,
    headers: Mapping[str, Any] | None,
    settings: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> UpstreamErrorClassification:
    """Classify only explicit structured codes and validated provider metadata."""
    code = _structured_code(error)
    metadata = error.get("metadata") if isinstance(error, Mapping) else None
    limit_source = (
        metadata.get("limit_source") if isinstance(metadata, Mapping) else None
    )
    retry_after = _parse_retry_after(headers, now=now)

    if provider == "openrouter":
        if limit_source == "openrouter_key_limit":
            return UpstreamErrorClassification(
                "quota_exhausted", retry_after_seconds=retry_after
            )
        if limit_source == "openrouter_in_flight_budget":
            return UpstreamErrorClassification(
                "transient", retry_after_seconds=retry_after
            )
    if provider == "openai" and code in _OPENAI_QUOTA_CODES:
        scope_type = _OPENAI_QUOTA_SCOPES.get(code)
        scope_id = _configured_scope(settings, scope_type)
        return UpstreamErrorClassification(
            "quota_exhausted",
            scope_type if scope_id is not None else None,
            scope_id,
            retry_after,
        )
    if provider == "openai" and code in _OPENAI_TRANSIENT_CODES:
        return UpstreamErrorClassification("transient", retry_after_seconds=retry_after)
    if provider == "azure" and code in _OPENAI_TRANSIENT_CODES:
        return UpstreamErrorClassification("transient", retry_after_seconds=retry_after)
    if code in _TERMINAL_CODES:
        return UpstreamErrorClassification("terminal", retry_after_seconds=retry_after)
    if status in {408, 425, 429} or status is not None and 500 <= status <= 599:
        return UpstreamErrorClassification("transient", retry_after_seconds=retry_after)
    if status is not None and 400 <= status <= 499:
        return UpstreamErrorClassification("terminal", retry_after_seconds=retry_after)
    return UpstreamErrorClassification("unknown", retry_after_seconds=retry_after)


def _structured_code(error: Mapping[str, Any] | None) -> str | None:
    if not isinstance(error, Mapping):
        return None
    code = error.get("code") or error.get("type")
    return code.casefold() if isinstance(code, str) else None


def _configured_scope(
    settings: Mapping[str, Any] | None, scope_type: str | None
) -> str | None:
    if scope_type is None or not isinstance(settings, Mapping):
        return None
    value = settings.get(scope_type)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _parse_retry_after(
    headers: Mapping[str, Any] | None, *, now: datetime | None
) -> float | None:
    if not isinstance(headers, Mapping):
        return None
    raw = next(
        (
            value
            for key, value in headers.items()
            if str(key).casefold() == "retry-after"
        ),
        None,
    )
    if not isinstance(raw, str | int | float) or isinstance(raw, bool):
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        clock = now or datetime.now(timezone.utc)
        if clock.tzinfo is None:
            raise ValueError("Retry-After clock must be timezone-aware")
        seconds = (retry_at - clock).total_seconds()
    else:
        if isinstance(raw, bool) or not str(raw).isascii() or not str(raw).isdecimal():
            return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, RETRY_AFTER_MAX_SECONDS)
