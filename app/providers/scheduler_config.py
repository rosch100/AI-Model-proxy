"""Shared validation for administrator-configured provider scheduler policies."""

from __future__ import annotations

import json
from collections.abc import Mapping

from app.persistence.provider_scheduler import (
    MAX_POSTGRES_INTEGER,
    BudgetMetric,
    BudgetScopeKind,
)

_POLICY_REQUIRED_KEYS = frozenset({"scope_kind", "metric", "limit", "window_seconds"})
_POLICY_KEYS = _POLICY_REQUIRED_KEYS | {"scope_id"}


def parse_scheduler_limits(value: object) -> list[dict[str, object]]:
    """Parse and strictly validate scheduler policy JSON or its decoded list."""
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("scheduler_limits must be valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("scheduler_limits must be a JSON array")

    scope_kinds = {kind.value for kind in BudgetScopeKind}
    metrics = {metric.value for metric in BudgetMetric}
    parsed: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(value):
        if (
            not isinstance(entry, Mapping)
            or not _POLICY_REQUIRED_KEYS.issubset(entry)
            or not set(entry).issubset(_POLICY_KEYS)
        ):
            raise ValueError(f"scheduler_limits[{index}] has an invalid shape")
        scope_kind = entry["scope_kind"]
        metric = entry["metric"]
        scope_id = entry.get("scope_id")
        limit = entry["limit"]
        window_seconds = entry["window_seconds"]
        if not isinstance(scope_kind, str) or scope_kind not in scope_kinds:
            raise ValueError(f"scheduler_limits[{index}].scope_kind is unsupported")
        if not isinstance(metric, str) or metric not in metrics:
            raise ValueError(f"scheduler_limits[{index}].metric is unsupported")
        if scope_id is not None and (
            not isinstance(scope_id, str) or not scope_id.strip()
        ):
            raise ValueError(f"scheduler_limits[{index}].scope_id must be non-empty")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or limit > MAX_POSTGRES_INTEGER
            or isinstance(window_seconds, bool)
            or not isinstance(window_seconds, int)
            or window_seconds <= 0
            or window_seconds > MAX_POSTGRES_INTEGER
        ):
            raise ValueError(
                f"scheduler_limits[{index}] values must fit positive "
                "32-bit PostgreSQL INTEGER columns"
            )
        key = (scope_kind, scope_id or "<runtime>", metric)
        if key in seen:
            raise ValueError(f"scheduler_limits[{index}] duplicates a scope and metric")
        seen.add(key)
        normalized_entry = {
            "scope_kind": scope_kind,
            "metric": metric,
            "limit": limit,
            "window_seconds": window_seconds,
        }
        if scope_id is not None:
            normalized_entry["scope_id"] = scope_id.strip()
        parsed.append(normalized_entry)
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant {value} is not allowed")
