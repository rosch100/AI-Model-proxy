"""Validated tenant and provider-profile routing policy values."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from app.persistence.provider_scheduler import (
    AIMD_INITIAL_CONCURRENCY,
    AIMD_MAX_CONCURRENCY,
    AIMD_MIN_CONCURRENCY,
)

_VALID_STRATEGIES = frozenset({"prioritized", "load_balanced"})
_VALID_BALANCERS = frozenset({"weighted_least_loaded"})
_VALID_COST_POLICIES = frozenset({"ignore", "prefer_lower_cost", "cost_tiers"})
_VALID_TIE_BREAKERS = frozenset({"profile_id"})
_MAX_WEIGHT = 100.0
_MAX_COST_TIER = 1000
_MAX_RETRY_WAIT_SECONDS = 300


@dataclass(frozen=True)
class TenantRoutingSettings:
    """Tenant-wide validated route selection policy."""

    strategy: str = "prioritized"
    load_balancing_method: str = "weighted_least_loaded"
    cost_policy: str = "ignore"
    headroom_weight: float = 0.5
    max_retry_wait_seconds: int = 30
    tie_breaker: str = "profile_id"


@dataclass(frozen=True)
class ConcurrencyBounds:
    """AIMD initial, minimum, and maximum concurrency for one scope."""

    initial: int = AIMD_INITIAL_CONCURRENCY
    minimum: int = AIMD_MIN_CONCURRENCY
    maximum: int = AIMD_MAX_CONCURRENCY


@dataclass(frozen=True)
class ProfileRoutingSettings:
    """Validated profile-level weights, cost tier, and concurrency bounds."""

    cost_tier: int
    load_balancing_weight: float
    profile_concurrency: ConcurrencyBounds
    model_concurrency: Mapping[str, ConcurrencyBounds]

    def concurrency_for_model(self, model_id: str) -> ConcurrencyBounds:
        """Return model-specific bounds or inherit the profile's bounds."""
        return self.model_concurrency.get(model_id, self.profile_concurrency)


def parse_tenant_routing_settings(**values: object) -> TenantRoutingSettings:
    """Validate tenant-level policy values, applying documented defaults."""
    allowed = {
        "strategy",
        "load_balancing_method",
        "cost_policy",
        "headroom_weight",
        "max_retry_wait_seconds",
        "tie_breaker",
    }
    if not set(values).issubset(allowed):
        raise ValueError("Tenant routing settings contain unsupported fields")
    strategy = values.get("strategy", "prioritized")
    method = values.get("load_balancing_method", "weighted_least_loaded")
    policy = values.get("cost_policy", "ignore")
    tie_breaker = values.get("tie_breaker", "profile_id")
    headroom_weight = values.get("headroom_weight", 0.5)
    max_wait = values.get("max_retry_wait_seconds", 30)
    if not isinstance(strategy, str) or strategy not in _VALID_STRATEGIES:
        raise ValueError("Unsupported tenant routing strategy")
    if not isinstance(method, str) or method not in _VALID_BALANCERS:
        raise ValueError("Unsupported load-balancing method")
    if not isinstance(policy, str) or policy not in _VALID_COST_POLICIES:
        raise ValueError("Unsupported provider cost policy")
    if not isinstance(tie_breaker, str) or tie_breaker not in _VALID_TIE_BREAKERS:
        raise ValueError("Unsupported provider route tie-breaker")
    if (
        isinstance(headroom_weight, bool)
        or not isinstance(headroom_weight, (int, float))
        or not math.isfinite(headroom_weight)
        or not 0 <= headroom_weight <= 1
    ):
        raise ValueError("headroom_weight must be finite and between 0 and 1")
    if (
        isinstance(max_wait, bool)
        or not isinstance(max_wait, int)
        or not 1 <= max_wait <= _MAX_RETRY_WAIT_SECONDS
    ):
        raise ValueError("max_retry_wait_seconds is outside its supported range")
    return TenantRoutingSettings(
        strategy=strategy,
        load_balancing_method=method,
        cost_policy=policy,
        headroom_weight=float(headroom_weight),
        max_retry_wait_seconds=max_wait,
        tie_breaker=tie_breaker,
    )


def parse_profile_routing_settings(
    value: object,
    *,
    catalog_model_ids: Collection[str] | None = None,
) -> ProfileRoutingSettings:
    """Parse profile routing settings and optionally verify catalog model IDs."""
    if not isinstance(value, Mapping):
        raise ValueError("Provider profile settings must be an object")
    routing = value.get("routing", {})
    if not isinstance(routing, Mapping):
        raise ValueError("Provider profile routing settings must be an object")
    allowed = {
        "cost_tier",
        "load_balancing_weight",
        "profile_concurrency",
        "model_concurrency",
    }
    if not set(routing).issubset(allowed):
        raise ValueError("Provider profile routing settings contain unsupported fields")
    cost_tier = routing.get("cost_tier", 0)
    weight = routing.get("load_balancing_weight", 1.0)
    if (
        isinstance(cost_tier, bool)
        or not isinstance(cost_tier, int)
        or not 0 <= cost_tier <= _MAX_COST_TIER
    ):
        raise ValueError("cost_tier is outside its supported range")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or not 0 < weight <= _MAX_WEIGHT
    ):
        raise ValueError("load_balancing_weight must be finite and between 0 and 100")
    profile_bounds = _parse_bounds(routing.get("profile_concurrency", {}))
    raw_model_bounds = routing.get("model_concurrency", {})
    if not isinstance(raw_model_bounds, Mapping):
        raise ValueError("model_concurrency must be an object keyed by target model")
    model_bounds = {}
    for model_id, bounds in raw_model_bounds.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_concurrency model IDs must be non-empty strings")
        if catalog_model_ids is not None and model_id not in catalog_model_ids:
            raise ValueError(
                "model_concurrency references a model outside the profile catalog"
            )
        model_bounds[model_id] = _parse_bounds(bounds)
    return ProfileRoutingSettings(
        cost_tier=cost_tier,
        load_balancing_weight=float(weight),
        profile_concurrency=profile_bounds,
        model_concurrency=model_bounds,
    )


def _parse_bounds(value: object) -> ConcurrencyBounds:
    if not isinstance(value, Mapping):
        raise ValueError("Concurrency bounds must be an object")
    if not set(value).issubset({"initial", "min", "max"}):
        raise ValueError("Concurrency bounds contain unsupported fields")
    initial = value.get("initial", AIMD_INITIAL_CONCURRENCY)
    minimum = value.get("min", AIMD_MIN_CONCURRENCY)
    maximum = value.get("max", AIMD_MAX_CONCURRENCY)
    for key, item in (("initial", initial), ("min", minimum), ("max", maximum)):
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not AIMD_MIN_CONCURRENCY <= item <= AIMD_MAX_CONCURRENCY
        ):
            raise ValueError(f"Concurrency {key} must be between 1 and 64")
    if minimum > initial or initial > maximum:
        raise ValueError("Concurrency bounds must satisfy min <= initial <= max")
    return ConcurrencyBounds(initial, minimum, maximum)
