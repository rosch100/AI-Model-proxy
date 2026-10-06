"""Validation tests for tenant and provider routing settings."""

from __future__ import annotations

import pytest

from app.providers.routing_config import (
    parse_profile_routing_settings,
    parse_tenant_routing_settings,
)


def test_tenant_routing_settings_use_safe_prioritized_defaults():
    """Tenant routing defaults to conservative prioritized behavior."""
    settings = parse_tenant_routing_settings()

    assert settings.strategy == "prioritized"
    assert settings.load_balancing_method == "weighted_least_loaded"
    assert settings.cost_policy == "ignore"
    assert settings.headroom_weight == 0.5
    assert settings.max_retry_wait_seconds == 30
    assert settings.tie_breaker == "profile_id"


def test_tenant_routing_settings_validate_modes_and_bounds():
    """Tenant modes and numeric options reject unsupported values."""
    settings = parse_tenant_routing_settings(
        strategy="load_balanced",
        load_balancing_method="weighted_least_loaded",
        cost_policy="prefer_lower_cost",
        headroom_weight=0.75,
        max_retry_wait_seconds=12,
        tie_breaker="profile_id",
    )

    assert settings.strategy == "load_balanced"
    assert settings.cost_policy == "prefer_lower_cost"
    assert settings.headroom_weight == 0.75
    assert settings.max_retry_wait_seconds == 12

    for kwargs in (
        {"strategy": "unknown"},
        {"load_balancing_method": "round_robin"},
        {"cost_policy": "invented"},
        {"headroom_weight": 1.1},
        {"max_retry_wait_seconds": 0},
        {"tie_breaker": "random"},
    ):
        with pytest.raises(ValueError):
            parse_tenant_routing_settings(**kwargs)


def test_profile_routing_settings_validate_weight_and_aimd_limits():
    """Profile weights, tiers, and AIMD bounds are validated."""
    settings = parse_profile_routing_settings(
        {
            "routing": {
                "cost_tier": 2,
                "load_balancing_weight": 1.5,
                "profile_concurrency": {"initial": 6, "min": 2, "max": 20},
                "model_concurrency": {"gpt-5.4": {"initial": 3, "min": 1, "max": 8}},
            }
        }
    )

    assert settings.cost_tier == 2
    assert settings.load_balancing_weight == 1.5
    assert settings.profile_concurrency.initial == 6
    assert settings.profile_concurrency.minimum == 2
    assert settings.profile_concurrency.maximum == 20
    assert settings.model_concurrency["gpt-5.4"].initial == 3

    for routing in (
        {"load_balancing_weight": 0},
        {"load_balancing_weight": float("nan")},
        {"cost_tier": -1},
        {"profile_concurrency": {"initial": 1, "min": 2, "max": 8}},
        {"model_concurrency": {"gpt-5.4": {"initial": 2, "min": 1, "max": 65}}},
        {"unknown": True},
    ):
        with pytest.raises(ValueError):
            parse_profile_routing_settings({"routing": routing})


def test_profile_routing_settings_default_model_limits_to_profile_limits():
    """Unspecified model limits inherit the profile's concurrency bounds."""
    settings = parse_profile_routing_settings({})
    customized = parse_profile_routing_settings(
        {
            "routing": {
                "profile_concurrency": {"initial": 8, "min": 2, "max": 16},
                "model_concurrency": {"gpt-5.4": {"initial": 3, "min": 1, "max": 6}},
            }
        }
    )

    assert settings.cost_tier == 0
    assert settings.load_balancing_weight == 1.0
    assert settings.profile_concurrency.initial == 4
    assert settings.profile_concurrency.minimum == 1
    assert settings.profile_concurrency.maximum == 64
    assert settings.model_concurrency == {}
    assert customized.concurrency_for_model("gpt-5.4").initial == 3
    assert (
        customized.concurrency_for_model("other-model")
        == customized.profile_concurrency
    )
