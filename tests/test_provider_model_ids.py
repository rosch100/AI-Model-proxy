"""Tests for public catalog model IDs used by tenant routing."""

import pytest

from app.exceptions import ServiceConfigurationError
from app.providers.model_ids import (
    account_model_id,
    qualified_model_id,
    tenant_catalog_model_ids,
)
from app.providers.routing import _route_targets
from app.tenants import DatabaseTenantSnapshot


def _profile(
    provider: str,
    name: str | None,
    catalog: tuple[str, ...],
) -> DatabaseTenantSnapshot:
    return DatabaseTenantSnapshot(
        id="tenant",
        api_key_hash="unused",
        custom_model_id="cursor-tenant",
        provider=provider,
        provider_settings={},
        inference_secret="secret",
        default_model=catalog[0] if catalog else None,
        profile_id=f"{provider}-profile",
        profile_name=name,
        catalog_model_ids=catalog,
    )


def test_qualified_model_ids_preserve_spaces_and_escape_reserved_account_chars():
    """Keep spaces readable and escape alias delimiters in account names."""
    account = "Research Team / A:B%ü"
    model = "org/model-x"

    assert account_model_id(account, model) == (
        "Research Team %2F A%3AB%25%C3%BC/org/model-x"
    )
    assert qualified_model_id("openrouter", account, model) == (
        "openrouter:Research Team %2F A%3AB%25%C3%BC/org/model-x"
    )


def test_unqualified_account_alias_pins_the_matching_profile():
    """Resolve the user-facing account/model shorthand to one profile."""
    profile = _profile("deepseek", "Research Team", ("deepseek-flash",))

    assert _route_targets(
        (profile,), "Research Team/deepseek-flash", "cursor-tenant"
    ) == ((profile, "deepseek-flash"),)


def test_unqualified_account_alias_rejects_duplicate_profile_names():
    """Require a provider prefix when account names do not identify one profile."""
    profiles = (
        _profile("openai", "Research Team", ("model-x",)),
        _profile("deepseek", "Research Team", ("model-x",)),
    )

    with pytest.raises(ServiceConfigurationError, match="include its provider name"):
        _route_targets(profiles, "Research Team/model-x", "cursor-tenant")


def test_provider_qualified_alias_rejects_multiple_matches():
    """Reject duplicate provider-qualified aliases rather than choosing one."""
    profiles = (
        _profile("openai", "Primary", ("model-x",)),
        _profile("openai", "Primary", ("model-x",)),
    )

    with pytest.raises(ServiceConfigurationError, match="ambiguous"):
        _route_targets(profiles, "openai:Primary/model-x", "cursor-tenant")


def test_native_catalog_id_wins_over_colliding_account_alias():
    """Resolve a native catalog ID even when it resembles another account alias."""
    native_profile = _profile("openai", None, ("Research Team/model-x",))
    alias_profile = _profile("deepseek", "Research Team", ("model-x",))
    profiles = (native_profile, alias_profile)

    assert _route_targets(profiles, "Research Team/model-x", "cursor-tenant") == (
        (native_profile, "Research Team/model-x"),
    )


def test_native_catalog_id_is_not_suppressed_by_colliding_account_alias():
    """Keep native catalog IDs visible when an account alias has the same ID."""
    profiles = (
        _profile("openai", None, ("Research Team/model-x",)),
        _profile("deepseek", "Research Team", ("model-x",)),
    )

    model_ids = tenant_catalog_model_ids(profiles)

    assert model_ids.count("Research Team/model-x") == 1
    assert "deepseek:Research Team/model-x" in model_ids


def test_native_catalog_id_wins_over_colliding_provider_qualified_alias():
    """Resolve native IDs before provider-qualified aliases with the same ID."""
    native_profile = _profile("openai", None, ("deepseek:Research Team/model-x",))
    alias_profile = _profile("deepseek", "Research Team", ("model-x",))

    assert _route_targets(
        (native_profile, alias_profile),
        "deepseek:Research Team/model-x",
        "cursor-tenant",
    ) == ((native_profile, "deepseek:Research Team/model-x"),)


def test_tenant_catalog_model_ids_include_custom_native_and_qualified_ids():
    """Include each unique native ID and every profile-specific alias."""
    profiles = (
        _profile("deepseek", "Research Team", ("deepseek-flash",)),
        _profile("openrouter", "Research Team", ("deepseek-flash", "org/model-x")),
    )

    assert tenant_catalog_model_ids(profiles) == (
        "cursor-tenant",
        "deepseek-flash",
        "org/model-x",
        "deepseek:Research Team/deepseek-flash",
        "openrouter:Research Team/deepseek-flash",
        "openrouter:Research Team/org/model-x",
    )


def test_tenant_catalog_model_ids_omit_duplicate_native_and_missing_account_aliases():
    """Deduplicate native IDs and omit aliases for unnamed accounts."""
    profiles = (
        _profile("openai", "Primary", ("gpt-5.4",)),
        _profile("openai", None, ("gpt-5.4", "gpt-5.5")),
    )

    assert tenant_catalog_model_ids(profiles) == (
        "cursor-tenant",
        "gpt-5.4",
        "gpt-5.5",
        "Primary/gpt-5.4",
        "openai:Primary/gpt-5.4",
    )


def test_tenant_catalog_model_ids_filter_to_azure_profiles():
    """Expose only Azure IDs on the explicit Azure model-list route."""
    profiles = (
        _profile("azure", "Azure Work", ("gpt-5.4",)),
        _profile("deepseek", "Research Team", ("deepseek-flash",)),
    )

    assert tenant_catalog_model_ids(profiles, azure_only=True) == (
        "cursor-tenant",
        "gpt-5.4",
        "Azure Work/gpt-5.4",
        "azure:Azure Work/gpt-5.4",
    )
