"""Build Cursor-facing model IDs from tenant provider catalogs."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import quote

from app.tenants import DatabaseTenantSnapshot


def account_model_id(profile_name: str, model_id: str) -> str:
    """Return an account-qualified ID while keeping account spaces readable."""
    if not profile_name or not model_id:
        raise ValueError("Profile name and model ID must be non-empty")
    account = quote(profile_name, safe=" ")
    return f"{account}/{model_id}"


def qualified_model_id(provider: str, profile_name: str, model_id: str) -> str:
    """Return a provider-and-account ID while keeping spaces readable."""
    if not provider:
        raise ValueError("Provider must be non-empty")
    return f"{provider}:{account_model_id(profile_name, model_id)}"


def tenant_catalog_model_ids(
    profiles: Sequence[DatabaseTenantSnapshot],
    *,
    azure_only: bool = False,
    include_custom_model_id: bool = True,
) -> tuple[str, ...]:
    """Project ready tenant profiles into unique IDs Cursor can select."""
    candidates = tuple(
        profile for profile in profiles if not azure_only or profile.provider == "azure"
    )
    if not candidates:
        return ()

    custom_model_id = candidates[0].custom_model_id
    model_ids = [custom_model_id] if include_custom_model_id else []
    native_ids = {custom_model_id.casefold()} if include_custom_model_id else set()
    account_name_counts = {}
    for profile in candidates:
        if profile.profile_name is not None:
            key = profile.profile_name.casefold()
            account_name_counts[key] = account_name_counts.get(key, 0) + 1

    for profile in candidates:
        for model_id in profile.catalog_model_ids:
            normalized = model_id.casefold()
            if normalized not in native_ids:
                model_ids.append(model_id)
                native_ids.add(normalized)

    aliases = []
    for profile in candidates:
        if profile.profile_name is None:
            continue
        for model_id in profile.catalog_model_ids:
            account_key = profile.profile_name.casefold()
            if account_name_counts[account_key] == 1:
                aliases.append(account_model_id(profile.profile_name, model_id))
            if profile.provider is not None:
                aliases.append(
                    qualified_model_id(profile.provider, profile.profile_name, model_id)
                )

    seen_aliases = set(native_ids)
    for alias in aliases:
        normalized = alias.casefold()
        if normalized not in seen_aliases:
            model_ids.append(alias)
            seen_aliases.add(normalized)
    return tuple(model_ids)
