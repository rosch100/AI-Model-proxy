"""Canonical Azure ARM scope identifiers used by cost binding flows."""

from __future__ import annotations

import unicodedata
from uuid import UUID


def canonical_subscription_id(subscription_id: str) -> str:
    """Validate and normalize an Azure subscription UUID."""
    return _subscription_id(subscription_id)


def canonical_resource_group_id(scope_id: str) -> str:
    """Validate and normalize an Azure Resource Group ARM ID."""
    parts = scope_id.strip("/").split("/")
    if (
        len(parts) != 4
        or parts[0].casefold() != "subscriptions"
        or parts[2].casefold() != "resourcegroups"
        or any(not part for part in parts)
    ):
        raise ValueError("Azure billing scope must be a Resource Group ARM ID.")
    subscription_id = _subscription_id(parts[1])
    _reject_url_components(scope_id)
    resource_group = parts[3]
    if not _is_valid_resource_name(resource_group, max_length=90):
        raise ValueError("Azure Resource Group name is invalid.")
    return f"/subscriptions/{subscription_id}/resourcegroups/{resource_group.lower()}"


def canonical_cost_scopes(
    subscription_id: str,
    resource_group_arm_id: str,
    cognitive_resource_arm_id: str,
) -> tuple[str, str]:
    """Validate and normalize Azure billing and usage scopes together."""
    subscription = canonical_subscription_id(subscription_id)
    resource_group = canonical_resource_group_id(resource_group_arm_id)
    cognitive_resource = canonical_cognitive_resource_id(cognitive_resource_arm_id)
    expected_prefix = f"/subscriptions/{subscription}/"
    if not resource_group.startswith(
        expected_prefix
    ) or not cognitive_resource.startswith(expected_prefix):
        raise ValueError("Azure ARM scopes must belong to the entered subscription.")
    parent_prefix = resource_group + "/providers/microsoft.cognitiveservices/accounts/"
    if not cognitive_resource.startswith(parent_prefix):
        raise ValueError("Azure usage scope must belong to the bound Resource Group.")
    return resource_group, cognitive_resource


def canonical_cognitive_resource_id(scope_id: str) -> str:
    """Validate and normalize an Azure Cognitive Services ARM ID."""
    parts = scope_id.strip("/").split("/")
    if (
        len(parts) != 8
        or parts[0].casefold() != "subscriptions"
        or parts[2].casefold() != "resourcegroups"
        or parts[4].casefold() != "providers"
        or parts[5].casefold() != "microsoft.cognitiveservices"
        or parts[6].casefold() != "accounts"
        or any(not part for part in parts)
    ):
        raise ValueError(
            "Azure usage scope must be a Cognitive Services account ARM ID."
        )
    subscription_id = _subscription_id(parts[1])
    _reject_url_components(scope_id)
    resource_group = parts[3]
    account_name = parts[7]
    if not _is_valid_resource_name(resource_group, max_length=90):
        raise ValueError("Azure Resource Group name is invalid.")
    if not _is_valid_cognitive_account_name(account_name):
        raise ValueError("Azure Cognitive Services account name is invalid.")
    return (
        f"/subscriptions/{subscription_id}/resourcegroups/{resource_group.lower()}"
        f"/providers/microsoft.cognitiveservices/accounts/{account_name.casefold()}"
    )


def _subscription_id(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ValueError("Azure scope has an invalid subscription GUID.") from exc


def _reject_url_components(scope_id: str) -> None:
    if "?" in scope_id or "#" in scope_id or "://" in scope_id:
        raise ValueError("Azure scope must not contain a URL, query, or fragment.")


def _is_valid_resource_name(name: str, *, max_length: int) -> bool:
    return (
        1 <= len(name) <= max_length
        and name[-1] != "."
        and all(
            unicodedata.category(character) in {"Lu", "Ll", "Lt", "Lm", "Lo", "Nd"}
            or character in "_.()-"
            for character in name
        )
    )


def _is_valid_cognitive_account_name(name: str) -> bool:
    return (
        2 <= len(name) <= 64
        and name[0].isalnum()
        and name[-1].isalnum()
        and all(
            character.isascii() and (character.isalnum() or character == "-")
            for character in name
        )
    )
