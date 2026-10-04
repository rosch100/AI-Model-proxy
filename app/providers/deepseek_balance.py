"""Strict live balance lookup for the DeepSeek account API."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import requests


class DeepSeekBalanceError(RuntimeError):
    """Raised when DeepSeek balance data cannot be retrieved or validated."""


@dataclass(frozen=True)
class DeepSeekBalanceInfo:
    """One provider-reported currency balance."""

    currency: str
    total_balance: Decimal
    granted_balance: Decimal
    topped_up_balance: Decimal


@dataclass(frozen=True)
class DeepSeekBalance:
    """A validated, ephemeral snapshot of DeepSeek account availability."""

    is_available: bool
    balance_infos: tuple[DeepSeekBalanceInfo, ...]


def _balance_value(value: Any, field: str) -> Decimal:
    """Parse a finite nonnegative Decimal without accepting booleans or floats."""
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal)):
        raise DeepSeekBalanceError(f"DeepSeek balance field {field} is invalid.")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DeepSeekBalanceError(
            f"DeepSeek balance field {field} is invalid."
        ) from exc
    if not amount.is_finite() or amount < 0:
        raise DeepSeekBalanceError(f"DeepSeek balance field {field} is invalid.")
    return amount


def fetch_deepseek_balance(api_key: str) -> DeepSeekBalance:
    """Fetch and strictly validate account balance using its inference API key."""
    try:
        response = requests.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
    except requests.RequestException as exc:
        raise DeepSeekBalanceError("DeepSeek balance request failed.") from exc
    if response.status_code >= 400:
        raise DeepSeekBalanceError(
            f"DeepSeek balance request failed with HTTP {response.status_code}."
        )
    try:
        payload = response.json(parse_float=Decimal)
    except (requests.exceptions.JSONDecodeError, ValueError) as exc:
        raise DeepSeekBalanceError("DeepSeek balance response is invalid.") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("is_available"), bool
    ):
        raise DeepSeekBalanceError("DeepSeek balance response is invalid.")
    raw_balances = payload.get("balance_infos")
    if not isinstance(raw_balances, list):
        raise DeepSeekBalanceError("DeepSeek balance response is invalid.")
    balances = []
    for row in raw_balances:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("currency"), str)
            or row["currency"] not in {"USD", "CNY"}
        ):
            raise DeepSeekBalanceError("DeepSeek balance response is invalid.")
        balances.append(
            DeepSeekBalanceInfo(
                currency=row["currency"],
                total_balance=_balance_value(row.get("total_balance"), "total_balance"),
                granted_balance=_balance_value(
                    row.get("granted_balance"), "granted_balance"
                ),
                topped_up_balance=_balance_value(
                    row.get("topped_up_balance"), "topped_up_balance"
                ),
            )
        )
    return DeepSeekBalance(payload["is_available"], tuple(balances))
