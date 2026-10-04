"""DeepSeek live balance validation and admin-action contracts."""

import re
import secrets
from decimal import Decimal

import pytest
import requests
from sqlalchemy import select

from app.admin.security import ADMIN_COOKIE_NAME
from app.persistence.admin_auth import authenticate_admin, create_admin_session
from app.persistence.admin_ops import create_provider_profile
from app.persistence.models import CostUsageRecord, ProviderProfile, Tenant
from app.persistence.passkeys import insert_passkey
from app.providers.deepseek_balance import (
    DeepSeekBalanceError,
    fetch_deepseek_balance,
)


def test_fetch_deepseek_balance_validates_multiple_currencies(requests_mock):
    """Return exact nonnegative decimal values from each reported currency."""
    request = requests_mock.get(
        "https://api.deepseek.com/user/balance",
        json={
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "USD",
                    "total_balance": "12.340",
                    "granted_balance": "2.00",
                    "topped_up_balance": "10.34",
                },
                {
                    "currency": "CNY",
                    "total_balance": 4,
                    "granted_balance": 0,
                    "topped_up_balance": 4,
                },
            ],
        },
    )

    balance = fetch_deepseek_balance("deepseek-key")

    assert request.last_request.headers["Authorization"] == "Bearer deepseek-key"
    assert balance.is_available is True
    assert balance.balance_infos[0].total_balance == Decimal("12.340")
    assert balance.balance_infos[1].currency == "CNY"


def test_fetch_deepseek_balance_preserves_json_number_precision(requests_mock):
    """Parse numeric JSON balances directly as Decimal without float loss."""
    requests_mock.get(
        "https://api.deepseek.com/user/balance",
        text=(
            '{"is_available":true,"balance_infos":[{"currency":"USD",'
            '"total_balance":0.12345678901234567890123456789,'
            '"granted_balance":0,"topped_up_balance":0}]}'
        ),
        headers={"Content-Type": "application/json"},
    )

    balance = fetch_deepseek_balance("deepseek-key")

    assert balance.balance_infos[0].total_balance == Decimal(
        "0.12345678901234567890123456789"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"is_available": 1, "balance_infos": []},
        {"is_available": True},
        {"is_available": True, "balance_infos": {}},
        {
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "EUR",
                    "total_balance": "1",
                    "granted_balance": "0",
                    "topped_up_balance": "1",
                }
            ],
        },
        {
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "USD",
                    "total_balance": "NaN",
                    "granted_balance": "0",
                    "topped_up_balance": "0",
                }
            ],
        },
        {
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "USD",
                    "total_balance": "-1",
                    "granted_balance": "0",
                    "topped_up_balance": "0",
                }
            ],
        },
    ],
)
def test_fetch_deepseek_balance_rejects_invalid_payloads(requests_mock, payload):
    """Malformed balance schema and unsupported values fail explicitly."""
    requests_mock.get("https://api.deepseek.com/user/balance", json=payload)

    with pytest.raises(DeepSeekBalanceError):
        fetch_deepseek_balance("deepseek-key")


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({"status_code": 401, "json": {"error": "no"}}, "HTTP 401"),
        ({"text": "not-json"}, "response is invalid"),
    ],
)
def test_fetch_deepseek_balance_surfaces_http_and_json_failures(
    requests_mock, response, error
):
    """HTTP status and invalid JSON errors are provider-safe and explicit."""
    requests_mock.get("https://api.deepseek.com/user/balance", **response)

    with pytest.raises(DeepSeekBalanceError, match=error):
        fetch_deepseek_balance("deepseek-key")


def _deepseek_admin_client(admin_app, *, profile_tenant="acme"):
    """Create an authenticated admin client and one encrypted DeepSeek profile."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        account = authenticate_admin(session, "ada", "correct-horse-battery")
        insert_passkey(
            session,
            account_id=account.id,
            credential_id=secrets.token_bytes(32),
            public_key=secrets.token_bytes(64),
            sign_count=0,
            user_handle=secrets.token_bytes(32),
            label="Primary",
            aaguid=None,
            backed_up=False,
        )
        principal = create_admin_session(session, account, enrollment_only=False)
        profile = create_provider_profile(
            session,
            database.secret_cipher,
            profile_tenant,
            "deepseek",
            "Production",
            {},
            "deepseek-v4-flash",
            "deepseek-key",
            "ada",
        )
        profile_id = profile.id
    client = admin_app.test_client()
    client.set_cookie(ADMIN_COOKIE_NAME, principal.token, path="/admin")
    return client, profile_id


def test_deepseek_balance_admin_action_renders_live_balance_without_cost_records(
    admin_app, requests_mock
):
    """The explicit balance action renders current data without cost persistence."""
    database = admin_app.extensions["database"]
    client, profile_id = _deepseek_admin_client(admin_app)
    page = client.get("/admin/settings/costs")
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"',
        page.get_data(as_text=True),
    ).group(1)
    request = requests_mock.get(
        "https://api.deepseek.com/user/balance",
        json={
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "USD",
                    "total_balance": "12.340",
                    "granted_balance": "2.00",
                    "topped_up_balance": "10.34",
                }
            ],
        },
    )

    response = client.post(
        f"/admin/settings/costs/deepseek-balance/{profile_id}",
        data={"csrf_token": csrf},
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "12.340 USD" in body
    assert "Live-Kontostand" in body
    assert "https://platform.deepseek.com/usage" in body
    assert request.last_request.headers["Authorization"] == "Bearer deepseek-key"
    with database.sessions() as session:
        assert session.scalar(select(CostUsageRecord.id)) is None


def test_deepseek_balance_admin_action_requires_csrf(admin_app, requests_mock):
    """Reject balance POST without valid CSRF before contacting DeepSeek."""
    client, profile_id = _deepseek_admin_client(admin_app)

    response = client.post(f"/admin/settings/costs/deepseek-balance/{profile_id}")

    assert response.status_code == 400
    assert requests_mock.call_count == 0


def test_deepseek_balance_admin_action_rejects_foreign_profile(
    admin_app, requests_mock
):
    """An account outside the logged-in tenant cannot be queried."""
    database = admin_app.extensions["database"]
    with database.sessions.begin() as session:
        session.add(
            Tenant(
                id="other",
                api_key_hash="unused",
                custom_model_id="cursor-other",
            )
        )
    client, _profile_id = _deepseek_admin_client(admin_app, profile_tenant="other")
    page = client.get("/admin/settings/costs")
    csrf = re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"',
        page.get_data(as_text=True),
    ).group(1)
    with database.sessions() as session:
        foreign_profile_id = session.scalar(
            select(ProviderProfile.id).where(ProviderProfile.tenant_id == "other")
        )

    response = client.post(
        f"/admin/settings/costs/deepseek-balance/{foreign_profile_id}",
        data={"csrf_token": csrf},
    )

    assert response.status_code == 404
    assert requests_mock.call_count == 0


def test_fetch_deepseek_balance_surfaces_transport_failures(requests_mock):
    """Network errors become a provider-specific failure without key reflection."""
    requests_mock.get(
        "https://api.deepseek.com/user/balance",
        exc=requests.ConnectTimeout("deepseek-key"),
    )

    with pytest.raises(DeepSeekBalanceError, match="request failed") as caught:
        fetch_deepseek_balance("deepseek-key")
    assert "deepseek-key" not in str(caught.value)
