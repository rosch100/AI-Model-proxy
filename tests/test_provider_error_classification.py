"""Provider-neutral classification of structured upstream errors."""

from datetime import datetime, timedelta, timezone

import pytest

from app.providers.error_classification import classify_upstream_error


@pytest.mark.parametrize(
    ("code", "scope_type"),
    [
        ("insufficient_quota", None),
        ("credit_balance_exhausted", None),
        ("organization_spend_limit_exceeded", "organization"),
        ("project_spend_limit_exceeded", "project"),
        ("organization_usage_limit_exceeded", "organization"),
    ],
)
def test_openai_budget_codes_are_classified_without_reading_message(code, scope_type):
    """Known durable OpenAI quota codes open the breaker, not error prose."""
    result = classify_upstream_error(
        "openai",
        429,
        {"code": code, "message": "this message must not influence classification"},
        {},
        {"organization": "org-example", "project": "proj-example"},
    )

    assert result.category == "quota_exhausted"
    assert result.scope_type == scope_type


@pytest.mark.parametrize(
    ("provider", "status", "error", "expected"),
    [
        ("openai", 429, {"code": "rate_limit_exceeded"}, "transient"),
        ("azure", 429, {"code": "rate_limit_exceeded"}, "transient"),
        (
            "openrouter",
            402,
            {
                "code": "provider_error",
                "metadata": {"limit_source": "openrouter_in_flight_budget"},
            },
            "transient",
        ),
        (
            "openrouter",
            402,
            {
                "code": "provider_error",
                "metadata": {"limit_source": "openrouter_key_limit"},
            },
            "quota_exhausted",
        ),
        (
            "openrouter",
            402,
            {"code": "payment_required", "message": "quota exceeded"},
            "terminal",
        ),
        (
            "openai",
            429,
            {"code": "some_new_code", "message": "quota exceeded"},
            "transient",
        ),
    ],
)
def test_transient_and_ambiguous_errors_never_open_quota_breaker(
    provider, status, error, expected
):
    """Rate limits and ambiguous billing errors are never durable quota signals."""
    result = classify_upstream_error(provider, status, error, {}, {})

    assert result.category == expected
    assert result.scope_type is None


def test_organization_scope_requires_explicit_configured_id():
    """Use a shared organization only when the profile explicitly configures it."""
    unconfigured = classify_upstream_error(
        "openai", 429, {"code": "organization_spend_limit_exceeded"}, {}, {}
    )
    configured = classify_upstream_error(
        "openai",
        429,
        {"code": "organization_spend_limit_exceeded"},
        {},
        {"organization": "org-123", "project": "proj-456"},
    )

    assert unconfigured.category == "quota_exhausted"
    assert unconfigured.scope_type is None
    assert configured.scope_type == "organization"
    assert configured.scope_id == "org-123"


def test_project_budget_only_shares_exact_configured_project_scope():
    """Project budget codes share only the configured project identity."""
    result = classify_upstream_error(
        "openai",
        429,
        {"code": "project_spend_limit_exceeded"},
        {},
        {"organization": "org-123", "project": "proj-456"},
    )

    assert result.scope_type == "project"
    assert result.scope_id == "proj-456"


@pytest.mark.parametrize(
    ("provider", "error"),
    [
        ("openai", {"code": "insufficient_quota"}),
        (
            "openrouter",
            {"metadata": {"limit_source": "openrouter_key_limit"}},
        ),
    ],
)
def test_quota_classification_preserves_retry_after(provider, error):
    """A durable quota classification carries its provider retry floor."""
    result = classify_upstream_error(
        provider,
        429,
        error,
        {"Retry-After": "7200"},
        {},
    )

    assert result.category == "quota_exhausted"
    assert result.retry_after_seconds == 7200


def test_retry_after_parses_seconds_and_http_date_with_upper_bound():
    """Parse both Retry-After forms and cap excessive provider-supplied delays."""
    now = datetime(2026, 10, 4, 10, tzinfo=timezone.utc)
    seconds = classify_upstream_error(
        "openrouter",
        402,
        {"metadata": {"limit_source": "openrouter_in_flight_budget"}},
        {"Retry-After": "12"},
        {},
        now=now,
    )
    date = classify_upstream_error(
        "openrouter",
        402,
        {"metadata": {"limit_source": "openrouter_in_flight_budget"}},
        {
            "retry-after": (now + timedelta(minutes=3)).strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            )
        },
        {},
        now=now,
    )
    decimal = classify_upstream_error(
        "openrouter",
        402,
        {"metadata": {"limit_source": "openrouter_in_flight_budget"}},
        {"retry-after": "12.5"},
        {},
        now=now,
    )
    excessive = classify_upstream_error(
        "openrouter",
        402,
        {"metadata": {"limit_source": "openrouter_in_flight_budget"}},
        {"retry-after": "999999999"},
        {},
        now=now,
    )

    assert seconds.retry_after_seconds == 12
    assert date.retry_after_seconds == 180
    assert decimal.retry_after_seconds is None
    assert excessive.retry_after_seconds == 7 * 24 * 60 * 60


@pytest.mark.parametrize("header", ["-1", "NaN", "12.5", "invalid"])
def test_invalid_retry_after_is_ignored(header):
    """Ignore malformed and negative provider Retry-After values."""
    result = classify_upstream_error(
        "openrouter",
        402,
        {"metadata": {"limit_source": "openrouter_in_flight_budget"}},
        {"retry-after": header},
        {},
        now=datetime.now(timezone.utc),
    )

    assert result.retry_after_seconds is None
