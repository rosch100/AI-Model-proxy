"""Persist and load per-request inference activity for the tenant overview."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from flask import current_app
from sqlalchemy import delete, select, update
from sqlalchemy.exc import SQLAlchemyError

from app.persistence.database import Database
from app.persistence.models import InferenceActivityEvent, ProviderAttemptEvent

ACTIVITY_WINDOW_MINUTES = 15
ACTIVITY_LOOKBACK_HOURS = 24
PROVIDER_ATTEMPT_RETENTION = timedelta(hours=24)
_PROVIDER_ATTEMPT_PRUNE_BATCH_SIZE = 1000
ACTIVITY_LOOKBACK_OPTIONS = (
    (1, "Letzte Stunde"),
    (6, "Letzte 6 Stunden"),
    (24, "Letzte 24 Stunden"),
    (168, "Letzte 7 Tage"),
    (720, "Letzte 30 Tage"),
)


@dataclass(frozen=True)
class ParsedTokenUsage:
    """Token counts reported by an upstream provider for one request."""

    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    total_tokens: int


def _as_token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def parse_provider_usage(usage: object) -> ParsedTokenUsage | None:
    """Read Azure Responses or Chat Completions usage without inventing counts."""
    if not isinstance(usage, dict):
        return None
    input_tokens = _as_token_count(
        usage.get("input_tokens", usage.get("prompt_tokens"))
    )
    output_tokens = _as_token_count(
        usage.get("output_tokens", usage.get("completion_tokens"))
    )
    if input_tokens is None or output_tokens is None:
        return None
    input_details = usage.get("input_tokens_details") or usage.get(
        "prompt_tokens_details"
    )
    output_details = usage.get("output_tokens_details") or usage.get(
        "completion_tokens_details"
    )
    cached_tokens = 0
    reasoning_tokens = 0
    if isinstance(input_details, dict):
        parsed_cached = _as_token_count(input_details.get("cached_tokens"))
        if parsed_cached is not None:
            cached_tokens = parsed_cached
    if isinstance(output_details, dict):
        parsed_reasoning = _as_token_count(output_details.get("reasoning_tokens"))
        if parsed_reasoning is not None:
            reasoning_tokens = parsed_reasoning
    total_tokens = _as_token_count(usage.get("total_tokens"))
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return ParsedTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def start_provider_attempt(
    *,
    tenant_id: str | None,
    provider: str | None,
    profile_id: str | None,
    inbound_model: object,
    routed_model: object,
    occurred_at: datetime | None = None,
) -> int | None:
    """Persist a pending routed provider attempt and return its identity."""
    if (
        tenant_id is None
        or provider not in {"azure", "openai", "openrouter", "deepseek"}
        or profile_id is None
        or not isinstance(inbound_model, str)
        or not inbound_model
        or not isinstance(routed_model, str)
        or not routed_model
    ):
        return None
    database = current_app.extensions.get("database")
    if not isinstance(database, Database):
        return
    moment = occurred_at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("Provider attempt timestamps must be timezone-aware")
    try:
        with database.sessions.begin() as session:
            expired_ids = session.scalars(
                select(ProviderAttemptEvent.id)
                .where(
                    ProviderAttemptEvent.occurred_at
                    < moment - PROVIDER_ATTEMPT_RETENTION
                )
                .order_by(ProviderAttemptEvent.occurred_at)
                .limit(_PROVIDER_ATTEMPT_PRUNE_BATCH_SIZE)
            ).all()
            if expired_ids:
                session.execute(
                    delete(ProviderAttemptEvent).where(
                        ProviderAttemptEvent.id.in_(expired_ids)
                    )
                )
            attempt = ProviderAttemptEvent(
                tenant_id=tenant_id,
                provider=provider,
                profile_id=profile_id,
                inbound_model=inbound_model,
                routed_model=routed_model,
                outcome="pending",
                occurred_at=moment,
            )
            session.add(attempt)
            session.flush()
            return attempt.id
    except SQLAlchemyError:
        current_app.logger.exception("Failed to start provider attempt")
        return None


def complete_provider_attempt(
    attempt_id: int | None,
    *,
    outcome: str,
    status_code: int | None,
    failure_details: dict[str, object] | None = None,
    completed_at: datetime | None = None,
) -> None:
    """Finish a pending provider attempt with its observed terminal outcome."""
    if attempt_id is None or outcome not in {"success", "failure", "aborted"}:
        return
    database = current_app.extensions.get("database")
    if not isinstance(database, Database):
        return
    moment = completed_at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("Provider attempt timestamps must be timezone-aware")
    try:
        with database.sessions.begin() as session:
            session.execute(
                update(ProviderAttemptEvent)
                .where(
                    ProviderAttemptEvent.id == attempt_id,
                    ProviderAttemptEvent.outcome == "pending",
                )
                .values(
                    outcome=outcome,
                    status_code=status_code,
                    failure_details=failure_details,
                    completed_at=moment,
                )
            )
    except SQLAlchemyError:
        current_app.logger.exception("Failed to complete provider attempt")


def record_inference_activity(
    *,
    tenant_id: str | None,
    provider: str | None,
    profile_id: str | None,
    inbound_model: object,
    routed_model: object,
    usage: ParsedTokenUsage | None,
    occurred_at: datetime | None = None,
) -> None:
    """Store one completed inference. Skip single-mode and invalid identities."""
    if tenant_id is None or provider not in {
        "azure",
        "openai",
        "openrouter",
        "deepseek",
    }:
        return
    if not isinstance(inbound_model, str) or not inbound_model:
        return
    routed = routed_model if isinstance(routed_model, str) and routed_model else None
    database = current_app.extensions.get("database")
    if not isinstance(database, Database):
        return
    moment = occurred_at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("Inference activity timestamps must be timezone-aware")
    try:
        with database.sessions.begin() as session:
            session.add(
                InferenceActivityEvent(
                    tenant_id=tenant_id,
                    provider=provider,
                    profile_id=profile_id,
                    inbound_model=inbound_model,
                    routed_model=routed,
                    input_tokens=None if usage is None else usage.input_tokens,
                    output_tokens=None if usage is None else usage.output_tokens,
                    cached_tokens=None if usage is None else usage.cached_tokens,
                    reasoning_tokens=None if usage is None else usage.reasoning_tokens,
                    total_tokens=None if usage is None else usage.total_tokens,
                    occurred_at=moment,
                )
            )
    except SQLAlchemyError:
        current_app.logger.exception("Failed to record inference activity")
