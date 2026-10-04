"""Tie persistent breaker permits to one provider profile attempt."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from app.persistence.provider_circuit_breaker import (
    BreakerScope,
    CircuitPermit,
    ProviderCircuitBreakerStore,
)
from app.providers.error_classification import UpstreamErrorClassification


@dataclass
class ProviderCircuitAttempt:
    """Resolve a routed profile's leases and report structured upstream outcomes."""

    store: ProviderCircuitBreakerStore
    permit: CircuitPermit
    tenant_id: str
    provider: str
    profile_id: str
    settings: Mapping[str, Any]

    def preflight_succeeded(self) -> None:
        """Close all due breakers after a valid response passes bounded preflight."""
        if self.permit.leases:
            self.store.resolve_probe(self.permit, category="success")
            self.permit = CircuitPermit(True)

    def failed(
        self,
        classification: UpstreamErrorClassification,
    ) -> None:
        """Open quota scope or clear an expired breaker after another probe error."""
        if classification.category == "quota_exhausted":
            scope = self._reported_scope(classification)
            matching = tuple(
                lease for lease in self.permit.leases if lease.scope == scope
            )
            if matching:
                self.store.resolve_probe(
                    CircuitPermit(False, matching),
                    category="quota_exhausted",
                    retry_after_seconds=classification.retry_after_seconds,
                )
                self.store.release(
                    CircuitPermit(
                        False,
                        tuple(
                            lease
                            for lease in self.permit.leases
                            if lease not in matching
                        ),
                    )
                )
            else:
                self.store.open_quota(
                    scope, retry_after_seconds=classification.retry_after_seconds
                )
                self.store.release(self.permit)
        elif self.permit.leases:
            self.store.resolve_probe(self.permit, category=classification.category)
        self.permit = CircuitPermit(True)

    def release(self) -> None:
        """Release any unfinished probe lease after an interrupted request."""
        if self.permit.leases:
            self.store.release(self.permit)
            self.permit = CircuitPermit(True)

    def _reported_scope(
        self, classification: UpstreamErrorClassification
    ) -> BreakerScope:
        if classification.scope_type is None or classification.scope_id is None:
            scope_type, scope_id = "profile", self.profile_id
        else:
            scope_type = classification.scope_type
            scope_id = classification.scope_id
        return self.store.scope(self.tenant_id, self.provider, scope_type, scope_id)


def retry_after_header(retry_at: datetime, *, now: datetime | None = None) -> str:
    """Render an absolute retry time as delta-seconds without rounding early."""
    now = now or datetime.now(timezone.utc)
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return str(max(1, math.ceil((retry_at - now).total_seconds())))
