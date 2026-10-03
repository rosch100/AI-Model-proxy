"""Azure adapter orchestrating request/response transformations."""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from flask import Request, Response

from ..common.logging import console
from ..common.recording import record_payload
from ..providers.failover_upstream import (
    ROUTED_READ_TIMEOUT_SECONDS,
    prepare_upstream,
    transport_failure,
)
from ..tenants import DatabaseTenantSnapshot

# Local adapters
from .request_adapter import RequestAdapter
from .response_adapter import ResponseAdapter

MAX_AZURE_RETRY_DELAY_SECONDS = 60.0
MIN_AZURE_RATE_LIMIT_DELAY_SECONDS = 15.0

# Shared per-deployment cooldown so concurrent Cursor streams do not retry
# into an already exhausted Azure token window.
_rate_limit_not_before: dict[str, float] = {}


@dataclass
class AzureRequestContext:
    """Share request parameters and retry state across HTTP and SSE handling."""

    request_kwargs: dict[str, Any]
    retries_used: int = 0
    # Monotonic deadline this request has already waited through, so a retry
    # sleep does not pay the same shared cooldown twice.
    satisfied_cooldown_until: float = 0.0


class AzureAdapter:
    """Orchestrate forwarding of a Flask Request to Azure's Responses API.

    Provides a Completions-compatible interface to the caller by composing a
    RequestAdapter (pre-request transformations) and a ResponseAdapter
    (post-request transformations). The adapters receive a reference to this
    instance for shared per-request state (models).
    """

    # Per-request state (streaming completions only)
    inbound_model: Optional[str] = None
    include_usage: bool = False

    def __init__(self) -> None:
        """Initialize child adapters and shared state references."""
        # Composition: child adapters get a reference to this orchestrator
        self.request_adapter = RequestAdapter(self)
        self.response_adapter = ResponseAdapter(self)

    # Public API
    def forward(
        self, req: Request, snapshot: DatabaseTenantSnapshot | None = None
    ) -> Response:
        """Forward the Flask request upstream and adapt the response back.

        High-level flow:
        1) RequestAdapter builds the upstream request kwargs and stores state
           on this adapter (models).
        2) Perform the upstream HTTP call using a short-lived requests call.
        3) ResponseAdapter converts the upstream response into a Flask Response.
        """
        request_kwargs = self.request_adapter.adapt(req, snapshot)

        try:
            record_payload(request_kwargs.get("json", {}), "upstream_request")
        except OSError as exc:
            console.print(f"[yellow]Recording failed (non-fatal): {exc}[/yellow]")

        request_context = AzureRequestContext(request_kwargs)
        resp = self._request_upstream(request_context)
        if resp.status_code != 200:
            return self._handle_azure_error(resp, request_kwargs)

        return self.response_adapter.adapt(resp, request_context)

    def forward_attempt(
        self, req: Request, snapshot: DatabaseTenantSnapshot
    ) -> Response:
        """Make one routed attempt; do not wait or replay within this provider."""
        request_kwargs = self.request_adapter.adapt(req, snapshot)
        request_kwargs["timeout"] = (10.0, ROUTED_READ_TIMEOUT_SECONDS)
        try:
            upstream = requests.request(**request_kwargs)
        except requests.RequestException as exc:
            raise transport_failure(exc) from exc
        prepared = prepare_upstream(upstream)
        # No AzureRequestContext: HTTP/SSE retries belong to the outer router.
        response = self.response_adapter.adapt(prepared)
        response.call_on_close(prepared.close)
        return response

    def _request_upstream(
        self, request_context: AzureRequestContext
    ) -> requests.Response:
        """Retry transient Azure rate limits before returning a response stream."""
        while True:
            self._wait_for_shared_cooldown(request_context)
            response = requests.request(**request_context.request_kwargs)

            retry_delay = self._azure_rate_limit_retry_delay(
                response, request_context.retries_used
            )
            if retry_delay is None:
                return response

            response.close()
            self._wait_before_retry(request_context, retry_delay)

    def _retry_stream_rate_limit(
        self,
        response: requests.Response,
        request_context: AzureRequestContext,
        event_data: Any = None,
    ) -> requests.Response | None:
        """Retry a streamed rate limit using the shared backoff state."""
        retry_delay = self._retry_delay_from_headers(
            self._merged_retry_headers(response.headers, event_data),
            request_context.retries_used,
        )
        if retry_delay is None:
            return None

        response.close()
        self._wait_before_retry(request_context, retry_delay)
        return self._request_upstream(request_context)

    @staticmethod
    def _wait_before_retry(
        request_context: AzureRequestContext, retry_delay: float
    ) -> None:
        retry_number = request_context.retries_used + 1
        console.print(
            f"[yellow]Azure rate limit; retry {retry_number} "
            f"in {retry_delay:.2f}s[/yellow]"
        )
        cooldown_until = time.monotonic() + retry_delay
        AzureAdapter._record_shared_cooldown_until(request_context, cooldown_until)
        time.sleep(retry_delay)
        request_context.satisfied_cooldown_until = max(
            request_context.satisfied_cooldown_until, cooldown_until
        )
        request_context.retries_used += 1

    @staticmethod
    def _azure_rate_limit_retry_delay(
        response: requests.Response, retry_number: int
    ) -> float | None:
        """Return a bounded delay only for Azure's retryable rate-limit errors."""
        if response.status_code != 429:
            return None

        try:
            error = response.json().get("error")
        except (AttributeError, ValueError):
            return None
        if not isinstance(error, dict) or error.get("code") != "rate_limit_exceeded":
            return None

        return AzureAdapter._retry_delay_from_headers(response.headers, retry_number)

    @staticmethod
    def _merged_retry_headers(headers, event_data: Any) -> dict[str, str]:
        """Prefer HTTP retry hints, then retry-after fields on the SSE error."""
        merged = {str(key).lower(): str(value) for key, value in headers.items()}
        for key, value in AzureAdapter._retry_hints_from_event(event_data).items():
            merged.setdefault(key, value)
        return merged

    @staticmethod
    def _retry_hints_from_event(event_data: Any) -> dict[str, str]:
        if not isinstance(event_data, dict):
            return {}
        response = event_data.get("response")
        response_error = response.get("error") if isinstance(response, dict) else None
        nested_error = event_data.get("error")
        sources = [event_data]
        if isinstance(response_error, dict):
            sources.append(response_error)
        if isinstance(nested_error, dict):
            sources.append(nested_error)
        hints: dict[str, str] = {}
        for source in sources:
            for key in (
                "retry-after-ms",
                "retry_after_ms",
                "retry-after",
                "retry_after",
            ):
                value = source.get(key)
                if value is None:
                    continue
                header = (
                    "retry-after-ms"
                    if key.replace("_", "-").endswith("-ms")
                    else "retry-after"
                )
                hints.setdefault(header, str(value))
        return hints

    @staticmethod
    def _rate_limit_key(request_context: AzureRequestContext) -> str:
        body = request_context.request_kwargs.get("json") or {}
        url = request_context.request_kwargs.get("url") or ""
        model = body.get("model") or ""
        return f"{url}|{model}"

    @staticmethod
    def _wait_for_shared_cooldown(request_context: AzureRequestContext) -> None:
        key = AzureAdapter._rate_limit_key(request_context)
        not_before = _rate_limit_not_before.get(key, 0.0)
        if not_before <= request_context.satisfied_cooldown_until:
            return
        remaining = not_before - time.monotonic()
        if remaining <= 0:
            request_context.satisfied_cooldown_until = max(
                request_context.satisfied_cooldown_until, time.monotonic()
            )
            return
        console.print(
            f"[yellow]Azure rate limit cooldown {remaining:.2f}s "
            "before next upstream attempt[/yellow]"
        )
        time.sleep(remaining)
        request_context.satisfied_cooldown_until = max(
            request_context.satisfied_cooldown_until, not_before
        )

    @staticmethod
    def _record_shared_cooldown_until(
        request_context: AzureRequestContext, not_before: float
    ) -> None:
        key = AzureAdapter._rate_limit_key(request_context)
        previous = _rate_limit_not_before.get(key, 0.0)
        if not_before > previous:
            _rate_limit_not_before[key] = not_before

    @staticmethod
    def note_empty_stream_error_precursor(
        request_context: AzureRequestContext,
    ) -> None:
        """Protect peers when Azure emits an empty error before response.failed."""
        AzureAdapter._record_shared_cooldown_until(
            request_context, time.monotonic() + MIN_AZURE_RATE_LIMIT_DELAY_SECONDS
        )

    @staticmethod
    def note_stream_rate_limit(
        request_context: AzureRequestContext,
        headers,
        event_data: Any,
    ) -> None:
        """Share a streamed Azure rate-limit cooldown without retrying this response."""
        retry_delay = AzureAdapter._retry_delay_from_headers(
            AzureAdapter._merged_retry_headers(headers, event_data),
            request_context.retries_used,
        )
        if retry_delay is not None:
            AzureAdapter._record_shared_cooldown_until(
                request_context, time.monotonic() + retry_delay
            )

    @staticmethod
    def _backoff_ceiling(retry_number: int) -> float:
        max_exponent = math.ceil(
            math.log2(
                MAX_AZURE_RETRY_DELAY_SECONDS / MIN_AZURE_RATE_LIMIT_DELAY_SECONDS
            )
        )
        return min(
            MIN_AZURE_RATE_LIMIT_DELAY_SECONDS * (2 ** min(retry_number, max_exponent)),
            MAX_AZURE_RETRY_DELAY_SECONDS,
        )

    @staticmethod
    def _azure_hint_delay_seconds(headers) -> float | None:
        """Parse Azure retry-after hints; ignore unusable values."""
        for header, milliseconds_per_unit in (
            ("retry-after-ms", 0.001),
            ("retry-after", 1.0),
        ):
            header_value = headers.get(header)
            if header_value is None and hasattr(headers, "get"):
                header_value = headers.get(header.title()) or headers.get(
                    header.upper()
                )
            if header_value is None:
                continue
            try:
                delay = float(header_value) * milliseconds_per_unit
            except (TypeError, ValueError):
                continue
            if not math.isfinite(delay) or delay < 0:
                return None
            return delay
        return None

    @staticmethod
    def _retry_delay_from_headers(headers, retry_number: int) -> float | None:
        """Floor short Azure hints at 15s so TPM windows can refill under load."""
        azure_delay = AzureAdapter._azure_hint_delay_seconds(headers)
        floor = MIN_AZURE_RATE_LIMIT_DELAY_SECONDS
        if azure_delay is not None:
            return min(max(azure_delay, floor), MAX_AZURE_RETRY_DELAY_SECONDS)

        backoff_ceiling = AzureAdapter._backoff_ceiling(retry_number)
        return random.uniform(max(floor, backoff_ceiling / 2.0), backoff_ceiling)

    def _handle_azure_error(self, resp: Response, request_kwargs) -> Response:

        try:
            resp_content = resp.json()
        except ValueError:
            resp_content = resp.text

        body = request_kwargs.get("json") or {}
        instructions = body.get("instructions") or "no instructions"
        body["instructions"] = instructions[:16] + "..."
        tools = body.get("tools")
        body["tools"] = f"...redacted {len(tools) if tools else 0} tools..."
        inp = body.get("input")
        body["input"] = f"...redacted {len(inp) if inp else 0} input items..."
        pck = body.get("prompt_cache_key") or "no prompt_cache_key"
        body["prompt_cache_key"] = re.sub(
            r"(...)(.*)(...)",
            "\\1***\\3",
            pck if isinstance(pck, str) else str(pck),
        )
        report = {
            "endpoint": re.sub(
                r"(//.)(.*?)(.\.)", "\\1***\\3", request_kwargs.get("url")
            ),
            "azure_status_code": resp.status_code,
            "azure_response": resp_content,
            "request_body": body,
        }
        # Precompute pretty JSON to avoid backslashes inside f-string expressions
        report_pretty = json.dumps(report, indent=4).replace("\n", "\n\t")
        error_message = (
            '\nCheck "azure_response" for the error details:\n'
            f"\t{report_pretty}\n"
            "If the issue persists, report it with the details above."
        )
        console.rule(f"[red]Request failed with status code {resp.status_code}[/red]")
        from rich.markup import escape as rich_escape

        console.print(rich_escape(error_message))
        return Response(
            error_message,
            status=resp.status_code if resp.status_code != 401 else 400,
        )
