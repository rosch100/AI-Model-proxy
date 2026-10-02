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
from ..tenants import DatabaseTenantSnapshot

# Local adapters
from .request_adapter import RequestAdapter
from .response_adapter import ResponseAdapter

MAX_AZURE_RATE_LIMIT_RETRIES = 5
MAX_AZURE_RETRY_DELAY_SECONDS = 60.0


@dataclass
class AzureRequestContext:
    """Share request parameters and retry budget across HTTP and SSE handling."""

    request_kwargs: dict[str, Any]
    retries_used: int = 0


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

    def _request_upstream(
        self, request_context: AzureRequestContext
    ) -> requests.Response:
        """Retry transient Azure rate limits before returning a response stream."""
        while True:
            response = requests.request(**request_context.request_kwargs)
            if request_context.retries_used >= MAX_AZURE_RATE_LIMIT_RETRIES:
                return response

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
    ) -> requests.Response | None:
        """Retry a streamed rate limit while sharing the HTTP retry budget."""
        if request_context.retries_used >= MAX_AZURE_RATE_LIMIT_RETRIES:
            return None

        retry_delay = self._retry_delay_from_headers(
            response.headers, request_context.retries_used
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
            f"[yellow]Azure rate limit; retry {retry_number}/"
            f"{MAX_AZURE_RATE_LIMIT_RETRIES} in {retry_delay:.2f}s[/yellow]"
        )
        time.sleep(retry_delay)
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
    def _retry_delay_from_headers(headers, retry_number: int) -> float | None:
        """Respect Azure's retry delay without retrying early when it exceeds the limit."""
        for header, milliseconds_per_unit in (
            ("retry-after-ms", 0.001),
            ("retry-after", 1.0),
        ):
            header_value = headers.get(header)
            if header_value is None:
                continue
            try:
                delay = float(header_value) * milliseconds_per_unit
            except ValueError:
                continue
            if not math.isfinite(delay) or delay < 0:
                return None
            if delay > MAX_AZURE_RETRY_DELAY_SECONDS:
                return None
            return delay

        backoff_ceiling = min(2**retry_number, MAX_AZURE_RETRY_DELAY_SECONDS)
        return random.uniform(0.0, backoff_ceiling)

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
