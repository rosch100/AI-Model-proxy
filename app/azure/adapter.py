"""Azure adapter orchestrating request/response transformations."""

from __future__ import annotations

import json
import math
import random
import re
import time
from typing import Optional

import requests
from flask import Request, Response

from ..common.logging import console
from ..common.recording import record_payload

# Local adapters
from .request_adapter import RequestAdapter
from .response_adapter import ResponseAdapter

MAX_AZURE_RATE_LIMIT_RETRIES = 2
MAX_AZURE_RETRY_DELAY_SECONDS = 30.0


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
    def forward(self, req: Request) -> Response:
        """Forward the Flask request upstream and adapt the response back.

        High-level flow:
        1) RequestAdapter builds the upstream request kwargs and stores state
           on this adapter (models).
        2) Perform the upstream HTTP call using a short-lived requests call.
        3) ResponseAdapter converts the upstream response into a Flask Response.
        """
        request_kwargs = self.request_adapter.adapt(req)

        try:
            record_payload(request_kwargs.get("json", {}), "upstream_request")
        except OSError as exc:
            console.print(f"[yellow]Recording failed (non-fatal): {exc}[/yellow]")

        resp = self._request_upstream(request_kwargs)
        if resp.status_code != 200:
            return self._handle_azure_error(resp, request_kwargs)

        return self.response_adapter.adapt(resp)

    def _request_upstream(self, request_kwargs) -> requests.Response:
        """Retry transient Azure rate limits before returning a response stream."""
        retries = 0
        while True:
            response = requests.request(**request_kwargs)
            if retries >= MAX_AZURE_RATE_LIMIT_RETRIES:
                return response

            retry_delay = self._azure_rate_limit_retry_delay(response, retries)
            if retry_delay is None:
                return response

            response.close()
            console.print(
                f"[yellow]Azure rate limit; retry {retries + 1}/"
                f"{MAX_AZURE_RATE_LIMIT_RETRIES} in {retry_delay:.2f}s[/yellow]"
            )
            time.sleep(retry_delay)
            retries += 1

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

        for header, milliseconds_per_unit in (
            ("retry-after-ms", 0.001),
            ("retry-after", 1.0),
        ):
            header_value = response.headers.get(header)
            if header_value is None:
                continue
            try:
                delay = float(header_value) * milliseconds_per_unit
            except ValueError:
                continue
            if (
                not math.isfinite(delay)
                or delay < 0
                or delay > MAX_AZURE_RETRY_DELAY_SECONDS
            ):
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
