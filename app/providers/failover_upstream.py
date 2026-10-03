"""Bounded, provider-neutral preflight before committing a client response."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import requests
from flask import Response, jsonify
from urllib3.exceptions import MaxRetryError, NewConnectionError

from app.common.sse import SSEDecoder, SSEEvent
from app.persistence.inference_activity import (
    parse_provider_usage,
    record_inference_activity,
)

MAX_PREFLIGHT_BYTES = 65536
MAX_PREFLIGHT_EVENTS = 32
MAX_PREFLIGHT_SECONDS = 5.0
ROUTED_READ_TIMEOUT_SECONDS = 30.0
RETRYABLE_STATUSES = frozenset({408, 429, *range(500, 600)})
_ERROR_CODE_STATUSES = {
    "rate_limit_exceeded": 429,
    "insufficient_quota": 429,
    "quota_exceeded": 429,
    "server_error": 500,
    "internal_error": 500,
    "internal_server_error": 500,
    "overloaded_error": 503,
    "api_error": 500,
    "invalid_api_key": 401,
    "authentication_error": 401,
    "permission_error": 403,
    "invalid_request_error": 400,
    "model_not_found": 404,
}
_END_OF_STREAM = object()


class _PreviewReader:
    """Own at most one pending socket read while bounding preflight wall time."""

    def __init__(self, iterator: Iterator[bytes]) -> None:
        self.iterator = iterator
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="provider-preview"
        )
        self.pending: Future | None = None

    def next(self, remaining: float) -> bytes | object:
        """Wait only until the preview deadline; retain a late read for streaming."""
        self.pending = self.executor.submit(next, self.iterator, _END_OF_STREAM)
        value = self.pending.result(timeout=max(0.0, remaining))
        self.pending = None
        return value

    def rest(self) -> Iterator[bytes]:
        """Hand the pending read back to the same stream, then read synchronously."""
        try:
            if self.pending is not None:
                value = self.pending.result()
                self.pending = None
                if value is _END_OF_STREAM:
                    return
                yield value
            self.close()
            yield from self.iterator
        finally:
            self.close()

    def close(self) -> None:
        """Stop scheduling reads; a blocked read is bounded by its socket timeout."""
        self.executor.shutdown(wait=False, cancel_futures=True)


_LIFECYCLE_EVENTS = frozenset(
    {"response.created", "response.in_progress", "response.queued"}
)


@dataclass
class UpstreamError(Exception):
    """An explicit attempt failure, never a successful dummy response."""

    status: int
    code: str
    message: str
    retryable: bool

    def response(self) -> Response:
        """Return a JSON error before any SSE headers have been committed."""
        response = jsonify({"error": {"code": self.code, "message": self.message}})
        response.status_code = self.status
        return response


@dataclass
class PreparedUpstream:
    """Replay inspected bytes exactly once, then continue the same connection."""

    upstream: Any = field(repr=False)
    chunks: Iterator[bytes] = field(repr=False)
    reader: _PreviewReader = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def status_code(self) -> int:
        """Expose the upstream HTTP status to existing response adapters."""
        return self.upstream.status_code

    @property
    def headers(self) -> Any:
        """Expose upstream metadata without forwarding credential headers."""
        return self.upstream.headers

    def iter_content(self, chunk_size: int = 1024) -> Iterator[bytes]:
        """Consume the single prepared stream without reopening the request."""
        return self.chunks

    def close(self) -> None:
        """Release the original HTTP connection."""
        if not self._closed:
            self._closed = True
            self.reader.close()
            self.upstream.close()


def transport_failure(exc: requests.RequestException) -> UpstreamError:
    """Distinguish connection establishment from ambiguous post-send read failures."""
    reason = exc.args[0] if exc.args else None
    if isinstance(reason, MaxRetryError):
        reason = reason.reason
    safe_connection_failure = isinstance(exc, requests.ConnectTimeout) or (
        isinstance(exc, requests.ConnectionError)
        and isinstance(reason, NewConnectionError)
    )
    return UpstreamError(
        502,
        "upstream_connection_failed",
        "Provider connection failed. The request was not replayed if its outcome was uncertain.",
        safe_connection_failure,
    )


def _error_failure(error: dict[str, Any], status: int | None = None) -> UpstreamError:
    raw_code = str(error.get("code") or error.get("type") or "upstream_error")
    code = raw_code if raw_code in _ERROR_CODE_STATUSES else "upstream_error"
    if status is None:
        raw_status = (
            error.get("status") or error.get("status_code") or error.get("code")
        )
        if isinstance(raw_status, int) and 400 <= raw_status < 600:
            status = raw_status
        elif (
            isinstance(raw_status, str)
            and raw_status.isdecimal()
            and 400 <= int(raw_status) < 600
        ):
            status = int(raw_status)
        else:
            status = _ERROR_CODE_STATUSES.get(code, 502)
            # Unknown structured errors are terminal, not guessed to be transient.
            if code not in _ERROR_CODE_STATUSES:
                return UpstreamError(
                    status, code, "Provider returned a streaming error.", False
                )
    # Never reflect upstream bodies/URLs/credentials into a client error.
    return UpstreamError(
        status,
        code,
        f"Provider returned an error (HTTP {status}).",
        status in RETRYABLE_STATUSES,
    )


def _event_failure(event: SSEEvent, data: Any) -> UpstreamError | None:
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        response = data.get("response")
        error = response.get("error") if isinstance(response, dict) else None
    if not isinstance(error, dict) and (
        event.event == "error" or data.get("type") == "error"
    ):
        error = data
    if not isinstance(error, dict):
        return None
    has_details = any(
        error.get(key) for key in ("code", "message", "status", "status_code")
    )
    if not has_details and error.get("type") not in _ERROR_CODE_STATUSES:
        return None
    return _error_failure(error)


def sanitize_error_event(event: SSEEvent) -> SSEEvent:
    """Retain the event shape but remove untrusted provider error details."""
    data = event.json
    failure = _event_failure(event, data)
    if failure is None:
        return event
    safe_error = {"code": failure.code, "message": failure.message}
    if isinstance(data.get("error"), dict):
        data = {**data, "error": safe_error}
    elif isinstance(data.get("response"), dict):
        data = {**data, "response": {**data["response"], "error": safe_error}}
    else:
        data = {"type": data.get("type", "error"), **safe_error}
    return SSEEvent(event.event, json.dumps(data), event.id, event.retry, event.index)


def _has_output(event: SSEEvent, data: Any) -> bool:
    name = event.event or (data.get("type", "") if isinstance(data, dict) else "")
    if name in _LIFECYCLE_EVENTS or name in {
        "response.content_part.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.output_item.done",
        "response.content_part.done",
        "response.output_text.done",
        "response.reasoning_text.done",
        "response.function_call_arguments.done",
    }:
        return False
    if name == "error" and _event_failure(event, data) is None:
        return False
    if name == "response.output_item.added" and isinstance(data, dict):
        item = data.get("item")
        return isinstance(item, dict) and item.get("type") in {
            "function_call",
            "custom_tool_call",
            "apply_patch_call",
            "shell_call",
            "local_shell_call",
            "mcp_call",
            "computer_call",
        }
    if not isinstance(data, dict):
        return bool(event.data)
    choices = data.get("choices")
    if isinstance(choices, list):
        return any(
            isinstance(choice, dict)
            and (
                choice.get("finish_reason") is not None
                or any(
                    value
                    for key, value in (choice.get("delta") or {}).items()
                    if key != "role"
                )
            )
            for choice in choices
        )
    return bool(name or data)


def prepare_upstream(upstream: Any) -> PreparedUpstream:
    """Inspect a bounded SSE prefix synchronously before Flask returns headers."""
    if upstream.status_code != 200:
        # HTTP status is sufficient. Reading the body could erase that status on
        # timeout and keep failover blocked on an arbitrarily large error payload.
        try:
            raise _error_failure({}, upstream.status_code)
        finally:
            upstream.close()
    iterator = iter(upstream.iter_content(chunk_size=1024))
    reader = _PreviewReader(iterator)
    buffered: list[bytes] = []
    size = 0
    events = 0
    decoder = SSEDecoder()
    deadline = time.monotonic() + MAX_PREFLIGHT_SECONDS
    try:
        while time.monotonic() < deadline:
            try:
                chunk = reader.next(deadline - time.monotonic())
            except FutureTimeoutError:
                break
            if chunk is _END_OF_STREAM:
                break
            if not chunk:
                continue
            buffered.append(chunk)
            size += len(chunk)
            if size >= MAX_PREFLIGHT_BYTES:
                break
            for event in decoder.feed(chunk):
                events += 1
                try:
                    data = event.json
                except (ValueError, UnicodeError) as exc:
                    raise UpstreamError(
                        502,
                        "invalid_upstream_event",
                        "Provider sent invalid SSE JSON.",
                        False,
                    ) from exc
                failure = _event_failure(event, data)
                if failure is not None:
                    raise failure
                if _has_output(event, data) or events >= MAX_PREFLIGHT_EVENTS:
                    return PreparedUpstream(
                        upstream, chain(buffered, reader.rest()), reader
                    )
            if time.monotonic() >= deadline:
                break
    except UpstreamError:
        reader.close()
        upstream.close()
        raise
    except requests.RequestException as exc:
        reader.close()
        upstream.close()
        raise transport_failure(exc) from exc
    return PreparedUpstream(upstream, chain(buffered, reader.rest()), reader)


def chat_stream(
    upstream: PreparedUpstream,
    model: str,
    *,
    activity_tenant_id: str | None = None,
    activity_provider: str | None = None,
    activity_profile_id: str | None = None,
    inbound_model: object = None,
    routed_model: object = None,
) -> Iterator[bytes]:
    """Restore logical model identity while preserving choices and tool-call IDs."""
    decoder = SSEDecoder()
    usage = None
    try:
        for chunk in upstream.iter_content():
            # Clear the recording buffer: this decoder is not a traffic recorder.
            decoder.full_buffer = b""
            for event in decoder.feed(chunk):
                if not event.data:
                    continue
                if event.data == "[DONE]":
                    yield b"data: [DONE]\n\n"
                    continue
                try:
                    data = sanitize_error_event(event).json
                except ValueError:
                    yield (
                        b'data: {"error":{"code":"invalid_upstream_event",'
                        b'"message":"Provider sent invalid SSE JSON."}}\n\n'
                    )
                    return
                if isinstance(data, dict):
                    parsed_usage = parse_provider_usage(data.get("usage"))
                    if parsed_usage is not None:
                        usage = parsed_usage
                    if "model" in data:
                        data = {**data, "model": model}
                yield (
                    "data: "
                    + json.dumps(data, separators=(",", ":"), ensure_ascii=False)
                    + "\n\n"
                ).encode("utf-8")
    except requests.RequestException:
        yield (
            b'data: {"error":{"code":"stream_interrupted",'
            b'"message":"Provider stream interrupted; not replayed."}}\n\n'
        )
    finally:
        if activity_tenant_id is not None:
            record_inference_activity(
                tenant_id=activity_tenant_id,
                provider=activity_provider,
                profile_id=activity_profile_id,
                inbound_model=inbound_model,
                routed_model=routed_model,
                usage=usage,
            )
        upstream.close()
