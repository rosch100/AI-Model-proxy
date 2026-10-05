"""Bounded, provider-neutral preflight before committing a client response."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from itertools import chain
from typing import Any
from urllib.parse import urlsplit

import requests
from flask import Response, current_app, g, has_request_context, jsonify
from urllib3.exceptions import MaxRetryError, NewConnectionError

from app.common.sse import SSEDecoder, SSEEvent
from app.persistence.inference_activity import (
    complete_provider_attempt,
    parse_provider_usage,
    record_inference_activity,
)
from app.providers.error_classification import (
    UpstreamErrorClassification,
    classify_upstream_error,
)

MAX_PREFLIGHT_BYTES = 65536
MAX_PREFLIGHT_EVENTS = 32
MAX_PREFLIGHT_SECONDS = 5.0
ROUTED_READ_TIMEOUT_SECONDS = 30.0
MAX_HTTP_ERROR_BODY_BYTES = 64 * 1024
RETRYABLE_STATUSES = frozenset({408, 429, *range(500, 600)})
_ERROR_CODE_STATUSES = {
    "rate_limit_exceeded": 429,
    "requests_rate_limit_exceeded": 429,
    "tokens_rate_limit_exceeded": 429,
    "insufficient_quota": 429,
    "credit_balance_exhausted": 429,
    "organization_spend_limit_exceeded": 429,
    "project_spend_limit_exceeded": 429,
    "organization_usage_limit_exceeded": 429,
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
    "payment_required": 402,
    "openrouter_key_limit": 402,
    "openrouter_in_flight_budget": 402,
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


_SAFE_PROVIDER_ERROR_CODES = frozenset(_ERROR_CODE_STATUSES)
_SAFE_PROVIDER_ERROR_TYPES = _SAFE_PROVIDER_ERROR_CODES | frozenset(
    {"provider_error", "error", "too_many_requests", "rate_limited"}
)
_SAFE_PROVIDER_METADATA_FIELDS = frozenset(
    {"provider_name", "is_byok", "is_free_tier", "billing_multiplier"}
)
_PROVIDER_RATE_LIMIT_HEADERS = frozenset(
    {
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
    }
)
_SAFE_PROVIDER_ERROR_PARAMS = frozenset(
    {
        "model",
        "messages",
        "input",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "tools",
        "tool_choice",
        "response_format",
        "reasoning_effort",
    }
)
_PROVIDER_PARAMETER_PATH = re.compile(
    r"(?:messages|input)\[\d{1,4}\]\.(?:content|role)\Z"
)
_PROVIDER_REQUEST_ID_PATTERNS = (
    re.compile(r"req_[0-9a-f]{24,64}\Z", re.IGNORECASE),
    re.compile(r"gen-[0-9a-f-]{32,36}\Z", re.IGNORECASE),
    re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z", re.IGNORECASE),
)
_OPENROUTER_LIMIT_SOURCES = frozenset(
    {"openrouter_key_limit", "openrouter_in_flight_budget"}
)
_PROVIDER_REQUEST_ID_HEADERS = (
    "x-request-id",
    "request-id",
    "openai-request-id",
    "x-openrouter-request-id",
    "apim-request-id",
    "x-ms-request-id",
)
_PROVIDER_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SAFE_PROVIDER_NAME_PATTERN = re.compile(r"[\w .()+/-]{1,80}\Z")
_SAFE_RATE_LIMIT_VALUE_PATTERN = re.compile(
    r"(?:[0-9]{1,13}(?:\.[0-9]{1,3})?|(?:[0-9]+(?:\.[0-9]+)?(?:ms|s|m|h|d))+|"
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.+-]{1,20}Z?)\Z"
)
_SENSITIVE_DIAGNOSTIC_PATTERN = re.compile(
    r"(?:sk[-_][a-z0-9_-]{8,}|github_pat_[a-z0-9_]{20,}|gh[pousr]_[a-z0-9]{20,}|"
    r"xox[baprs]-[a-z0-9-]{20,}|bearer\s|api[-_]?key|secret|token|password|"
    r"[a-f0-9]{32,})",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_SCHEMELESS_URL_PATTERN = re.compile(r"(?i)(?<![\w-])url:\s*.*\Z")
_BEARER_CREDENTIAL_PATTERN = re.compile(r"(?i)bearer\s+\S+")


def _safe_exception_message(exc: BaseException) -> str | None:
    """Retain a bounded transport reason while stripping URLs and credentials."""
    message = str(exc).replace("\r", " ").replace("\n", " ").strip()
    if not message:
        return None

    def strip_url(match: re.Match[str]) -> str:
        try:
            parsed = urlsplit(match.group(0).rstrip(".,);"))
            host = parsed.hostname
            port = parsed.port
        except ValueError:
            return "<upstream-url>"
        if not host:
            return "<upstream-url>"
        if port is not None:
            host = f"{host}:{port}"
        return f"{parsed.scheme}://{host}"

    message = _URL_PATTERN.sub(strip_url, message)
    message = _SCHEMELESS_URL_PATTERN.sub("url: <path>", message)
    message = _BEARER_CREDENTIAL_PATTERN.sub("Bearer <redacted>", message)
    if _SENSITIVE_DIAGNOSTIC_PATTERN.search(message):
        return "<redacted-sensitive-transport-detail>"
    return message[:512]


def _transport_error_details(
    exc: requests.RequestException,
    *,
    headers: Any = None,
    upstream_url: str | None = None,
) -> dict[str, Any]:
    """Capture safe exception-chain facts without logging arbitrary request data."""
    causes: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(causes) < 5:
        causes.append(type(current).__name__)
        current = current.__cause__ or current.__context__

    diagnostics: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception_chain": causes,
    }
    message = _safe_exception_message(exc)
    if message is not None:
        diagnostics["message"] = message
    for attribute in ("errno", "winerror"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            diagnostics[attribute] = value
    request = getattr(exc, "request", None)
    request_url = getattr(request, "url", None)
    if not isinstance(request_url, str):
        request_url = upstream_url
    if isinstance(request_url, str):
        try:
            hostname = urlsplit(request_url).hostname
        except ValueError:
            hostname = None
        if hostname:
            diagnostics["upstream_host"] = hostname[:253]
    if isinstance(headers, Mapping):
        safe_header_diagnostics = _provider_error_details({}, headers)[
            "provider_diagnostics"
        ]
        diagnostics.update(safe_header_diagnostics)
    return diagnostics


def _recognized_provider_request_id(headers: Mapping[str, Any]) -> str | None:
    """Retain a provider correlation ID only when it matches a known format."""
    for header in _PROVIDER_REQUEST_ID_HEADERS:
        value = headers.get(header)
        if isinstance(value, str) and any(
            pattern.fullmatch(value.strip())
            for pattern in _PROVIDER_REQUEST_ID_PATTERNS
        ):
            return value.strip()
    return None


def _recognized_error_value(
    value: object, allowed_values: frozenset[str]
) -> str | None:
    """Retain known error identifiers or safe, bounded provider codes."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if normalized in allowed_values:
        return normalized
    if _PROVIDER_ERROR_CODE_PATTERN.fullmatch(
        normalized
    ) and not _SENSITIVE_DIAGNOSTIC_PATTERN.search(normalized):
        return normalized
    return None


def _recognized_error_parameter(value: object) -> str | None:
    """Retain known field names and bounded indexed message/input paths."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if normalized in _SAFE_PROVIDER_ERROR_PARAMS:
        return normalized
    return normalized if _PROVIDER_PARAMETER_PATH.fullmatch(normalized) else None


def _safe_metadata_value(key: str, value: object) -> str | bool | int | float | None:
    """Keep only bounded documented scalar metadata that is safe for logs."""
    if key in {"is_byok", "is_free_tier"}:
        return value if isinstance(value, bool) else None
    if key == "billing_multiplier":
        if isinstance(value, int) and not isinstance(value, bool):
            return value if 0 <= value <= 100 else None
        if isinstance(value, float):
            return value if math.isfinite(value) and 0 <= value <= 100 else None
        return None
    if key == "provider_name":
        if (
            isinstance(value, str)
            and _SAFE_PROVIDER_NAME_PATTERN.fullmatch(value)
            and not _SENSITIVE_DIAGNOSTIC_PATTERN.search(value)
        ):
            return value
    return None


def _provider_error_details(error: dict[str, Any], headers: Any) -> dict[str, Any]:
    """Extract bounded provider diagnostics without raw message, body, or secrets."""
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    normalized_headers = (
        {str(key).casefold(): value for key, value in headers.items()}
        if isinstance(headers, Mapping)
        else {}
    )
    request_id = _recognized_provider_request_id(normalized_headers)
    limit_source = metadata.get("limit_source")
    safe_limit_source = (
        limit_source
        if isinstance(limit_source, str)
        and _PROVIDER_ERROR_CODE_PATTERN.fullmatch(limit_source)
        and not _SENSITIVE_DIAGNOSTIC_PATTERN.search(limit_source)
        else None
    )
    error_code = _recognized_error_value(error.get("code"), _SAFE_PROVIDER_ERROR_CODES)
    error_type = _recognized_error_value(error.get("type"), _SAFE_PROVIDER_ERROR_TYPES)
    error_param = _recognized_error_parameter(error.get("param"))
    provider_diagnostics = {
        key: value
        for key, value in {
            "provider_error_code": error_code,
            "provider_error_type": error_type,
            "provider_error_param": error_param,
            "provider_limit_source": safe_limit_source,
            "provider_request_id": request_id,
        }.items()
        if value is not None
    }
    for key in _SAFE_PROVIDER_METADATA_FIELDS:
        value = _safe_metadata_value(key, metadata.get(key))
        if value is not None:
            provider_diagnostics[key] = value

    retry_after = classify_upstream_error(
        "unknown", None, None, normalized_headers, None
    ).retry_after_seconds
    if retry_after is not None:
        provider_diagnostics["retry_after_seconds"] = retry_after

    rate_limit_headers = {
        key: value.strip()
        for key, value in normalized_headers.items()
        if key in _PROVIDER_RATE_LIMIT_HEADERS
        and isinstance(value, str)
        and len(value) <= 32
        and _SAFE_RATE_LIMIT_VALUE_PATTERN.fullmatch(value.strip())
    }
    if rate_limit_headers:
        provider_diagnostics["rate_limit_headers"] = rate_limit_headers
    return {
        "provider_error_code": error_code,
        "provider_error_type": error_type,
        "provider_error_param": error_param,
        "provider_limit_source": safe_limit_source,
        "provider_request_id": request_id,
        "provider_diagnostics": provider_diagnostics,
    }


@dataclass
class UpstreamError(Exception):
    """An explicit attempt failure, never a successful dummy response."""

    status: int
    code: str
    message: str
    retryable: bool
    classification: UpstreamErrorClassification = field(
        default_factory=lambda: UpstreamErrorClassification("unknown")
    )
    provider_error_code: str | None = None
    provider_error_type: str | None = None
    provider_error_param: str | None = None
    provider_limit_source: str | None = None
    provider_request_id: str | None = None
    provider_diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)

    def response(self) -> Response:
        """Return a safe JSON error before any SSE headers have been committed."""
        response = jsonify({"error": {"code": self.code, "message": self.message}})
        response.status_code = self.status
        retry_after = self.classification.retry_after_seconds
        if retry_after is not None:
            response.headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
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
        provider_error_type=type(exc).__name__,
        provider_diagnostics=_transport_error_details(exc),
    )


def _error_failure(
    error: dict[str, Any],
    status: int | None = None,
    *,
    provider: str | None = None,
    headers: Any = None,
    settings: Any = None,
) -> UpstreamError:
    diagnostics = _provider_error_details(error, headers)
    raw_code = str(error.get("code") or error.get("type") or "upstream_error")
    metadata = error.get("metadata")
    limit_source = metadata.get("limit_source") if isinstance(metadata, dict) else None
    recognized_openrouter_limit = provider == "openrouter" and limit_source in {
        "openrouter_key_limit",
        "openrouter_in_flight_budget",
    }
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
        elif recognized_openrouter_limit:
            status = 402
        else:
            status = _ERROR_CODE_STATUSES.get(code, 502)
            # Unknown structured errors are terminal, not guessed to be transient.
            if code not in _ERROR_CODE_STATUSES:
                return UpstreamError(
                    status,
                    "upstream_error",
                    "Provider returned a streaming error.",
                    False,
                    **diagnostics,
                )
    classification = classify_upstream_error(
        provider or "unknown", status, error, headers, settings
    )
    return UpstreamError(
        status,
        code,
        f"Provider returned an error (HTTP {status}).",
        status in RETRYABLE_STATUSES
        or status == 402
        or classification.category in {"quota_exhausted", "transient"},
        classification,
        **diagnostics,
    )


def _event_failure(
    event: SSEEvent,
    data: Any,
    *,
    provider: str | None = None,
    headers: Any = None,
    settings: Any = None,
) -> UpstreamError | None:
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
    ) or isinstance(error.get("metadata"), dict)
    if not has_details and error.get("type") not in _ERROR_CODE_STATUSES:
        return None
    return _error_failure(
        error,
        provider=provider,
        headers=headers,
        settings=settings,
    )


def sanitize_error_event(
    event: SSEEvent,
    *,
    provider: str | None = None,
    headers: Any = None,
    settings: Any = None,
) -> SSEEvent:
    """Retain the event shape but remove untrusted provider error details."""
    data = event.json
    failure = _event_failure(
        event,
        data,
        provider=provider,
        headers=headers,
        settings=settings,
    )
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


def _structured_http_error(upstream: Any) -> dict[str, Any]:
    """Read only a bounded JSON error object; ignore oversized or invalid bodies."""
    body = bytearray()
    try:
        for chunk in upstream.iter_content(chunk_size=4096):
            if not chunk:
                continue
            remaining = MAX_HTTP_ERROR_BODY_BYTES + 1 - len(body)
            body.extend(chunk[:remaining])
            if len(body) > MAX_HTTP_ERROR_BODY_BYTES:
                return {}
    except requests.RequestException:
        return {}
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    error = payload.get("error", payload)
    return error if isinstance(error, dict) else {}


def prepare_upstream(
    upstream: Any,
    *,
    provider: str | None = None,
    settings: Any = None,
) -> PreparedUpstream:
    """Inspect a bounded SSE prefix synchronously before Flask returns headers."""
    if upstream.status_code != 200:
        try:
            failure = _error_failure(
                _structured_http_error(upstream),
                upstream.status_code,
                provider=provider,
                headers=upstream.headers,
                settings=settings,
            )
            raise failure
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
                failure = _event_failure(
                    event,
                    data,
                    provider=provider,
                    headers=upstream.headers,
                    settings=settings,
                )
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
    attempt_id: int | None = None,
    provider: str | None = None,
    settings: Any = None,
    circuit_attempt: Any = None,
) -> Iterator[bytes]:
    """Restore logical model identity while preserving choices and tool-call IDs."""
    decoder = SSEDecoder()
    usage = None
    completed_successfully = False
    status_code = None
    outcome = "failure"
    try:
        for chunk in upstream.iter_content():
            # Clear the recording buffer: this decoder is not a traffic recorder.
            decoder.full_buffer = b""
            for event in decoder.feed(chunk):
                if not event.data:
                    continue
                if event.data == "[DONE]":
                    completed_successfully = status_code is None
                    outcome = "success" if completed_successfully else "failure"
                    yield b"data: [DONE]\n\n"
                    continue
                try:
                    original_data = event.json
                    failure = _event_failure(
                        event,
                        original_data,
                        provider=provider,
                        headers=upstream.headers,
                        settings=settings,
                    )
                    data = sanitize_error_event(
                        event,
                        provider=provider,
                        headers=upstream.headers,
                        settings=settings,
                    ).json
                except ValueError:
                    status_code = 502
                    yield (
                        b'data: {"error":{"code":"invalid_upstream_event",'
                        b'"message":"Provider sent invalid SSE JSON."}}\n\n'
                    )
                    return
                if failure is not None:
                    status_code = failure.status
                    if circuit_attempt is not None:
                        circuit_attempt.failed(failure.classification)
                    if has_request_context():
                        current_app.logger.warning(
                            "Provider stream failed: request_id=%s provider=%s "
                            "status=%s error_code=%s error_category=%s "
                            "provider_error_code=%s provider_error_type=%s "
                            "provider_error_param=%s provider_limit_source=%s "
                            "provider_request_id=%s provider_diagnostics=%s",
                            getattr(g, "proxy_request_id", "unavailable"),
                            provider or "unknown",
                            failure.status,
                            failure.code,
                            failure.classification.category,
                            failure.provider_error_code or "unknown",
                            failure.provider_error_type or "unknown",
                            failure.provider_error_param or "unknown",
                            failure.provider_limit_source or "unknown",
                            failure.provider_request_id or "unknown",
                            json.dumps(
                                failure.provider_diagnostics,
                                sort_keys=True,
                                ensure_ascii=False,
                            ),
                        )
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
    except requests.RequestException as exc:
        outcome = "failure"
        if has_request_context():
            current_app.logger.warning(
                "Provider stream interrupted: request_id=%s provider=%s error_type=%s "
                "provider_diagnostics=%s",
                getattr(g, "proxy_request_id", "unavailable"),
                provider or "unknown",
                type(exc).__name__,
                json.dumps(
                    _transport_error_details(exc),
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            )
        yield (
            b'data: {"error":{"code":"stream_interrupted",'
            b'"message":"Provider stream interrupted; not replayed."}}\n\n'
        )
    except GeneratorExit:
        outcome = "aborted"
        raise
    finally:
        if attempt_id is not None:
            complete_provider_attempt(
                attempt_id,
                outcome=outcome,
                status_code=200 if completed_successfully else status_code,
            )
        if circuit_attempt is not None:
            circuit_attempt.release()
        if activity_tenant_id is not None and completed_successfully:
            record_inference_activity(
                tenant_id=activity_tenant_id,
                provider=activity_provider,
                profile_id=activity_profile_id,
                inbound_model=inbound_model,
                routed_model=routed_model,
                usage=usage,
            )
        upstream.close()
