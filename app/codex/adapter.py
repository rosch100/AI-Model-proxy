"""Flask adapter for the Codex provider."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import requests
from flask import Request, Response, jsonify, stream_with_context

from .auth_state import AuthStateError, CodexAuthManager, CodexAuthStore
from .request_adapter import CursorRequestAdapter, UnsupportedCursorShape
from .response_adapter import adapt_responses_sse_to_chat_sse
from .settings import CodexSettings
from .upstream import build_upstream_headers, post_responses

logger = logging.getLogger(__name__)

_NOT_READY_ERROR = "Codex authentication is not ready"
_REQUEST_REJECTED_ERROR = "The Codex request could not be accepted"
_UPSTREAM_FAILED_ERROR = "Codex upstream request failed"
_AUTH_FAILED_ERROR = "Codex authentication failed"


class CodexAdapter:
    """Forward Flask requests to the ChatGPT Codex backend."""

    def __init__(self, settings: CodexSettings | None = None) -> None:
        """Initialize the adapter with current Flask-backed settings."""
        self.settings = settings or CodexSettings()

    def ready(self) -> Response:
        """Return Codex readiness based on local auth state."""
        try:
            CodexAuthStore(self.settings.codex_auth_path).validate_ready()
        except AuthStateError:
            logger.exception("Codex auth state is not ready")
            return jsonify({"status": "not_ready", "error": _NOT_READY_ERROR}), 503
        return jsonify({"status": "ready"})

    def forward(self, req: Request, provider_path: str) -> Response:
        """Forward a request to Codex and adapt the response when needed."""
        payload = req.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": {"message": "Request body must be JSON"}}), 400

        downstream_headers = dict(req.headers)
        try:
            adapted = CursorRequestAdapter(self.settings).adapt(
                provider_path, payload, downstream_headers
            )
            auth_manager = CodexAuthManager(
                CodexAuthStore(self.settings.codex_auth_path),
                refresh_skew_seconds=self.settings.token_refresh_skew_seconds,
            )
            auth = auth_manager.current()
        except (UnsupportedCursorShape, AuthStateError):
            logger.exception("Rejected Codex request before contacting upstream")
            return jsonify({"error": {"message": _REQUEST_REJECTED_ERROR}}), 400

        headers = build_upstream_headers(
            self.settings,
            auth,
            session_id=adapted.session_id,
            thread_id=adapted.thread_id,
            downstream_headers=downstream_headers,
        )
        try:
            upstream_response = post_responses(
                self.settings.codex_responses_url,
                headers,
                adapted.body,
                timeout=self.settings.request_timeout_seconds,
            )
            if upstream_response.status_code == 401:
                upstream_response.close()
                auth = auth_manager.refresh(force=True)
                headers = build_upstream_headers(
                    self.settings,
                    auth,
                    session_id=adapted.session_id,
                    thread_id=adapted.thread_id,
                    downstream_headers=downstream_headers,
                )
                upstream_response = post_responses(
                    self.settings.codex_responses_url,
                    headers,
                    adapted.body,
                    timeout=self.settings.request_timeout_seconds,
                )
        except requests.RequestException:
            logger.exception("Codex upstream request failed")
            return jsonify({"error": {"message": _UPSTREAM_FAILED_ERROR}}), 502
        except AuthStateError:
            logger.exception("Codex authentication failed during an upstream request")
            return jsonify({"error": {"message": _AUTH_FAILED_ERROR}}), 400

        def stream_response():
            try:
                chunks = _response_chunks(upstream_response)
                if (
                    _wants_chat_completion_response(provider_path)
                    and upstream_response.status_code == 200
                ):
                    chunks = adapt_responses_sse_to_chat_sse(
                        chunks,
                        model=str(adapted.body.get("model", "")),
                        reasoning_display_mode=self.settings.reasoning_display_mode,
                    )
                yield from chunks
            finally:
                close = getattr(upstream_response, "close", None)
                if callable(close):
                    close()

        return Response(
            stream_with_context(stream_response()),
            status=upstream_response.status_code,
            headers={
                "content-type": upstream_response.headers.get(
                    "content-type", "text/event-stream"
                ),
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            },
        )


def _wants_chat_completion_response(path: str) -> bool:
    clean_path = path.strip("/")
    return clean_path == "chat/completions" or clean_path.endswith("/chat/completions")


def _response_chunks(response) -> Iterator[bytes]:
    iter_content = getattr(response, "iter_content", None)
    if callable(iter_content):
        yield from iter_content(chunk_size=8192)
        return

    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        yield content
    elif isinstance(content, str):
        yield content.encode()
