"""Hard preview deadline despite upstream reads that stay busy or blocked."""

import time
from threading import Event
from unittest.mock import Mock

from app.providers import failover_upstream


def test_preview_releases_headers_at_deadline_and_replays_pending_read(monkeypatch):
    """A blocked prefix read cannot keep Flask inside the failover window forever."""
    monkeypatch.setattr(failover_upstream, "MAX_PREFLIGHT_SECONDS", 0.03)
    release = Event()
    content = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'

    def chunks():
        release.wait(timeout=2)
        yield content

    raw = Mock(status_code=200)
    raw.headers = {"Content-Type": "text/event-stream"}
    raw.iter_content.return_value = chunks()
    started = time.monotonic()
    try:
        prepared = failover_upstream.prepare_upstream(raw)
        assert time.monotonic() - started < 0.5
        release.set()
        assert b"".join(prepared.iter_content()) == content
    finally:
        release.set()
        raw.close()


def test_error_received_after_preview_deadline_is_not_retried(monkeypatch):
    """Once headers can be released, a pending error stays in the same stream."""
    monkeypatch.setattr(failover_upstream, "MAX_PREFLIGHT_SECONDS", 0.03)
    release = Event()
    content = b'data: {"error":{"code":"rate_limit_exceeded"}}\n\n'

    def chunks():
        release.wait(timeout=2)
        yield content

    raw = Mock(status_code=200)
    raw.iter_content.return_value = chunks()
    try:
        prepared = failover_upstream.prepare_upstream(raw)
        release.set()
        assert b"".join(prepared.iter_content()) == content
        raw.close.assert_not_called()
    finally:
        release.set()
        raw.close()
