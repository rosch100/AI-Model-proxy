"""Functional tests using WebTest.

See: http://webtest.readthedocs.org/
"""

import os

import pytest

from app.common import recording

from .replay_base import ReplyBase


class TestDecodeUtf8CompletePrefix:
    """Unit tests for mid-stream UTF-8 decoding used by SSE recording."""

    def test_keeps_complete_multibyte_text(self):
        """Decode complete UTF-8 including multi-byte characters."""
        assert recording.decode_utf8_complete_prefix("€-total".encode()) == "€-total"

    def test_drops_incomplete_trailing_multibyte_sequence(self):
        """Drop only a truncated trailing multi-byte sequence while streaming."""
        # Euro sign is e2 82 ac; leave the first byte only at the end.
        payload = b"prefix-" + "€".encode()[:1]
        assert recording.decode_utf8_complete_prefix(payload) == "prefix-"

    def test_rejects_invalid_utf8_in_the_middle(self):
        """Still raise when invalid UTF-8 appears before the buffer end."""
        with pytest.raises(UnicodeDecodeError):
            recording.decode_utf8_complete_prefix(b"ok\xffmore")

    def test_empty_incomplete_buffer_becomes_empty_string(self):
        """A buffer that is only an incomplete sequence decodes to empty."""
        assert recording.decode_utf8_complete_prefix("€".encode()[:1]) == ""


class TestRecordSseIncompleteUtf8:
    """record_sse must not retain incomplete/unredactable SSE tails."""

    def test_record_sse_omits_incomplete_event_without_delimiter(
        self, app, tmp_path, monkeypatch
    ):
        """An incomplete event with no blank-line delimiter is not written."""
        monkeypatch.setattr(recording, "RECORDINGS_DIR", str(tmp_path))
        monkeypatch.setattr(recording, "__LAST_RECORDING_INDEX", 1)
        app.config["RECORD_TRAFFIC"] = True

        incomplete = b'data: {"text":"secret-caf' + "é".encode()[:1]
        with app.app_context():
            recording.record_sse(incomplete, "upstream_response")

        assert not (tmp_path / "1" / "upstream_response.sse").exists()

    def test_record_sse_keeps_complete_events_and_drops_truncated_tail(
        self, app, tmp_path, monkeypatch
    ):
        """Only events closed by a blank line are kept; truncated tails are omitted."""
        monkeypatch.setattr(recording, "RECORDINGS_DIR", str(tmp_path))
        monkeypatch.setattr(recording, "__LAST_RECORDING_INDEX", 1)
        app.config["RECORD_TRAFFIC"] = True

        complete = b'data: {"text":"safe"}\n\n'
        truncated = b'data: {"text":"secret-leak'
        with app.app_context():
            recording.record_sse(complete + truncated, "upstream_response")

        recorded = (tmp_path / "1" / "upstream_response.sse").read_bytes()
        assert recorded == b'data: {"text":"REDACTED"}\n\n'
        assert b"secret-leak" not in recorded


class TestRecording(ReplyBase):
    """Test different scenarios with traffic recording enabled."""

    def modify_settings(self, app):
        """Enables traffic recording."""
        app.config["RECORD_TRAFFIC"] = True

    def test_multiple_requests(self, testapp, requests_mock, monkeypatch, tmp_path):
        """Test two consecutive requests."""
        monkeypatch.setattr(recording, "RECORDINGS_DIR", tmp_path)
        monkeypatch.setattr(recording, "__LAST_RECORDING_INDEX", -1)

        super().test(testapp, requests_mock)

        directories = os.listdir(tmp_path)
        assert len(directories) == 1, "First directory created"

        directory = directories[0]
        assert directory == "1"
        assert os.path.exists(
            os.path.join(tmp_path, directory, "upstream_request.json")
        )
        assert os.path.exists(
            os.path.join(tmp_path, directory, "upstream_response.sse")
        )
        assert os.path.exists(
            os.path.join(tmp_path, directory, "downstream_request.json")
        )
        assert os.path.exists(
            os.path.join(tmp_path, directory, "downstream_response.sse")
        )

        super().test(testapp, requests_mock)

        directories = os.listdir(tmp_path)
        assert len(directories) == 2, "Second directory created"

    def test_creates_folder(self, testapp, requests_mock, monkeypatch, tmp_path):
        """Test recordings folder is created."""
        recordings_path = os.path.join(tmp_path, "recordings")
        monkeypatch.setattr(recording, "RECORDINGS_DIR", recordings_path)
        monkeypatch.setattr(recording, "__LAST_RECORDING_INDEX", -1)

        assert not os.path.exists(recordings_path)

        super().test(testapp, requests_mock)

        assert os.path.exists(recordings_path)
        assert os.path.exists(os.path.join(recordings_path, "1"))

    def test_increments_index(self, testapp, requests_mock, monkeypatch, tmp_path):
        """Test that the index for the next recording is incremented, and ignores unrelated folders."""
        monkeypatch.setattr(recording, "RECORDINGS_DIR", tmp_path)
        monkeypatch.setattr(recording, "__LAST_RECORDING_INDEX", -1)

        # Last recording index 123
        os.makedirs(os.path.join(tmp_path, "123"))

        # Unrelated folder
        os.makedirs(os.path.join(tmp_path, "foo"))

        # Smaller index than last recording index
        os.makedirs(os.path.join(tmp_path, "-10"))

        super().test(testapp, requests_mock)

        assert os.path.exists(os.path.join(tmp_path, "124"))
