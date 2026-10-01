"""Functional tests using WebTest.

See: http://webtest.readthedocs.org/
"""

import json

from app.models import SUPPORTED_MODELS_TEXT

from .replay_base import ReplyBase


class TestBadSummaryLevel(ReplyBase):
    """Test a single ping-pong interaction, no tool calls."""

    expected_upstream_request_body = None
    expected_downstream_status_code = 400
    expected_downstream_response_body = b"""Service configuration error, check your .env file.

\tAZURE_SUMMARY_LEVEL must be either auto, detailed, or concise.
\t
\tGot: foo"""

    def modify_settings(self, app) -> None:
        """Set invalid summary level in settings."""
        app.config["AZURE_SUMMARY_LEVEL"] = "foo"

    @property
    def downstream_request_body(self) -> str:
        """Use a supported bare model with native reasoning."""
        return super().downstream_request_body.replace(
            '"model": "gpt-5"',
            '"model": "gpt-5.4", "reasoning": {"effort": "medium"}',
        )


class TestBadModelName(ReplyBase):
    """Test a single ping-pong interaction, no tool calls."""

    expected_upstream_request_body = None
    expected_downstream_status_code = 400
    expected_downstream_response_body = (
        "Cursor configuration error, check your Cursor settings.\n\n\t"
        + (
            "Model name must be one of:\n"
            f"{SUPPORTED_MODELS_TEXT}\n\n"
            "Got: foo-minimal"
        ).replace("\n", "\n\t")
    ).encode()

    @property
    def downstream_request_body(self) -> str:
        """Set invalid model name in request body."""
        return super().downstream_request_body.replace(
            '"model": "gpt-5"', '"model": "foo-minimal"'
        )


class TestBareModelWithoutReasoning(ReplyBase):
    """Use the default high reasoning effort when Cursor omits the field."""

    expected_downstream_status_code = 200

    @property
    def expected_upstream_request_body(self) -> str:
        """Expect a bare model request to use high reasoning effort."""
        payload = json.loads(super().expected_upstream_request_body)
        payload["model"] = "gpt-5.4"
        payload["reasoning"]["effort"] = "high"
        return json.dumps(payload)

    @property
    def expected_downstream_response_body(self) -> bytes:
        """Expect the Cursor-facing response to retain the requested model id."""
        return super().expected_downstream_response_body.replace(
            b'"model":"gpt-5"', b'"model":"gpt-5.4"'
        )

    @property
    def downstream_request_body(self) -> str:
        """Use a bare model name without the native reasoning field."""
        payload = json.loads(super().downstream_request_body)
        payload["model"] = "gpt-5.4"
        payload.pop("reasoning", None)
        return json.dumps(payload, indent=2) + "\n"
