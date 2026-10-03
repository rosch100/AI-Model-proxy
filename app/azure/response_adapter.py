"""Response adaptation helpers for Azure Responses API streams.

This module defines ResponseAdapter, which converts Azure SSE streams into
OpenAI Chat Completions-compatible streaming responses.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from string import ascii_letters, digits
from typing import Any, Dict, Iterable, Optional

import requests
from flask import Response, current_app, stream_with_context
from rich.live import Live
from rich.markup import escape as rich_escape

from ..common.logging import console, create_message_panel
from ..common.sse import chunks_to_sse, sse_to_events
from ..exceptions import ClientClosedConnection
from ..providers.failover_upstream import PreparedUpstream, sanitize_error_event
from ..reasoning_display import (
    parse_reasoning_display_mode,
    reasoning_content_delta,
    reasoning_end_delta,
    reasoning_start_delta,
)

# Events we intentionally skip without logging (pure lifecycle / status noise)
_SILENT_EVENTS = {
    "response.created",
    "response.in_progress",
    "response.queued",
    "response.completed",
    "response.output_item.done",
    "response.output_text.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.function_call_arguments.done",
    "response.reasoning_text.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.refusal.done",
    "response.output_text.annotation.added",
    # Tool status events (progress / searching / interpreting)
    "response.file_search_call.in_progress",
    "response.file_search_call.searching",
    "response.file_search_call.completed",
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
    "response.web_search_call.completed",
    "response.code_interpreter_call.in_progress",
    "response.code_interpreter_call.interpreting",
    "response.code_interpreter_call.completed",
    "response.image_generation_call.in_progress",
    "response.image_generation_call.generating",
    "response.image_generation_call.completed",
    "response.image_generation_call.partial_image",
    "response.mcp_call.in_progress",
    "response.mcp_call.completed",
    "response.mcp_list_tools.in_progress",
    "response.mcp_list_tools.completed",
    "response.audio.done",
    "response.audio.transcript.done",
}


@dataclass
class _ResponseStreamState:
    """Track mutable state for one adapted upstream response stream."""

    upstream_resp: Any
    completion_msg: Dict[str, Any]
    events: int = 0
    has_emitted_output: bool = False


class ResponseAdapter:
    """Handle post-request adaptation from Azure Responses API to Flask.

    Translates Azure SSE events into OpenAI Chat Completions chunks, including
    native reasoning deltas and function call streaming.
    """

    # Per-request chat completion id (for streaming)
    _chat_completion_id: Optional[str]
    _reasoning_open: bool
    _reasoning_pending_whitespace: str
    _reasoning_display_mode: str
    _tool_calls: int
    _usage: Optional[Dict[str, Any]]

    def __init__(self, adapter: Any) -> None:
        """Initialize the adapter with a reference to the AzureAdapter."""
        self.adapter = adapter  # AzureAdapter instance for shared config/env

    # ---- Helpers ----
    @staticmethod
    def _create_chat_completion_id() -> str:
        """Return a new pseudo-random chat completion id."""
        alphabet = ascii_letters + digits
        return "chatcmpl-" + "".join(random.choices(alphabet, k=24))

    def _build_completion_chunk(
        self,
        *,
        delta: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a Chat Completions chunk dict with the provided delta."""
        return {
            "id": self._chat_completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.adapter.inbound_model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _build_usage_chunk(self) -> Optional[Dict[str, Any]]:
        """Build a terminal Chat Completions usage chunk.

        Maps Azure Responses API usage fields to the OpenAI Chat Completions
        format, including prompt_tokens_details.cached_tokens and
        completion_tokens_details.reasoning_tokens so Cursor can see cache
        hit rates and reasoning overhead.
        """
        if not isinstance(self._usage, dict):
            return None

        input_details = self._usage.get("input_tokens_details")
        cached_tokens = (
            input_details.get("cached_tokens", 0)
            if isinstance(input_details, dict)
            else 0
        )

        output_details = self._usage.get("output_tokens_details")
        reasoning_tokens = (
            output_details.get("reasoning_tokens", 0)
            if isinstance(output_details, dict)
            else 0
        )

        return {
            "id": self._chat_completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.adapter.inbound_model,
            "choices": [],
            "usage": {
                "prompt_tokens": self._usage.get("input_tokens", 0),
                "completion_tokens": self._usage.get("output_tokens", 0),
                "total_tokens": self._usage.get("total_tokens", 0),
                "prompt_tokens_details": {
                    "cached_tokens": cached_tokens,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": reasoning_tokens,
                },
            },
        }

    # ---- Helpers for native Responses API tool types ----
    def _native_tool_to_function_call(
        self, item: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Convert a native Responses API tool call item into a Chat Completions tool_calls chunk.

        Handles native tool types (apply_patch_call, shell_call, mcp_call, etc.)
        by wrapping them as function calls so Cursor can process them.
        """
        item_type = item.get("type", "")
        call_id = item.get("call_id") or item.get("id") or ""

        # Map native type to a function name Cursor expects
        native_type_to_name = {
            "apply_patch_call": "ApplyPatch",
            "shell_call": "Shell",
            "local_shell_call": "Shell",
            "mcp_call": "CallMcpTool",
            "computer_call": "ComputerUse",
        }

        name = native_type_to_name.get(item_type)
        if not name:
            return None

        console.print(
            f"[bold magenta]NATIVE TOOL:[/bold magenta] Converting {item_type} → {name} "
            f"(call_id={call_id})"
        )

        # Build the arguments JSON from the item's fields
        if item_type == "apply_patch_call":
            # Extract diff/operation from the native format
            operation = item.get("operation", {})
            args = {
                "diff": operation.get("diff", ""),
                "path": operation.get("path", ""),
            }
        elif item_type in ("shell_call", "local_shell_call"):
            action = item.get("action", {})
            args = {
                "command": action.get("command", []),
                "working_directory": action.get("working_directory", ""),
            }
        elif item_type == "mcp_call":
            args = {
                "server_label": item.get("server_label", ""),
                "tool_name": item.get("name", ""),
                "arguments": item.get("arguments", "{}"),
            }
        else:
            args = {}

        arguments_json = json.dumps(args, ensure_ascii=False)

        self._tool_calls += 1
        return self._build_completion_chunk(
            delta={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": self._tool_calls - 1,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments_json,
                        },
                    }
                ],
            }
        )

    # ---- Event handlers (per SSE event) ----
    def _output_item__added(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.output_item.added events and emit a single chunk."""

        item = obj.get("item", {}) if isinstance(obj, dict) else {}
        item_type = item.get("type")

        if item_type == "reasoning":
            return None
        if item_type == "function_call":
            self._tool_calls += 1
            name = item.get("name")
            arguments = item.get("arguments")
            call_id = item.get("call_id")
            return self._build_completion_chunk(
                delta={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": self._tool_calls - 1,
                            "id": call_id or "",
                            "type": "function",
                            "function": {
                                "name": name or "",
                                "arguments": arguments or "",
                            },
                        }
                    ],
                }
            )
        # Handle custom_tool_call — Cursor's tools come through as this type
        if item_type == "custom_tool_call":
            self._tool_calls += 1
            name = item.get("name", "")
            call_id = item.get("call_id") or item.get("id") or ""
            # In streaming, `input` is empty here; deltas arrive via
            # response.custom_tool_call_input.delta events.
            arguments = item.get("input", "")
            return self._build_completion_chunk(
                delta={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": self._tool_calls - 1,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            )

        # Handle native Responses API tool types (apply_patch_call, shell_call, etc.)
        if item_type in (
            "apply_patch_call",
            "shell_call",
            "local_shell_call",
            "mcp_call",
            "computer_call",
        ):
            return self._native_tool_to_function_call(item)

        if item_type == "message":
            # Message items (e.g. output_item.added with type=message) are typically
            # the container for output text. No action needed here.
            return None

        # Log unexpected item types for debugging
        if item_type:
            console.print(f"[bold yellow]UNKNOWN ITEM TYPE:[/bold yellow] {item_type}")

        return None

    def _function_call_arguments__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.function_call.arguments.delta events."""
        arguments_delta = obj.get("delta", "") if isinstance(obj, dict) else ""
        return self._build_completion_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": self._tool_calls - 1,
                        "function": {"arguments": arguments_delta},
                    }
                ]
            }
        )

    def _custom_tool_call_input__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.custom_tool_call_input.delta events (streaming tool arguments)."""
        input_delta = obj.get("delta", "") if isinstance(obj, dict) else ""
        return self._build_completion_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": self._tool_calls - 1,
                        "function": {"arguments": input_delta},
                    }
                ]
            }
        )

    def _custom_tool_call_input__done(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.custom_tool_call_input.done — end marker, no-op."""
        return None

    # ---- Error event (no "response." prefix!) ----
    def _error(self, obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Handle 'error' SSE event by logging until response.failed arrives."""
        error = self._sse_error_details(obj)
        code = error.get("code", "")
        message = error.get("message", "")
        console.print(
            f"[bold red]STREAM ERROR:[/bold red] {rich_escape(f'code={code} message={message}')}"
        )
        return None

    # ---- Refusal events ----
    def _refusal__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.refusal.delta — model is refusing the request."""
        return self._build_completion_chunk(
            delta={
                "role": "assistant",
                "content": (obj.get("delta", "") if isinstance(obj, dict) else ""),
            }
        )

    def _build_reasoning_chunk(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Build a Chat Completions chunk with Cursor-native reasoning content."""
        delta = obj.get("delta", "") if isinstance(obj, dict) else ""
        if not delta:
            return None

        start_delta = (
            reasoning_start_delta(self._reasoning_display_mode)
            if not self._reasoning_open
            else None
        )
        if start_delta is not None and not delta.strip():
            self._reasoning_pending_whitespace += delta
            return None

        reasoning_text = self._reasoning_pending_whitespace + delta
        self._reasoning_pending_whitespace = ""
        reasoning_delta = reasoning_content_delta(
            reasoning_text, self._reasoning_display_mode
        )
        if start_delta is not None:
            reasoning_delta["content"] = start_delta["content"] + reasoning_text
            self._reasoning_open = True

        return self._build_completion_chunk(delta=reasoning_delta)

    def _close_reasoning_chunk(self) -> Optional[Dict[str, Any]]:
        """Close any visible reasoning wrapper before normal output resumes."""
        if not self._reasoning_open:
            return None
        self._reasoning_open = False
        closing_delta = reasoning_end_delta(self._reasoning_display_mode)
        if closing_delta is None:
            return None
        return self._build_completion_chunk(delta=closing_delta)

    # ---- Reasoning text (raw, not summary) ----
    def _reasoning_text__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.reasoning_text.delta — raw reasoning content.

        Cursor renders this separately from normal assistant content when it
        arrives as a native reasoning delta.
        """
        return self._build_reasoning_chunk(obj)

    # ---- MCP call argument streaming ----
    def _mcp_call_arguments__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.mcp_call_arguments.delta — streaming MCP tool args."""
        arguments_delta = obj.get("delta", "") if isinstance(obj, dict) else ""
        return self._build_completion_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": self._tool_calls - 1,
                        "function": {"arguments": arguments_delta},
                    }
                ]
            }
        )

    def _mcp_call_arguments__done(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.mcp_call_arguments.done — no-op end marker."""
        return None

    # ---- MCP call failure ----
    def _mcp_call__failed(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.mcp_call.failed — log MCP call failure."""
        console.print(
            f"[bold red]MCP CALL FAILED:[/bold red] {rich_escape(str(obj)[:300])}"
        )
        return None

    # ---- MCP list tools failure ----
    def _mcp_list_tools__failed(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.mcp_list_tools.failed — log MCP list failure."""
        console.print(
            f"[bold red]MCP LIST TOOLS FAILED:[/bold red] {rich_escape(str(obj)[:300])}"
        )
        return None

    # ---- Code interpreter code streaming ----
    def _code_interpreter_call_code__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.code_interpreter_call_code.delta — streaming code."""
        # Emit as text content so the user sees the code being generated
        return self._build_completion_chunk(
            delta={
                "role": "assistant",
                "content": (obj.get("delta", "") if isinstance(obj, dict) else ""),
            }
        )

    # ---- Audio events ----
    def _audio__delta(self, obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Handle response.audio.delta — audio chunk (base64). Pass-through as-is."""
        # Audio can't be represented in Chat Completions text stream; skip.
        return None

    def _audio__transcript__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.audio.transcript.delta — audio transcript text."""
        return self._build_completion_chunk(
            delta={
                "role": "assistant",
                "content": (obj.get("delta", "") if isinstance(obj, dict) else ""),
            }
        )

    def _reasoning_summary_text__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle reasoning.summary_text.delta events as native reasoning."""
        return self._build_reasoning_chunk(obj)

    def _reasoning_summary_text__done(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle reasoning.summary_text.done events."""
        return None

    def _output_text__delta(
        self, obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle response.output_text.delta events and emit text chunk."""
        return self._build_completion_chunk(
            delta={
                "role": "assistant",
                "content": (obj.get("delta", "") if isinstance(obj, dict) else ""),
            }
        )

    def _completed(self, obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Capture final Azure usage for an optional terminal usage chunk."""
        response = obj.get("response", {}) if isinstance(obj, dict) else {}
        usage = response.get("usage")
        self._usage = usage if isinstance(usage, dict) else None

        # Log usage details including cache hit info
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            input_details = usage.get("input_tokens_details", {})
            cached_tokens = (
                input_details.get("cached_tokens", 0)
                if isinstance(input_details, dict)
                else 0
            )
            output_details = usage.get("output_tokens_details", {})
            reasoning_tokens = (
                output_details.get("reasoning_tokens", 0)
                if isinstance(output_details, dict)
                else 0
            )
            cache_pct = (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
            console.print(
                f"[bold green]USAGE:[/bold green] "
                f"input={input_tokens} (cached={cached_tokens}, {cache_pct:.0f}%) "
                f"output={output_tokens} (reasoning={reasoning_tokens}) "
                f"total={total_tokens}"
            )
        return None

    def _incomplete(self, obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Handle response.incomplete events (model output was truncated).

        This occurs when the model hits max_output_tokens. We log and pass
        the reason through so the downstream client knows the response was cut short.
        """
        reason = (
            obj.get("response", {})
            .get("incomplete_details", {})
            .get("reason", "unknown")
            if isinstance(obj, dict)
            else "unknown"
        )
        console.print(f"[bold red]RESPONSE INCOMPLETE:[/bold red] reason={reason}")
        return self._build_completion_chunk(
            delta={
                "role": "assistant",
                "content": f"\n\n[Response was truncated: {reason}]",
            }
        )

    def _failed(self, obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Handle response.failed events and emit a single chunk."""
        error = obj.get("response", {}).get("error", {})
        return self._build_completion_chunk(
            delta={
                "role": "assistant",
                "content": "Azure raised a '"
                + error.get("code", "")
                + "' error with the following message:\n\n\n"
                + "_**"
                + error.get("message", "")
                + "**_\n\n\n"
                "This may be an Azure-side issue rather than a proxy bug.",
            }
        )

    @staticmethod
    def _error_dict_is_rate_limit(error: Any) -> bool:
        """Match Azure TPM/RPM throttles by code or message text."""
        if not isinstance(error, dict):
            return False
        if error.get("code") == "rate_limit_exceeded":
            return True
        message = str(error.get("message") or "").lower()
        return "rate limit" in message or "token rate" in message

    @staticmethod
    def _sse_error_details(event_data: Any) -> dict[str, Any]:
        """Return Azure error details from either supported SSE payload shape."""
        if not isinstance(event_data, dict):
            return {}
        nested_error = event_data.get("error")
        return nested_error if isinstance(nested_error, dict) else event_data

    @staticmethod
    def _is_rate_limit_failure(raw_event: str, event_data: Any) -> bool:
        """Identify Azure's retryable streaming rate-limit failure event."""
        if not isinstance(event_data, dict):
            return False
        if raw_event == "error":
            return ResponseAdapter._error_dict_is_rate_limit(
                ResponseAdapter._sse_error_details(event_data)
            )
        if raw_event != "response.failed":
            return False
        response = event_data.get("response")
        if not isinstance(response, dict):
            return False
        return ResponseAdapter._error_dict_is_rate_limit(response.get("error"))

    @staticmethod
    def _error_from_response(upstream_resp: Any) -> Dict[str, str]:
        """Extract a useful error when a stream retry returns a non-200 response."""
        try:
            payload = upstream_resp.json()
        except (AttributeError, ValueError):
            payload = {}
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return {
                "code": str(error.get("code") or upstream_resp.status_code),
                "message": str(error.get("message") or upstream_resp.text),
            }
        return {
            "code": str(upstream_resp.status_code),
            "message": str(upstream_resp.text),
        }

    def _absorb_empty_error_precursor(
        self, event: Any, state: _ResponseStreamState, request_context: Any
    ) -> bool:
        """Skip empty SSE error noise and start shared cooldown before peers stampede."""
        if request_context is None or state.has_emitted_output:
            return False
        if (event.event or "") != "error":
            return False
        data = event.json if isinstance(event.json, dict) else {}
        code = str(data.get("code") or "").strip()
        message = str(data.get("message") or "").strip()
        if code or message or data.get("error"):
            return False
        self.adapter.note_empty_stream_error_precursor(request_context)
        console.print(
            "[yellow]Azure empty stream error before output; "
            "holding shared cooldown for likely rate limit[/yellow]"
        )
        return True

    def _retry_stream_if_possible(
        self, event: Any, state: _ResponseStreamState, request_context: Any
    ) -> Any:
        """Retry a rate-limit event only before downstream-visible output."""
        if request_context is None or state.has_emitted_output:
            return None
        raw_event = event.event or ""
        if not self._is_rate_limit_failure(raw_event, event.json):
            return None
        return self.adapter._retry_stream_rate_limit(
            state.upstream_resp, request_context, event.json
        )

    def _stream_upstream_events(
        self,
        state: _ResponseStreamState,
        request_context: Any,
        live: Live,
    ) -> Iterable[Dict[str, Any]]:
        """Adapt events, restarting the upstream stream only before output."""
        while True:
            if state.upstream_resp.status_code != 200:
                error_chunk = self._failed(
                    {
                        "response": {
                            "error": self._error_from_response(state.upstream_resp)
                        }
                    }
                )
                if error_chunk is not None:
                    yield error_chunk
                    self._record_completion_chunk(error_chunk, state)
                return

            retry_stream = False
            empty_error_precursor = False
            stream_error_after_output: Optional[Dict[str, str]] = None
            stream_rate_limit_noted = False
            terminal_event_seen = False
            for event in sse_to_events(
                state.upstream_resp.iter_content(chunk_size=128)
            ):
                if isinstance(state.upstream_resp, PreparedUpstream):
                    event = sanitize_error_event(event)
                if event.event in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                }:
                    terminal_event_seen = True
                absorbed_precursor = self._absorb_empty_error_precursor(
                    event, state, request_context
                )
                if absorbed_precursor:
                    empty_error_precursor = True
                    continue
                raw_event = event.event or ""
                is_rate_limit_failure = self._is_rate_limit_failure(
                    raw_event, event.json
                )
                retry_response = self._retry_stream_if_possible(
                    event, state, request_context
                )
                if retry_response is not None:
                    state.upstream_resp = retry_response
                    retry_stream = True
                    break
                if raw_event == "error":
                    error = self._sse_error_details(event.json)
                    if state.has_emitted_output:
                        stream_error_after_output = {
                            "code": str(error.get("code") or "stream_error"),
                            "message": str(
                                error.get("message")
                                or "Azure ended the stream after partial output."
                            ),
                        }
                    elif error.get("code") or error.get("message"):
                        error_chunk = self._failed({"response": {"error": error}})
                        if error_chunk is not None:
                            state.has_emitted_output = True
                            yield error_chunk
                            self._record_completion_chunk(error_chunk, state)
                        return
                if (
                    is_rate_limit_failure
                    and state.has_emitted_output
                    and request_context is not None
                ):
                    self.adapter.note_stream_rate_limit(
                        request_context, state.upstream_resp.headers, event.json
                    )
                    stream_rate_limit_noted = True
                for chunk in self._adapt_event(event, state, live):
                    state.has_emitted_output = True
                    yield chunk

            if retry_stream:
                self._reset_stream_attempt(state)
                continue
            if (
                empty_error_precursor
                and not state.has_emitted_output
                and not terminal_event_seen
                and request_context is not None
            ):
                retry_response = self.adapter._retry_stream_rate_limit(
                    state.upstream_resp,
                    request_context,
                    {
                        "code": "rate_limit_exceeded",
                        "message": "token rate limit",
                    },
                )
                if retry_response is not None:
                    state.upstream_resp = retry_response
                    self._reset_stream_attempt(state)
                    continue
            incomplete_after_output = state.has_emitted_output and (
                stream_error_after_output or empty_error_precursor
            )
            if incomplete_after_output and not terminal_event_seen:
                is_rate_limit_or_unknown_error = stream_error_after_output and (
                    stream_error_after_output["code"] == "rate_limit_exceeded"
                    or stream_error_after_output["code"] == "stream_error"
                )
                if (
                    request_context is not None
                    and not empty_error_precursor
                    and not stream_rate_limit_noted
                    and is_rate_limit_or_unknown_error
                ):
                    self.adapter.note_empty_stream_error_precursor(request_context)
                if stream_error_after_output is None:
                    error = {
                        "code": "rate_limit_exceeded",
                        "message": (
                            "Azure emitted an empty stream error before the response "
                            "ended; the response may be incomplete. It was not "
                            "restarted to avoid duplicating output or tool calls."
                        ),
                    }
                else:
                    error = {
                        **stream_error_after_output,
                        "message": (
                            f"{stream_error_after_output['message']} The response may "
                            "be incomplete; it was not restarted to avoid duplicating "
                            "output or tool calls."
                        ),
                    }
                error_chunk = self._failed({"response": {"error": error}})
                if error_chunk is not None:
                    yield error_chunk
                    self._record_completion_chunk(error_chunk, state)
            return

    def _reset_stream_attempt(self, state: _ResponseStreamState) -> None:
        """Discard adaptation state belonging to a failed upstream attempt."""
        state.events = 0
        state.completion_msg = {"role": "assistant", "content": "", "tool_calls": []}
        self._reasoning_open = False
        self._reasoning_pending_whitespace = ""
        self._tool_calls = 0
        self._usage = None

    def _adapt_event(
        self, event: Any, state: _ResponseStreamState, live: Live
    ) -> Iterable[Dict[str, Any]]:
        """Convert one upstream event and record any emitted completion data."""
        if current_app.config["LOG_COMPLETION"]:
            if state.events > 1:
                live.update(create_message_panel(state.completion_msg, 1, 1))
            state.events += 1

        raw_event = event.event or ""
        event_data = event.json
        handler_name = "_" + raw_event.replace("response.", "", 1).replace(".", "__")
        handler = getattr(self, handler_name, None)
        if handler is None:
            self._log_unhandled_event(raw_event, handler_name, event_data)
            return

        closing = self._reasoning_chunk_before_event(raw_event, event_data)
        if closing is not None:
            yield closing
            self._record_completion_chunk(closing, state)

        chunk = handler(event_data)
        if chunk is not None:
            yield chunk
            self._record_completion_chunk(chunk, state)

    def _reasoning_chunk_before_event(
        self, raw_event: str, event_data: Any
    ) -> Optional[Dict[str, Any]]:
        """Close an open reasoning block when the next output event requires it."""
        if raw_event not in {
            "response.output_text.delta",
            "response.output_item.added",
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            return None

        item = event_data.get("item") if isinstance(event_data, dict) else None
        if (
            raw_event == "response.output_item.added"
            and isinstance(item, dict)
            and item.get("type") == "reasoning"
        ):
            return None

        self._reasoning_pending_whitespace = ""
        if self._reasoning_open:
            return self._close_reasoning_chunk()
        return None

    @staticmethod
    def _log_unhandled_event(
        raw_event: str, handler_name: str, event_data: Any
    ) -> None:
        """Log unexpected upstream events while suppressing known lifecycle noise."""
        if raw_event in _SILENT_EVENTS:
            return
        console.print(
            f"[bold yellow]UNHANDLED EVENT:[/bold yellow] "
            f"{rich_escape(f'{raw_event} → {handler_name} data={str(event_data)[:300]}')}"
        )

    @staticmethod
    def _record_completion_chunk(
        chunk: Dict[str, Any], state: _ResponseStreamState
    ) -> None:
        """Accumulate visible text and tool arguments for completion logging."""
        if not current_app.config["LOG_COMPLETION"]:
            return
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content is not None:
            state.completion_msg["content"] += content
            return

        for tool_call_delta in delta.get("tool_calls", []):
            function = tool_call_delta.get("function", {})
            name = function.get("name")
            arguments = function.get("arguments", "")
            if name:
                state.completion_msg["tool_calls"].append(
                    {
                        "id": tool_call_delta.get("id", ""),
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            else:
                state.completion_msg["tool_calls"][-1]["function"][
                    "arguments"
                ] += arguments

    def _finish_stream(
        self, state: _ResponseStreamState, live: Live
    ) -> Iterable[Dict[str, Any]]:
        """Emit the terminal completion, usage, and logging updates."""
        finish = "tool_calls" if self._tool_calls > 0 else "stop"
        console.print(
            f"[bold cyan]STREAM_END:[/bold cyan] events={state.events}, "
            f"tool_calls={self._tool_calls}, finish_reason={finish}"
        )

        closing = self._close_reasoning_chunk()
        if closing is not None:
            yield closing
        finish_reason = "tool_calls" if self._tool_calls > 0 else "stop"
        yield self._build_completion_chunk(finish_reason=finish_reason)

        if self.adapter.include_usage:
            usage_chunk = self._build_usage_chunk()
            if usage_chunk is not None:
                yield usage_chunk
        if current_app.config["LOG_COMPLETION"]:
            live.update(create_message_panel(state.completion_msg, 1, 1))

    def _adapt_stream(
        self, upstream_resp: Any, request_context: Any
    ) -> Iterable[Dict[str, Any]]:
        """Manage per-stream state, logging, and upstream response cleanup."""
        state = _ResponseStreamState(
            upstream_resp=upstream_resp,
            completion_msg={"role": "assistant", "content": "", "tool_calls": []},
        )
        try:
            with Live(None, console=console, refresh_per_second=2) as live:
                yield from self._stream_upstream_events(state, request_context, live)
                yield from self._finish_stream(state, live)
        finally:
            state.upstream_resp.close()

    def adapt(self, upstream_resp: Any, request_context: Any = None) -> Response:
        """Adapt an upstream Azure streaming response into SSE for Flask."""

        @stream_with_context
        def generate() -> Iterable[bytes]:
            # Generate once per stream
            self._chat_completion_id = self._create_chat_completion_id()
            # Initialize per-stream state on the instance
            self._reasoning_open = False
            self._reasoning_pending_whitespace = ""
            self._reasoning_display_mode = parse_reasoning_display_mode(
                current_app.config["REASONING_DISPLAY_MODE"]
            )
            self._tool_calls = 0
            self._usage = None

            try:
                yield from chunks_to_sse(
                    self._adapt_stream(upstream_resp, request_context)
                )
            except requests.RequestException:
                if not isinstance(upstream_resp, PreparedUpstream):
                    raise
                error = self._failed(
                    {
                        "response": {
                            "error": {
                                "code": "stream_interrupted",
                                "message": (
                                    "Provider stream interrupted; not replayed "
                                    "to avoid duplicate output or tool calls."
                                ),
                            }
                        }
                    }
                )
                yield from chunks_to_sse([error])
            except GeneratorExit:
                # Downstream client closed the connection mid-stream
                # Translate to a clearer exception for the caller.
                raise ClientClosedConnection(
                    "Client closed connection during streaming response"
                ) from None

        headers = {}
        headers["Content-Type"] = "text/event-stream; charset=utf-8"
        headers["Cache-Control"] = "no-cache"
        headers["Connection"] = "keep-alive"
        headers["X-Accel-Buffering"] = "no"
        return Response(
            generate(),
            status=getattr(upstream_resp, "status_code", 200),
            headers=headers,
        )
