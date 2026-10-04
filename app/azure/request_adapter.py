"""Request adaptation helpers for Azure Responses API.

This module defines RequestAdapter, which transforms incoming OpenAI-style
requests into Azure Responses API request parameters.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from flask import Request, current_app

from ..auth import current_tenant
from ..common.logging import console
from ..exceptions import CursorConfigurationError, ServiceConfigurationError
from ..models import (
    SUPPORTED_MODELS,
    SUPPORTED_MODELS_TEXT,
    select_fallback_public_model,
)
from ..tenants import AUTH_MODE_TENANT, DatabaseTenantSnapshot, TenantConfig

_STRIPPED_UPSTREAM_HEADERS = frozenset(
    {
        "authorization",
        "api-key",
        "host",
        "tenant",
        "tenant-id",
        "x-tenant",
        "x-tenant-id",
        "x-tenant-key",
    }
)


class RequestAdapter:
    """Handle pre-request adaptation for the Azure Responses API.

    Transforms OpenAI Completions/Chat-style inputs into Azure Responses API
    request parameters suitable for streaming completions in this codebase.
    Returns request_kwargs for requests.request(**kwargs). Also sets
    per-request state on the adapter (model).
    """

    def __init__(self, adapter: Any) -> None:
        """Initialize the adapter with a reference to the AzureAdapter."""
        self.adapter = adapter  # AzureAdapter instance for shared config/env

    # ---- Helpers (kept local to minimize cross-module coupling) ----

    @staticmethod
    def _safe_call_id(call_id: Any) -> Any:
        """Shorten call ids over Azure's 64-char Responses API limit.

        Cursor can send longer tool_call ids. Hashing deterministically keeps a
        function_call and its function_call_output consistent within a request.
        """
        if not isinstance(call_id, str) or len(call_id) <= 64:
            return call_id
        digest = hashlib.sha256(call_id.encode()).hexdigest()[:24]
        return f"{call_id[:39]}_{digest}"

    def _content_to_text(self, content: Any) -> str:
        """Convert message content (string or list of parts) to a string for Azure."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        parts.append("[image]")
                    else:
                        parts.append(f"[{part.get('type', 'unknown')}]")
                else:
                    parts.append(str(part))
            return "\n".join(parts) if parts else ""
        return str(content)

    def _copy_request_headers_for_azure(
        self, src: Request, *, api_key: str, session_id: str | None = None
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for key, value in src.headers.items():
            if key.casefold() in _STRIPPED_UPSTREAM_HEADERS:
                continue
            headers[key] = value
        # Azure prefers api-key header
        headers["api-key"] = api_key

        # Cache-routing headers matching Codex CLI (codex-rs).
        # session_id pins all requests in a conversation to the same Azure
        # backend machine so the prompt cache is reused across turns.
        # x-client-request-id provides per-conversation request correlation.
        if session_id:
            headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id

        return headers

    def _azure_runtime_settings(
        self, tenant: TenantConfig | DatabaseTenantSnapshot | None
    ) -> tuple[str, str, dict[str, str]]:
        """Resolve Azure URL, API key, and deployment map for this snapshot."""
        if tenant is not None:
            if isinstance(tenant, DatabaseTenantSnapshot):
                if tenant.profile_id is None:
                    raise ServiceConfigurationError(
                        "No active provider profile is configured for this tenant."
                    )
                if tenant.profile_deleted:
                    raise ServiceConfigurationError(
                        "The active provider profile has been removed."
                    )
                if tenant.provider != "azure":
                    raise ServiceConfigurationError(
                        "The active tenant provider is not available on the Azure route."
                    )
                if tenant.azure_api_key is None or not tenant.azure_model_deployments:
                    raise ServiceConfigurationError(
                        "The active Azure profile is missing credentials or model deployments."
                    )
            return (
                tenant.azure_responses_api_url,
                tenant.azure_api_key,
                dict(tenant.azure_model_deployments),
            )
        if current_app.config.get("AUTH_MODE") == AUTH_MODE_TENANT:
            raise ServiceConfigurationError(
                "AUTH_MODE=tenant requires an authenticated tenant; "
                "refusing to use global Azure credentials."
            )
        settings = current_app.config
        return (
            settings["AZURE_RESPONSES_API_URL"],
            settings["AZURE_API_KEY"],
            settings["AZURE_MODEL_DEPLOYMENTS"],
        )

    @staticmethod
    def _tenant_scoped_cache_key(
        conversation_id: str | None,
        tenant: TenantConfig | DatabaseTenantSnapshot | None,
    ) -> str | None:
        """Partition Azure cache/session keys per tenant when authenticated."""
        if not conversation_id:
            return None
        if tenant is None:
            return conversation_id
        return f"{tenant.id}:{conversation_id}"

    def _messages_to_responses_input_and_instructions(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role in {"system", "developer"}:
                instructions_parts.append(self._content_to_text(content))
                continue
            # For user/assistant/tools as inputs
            if role == "tool":
                call_id = self._safe_call_id(m.get("tool_call_id"))
                item = {
                    "type": "function_call_output",
                    "output": self._content_to_text(content),
                    "status": "completed",
                    "call_id": call_id,
                }
                input_items.append(item)
            else:
                text_content = self._content_to_text(content)
                item = {
                    "role": role or "user",
                    "content": [
                        {
                            "type": "input_text" if role == "user" else "output_text",
                            "text": text_content,
                        },
                    ],
                }
                input_items.append(item)

                if tool_calls := m.get("tool_calls"):
                    for tool_call in tool_calls or []:
                        if not isinstance(tool_call, dict):
                            continue
                        function = tool_call.get("function") or {}
                        call_id = self._safe_call_id(tool_call.get("id"))
                        item = {
                            "type": "function_call",
                            "name": function.get("name"),
                            "arguments": function.get("arguments"),
                            "call_id": call_id,
                        }
                        input_items.append(item)

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None
        return {
            "instructions": instructions,
            "input": input_items if input_items else None,
        }

    def _transform_tools_for_responses(self, tools: Any) -> Any:
        out: List[Dict[str, Any]] = []
        if not isinstance(tools, list):
            current_app.logger.debug(
                "Skipping tool transformation because tools payload is not a list: %r",
                tools,
            )
            return out

        # Debug: log the shape of first tool to understand what Cursor sends
        if tools:
            sample = tools[0] if isinstance(tools[0], dict) else {}
            console.print(
                f"[bold yellow]TOOL_DEBUG:[/bold yellow] count={len(tools)}, "
                f"first_keys={list(sample.keys())[:10]}, type={sample.get('type')}, "
                f"has_function={'function' in sample}, has_name={'name' in sample}"
            )

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not function:
                # Tool might already be in Responses API format (has "name" at top level)
                if tool.get("name"):
                    out.append(tool)
                else:
                    console.print(
                        f"[bold red]TOOL_SKIPPED:[/bold red] tool has no 'function' and no 'name'. "
                        f"keys={list(tool.keys())[:10]}"
                    )
                continue
            transformed: Dict[str, Any] = {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
                "strict": False,
            }
            out.append(transformed)

        console.print(
            f"[bold yellow]TOOL_TRANSFORM:[/bold yellow] {len(tools)} tools in → {len(out)} tools out"
        )
        if out:
            console.print(
                f"[bold yellow]TOOL_TRANSFORM:[/bold yellow] first out name={out[0].get('name')}"
            )
        return out

    def _transform_tool_choice_for_responses(self, tool_choice: Any) -> Any:
        """Convert Chat Completions forced function choice to Responses shape."""
        if not isinstance(tool_choice, dict):
            return tool_choice

        if tool_choice.get("type") != "function":
            return tool_choice

        function = tool_choice.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            return tool_choice

        return {
            "type": "function",
            "name": function["name"],
        }

    def _resolve_model_and_reasoning(
        self, payload: Dict[str, Any], deployment_map: dict[str, str]
    ) -> Dict[str, Any]:
        """Resolve the Azure deployment and reasoning settings for this request."""
        inbound_model = payload.get("model")
        if not isinstance(inbound_model, str) or not inbound_model.strip():
            raise CursorConfigurationError("Model name must be a non-empty string.")
        if any(character.isspace() for character in inbound_model):
            raise CursorConfigurationError("Model name must not contain whitespace.")

        model_key = inbound_model.lower()
        # Allow effort-suffixed names (e.g. gpt-5.6-sol-high) since Cursor only
        # sends reasoning.effort for model names it recognizes.
        suffix_effort = None
        if model_key not in SUPPORTED_MODELS:
            for effort in ("minimal", "low", "medium", "high"):
                base = model_key.removesuffix(f"-{effort}")
                if base != model_key and base in SUPPORTED_MODELS:
                    model_key = base
                    suffix_effort = effort
                    break
        supported = model_key in SUPPORTED_MODELS
        if model_key not in deployment_map:
            fallback_model = select_fallback_public_model(deployment_map)
            if fallback_model is None:
                if not supported:
                    raise CursorConfigurationError(
                        "Model name must be one of:\n"
                        f"{SUPPORTED_MODELS_TEXT}\n"
                        f"\nGot: {inbound_model}"
                    )
                raise CursorConfigurationError(
                    f"Model {model_key!r} is supported by the proxy but not configured "
                    "for this Azure resource."
                )
            console.print(
                f"MODEL_FALLBACK: inbound={inbound_model!r} → {fallback_model}",
                style="yellow",
                markup=False,
            )
            model_key = fallback_model
        azure_deployment = deployment_map[model_key]
        inbound_reasoning = (
            payload.get("reasoning") if isinstance(payload, dict) else None
        )
        inbound_effort = (
            inbound_reasoning.get("effort")
            if isinstance(inbound_reasoning, dict)
            else None
        )
        inbound_summary = (
            inbound_reasoning.get("summary")
            if isinstance(inbound_reasoning, dict)
            else None
        )

        # Precedence: explicit request field, then model-name suffix, then high.
        reasoning_effort = inbound_effort or suffix_effort or "high"

        return {
            "azure_deployment": azure_deployment,
            "reasoning_effort": reasoning_effort,
            "inbound_summary": inbound_summary,
        }

    # ---- Main adaptation (always streaming completions-like) ----
    def adapt(
        self,
        req: Request,
        tenant: TenantConfig | DatabaseTenantSnapshot | None = None,
        *,
        target_model: str | None = None,
    ) -> Dict[str, Any]:
        """Build requests.request kwargs for the Azure Responses API call.

        Maps inputs to the Responses schema and returns a dict suitable for
        requests.request(**kwargs).
        """
        # Reset per-request state
        self.adapter.inbound_model = None
        self.adapter.activity_tenant_id = None
        self.adapter.activity_provider = None
        self.adapter.activity_profile_id = None
        self.adapter.activity_routed_model = None

        # Parse request body (Cursor sometimes sends malformed payloads)
        payload = req.get_json(silent=True, force=False)
        if payload is None:
            payload = {}
        elif not isinstance(payload, dict):
            raise CursorConfigurationError("Request body must be a JSON object.")

        # Determine target model
        inbound_model = payload.get("model")
        self.adapter.inbound_model = inbound_model

        settings = current_app.config
        if tenant is None:
            tenant = current_tenant()
        azure_url, azure_api_key, deployment_map = self._azure_runtime_settings(tenant)
        if isinstance(tenant, DatabaseTenantSnapshot) and target_model is not None:
            matching_model = next(
                (
                    model
                    for model in deployment_map
                    if model.casefold() == target_model.casefold()
                ),
                None,
            )
            if matching_model is None:
                raise ServiceConfigurationError(
                    "The selected Azure profile model is not present in "
                    "the tenant deployment map."
                )
            payload = {**payload, "model": matching_model}
        elif (
            isinstance(tenant, DatabaseTenantSnapshot)
            and inbound_model == tenant.custom_model_id
        ):
            if tenant.default_model is None:
                raise ServiceConfigurationError(
                    "The active Azure profile has no default model configured."
                )
            if (
                tenant.default_model not in deployment_map
                and tenant.default_model.lower() not in deployment_map
            ):
                raise ServiceConfigurationError(
                    "The active Azure profile default model is not present in "
                    "the tenant deployment map."
                )
            payload = {**payload, "model": tenant.default_model}

        if isinstance(tenant, DatabaseTenantSnapshot):
            routed_model = payload.get("model")
            self.adapter.activity_tenant_id = tenant.id
            self.adapter.activity_provider = tenant.provider or "azure"
            self.adapter.activity_profile_id = tenant.profile_id
            self.adapter.activity_routed_model = (
                routed_model if isinstance(routed_model, str) else None
            )

        # Derive conversation_id from Cursor's metadata.cursorConversationId.
        # This is unique per conversation, matching Codex CLI's use of
        # conversation_id for session_id, x-client-request-id, and
        # prompt_cache_key.  The ``user`` field is a per-*user* hash that is
        # the same across all conversations and must NOT be used for routing.
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        conversation_id = (
            metadata.get("cursorConversationId") if isinstance(metadata, dict) else None
        )
        cache_key = self._tenant_scoped_cache_key(conversation_id, tenant)

        upstream_headers = self._copy_request_headers_for_azure(
            req, api_key=azure_api_key, session_id=cache_key
        )

        # Map Chat/Completions to Responses (always streaming)
        # Cursor may send either:
        #   - Chat Completions format: {"messages": [...]}
        #   - Responses API format:    {"input": [...], "instructions": "..."}
        messages = payload.get("messages")
        raw_input = payload.get("input")

        if messages and isinstance(messages, list):
            # Standard Chat Completions → convert to Responses format
            responses_body = self._messages_to_responses_input_and_instructions(
                messages
            )
        elif raw_input is not None:
            # Already in Responses API format — pass through
            responses_body = {
                "input": raw_input,
                "instructions": payload.get("instructions"),
            }
        else:
            responses_body = {"input": "", "instructions": None}

        resolved_reasoning = self._resolve_model_and_reasoning(payload, deployment_map)
        azure_deployment = resolved_reasoning["azure_deployment"]
        reasoning_effort = resolved_reasoning["reasoning_effort"]
        inbound_summary = resolved_reasoning["inbound_summary"]

        # Log request details including cache-relevant fields
        input_len = len(raw_input) if raw_input else 0
        req_fmt = "resp" if "input" in payload else "chat"
        conv_preview = conversation_id[:12] + "…" if conversation_id else "None"
        console.print(
            f"[bold cyan]REQUEST:[/bold cyan] "
            f"model={azure_deployment} effort={reasoning_effort} "
            f"fmt={req_fmt} items={input_len} "
            f"conv={conv_preview}"
        )

        responses_body["model"] = azure_deployment

        # Transform tools and tool choice
        responses_body["tools"] = self._transform_tools_for_responses(
            payload.get("tools", [])
        )
        responses_body["tool_choice"] = self._transform_tool_choice_for_responses(
            payload.get("tool_choice")
        )

        # Matching Codex CLI: prompt_cache_key = conversation_id so each
        # conversation gets its own cache partition on the Azure backend.
        # Only set when we actually have a conversation ID; sending None
        # could confuse Azure. Tenant mode prefixes the tenant id.
        if cache_key:
            responses_body["prompt_cache_key"] = cache_key

        # Always streaming
        responses_body["stream"] = True

        responses_body["reasoning"] = {
            "effort": reasoning_effort,
        }

        if inbound_summary is not None:
            responses_body["reasoning"]["summary"] = inbound_summary
        # Concise is not supported by GPT-5,
        # but allowing it for now to be able to test it on other models
        elif settings["AZURE_SUMMARY_LEVEL"] in {"auto", "detailed", "concise"}:
            responses_body["reasoning"]["summary"] = settings["AZURE_SUMMARY_LEVEL"]
        else:
            raise ServiceConfigurationError(
                "AZURE_SUMMARY_LEVEL must be either auto, detailed, or concise."
                f"\n\nGot: {settings['AZURE_SUMMARY_LEVEL']}"
            )

        # No need to pass verbosity if it's set to medium, as it's the model's default
        if settings["AZURE_VERBOSITY_LEVEL"] in {"low", "high"}:
            responses_body["text"] = {"verbosity": settings["AZURE_VERBOSITY_LEVEL"]}

        # Matching Codex CLI: store=True for Azure enables server-side
        # response storage which is used for prompt caching.
        responses_body["store"] = True

        # Forward the include field (e.g. ["reasoning.encrypted_content"])
        # so Azure returns all the data Cursor expects.
        include = payload.get("include")
        if isinstance(include, list) and include:
            responses_body["include"] = include

        # Codex CLI sends parallel_tool_calls=true; match that behaviour.
        responses_body["parallel_tool_calls"] = True

        # Forward service_tier if Cursor provides one (e.g. "default", "flex").
        service_tier = payload.get("service_tier")
        if service_tier is not None:
            responses_body["service_tier"] = service_tier
        payload_stream_options = payload.get("stream_options")
        merged_stream_options = (
            dict(payload_stream_options)
            if isinstance(payload_stream_options, dict)
            else {}
        )
        # Cursor may send include_usage (Chat Completions param) which Azure's
        # Responses API does not accept.  Store the flag on the shared adapter
        # so the response side can emit a usage chunk, then strip it.
        self.adapter.include_usage = bool(
            merged_stream_options.pop("include_usage", False)
        )
        merged_stream_options["include_obfuscation"] = False
        responses_body["stream_options"] = merged_stream_options

        if settings["AZURE_TRUNCATION"] == "auto":
            responses_body["truncation"] = settings["AZURE_TRUNCATION"]

        request_kwargs: Dict[str, Any] = {
            "method": "POST",
            "url": azure_url,
            "headers": upstream_headers,
            "json": responses_body,
            "data": None,
            "stream": True,
            "timeout": (60, None),
        }
        return request_kwargs
