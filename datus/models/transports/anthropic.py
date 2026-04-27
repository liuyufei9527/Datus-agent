# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Anthropic Messages transport.

Translates between OpenAI-style chat-message dicts (the in-house format
the rest of Datus speaks) and the Anthropic Messages API.

Heavily inspired by hermes/agent/anthropic_adapter.py — kept compact
because Datus does not (yet) need PKCE OAuth, Bedrock chaining, or
adaptive thinking budgets. Those can be added as the openai-only path
grows.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from datus.models.transports.base import ProviderTransport
from datus.models.transports.types import NormalizedResponse, ToolCall, Usage, build_tool_call

_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def _sanitize_tool_id(tool_id: Optional[str]) -> str:
    """Anthropic accepts ``[a-zA-Z0-9_-]+`` for tool_use_id."""
    if not tool_id:
        return "tool_0"
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
    return cleaned or "tool_0"


def _normalise_model(model: str) -> str:
    """OpenRouter-style ``anthropic/claude-...`` → bare model id."""
    return model.split("/", 1)[1] if model.lower().startswith("anthropic/") else model


class AnthropicTransport(ProviderTransport):
    """Format conversion + normalisation for the Messages API."""

    @property
    def api_mode(self) -> str:
        return "anthropic_messages"

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def convert_messages(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        """Return ``(system_prompt, anthropic_messages)``.

        OpenAI-style ``role=tool`` results are rewritten as ``role=user``
        messages whose content is a list of ``tool_result`` blocks, and
        assistant ``tool_calls`` become ``tool_use`` blocks. Empty roles
        get a placeholder string because Anthropic rejects empties.
        """
        system: Optional[Any] = None
        out: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            msg_type = msg.get("type")
            content = msg.get("content")

            if role == "system":
                system = content if isinstance(content, (str, list)) else str(content)
                continue

            if msg_type == "function_call":
                blocks = [
                    {
                        "type": "tool_use",
                        "id": _sanitize_tool_id(msg.get("call_id") or msg.get("id")),
                        "name": msg.get("name", ""),
                        "input": _safe_json_loads(msg.get("arguments", "{}")),
                    }
                ]
                out.append({"role": "assistant", "content": blocks})
                continue

            if msg_type == "function_call_output":
                output_text = msg.get("output", "")
                if not isinstance(output_text, str):
                    output_text = json.dumps(output_text)
                self._append_tool_result(
                    out,
                    tool_use_id=_sanitize_tool_id(msg.get("call_id")),
                    text=output_text or "(no output)",
                )
                continue

            if role == "assistant":
                blocks: List[Dict[str, Any]] = []
                if isinstance(content, str) and content:
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            ptype = part.get("type")
                            if ptype == "text" and part.get("text"):
                                blocks.append({"type": "text", "text": part["text"]})
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc.get("function") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": _sanitize_tool_id(tc.get("id")),
                            "name": fn.get("name", ""),
                            "input": _safe_json_loads(fn.get("arguments", "{}")),
                        }
                    )
                if not blocks:
                    blocks = [{"type": "text", "text": "(empty)"}]
                out.append({"role": "assistant", "content": blocks})
                continue

            if role == "tool":
                output_text = content if isinstance(content, str) else json.dumps(content)
                self._append_tool_result(
                    out,
                    tool_use_id=_sanitize_tool_id(msg.get("tool_call_id")),
                    text=output_text or "(no output)",
                )
                continue

            # Default: user message.
            if isinstance(content, list):
                blocks = [
                    {"type": "text", "text": part.get("text", "")}
                    for part in content
                    if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
                ]
                if not blocks:
                    blocks = [{"type": "text", "text": "(empty message)"}]
                out.append({"role": "user", "content": blocks})
            else:
                text = str(content or "(empty message)")
                out.append({"role": "user", "content": text})

        out = self._enforce_alternation(out)
        return system, out

    @staticmethod
    def _append_tool_result(out: List[Dict[str, Any]], *, tool_use_id: str, text: str) -> None:
        block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
        if (
            out
            and out[-1]["role"] == "user"
            and isinstance(out[-1]["content"], list)
            and out[-1]["content"]
            and isinstance(out[-1]["content"][0], dict)
            and out[-1]["content"][0].get("type") == "tool_result"
        ):
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    @staticmethod
    def _enforce_alternation(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fixed: List[Dict[str, Any]] = []
        for msg in messages:
            if fixed and fixed[-1]["role"] == msg["role"]:
                # Merge into the previous same-role message.
                prev_content = fixed[-1]["content"]
                curr_content = msg["content"]
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(curr_content, str):
                    curr_content = [{"type": "text", "text": curr_content}]
                fixed[-1]["content"] = list(prev_content) + list(curr_content)
            else:
                fixed.append(msg)
        return fixed

    def convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not tools:
            return []
        result: List[Dict[str, Any]] = []
        for entry in tools:
            fn = entry.get("function") or {}
            result.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return result

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        system, anthropic_messages = self.convert_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": _normalise_model(model),
            "messages": anthropic_messages,
            "max_tokens": int(params.get("max_tokens") or 4096),
        }
        if system is not None:
            kwargs["system"] = system
        anthropic_tools = self.convert_tools(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            tool_choice = params.get("tool_choice")
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        for key in ("temperature", "top_p", "stop_sequences", "metadata", "extra_headers"):
            value = params.get(key)
            if value is not None:
                kwargs[key] = value
        thinking = params.get("thinking")
        if thinking:
            kwargs["thinking"] = thinking
        return kwargs

    def normalize_response(self, response: Any, **kwargs: Any) -> NormalizedResponse:
        content_blocks = getattr(response, "content", None) or []
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type == "thinking":
                reasoning_parts.append(getattr(block, "thinking", "") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    build_tool_call(
                        id=getattr(block, "id", None),
                        name=getattr(block, "name", "") or "",
                        arguments=getattr(block, "input", {}) or {},
                    )
                )
        finish_raw = getattr(response, "stop_reason", None) or "end_turn"
        finish_reason = self.map_finish_reason(finish_raw)
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        usage_obj = getattr(response, "usage", None)
        usage: Optional[Usage] = None
        if usage_obj is not None:
            input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
            cached = int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_tokens=cached,
            )

        return NormalizedResponse(
            content="".join(text_parts) or None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            reasoning="\n".join(reasoning_parts) or None,
            usage=usage,
        )

    def map_finish_reason(self, raw_reason: str) -> str:
        return _FINISH_REASON_MAP.get(raw_reason, "stop")


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
