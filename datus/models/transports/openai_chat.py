# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OpenAI Chat Completions transport.

Wraps the openai SDK's ``chat.completions`` endpoint. Same wire format
serves OpenAI itself plus every OpenAI-compatible provider (DeepSeek,
Kimi/Moonshot, DashScope/Qwen, GLM, MiniMax, OpenRouter), so this
transport is the workhorse — most adapters in
:data:`MODEL_TYPE_MAP` ultimately drive it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from datus.models.transports.base import ProviderTransport
from datus.models.transports.types import NormalizedResponse, ToolCall, Usage, build_tool_call

_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",  # legacy
    "content_filter": "content_filter",
}


class OpenAIChatTransport(ProviderTransport):
    """Format conversion + normalisation for ``/v1/chat/completions``."""

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
        """Convert Datus message items into OpenAI chat-message dicts.

        Datus stores ``function_call`` / ``function_call_output`` items
        in the session (matching the SDK's resume format). The OpenAI
        Chat Completions endpoint expects ``role: assistant`` with
        ``tool_calls`` and ``role: tool`` with ``tool_call_id``, so we
        translate on the fly here.
        """
        out: List[Dict[str, Any]] = []
        pending_tool_calls: List[Dict[str, Any]] = []
        for msg in messages:
            msg_type = msg.get("type")
            role = msg.get("role")
            if msg_type == "function_call":
                pending_tool_calls.append(
                    {
                        "id": msg.get("call_id") or msg.get("id"),
                        "type": "function",
                        "function": {
                            "name": msg.get("name", ""),
                            "arguments": msg.get("arguments", "{}"),
                        },
                    }
                )
                continue
            if msg_type == "function_call_output":
                # Flush any pending assistant tool_calls first.
                if pending_tool_calls:
                    out.append({"role": "assistant", "content": None, "tool_calls": pending_tool_calls})
                    pending_tool_calls = []
                output = msg.get("output", "")
                if not isinstance(output, str):
                    output = str(output)
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("call_id"),
                        "content": output,
                    }
                )
                continue
            if pending_tool_calls:
                out.append({"role": "assistant", "content": None, "tool_calls": pending_tool_calls})
                pending_tool_calls = []
            if role:
                clone = dict(msg)
                # If assistant content is a list of blocks (Anthropic-shaped),
                # collapse it to its text part since OpenAI wants a string.
                content = clone.get("content")
                if isinstance(content, list):
                    parts = [
                        block.get("text", "") for block in content if isinstance(block, dict) and block.get("text")
                    ]
                    clone["content"] = "\n".join(parts)
                out.append(clone)
        if pending_tool_calls:
            out.append({"role": "assistant", "content": None, "tool_calls": pending_tool_calls})
        return out

    def convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        return list(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": self.convert_messages(messages),
        }
        formatted_tools = self.convert_tools(tools)
        if formatted_tools:
            kwargs["tools"] = formatted_tools
            tool_choice = params.pop("tool_choice", None)
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stop",
            "seed",
            "stream_options",
            "extra_headers",
            "extra_body",
        ):
            if key in params and params[key] is not None:
                kwargs[key] = params[key]
        return kwargs

    def normalize_response(self, response: Any, **kwargs: Any) -> NormalizedResponse:
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        finish_reason = self.map_finish_reason(getattr(choice, "finish_reason", "stop") or "stop")

        tool_calls: Optional[List[ToolCall]] = None
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None)
                tool_calls.append(
                    build_tool_call(
                        id=getattr(tc, "id", None),
                        name=getattr(fn, "name", "") if fn else "",
                        arguments=getattr(fn, "arguments", "{}") if fn else "{}",
                    )
                )
            if not finish_reason or finish_reason == "stop":
                finish_reason = "tool_calls"

        usage_obj = getattr(response, "usage", None)
        usage: Optional[Usage] = None
        if usage_obj is not None:
            cached = 0
            prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
            if prompt_details is not None:
                cached = int(getattr(prompt_details, "cached_tokens", 0) or 0)
            usage = Usage(
                prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
                cached_tokens=cached,
            )

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            reasoning=reasoning,
            usage=usage,
        )

    def map_finish_reason(self, raw_reason: str) -> str:
        return _FINISH_REASON_MAP.get(raw_reason, "stop")
