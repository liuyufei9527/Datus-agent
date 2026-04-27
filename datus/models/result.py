# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Run result containers for the Datus agent loop.

Replaces ``agents.RunResult`` / ``agents.RunResultBase`` from the
openai-agents-sdk.  ``extract_sql_contexts`` walks ``new_items`` to
extract SQL execution traces; the rest of the project consumes
``final_output`` and ``context_wrapper.usage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from datus.models.hooks import RunContextWrapper, Usage

ItemType = Literal["message", "tool_call", "tool_result", "reasoning"]


@dataclass
class NewItem:
    """A normalised piece of conversation history produced during a run.

    ``type`` discriminates how callers should interpret ``data``:

    * ``"message"`` — assistant text. ``data["content"]`` is a string.
    * ``"tool_call"`` — function call. ``data`` mirrors the OpenAI
      ``function_call`` shape: ``{"call_id", "name", "arguments"}``.
    * ``"tool_result"`` — function output. ``data``: ``{"call_id",
      "output"}``.
    * ``"reasoning"`` — model-internal thinking trace. ``data``:
      ``{"text"}``.
    """

    type: ItemType
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Final result of an agent run."""

    final_output: Any = None
    new_items: List[NewItem] = field(default_factory=list)
    context_wrapper: RunContextWrapper = field(default_factory=RunContextWrapper)
    turn_count: int = 0
    finish_reason: str = "stop"

    def to_input_list(self) -> List[Dict[str, Any]]:
        """Render the run as the chat history for a follow-up call."""
        out: List[Dict[str, Any]] = []
        for item in self.new_items:
            if item.type == "message":
                out.append({"role": "assistant", "content": item.data.get("content", "")})
            elif item.type == "tool_call":
                out.append(
                    {
                        "type": "function_call",
                        "call_id": item.data.get("call_id"),
                        "name": item.data.get("name"),
                        "arguments": item.data.get("arguments", "{}"),
                    }
                )
            elif item.type == "tool_result":
                out.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.data.get("call_id"),
                        "output": item.data.get("output", ""),
                    }
                )
            elif item.type == "reasoning":
                out.append({"role": "assistant", "content": [{"type": "thinking", "text": item.data.get("text", "")}]})
        return out


# Backwards-compatible alias for the SDK type name.
RunResultBase = RunResult


def make_usage(
    *,
    requests: int = 1,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: Optional[int] = None,
    cached_tokens: int = 0,
) -> Usage:
    """Convenience constructor used by the transports' ``normalize_response``."""
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return Usage(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
    )
