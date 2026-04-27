# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared response types for the provider transport layer.

Every transport (Chat Completions, Anthropic Messages, Gemini, Codex
Responses) normalises its raw provider response into the dataclasses
defined here, so the provider-agnostic ``agent_loop`` can branch only on
``finish_reason`` and ``tool_calls``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A normalised tool/function call from any provider.

    ``id`` is the protocol's canonical identifier carried back to the
    provider in the ``tool_call_id`` / ``tool_use_id`` field of the
    follow-up message; the ``agent_loop`` synthesises one if the provider
    omitted it. ``provider_data`` carries protocol-specific metadata
    (Anthropic ``cache_control``, Gemini thought signatures, ...).
    """

    id: Optional[str]
    name: str
    arguments: str  # JSON string
    provider_data: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class NormalizedResponse:
    """Provider-agnostic single-turn response."""

    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    finish_reason: str  # "stop" | "tool_calls" | "length" | "content_filter"
    reasoning: Optional[str] = None
    usage: Optional[Usage] = None
    provider_data: Optional[Dict[str, Any]] = field(default=None, repr=False)


def build_tool_call(
    id: Optional[str],
    name: str,
    arguments: Any,
    **provider_fields: Any,
) -> ToolCall:
    """Construct a :class:`ToolCall`, JSON-serialising dict arguments."""
    args_str = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
    pd = dict(provider_fields) if provider_fields else None
    return ToolCall(id=id, name=name, arguments=args_str, provider_data=pd)


def map_finish_reason(reason: Optional[str], mapping: Dict[str, str]) -> str:
    """Translate a provider stop reason via *mapping*; fallback to "stop"."""
    if reason is None:
        return "stop"
    return mapping.get(reason, "stop")
