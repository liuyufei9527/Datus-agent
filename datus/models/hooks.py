# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Lifecycle hooks for the Datus agent loop.

Replaces ``agents.lifecycle.{RunHooks,AgentHooks}`` from the
openai-agents-sdk.  Subclasses override the methods they care about; the
agent loop awaits each callback at the same lifecycle points as the SDK
did so existing skill-validators / permission gates / generation hooks
keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datus.models.tool import FunctionTool


@dataclass
class Agent:
    """Lightweight stand-in for ``agents.Agent``.

    Most call sites use ``Agent`` as nothing more than a name container
    passed into hook callbacks. The full SDK class wraps tools, model
    settings, instructions, etc.; we keep only the surface that other
    Datus modules read so the import path is preserved.
    """

    name: str = "default_agent"
    instructions: Optional[str] = None
    tools: List[FunctionTool] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContextWrapper:
    """Mirrors the SDK's ``RunContextWrapper`` for hook callbacks.

    ``context`` carries arbitrary user state (the value the caller
    passes to ``Runner.run(... context=...)``).  ``usage`` accumulates
    token counts across the run.
    """

    context: Any = None
    usage: Optional["Usage"] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Aggregate token usage across all turns of one agent run."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens


@dataclass
class AgentHookContext(RunContextWrapper):
    """Specialised context passed into ``on_agent_start`` / ``on_agent_end``."""


class RunHooks:
    """Base class for run-level lifecycle hooks.

    Subclass and override only the methods you need.  The default
    implementations are no-ops; the agent loop awaits the callbacks
    serially, so blocking inside a hook will stall the whole turn.
    """

    async def on_llm_start(
        self,
        context: RunContextWrapper,
        agent: Any,
        system_prompt: Optional[str],
        input_items: List[Dict[str, Any]],
    ) -> None:
        """Fired immediately before every LLM call."""

    async def on_llm_end(
        self,
        context: RunContextWrapper,
        agent: Any,
        response: Any,
    ) -> None:
        """Fired immediately after every LLM call (success or graceful stop)."""

    async def on_agent_start(self, context: AgentHookContext, agent: Any) -> None:
        """Fired once per agent activation (before the first LLM turn)."""

    async def on_agent_end(self, context: AgentHookContext, agent: Any, output: Any) -> None:
        """Fired when the agent yields its final output."""

    async def on_handoff(
        self,
        context: RunContextWrapper,
        from_agent: Any,
        to_agent: Any,
    ) -> None:
        """Fired when control passes from one agent to another."""

    async def on_tool_start(
        self,
        context: RunContextWrapper,
        agent: Any,
        tool: FunctionTool,
    ) -> None:
        """Fired before each tool invocation."""

    async def on_tool_end(
        self,
        context: RunContextWrapper,
        agent: Any,
        tool: FunctionTool,
        result: str,
    ) -> None:
        """Fired after each tool invocation (success or returned-error)."""


# ``AgentHooks`` and ``RunHooks`` are aliases in the SDK because they
# share the same callback shape; keep that here so existing subclasses
# (``ValidationHook``, permission hooks, generation hooks) compile
# unchanged once they switch their import path.
AgentHooks = RunHooks


class CompositeHooks(RunHooks):
    """Fan a lifecycle event out to a list of child hook objects."""

    def __init__(self, hooks: List[RunHooks]) -> None:
        self._hooks = list(hooks)

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            await h.on_llm_start(*args, **kwargs)

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            await h.on_llm_end(*args, **kwargs)

    async def on_agent_start(self, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            await h.on_agent_start(*args, **kwargs)

    async def on_agent_end(self, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            await h.on_agent_end(*args, **kwargs)

    async def on_handoff(self, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            await h.on_handoff(*args, **kwargs)

    async def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            await h.on_tool_start(*args, **kwargs)

    async def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            await h.on_tool_end(*args, **kwargs)
