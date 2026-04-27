# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.hooks``."""

import asyncio

from datus.models.hooks import (
    Agent,
    AgentHookContext,
    AgentHooks,
    CompositeHooks,
    RunContextWrapper,
    RunHooks,
    Usage,
)


class TestAgent:
    def test_default_agent(self):
        agent = Agent()
        assert agent.name == "default_agent"
        assert agent.tools == []
        assert agent.extra == {}

    def test_named_agent(self):
        agent = Agent(name="custom", instructions="be brief")
        assert agent.name == "custom"
        assert agent.instructions == "be brief"


class TestUsage:
    def test_add_accumulates(self):
        a = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        b = Usage(requests=1, input_tokens=2, output_tokens=3, total_tokens=5)
        a.add(b)
        assert a.requests == 2
        assert a.input_tokens == 12
        assert a.output_tokens == 8
        assert a.total_tokens == 20


class TestRunHooks:
    def test_default_callbacks_are_noops(self):
        hooks = RunHooks()
        ctx = RunContextWrapper(usage=Usage())
        # All callbacks resolve to None without raising.
        asyncio.run(hooks.on_llm_start(ctx, None, None, []))
        asyncio.run(hooks.on_llm_end(ctx, None, None))
        asyncio.run(hooks.on_agent_start(AgentHookContext(usage=Usage()), None))
        asyncio.run(hooks.on_agent_end(AgentHookContext(usage=Usage()), None, None))
        asyncio.run(hooks.on_handoff(ctx, None, None))
        asyncio.run(hooks.on_tool_start(ctx, None, None))
        asyncio.run(hooks.on_tool_end(ctx, None, None, ""))


class TestCompositeHooks:
    def test_fans_out_to_children(self):
        events = []

        class Recorder(RunHooks):
            def __init__(self, label: str) -> None:
                self.label = label

            async def on_llm_start(self, *args, **kwargs):
                events.append((self.label, "llm_start"))

            async def on_tool_end(self, *args, **kwargs):
                events.append((self.label, "tool_end"))

        composite = CompositeHooks([Recorder("a"), Recorder("b")])
        asyncio.run(composite.on_llm_start(RunContextWrapper(), None, None, []))
        asyncio.run(composite.on_tool_end(RunContextWrapper(), None, None, ""))
        assert events == [
            ("a", "llm_start"),
            ("b", "llm_start"),
            ("a", "tool_end"),
            ("b", "tool_end"),
        ]


def test_agent_hooks_alias():
    assert AgentHooks is RunHooks
