# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.agent_loop``."""

import json
from types import SimpleNamespace

import pytest

from datus.models.agent_loop import AgentLoop, _ChatStreamCollector, _new_call_id, _short
from datus.models.session import SQLiteSession
from datus.models.tool import FunctionTool, ToolContext
from datus.models.transports.openai_chat import OpenAIChatTransport
from datus.models.transports.types import NormalizedResponse, ToolCall, Usage

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal model stub satisfying ``AgentLoop.run`` / ``run_streamed``."""

    def __init__(self, responses, stream_chunks=None):
        self._responses = list(responses)
        self._stream_chunks = stream_chunks or []
        self.model_name = "fake-model"

    async def complete_once(self, messages, tools_schema, **kwargs):
        return self._responses.pop(0)

    async def stream_once(self, messages, tools_schema, **kwargs):
        for chunk in self._stream_chunks:
            yield chunk


def _make_local_tool(name="echo", returns=None):
    async def invoke(ctx: ToolContext, args_str: str):
        return returns if returns is not None else args_str

    return FunctionTool(
        name=name,
        description="test tool",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=invoke,
    )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def test_new_call_id_unique():
    assert _new_call_id() != _new_call_id()
    assert _new_call_id("foo").startswith("foo_")


def test_short_truncation():
    assert _short("a" * 10, limit=5) == "aaaaa..."
    assert _short("hi", limit=5) == "hi"


# ---------------------------------------------------------------------------
# AgentLoop.run — single-turn, no tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_final_text_when_no_tools(tmp_path):
    response = NormalizedResponse(
        content="hello",
        tool_calls=None,
        finish_reason="stop",
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    loop = AgentLoop(
        model=_FakeModel([response]),
        transport=OpenAIChatTransport(),
        local_tools=[],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=3,
        session=None,
        db_type="snowflake",
    )
    result = await loop.run("hi")
    assert result["content"] == "hello"
    assert result["turns_used"] == 1
    assert result["sql_contexts"] == []


# ---------------------------------------------------------------------------
# AgentLoop.run — local tool dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_dispatches_local_tool_then_returns(tmp_path):
    tool = _make_local_tool(returns={"result": "ok"})
    first = NormalizedResponse(
        content="thinking",
        tool_calls=[ToolCall(id="tc-1", name=tool.name, arguments="{}")],
        finish_reason="tool_calls",
    )
    second = NormalizedResponse(content="done", tool_calls=None, finish_reason="stop")
    loop = AgentLoop(
        model=_FakeModel([first, second]),
        transport=OpenAIChatTransport(),
        local_tools=[tool],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=3,
        session=None,
        db_type="snowflake",
    )
    result = await loop.run("trigger")
    assert result["content"] == "done"
    assert result["turns_used"] == 2


# ---------------------------------------------------------------------------
# AgentLoop.run_streamed — emits ActionHistory items
# ---------------------------------------------------------------------------


def _delta_chunk(content=None, tool_call=None):
    delta = SimpleNamespace(content=content, reasoning_content=None, tool_calls=tool_call)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.mark.asyncio
async def test_run_streamed_emits_response_action(tmp_path):
    chunks = [_delta_chunk(content="hello"), _delta_chunk(content=" world")]
    response = NormalizedResponse(content="hello world", tool_calls=None, finish_reason="stop")
    loop = AgentLoop(
        model=_FakeModel([response], stream_chunks=chunks),
        transport=OpenAIChatTransport(),
        local_tools=[],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=2,
        session=None,
        db_type="",
    )
    actions = []
    async for action in loop.run_streamed("hi"):
        actions.append(action)
    # Expect at least one thinking_delta + a final response action.
    types = [a.action_type for a in actions]
    assert "thinking_delta" in types
    assert "response" in types


# ---------------------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_persists_messages_into_session(tmp_path):
    session = SQLiteSession(session_id="t1", db_path=str(tmp_path / "t1.jsonl"))
    response = NormalizedResponse(content="answer", tool_calls=None, finish_reason="stop")
    loop = AgentLoop(
        model=_FakeModel([response]),
        transport=OpenAIChatTransport(),
        local_tools=[],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=1,
        session=session,
        db_type="",
    )
    await loop.run("question")
    items = await session.get_items()
    roles = [item.get("role") for item in items]
    assert "user" in roles
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# _ChatStreamCollector
# ---------------------------------------------------------------------------


class TestChatStreamCollector:
    def test_text_delta_accumulates(self):
        collector = _ChatStreamCollector()
        actions = collector.feed(_delta_chunk(content="hi "))
        actions += collector.feed(_delta_chunk(content="there"))
        assert collector.text == "hi there"
        assert all(a.action_type == "thinking_delta" for a in actions)

    def test_finalize_returns_response_action_when_text(self):
        collector = _ChatStreamCollector()
        collector.feed(_delta_chunk(content="hi"))
        final = collector.finalize()
        assert final is not None
        assert final.action_type == "response"

    def test_finalize_returns_none_when_no_text(self):
        collector = _ChatStreamCollector()
        assert collector.finalize() is None

    def test_tool_call_args_accumulate(self):
        collector = _ChatStreamCollector()
        tc1 = SimpleNamespace(index=0, id="tc-1", function=SimpleNamespace(name="run", arguments='{"q":'))
        tc2 = SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments="1}"))
        collector.feed(_delta_chunk(tool_call=[tc1]))
        collector.feed(_delta_chunk(tool_call=[tc2]))
        calls = collector.tool_calls
        assert len(calls) == 1
        assert calls[0].arguments == '{"q":1}'


# ---------------------------------------------------------------------------
# Tool dispatch helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error(tmp_path):
    loop = AgentLoop(
        model=_FakeModel([]),
        transport=OpenAIChatTransport(),
        local_tools=[],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=1,
        session=None,
    )
    out = await loop._dispatch_tool_call(ToolCall(id="x", name="missing", arguments="{}"), call_id="x", tool_lookup={})
    payload = json.loads(out)
    assert payload["success"] == 0


@pytest.mark.asyncio
async def test_dispatch_local_tool_failure_caught(tmp_path):
    async def boom(ctx, args):
        raise RuntimeError("oops")

    tool = FunctionTool(
        name="boom",
        description="",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=boom,
    )
    loop = AgentLoop(
        model=_FakeModel([]),
        transport=OpenAIChatTransport(),
        local_tools=[tool],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=1,
        session=None,
    )
    out = await loop._dispatch_tool_call(
        ToolCall(id="x", name="boom", arguments="{}"),
        call_id="x",
        tool_lookup={"boom": ("local", tool)},
    )
    payload = json.loads(out)
    assert payload["success"] == 0
    assert "oops" in payload["error"]


# ---------------------------------------------------------------------------
# Output coercion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coerce_output_returns_string_when_output_type_is_str(tmp_path):
    response = NormalizedResponse(content="just text", tool_calls=None, finish_reason="stop")
    loop = AgentLoop(
        model=_FakeModel([response]),
        transport=OpenAIChatTransport(),
        local_tools=[],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=1,
        session=None,
    )
    result = await loop.run("x")
    assert result["content"] == "just text"


# ---------------------------------------------------------------------------
# Interrupt
# ---------------------------------------------------------------------------


class _Interrupt:
    def __init__(self, value=True):
        self.is_interrupted = value


def test_raise_if_interrupted_raises():
    loop = AgentLoop(
        model=_FakeModel([]),
        transport=OpenAIChatTransport(),
        local_tools=[],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=1,
        session=None,
        interrupt_controller=_Interrupt(True),
    )
    from datus.cli.execution_state import ExecutionInterrupted

    with pytest.raises(ExecutionInterrupted):
        loop._raise_if_interrupted()


def test_raise_if_interrupted_callable_flag():
    flag = {"value": False}

    class CallableInterrupt:
        @property
        def is_interrupted(self):
            return lambda: flag["value"]

    flag["value"] = True
    loop = AgentLoop(
        model=_FakeModel([]),
        transport=OpenAIChatTransport(),
        local_tools=[],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=1,
        session=None,
        interrupt_controller=CallableInterrupt(),
    )
    from datus.cli.execution_state import ExecutionInterrupted

    with pytest.raises(ExecutionInterrupted):
        loop._raise_if_interrupted()


def test_raise_if_interrupted_noop_when_no_controller():
    loop = AgentLoop(
        model=_FakeModel([]),
        transport=OpenAIChatTransport(),
        local_tools=[],
        mcp_servers={},
        instruction="",
        output_type=str,
        max_turns=1,
        session=None,
    )
    loop._raise_if_interrupted()  # no exception
