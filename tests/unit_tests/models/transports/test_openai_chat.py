# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.transports.openai_chat``."""

from types import SimpleNamespace

from datus.models.transports.openai_chat import OpenAIChatTransport


def _make_choice(content=None, tool_calls=None, reasoning_content=None, finish_reason="stop"):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(message=message, finish_reason=finish_reason)


def _make_response(choice, usage=None):
    return SimpleNamespace(choices=[choice], usage=usage)


class TestConvertMessages:
    def test_function_call_pair_collapses_to_assistant_tool_calls(self):
        transport = OpenAIChatTransport()
        messages = [
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c1", "name": "run", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ]
        out = transport.convert_messages(messages)
        assert out[0] == {"role": "user", "content": "hi"}
        assistant = out[1]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"][0]["id"] == "c1"
        assert assistant["tool_calls"][0]["function"]["name"] == "run"
        assert out[2]["role"] == "tool"
        assert out[2]["tool_call_id"] == "c1"
        assert out[2]["content"] == "42"

    def test_assistant_block_list_collapses_to_text(self):
        transport = OpenAIChatTransport()
        out = transport.convert_messages([{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}])
        assert out[0]["content"] == "hello"

    def test_dangling_function_call_flushed_at_end(self):
        transport = OpenAIChatTransport()
        out = transport.convert_messages([{"type": "function_call", "call_id": "c2", "name": "n", "arguments": "{}"}])
        assert out[-1]["role"] == "assistant"
        assert out[-1]["tool_calls"][0]["id"] == "c2"


class TestConvertTools:
    def test_none_returns_none(self):
        transport = OpenAIChatTransport()
        assert transport.convert_tools(None) is None
        assert transport.convert_tools([]) is None

    def test_passthrough(self):
        transport = OpenAIChatTransport()
        tools = [{"type": "function", "function": {"name": "t"}}]
        assert transport.convert_tools(tools) == tools


class TestBuildKwargs:
    def test_includes_optional_params_only_when_set(self):
        transport = OpenAIChatTransport()
        kwargs = transport.build_kwargs(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            temperature=0.5,
            top_p=None,  # explicitly None should be skipped
        )
        assert kwargs["model"] == "gpt-4.1"
        assert kwargs["temperature"] == 0.5
        assert "top_p" not in kwargs
        assert "tools" not in kwargs

    def test_tools_and_choice(self):
        transport = OpenAIChatTransport()
        tools = [{"type": "function", "function": {"name": "t"}}]
        kwargs = transport.build_kwargs(
            model="gpt-4.1",
            messages=[],
            tools=tools,
            tool_choice="auto",
        )
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


class TestNormalizeResponse:
    def test_text_only_response(self):
        transport = OpenAIChatTransport()
        choice = _make_choice(content="hello")
        usage = SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        )
        norm = transport.normalize_response(_make_response(choice, usage))
        assert norm.content == "hello"
        assert norm.tool_calls is None
        assert norm.finish_reason == "stop"
        assert norm.usage.total_tokens == 8
        assert norm.usage.cached_tokens == 2

    def test_tool_call_response(self):
        transport = OpenAIChatTransport()
        tool_call = SimpleNamespace(
            id="call-x",
            function=SimpleNamespace(name="run", arguments='{"q":1}'),
        )
        choice = _make_choice(tool_calls=[tool_call], finish_reason="tool_calls")
        norm = transport.normalize_response(_make_response(choice))
        assert norm.tool_calls is not None
        assert norm.tool_calls[0].id == "call-x"
        assert norm.tool_calls[0].name == "run"
        assert norm.finish_reason == "tool_calls"

    def test_reasoning_content_surfaced(self):
        transport = OpenAIChatTransport()
        choice = _make_choice(content="", reasoning_content="thinking…")
        norm = transport.normalize_response(_make_response(choice))
        assert norm.reasoning == "thinking…"

    def test_finish_reason_default_when_missing(self):
        transport = OpenAIChatTransport()
        choice = _make_choice(content="x", finish_reason=None)
        norm = transport.normalize_response(_make_response(choice))
        assert norm.finish_reason == "stop"


class TestApiMode:
    def test_api_mode_constant(self):
        assert OpenAIChatTransport().api_mode == "chat_completions"
