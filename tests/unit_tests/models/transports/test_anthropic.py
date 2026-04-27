# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.transports.anthropic``."""

from types import SimpleNamespace

from datus.models.transports.anthropic import AnthropicTransport


class TestConvertMessages:
    def test_system_extracted_from_messages(self):
        transport = AnthropicTransport()
        system, msgs = transport.convert_messages(
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        )
        assert system == "be brief"
        assert msgs[0]["role"] == "user"

    def test_function_call_becomes_tool_use(self):
        transport = AnthropicTransport()
        _, msgs = transport.convert_messages(
            [{"type": "function_call", "call_id": "c1", "name": "run", "arguments": '{"q": 1}'}]
        )
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"][0]["type"] == "tool_use"
        assert msgs[0]["content"][0]["input"] == {"q": 1}

    def test_function_call_output_becomes_tool_result(self):
        transport = AnthropicTransport()
        _, msgs = transport.convert_messages(
            [
                {"type": "function_call", "call_id": "c1", "name": "run", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "42"},
            ]
        )
        # Anthropic expects user message with tool_result block.
        tool_result_msg = msgs[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["content"] == "42"

    def test_assistant_role_with_tool_calls(self):
        transport = AnthropicTransport()
        _, msgs = transport.convert_messages(
            [
                {
                    "role": "assistant",
                    "content": "thinking...",
                    "tool_calls": [{"id": "tc-1", "function": {"name": "run", "arguments": '{"a":1}'}}],
                }
            ]
        )
        assistant_blocks = msgs[0]["content"]
        # Text + tool_use blocks
        kinds = [b["type"] for b in assistant_blocks]
        assert "text" in kinds
        assert "tool_use" in kinds

    def test_empty_assistant_replaced_with_placeholder(self):
        transport = AnthropicTransport()
        _, msgs = transport.convert_messages([{"role": "assistant", "content": ""}])
        assert msgs[0]["content"][0]["type"] == "text"
        assert "(empty)" in msgs[0]["content"][0]["text"]

    def test_tool_role_legacy_input_format(self):
        transport = AnthropicTransport()
        _, msgs = transport.convert_messages([{"role": "tool", "tool_call_id": "tc-2", "content": "ok"}])
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["tool_use_id"] == "tc-2"

    def test_consecutive_user_messages_merge(self):
        transport = AnthropicTransport()
        _, msgs = transport.convert_messages([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
        assert len(msgs) == 1
        # Strings get wrapped into a list when merging.
        assert any(isinstance(p, dict) and p.get("text") == "a" for p in msgs[0]["content"])


class TestConvertTools:
    def test_none_returns_empty_list(self):
        assert AnthropicTransport().convert_tools(None) == []

    def test_translates_openai_tools_to_anthropic_input_schema(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "do",
                    "description": "do it",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = AnthropicTransport().convert_tools(tools)
        assert result[0]["name"] == "do"
        assert result[0]["input_schema"]["type"] == "object"


class TestBuildKwargs:
    def test_kwargs_include_max_tokens_default(self):
        transport = AnthropicTransport()
        kwargs = transport.build_kwargs(model="claude-sonnet-4-5", messages=[])
        assert kwargs["max_tokens"] == 4096
        assert kwargs["model"] == "claude-sonnet-4-5"

    def test_anthropic_prefix_is_stripped(self):
        transport = AnthropicTransport()
        kwargs = transport.build_kwargs(model="anthropic/claude-haiku-4-5", messages=[])
        assert kwargs["model"] == "claude-haiku-4-5"

    def test_thinking_payload_passes_through(self):
        transport = AnthropicTransport()
        kwargs = transport.build_kwargs(
            model="claude-sonnet-4-5",
            messages=[],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        assert kwargs["thinking"]["budget_tokens"] == 1024


class TestNormalizeResponse:
    def test_text_only_block(self):
        transport = AnthropicTransport()
        block = SimpleNamespace(type="text", text="hi")
        usage = SimpleNamespace(input_tokens=3, output_tokens=2, cache_read_input_tokens=1)
        response = SimpleNamespace(content=[block], stop_reason="end_turn", usage=usage)
        norm = transport.normalize_response(response)
        assert norm.content == "hi"
        assert norm.finish_reason == "stop"
        assert norm.usage.cached_tokens == 1
        assert norm.tool_calls is None

    def test_tool_use_block_translates_to_toolcall(self):
        transport = AnthropicTransport()
        thinking = SimpleNamespace(type="thinking", thinking="r")
        tool = SimpleNamespace(type="tool_use", id="tu-1", name="do", input={"x": 1})
        response = SimpleNamespace(content=[thinking, tool], stop_reason="tool_use", usage=None)
        norm = transport.normalize_response(response)
        assert norm.reasoning == "r"
        assert norm.finish_reason == "tool_calls"
        assert norm.tool_calls[0].id == "tu-1"
        assert norm.tool_calls[0].name == "do"


class TestApiMode:
    def test_api_mode_constant(self):
        assert AnthropicTransport().api_mode == "anthropic_messages"
