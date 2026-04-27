# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.result``."""

from datus.models.hooks import RunContextWrapper
from datus.models.result import NewItem, RunResult, RunResultBase, make_usage


class TestMakeUsage:
    def test_total_defaults_to_input_plus_output(self):
        usage = make_usage(input_tokens=10, output_tokens=5)
        assert usage.total_tokens == 15

    def test_explicit_total_is_kept(self):
        usage = make_usage(input_tokens=10, output_tokens=5, total_tokens=42)
        assert usage.total_tokens == 42

    def test_cached_tokens_propagates(self):
        usage = make_usage(input_tokens=10, output_tokens=5, cached_tokens=3)
        assert usage.cached_tokens == 3


class TestRunResultToInputList:
    def test_message_round_trip(self):
        result = RunResult(new_items=[NewItem(type="message", data={"content": "hi"})])
        out = result.to_input_list()
        assert out == [{"role": "assistant", "content": "hi"}]

    def test_tool_call_and_result_round_trip(self):
        result = RunResult(
            new_items=[
                NewItem(type="tool_call", data={"call_id": "c1", "name": "run", "arguments": "{}"}),
                NewItem(type="tool_result", data={"call_id": "c1", "output": "42"}),
            ]
        )
        out = result.to_input_list()
        assert out[0] == {"type": "function_call", "call_id": "c1", "name": "run", "arguments": "{}"}
        assert out[1] == {"type": "function_call_output", "call_id": "c1", "output": "42"}

    def test_reasoning_item(self):
        result = RunResult(new_items=[NewItem(type="reasoning", data={"text": "thinking"})])
        out = result.to_input_list()
        assert out[0]["role"] == "assistant"
        assert out[0]["content"][0]["text"] == "thinking"


class TestRunResultDefaults:
    def test_defaults(self):
        result = RunResult()
        assert result.final_output is None
        assert result.new_items == []
        assert isinstance(result.context_wrapper, RunContextWrapper)
        assert result.turn_count == 0
        assert result.finish_reason == "stop"


def test_run_result_base_alias():
    assert RunResultBase is RunResult
