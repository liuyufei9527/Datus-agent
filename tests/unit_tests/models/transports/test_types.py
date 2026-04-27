# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.transports.types``."""

import json

from datus.models.transports.types import (
    NormalizedResponse,
    ToolCall,
    Usage,
    build_tool_call,
    map_finish_reason,
)


class TestBuildToolCall:
    def test_dict_arguments_are_json_encoded(self):
        call = build_tool_call(id="call-1", name="run", arguments={"q": "x"})
        assert call.id == "call-1"
        assert call.name == "run"
        assert json.loads(call.arguments) == {"q": "x"}
        assert call.provider_data is None

    def test_string_arguments_are_passed_through(self):
        call = build_tool_call(id=None, name="t", arguments='{"k":1}')
        assert call.id is None
        assert call.arguments == '{"k":1}'

    def test_provider_fields_collected_into_provider_data(self):
        call = build_tool_call(id="x", name="t", arguments="{}", call_id="fc_1", response_item_id="r_1")
        assert call.provider_data == {"call_id": "fc_1", "response_item_id": "r_1"}


class TestMapFinishReason:
    def test_known_reason_maps(self):
        mapping = {"end_turn": "stop", "tool_use": "tool_calls"}
        assert map_finish_reason("end_turn", mapping) == "stop"

    def test_unknown_reason_falls_back_to_stop(self):
        assert map_finish_reason("unexpected", {}) == "stop"

    def test_none_reason_falls_back_to_stop(self):
        assert map_finish_reason(None, {}) == "stop"


class TestNormalizedResponse:
    def test_minimal_normalised_response(self):
        response = NormalizedResponse(content="hello", tool_calls=None, finish_reason="stop")
        assert response.content == "hello"
        assert response.tool_calls is None
        assert response.reasoning is None
        assert response.usage is None
        assert response.provider_data is None

    def test_with_tool_calls_and_usage(self):
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        call = ToolCall(id="call-x", name="run", arguments="{}")
        response = NormalizedResponse(
            content=None,
            tool_calls=[call],
            finish_reason="tool_calls",
            usage=usage,
        )
        assert response.tool_calls[0].name == "run"
        assert response.usage.total_tokens == 15
