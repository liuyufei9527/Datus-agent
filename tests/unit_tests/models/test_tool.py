# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.tool``."""

import asyncio

import pytest
from pydantic import BaseModel, Field

from datus.models.tool import (
    FunctionTool,
    Tool,
    ToolContext,
    function_tool,
    tool_to_openai_schema,
)


class TestFunctionToolDecorator:
    def test_decorator_form_returns_function_tool(self):
        @function_tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        assert isinstance(add, FunctionTool)
        assert add.name == "add"
        assert "Add two" in add.description
        assert add.params_json_schema["type"] == "object"
        assert "a" in add.params_json_schema["properties"]
        assert "b" in add.params_json_schema["properties"]

    def test_invocation_executes_callable(self):
        @function_tool
        def echo(value: str) -> str:
            """Echo the value."""
            return value

        ctx = ToolContext(tool_call_id="x", tool_name="echo")
        result = asyncio.run(echo.on_invoke_tool(ctx, '{"value": "hello"}'))
        assert result == "hello"

    def test_invalid_json_returns_error_envelope(self):
        @function_tool
        def echo(value: str) -> str:
            return value

        ctx = ToolContext()
        result = asyncio.run(echo.on_invoke_tool(ctx, "not-json"))
        assert isinstance(result, dict)
        assert result.get("success") == 0
        assert "Invalid JSON" in result.get("error", "")

    def test_async_callable_supported(self):
        @function_tool
        async def adder(x: int, y: int) -> int:
            return x + y

        ctx = ToolContext()
        result = asyncio.run(adder.on_invoke_tool(ctx, '{"x": 1, "y": 2}'))
        assert result == 3

    def test_overrides_apply(self):
        @function_tool(name_override="aliased", description_override="custom")
        def f() -> None:
            return None

        assert f.name == "aliased"
        assert f.description == "custom"

    def test_input_model_drives_schema(self):
        class Args(BaseModel):
            q: str = Field(..., description="search term")

        @function_tool(input_model=Args)
        def search(args: Args) -> str:
            return args.q

        assert "q" in search.params_json_schema["properties"]
        result = asyncio.run(search.on_invoke_tool(ToolContext(), '{"q": "ok"}'))
        assert result == "ok"


class TestTool:
    def test_tool_alias_collapses_to_function_tool(self):
        assert Tool is FunctionTool


class TestToolToOpenAISchema:
    def test_schema_shape(self):
        @function_tool
        def f(value: str) -> str:
            """Doc."""
            return value

        schema = tool_to_openai_schema(f)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "f"
        assert schema["function"]["description"]
        assert "parameters" in schema["function"]


class TestToolContext:
    def test_default_construction(self):
        ctx = ToolContext()
        assert ctx.tool_call_id is None
        assert ctx.extra == {}

    def test_from_agent_context_collects_extra(self):
        ctx = ToolContext.from_agent_context(tool_call_id="x", run_id="r")
        assert ctx.tool_call_id == "x"
        assert ctx.extra == {"run_id": "r"}


class TestRaisingTool:
    def test_exception_inside_tool_returns_error_envelope(self):
        @function_tool
        def boom() -> str:
            raise ValueError("crash")

        result = asyncio.run(boom.on_invoke_tool(ToolContext(), "{}"))
        assert result["success"] == 0
        assert "crash" in result["error"]


@pytest.mark.parametrize(
    "args_str, expected",
    [
        ("{}", {}),
        ("", {}),
    ],
)
def test_empty_args_handled(args_str, expected):
    @function_tool
    def f(**kwargs) -> dict:
        return kwargs

    result = asyncio.run(f.on_invoke_tool(ToolContext(), args_str))
    assert result == expected
