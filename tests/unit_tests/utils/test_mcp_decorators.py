# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/mcp_decorators.py."""

import pytest

# Import base first to break circular dependency
import datus.tools.func_tool.base  # noqa: F401

from datus.utils.mcp_decorators import (
    MCPToolConfig,
    ToolClassConfig,
    _GLOBAL_TOOL_REGISTRY,
    clear_tool_registry,
    create_dynamic_tool_wrapper,
    create_static_tool_wrapper,
    get_mcp_tools,
    get_tool_registry,
    mcp_tool,
    mcp_tool_class,
)


class TestMcpToolDecorator:
    def test_marks_method(self):
        @mcp_tool()
        def my_tool(self, param: str):
            """My tool description."""
            return param

        assert hasattr(my_tool, "_mcp_config")
        assert isinstance(my_tool._mcp_config, MCPToolConfig)
        assert my_tool._mcp_config.availability_check is None

    def test_marks_method_with_availability(self):
        @mcp_tool(availability_check="has_feature")
        def my_tool(self, param: str):
            return param

        assert my_tool._mcp_config.availability_check == "has_feature"

    def test_function_still_callable(self):
        @mcp_tool()
        def my_tool(self, x: int):
            return x + 1

        assert my_tool(None, 5) == 6


class TestGetMcpTools:
    def test_gets_decorated_methods(self):
        class MyTools:
            @mcp_tool()
            def tool_a(self):
                pass

            @mcp_tool(availability_check="has_x")
            def tool_b(self, query: str):
                pass

            def not_a_tool(self):
                pass

        tools = get_mcp_tools(MyTools)
        names = [t[0] for t in tools]
        assert "tool_a" in names
        assert "tool_b" in names
        assert "not_a_tool" not in names

    def test_empty_class(self):
        class Empty:
            pass

        tools = get_mcp_tools(Empty)
        assert tools == []


class TestCreateDynamicToolWrapper:
    def test_basic_wrapper(self):
        class MyTools:
            @mcp_tool()
            def my_method(self, query: str):
                """Search for stuff."""
                return {"success": 1, "result": query}

        class FakeContext:
            has_tools = True

        ctx = FakeContext()
        instance = MyTools()
        ctx.my_tools = instance

        def context_getter():
            return ctx

        def format_result(r):
            return {"success": 1, "result": str(r)}

        tools = get_mcp_tools(MyTools)
        name, method, config = tools[0]

        wrapper = create_dynamic_tool_wrapper(
            method_name=name,
            method=method,
            config=config,
            context_getter=context_getter,
            instance_attr="my_tools",
            availability_attr="has_tools",
            format_result=format_result,
        )

        assert wrapper.__name__ == "my_method"
        assert wrapper.__doc__ == "Search for stuff."
        result = wrapper(query="test")
        assert result["success"] == 1

    def test_tool_not_available(self):
        @mcp_tool()
        def tool_fn(self, q: str):
            return q

        class FakeContext:
            has_tools = False

        def context_getter():
            return FakeContext()

        wrapper = create_dynamic_tool_wrapper(
            method_name="tool_fn",
            method=tool_fn,
            config=MCPToolConfig(),
            context_getter=context_getter,
            instance_attr="my_tools",
            availability_attr="has_tools",
            format_result=lambda r: r,
        )

        result = wrapper(q="test")
        assert result["success"] == 0
        assert "not available" in result["error"]

    def test_feature_check_fails(self):
        @mcp_tool(availability_check="has_feature")
        def tool_fn(self, q: str):
            return q

        class FakeInstance:
            has_feature = False

        class FakeContext:
            has_tools = True
            my_tools = FakeInstance()

        def context_getter():
            return FakeContext()

        wrapper = create_dynamic_tool_wrapper(
            method_name="tool_fn",
            method=tool_fn,
            config=MCPToolConfig(availability_check="has_feature"),
            context_getter=context_getter,
            instance_attr="my_tools",
            availability_attr="has_tools",
            format_result=lambda r: r,
        )

        result = wrapper(q="test")
        assert result["success"] == 0
        assert "Feature not available" in result["error"]


class TestCreateStaticToolWrapper:
    def test_basic_wrapper(self):
        class MyTools:
            @mcp_tool()
            def my_method(self, query: str):
                """Do something."""
                return {"result": query}

        instance = MyTools()

        def format_result(r):
            return {"success": 1, "result": str(r)}

        wrapper = create_static_tool_wrapper(
            method_name="my_method",
            bound_method=instance.my_method,
            config=MCPToolConfig(),
            format_result=format_result,
        )

        assert wrapper.__name__ == "my_method"
        result = wrapper(query="hello")
        assert result["success"] == 1

    def test_feature_check_fails(self):
        class MyTools:
            has_feature = False

            @mcp_tool(availability_check="has_feature")
            def my_method(self, q: str):
                return q

        instance = MyTools()
        wrapper = create_static_tool_wrapper(
            method_name="my_method",
            bound_method=instance.my_method,
            config=MCPToolConfig(availability_check="has_feature"),
            format_result=lambda r: r,
        )

        result = wrapper(q="test")
        assert result["success"] == 0
        assert "Feature not available" in result["error"]


class TestMcpToolClass:
    def setup_method(self):
        """Save and clear registry before each test."""
        self._original_registry = list(_GLOBAL_TOOL_REGISTRY)
        clear_tool_registry()

    def teardown_method(self):
        """Restore registry after each test."""
        _GLOBAL_TOOL_REGISTRY.clear()
        _GLOBAL_TOOL_REGISTRY.extend(self._original_registry)

    def test_registers_class(self):
        @mcp_tool_class(name="test_tool", availability_property="has_test")
        class TestTool:
            @classmethod
            def create_dynamic(cls, agent_config, sub_agent_name=None):
                pass

            @classmethod
            def create_static(cls, agent_config, sub_agent_name=None, database_name=None):
                pass

        registry = get_tool_registry()
        assert len(registry) == 1
        assert registry[0].name == "test_tool"
        assert registry[0].tool_class is TestTool
        assert registry[0].availability_property == "has_test"

    def test_missing_create_dynamic(self):
        with pytest.raises(TypeError, match="create_dynamic"):

            @mcp_tool_class(name="bad", availability_property="has_x")
            class BadTool:
                @classmethod
                def create_static(cls, agent_config, sub_agent_name=None, database_name=None):
                    pass

    def test_missing_create_static(self):
        with pytest.raises(TypeError, match="create_static"):

            @mcp_tool_class(name="bad", availability_property="has_x")
            class BadTool:
                @classmethod
                def create_dynamic(cls, agent_config, sub_agent_name=None):
                    pass


class TestClearToolRegistry:
    def test_clear(self):
        original = list(_GLOBAL_TOOL_REGISTRY)
        try:
            _GLOBAL_TOOL_REGISTRY.append(ToolClassConfig(name="x", tool_class=object, availability_property="y"))
            assert len(_GLOBAL_TOOL_REGISTRY) > 0
            clear_tool_registry()
            assert len(_GLOBAL_TOOL_REGISTRY) == 0
        finally:
            _GLOBAL_TOOL_REGISTRY.clear()
            _GLOBAL_TOOL_REGISTRY.extend(original)


class TestGetToolRegistry:
    def test_returns_copy(self):
        registry = get_tool_registry()
        assert isinstance(registry, list)
        # Modifying the returned list shouldn't affect the original
        registry.append("junk")
        assert "junk" not in _GLOBAL_TOOL_REGISTRY
