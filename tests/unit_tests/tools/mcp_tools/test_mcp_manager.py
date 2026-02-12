# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for mcp_manager.py – 20.8% coverage target."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.tools.mcp_tools.mcp_config import (
    HTTPServerConfig,
    MCPConfig,
    MCPServerType,
    SSEServerConfig,
    STDIOServerConfig,
    ToolFilterConfig,
)
from datus.tools.mcp_tools.mcp_manager import (
    MCPManager,
    _validate_server_exists,
    create_static_tool_filter,
)


# ---------------------------------------------------------------------------
# create_static_tool_filter
# ---------------------------------------------------------------------------


class TestCreateStaticToolFilter:
    def test_defaults(self):
        f = create_static_tool_filter()
        assert f.enabled is True
        assert f.allowed_tool_names is None
        assert f.blocked_tool_names is None

    def test_with_lists(self):
        f = create_static_tool_filter(
            allowed_tool_names=["a"],
            blocked_tool_names=["b"],
            enabled=False,
        )
        assert f.allowed_tool_names == ["a"]
        assert f.blocked_tool_names == ["b"]
        assert f.enabled is False


# ---------------------------------------------------------------------------
# _validate_server_exists
# ---------------------------------------------------------------------------


class TestValidateServerExists:
    def test_exists(self):
        manager = MagicMock()
        manager.get_server_config.return_value = STDIOServerConfig(name="x", command="cmd")
        ok, msg, cfg = _validate_server_exists(manager, "x")
        assert ok is True
        assert cfg is not None

    def test_not_exists(self):
        manager = MagicMock()
        manager.get_server_config.return_value = None
        ok, msg, cfg = _validate_server_exists(manager, "x")
        assert ok is False
        assert "not found" in msg


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


class TestMCPManager:
    @pytest.fixture
    def manager(self, tmp_path):
        """Create MCPManager with mocked path manager."""
        config_path = tmp_path / "conf" / ".mcp.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        pm = MagicMock()
        pm.mcp_config_path.return_value = config_path
        pm.ensure_dirs.return_value = None

        with patch("datus.utils.path_manager.get_path_manager", return_value=pm):
            mgr = MCPManager()
        return mgr

    # --- load/save config ---

    def test_load_config_empty(self, manager):
        # Fresh manager, no file
        assert manager.config is not None

    def test_save_and_load_config(self, manager):
        server = STDIOServerConfig(name="s1", command="cmd")
        manager.config.add_server(server)
        assert manager.save_config() is True

        # Reload
        assert manager.load_config() is True
        assert manager.config.get_server("s1") is not None

    def test_load_config_invalid_json(self, manager):
        manager.config_path.write_text("not json", encoding="utf-8")
        assert manager.load_config() is False

    def test_load_config_no_mcpservers_key(self, manager):
        manager.config_path.write_text('{"other": "data"}', encoding="utf-8")
        result = manager.load_config()
        assert result is True
        assert len(manager.config.servers) == 0

    def test_load_config_empty_file(self, manager):
        manager.config_path.write_text("", encoding="utf-8")
        result = manager.load_config()
        assert result is True

    # --- add/remove/list/get ---

    def test_add_server(self, manager):
        server = STDIOServerConfig(name="new", command="cmd")
        ok, msg = manager.add_server(server)
        assert ok is True
        assert manager.get_server_config("new") is not None

    def test_add_server_duplicate(self, manager):
        server = STDIOServerConfig(name="dup", command="cmd")
        manager.add_server(server)
        ok, msg = manager.add_server(server)
        assert ok is False
        assert "already exists" in msg

    def test_remove_server(self, manager):
        server = STDIOServerConfig(name="rem", command="cmd")
        manager.add_server(server)
        ok, msg = manager.remove_server("rem")
        assert ok is True
        assert manager.get_server_config("rem") is None

    def test_remove_server_not_found(self, manager):
        ok, msg = manager.remove_server("nonexistent")
        assert ok is False

    def test_list_servers(self, manager):
        manager.add_server(STDIOServerConfig(name="s1", command="c1"))
        manager.add_server(SSEServerConfig(name="s2", url="http://localhost"))
        all_servers = manager.list_servers()
        assert len(all_servers) == 2

    def test_list_servers_filtered(self, manager):
        manager.add_server(STDIOServerConfig(name="s1", command="c1"))
        manager.add_server(SSEServerConfig(name="s2", url="http://localhost"))
        stdio_only = manager.list_servers(server_type=MCPServerType.STDIO)
        assert len(stdio_only) == 1

    def test_get_server_config(self, manager):
        manager.add_server(STDIOServerConfig(name="s1", command="cmd"))
        assert manager.get_server_config("s1") is not None
        assert manager.get_server_config("nope") is None

    # --- tool filter ---

    def test_set_tool_filter(self, manager):
        manager.add_server(STDIOServerConfig(name="s1", command="cmd"))
        f = ToolFilterConfig(allowed_tool_names=["t1"])
        ok, msg = manager.set_tool_filter("s1", f)
        assert ok is True

        ok, msg, flt = manager.get_tool_filter("s1")
        assert ok is True
        assert flt.allowed_tool_names == ["t1"]

    def test_set_tool_filter_not_found(self, manager):
        f = ToolFilterConfig()
        ok, msg = manager.set_tool_filter("nonexistent", f)
        assert ok is False

    def test_get_tool_filter_not_found(self, manager):
        ok, msg, flt = manager.get_tool_filter("nonexistent")
        assert ok is False
        assert flt is None

    # --- _create_server_instance ---

    def test_create_stdio_server_instance(self, manager):
        config = STDIOServerConfig(name="s", command="cmd", args=["a"], env={"K": "V"})
        instance, details = manager._create_server_instance(config)
        assert instance is not None
        assert details["command"] == "cmd"

    def test_create_sse_server_instance(self, manager):
        config = SSEServerConfig(name="s", url="http://localhost:3000/sse")
        instance, details = manager._create_server_instance(config)
        assert instance is not None
        assert details["url"] == "http://localhost:3000/sse"

    def test_create_sse_server_no_url(self, manager):
        config = SSEServerConfig(name="s", url="http://localhost")
        # Manually clear url in expanded config
        with patch.object(manager, "_create_sse_server", return_value=(None, {"error": "URL is required"})):
            instance, details = manager._create_server_instance(config)
            assert instance is None

    def test_create_http_server_instance(self, manager):
        config = HTTPServerConfig(name="s", url="http://localhost:3000/api")
        instance, details = manager._create_server_instance(config)
        assert instance is not None
        assert details["url"] == "http://localhost:3000/api"

    # --- cleanup ---

    def test_cleanup(self, manager):
        manager.cleanup()  # Should not raise

    # --- async operations (mock) ---

    @pytest.mark.asyncio
    async def test_check_connectivity_not_found(self, manager):
        ok, msg, details = await manager.check_connectivity("nonexistent")
        assert ok is False
        assert "not found" in msg

    @pytest.mark.asyncio
    async def test_list_tools_not_found(self, manager):
        ok, msg, tools = await manager.list_tools("nonexistent")
        assert ok is False
        assert "not found" in msg

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, manager):
        ok, msg, data = await manager.call_tool("nonexistent", "tool", {})
        assert ok is False

    @pytest.mark.asyncio
    async def test_call_tool_blocked_by_filter(self, manager):
        manager.add_server(
            STDIOServerConfig(
                name="s1",
                command="cmd",
                tool_filter=ToolFilterConfig(blocked_tool_names=["blocked_tool"]),
            )
        )
        ok, msg, data = await manager.call_tool("s1", "blocked_tool", {})
        assert ok is False
        assert "blocked" in msg.lower()

    @pytest.mark.asyncio
    async def test_list_filtered_tools(self, manager):
        ok, msg, tools = await manager.list_filtered_tools("nonexistent")
        assert ok is False

    # --- _dispatch_operation ---

    @pytest.mark.asyncio
    async def test_dispatch_unknown_operation(self, manager):
        ok, data = await manager._dispatch_operation(
            MagicMock(), "test", "unknown_op", MagicMock(), MagicMock()
        )
        assert ok is False
        assert "unknown" in data["error"].lower()

    # --- _handle_call_tool ---

    @pytest.mark.asyncio
    async def test_handle_call_tool_no_tool_name(self, manager):
        ok, data = await manager._handle_call_tool(MagicMock(), "test")
        assert ok is False
        assert "required" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self, manager):
        mock_server = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = False
        content_item = MagicMock()
        content_item.type = "text"
        content_item.text = "result text"
        mock_result.content = [content_item]
        mock_server.call_tool.return_value = mock_result

        ok, data = await manager._handle_call_tool(
            mock_server, "test", tool_name="my_tool", arguments={"key": "val"}
        )
        assert ok is True
        assert data["content"][0]["text"] == "result text"

    # --- _handle_list_tools ---

    @pytest.mark.asyncio
    async def test_handle_list_tools(self, manager):
        mock_server = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "desc"
        mock_tool.inputSchema = {"type": "object"}
        mock_server.list_tools.return_value = [mock_tool]

        ok, data = await manager._handle_list_tools(mock_server, "test", MagicMock(), MagicMock())
        assert ok is True
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "tool1"

    @pytest.mark.asyncio
    async def test_handle_list_tools_empty(self, manager):
        mock_server = AsyncMock()
        mock_server.list_tools.return_value = None
        ok, data = await manager._handle_list_tools(mock_server, "test", MagicMock(), MagicMock())
        assert ok is True
        assert len(data["tools"]) == 0

    @pytest.mark.asyncio
    async def test_handle_list_tools_model_dump(self, manager):
        mock_server = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "desc"
        mock_schema = MagicMock()
        mock_schema.model_dump.return_value = {"type": "object"}
        mock_tool.inputSchema = mock_schema
        mock_server.list_tools.return_value = [mock_tool]

        ok, data = await manager._handle_list_tools(mock_server, "test", MagicMock(), MagicMock())
        assert data["tools"][0]["inputSchema"] == {"type": "object"}

    # --- _handle_connectivity_test ---

    @pytest.mark.asyncio
    async def test_handle_connectivity_test(self, manager):
        mock_server = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "t1"
        mock_server.list_tools.return_value = [mock_tool]

        ok, data = await manager._handle_connectivity_test(mock_server, "test", MagicMock(), MagicMock())
        assert ok is True
        assert data["connected"] is True
        assert data["tool_count"] == 1

    @pytest.mark.asyncio
    async def test_handle_connectivity_test_no_list_tools(self, manager):
        mock_server = MagicMock(spec=[])  # No list_tools
        ok, data = await manager._handle_connectivity_test(mock_server, "test", MagicMock(), MagicMock())
        assert ok is True
        assert data["tools_available"] is False

    @pytest.mark.asyncio
    async def test_handle_connectivity_test_list_tools_error(self, manager):
        mock_server = AsyncMock()
        mock_server.list_tools.side_effect = Exception("fail")
        ok, data = await manager._handle_connectivity_test(mock_server, "test", MagicMock(), MagicMock())
        assert ok is True
        assert data["tools_available"] is False

    # --- _cleanup_server_instance ---

    @pytest.mark.asyncio
    async def test_cleanup_with_cleanup_method(self, manager):
        mock_server = AsyncMock()
        await manager._cleanup_server_instance(mock_server)
        mock_server.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_error_suppressed(self, manager):
        mock_server = AsyncMock()
        mock_server.cleanup.side_effect = Exception("error")
        await manager._cleanup_server_instance(mock_server)  # Should not raise
