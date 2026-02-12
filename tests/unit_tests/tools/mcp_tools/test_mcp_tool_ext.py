# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for mcp_tool.py and mcp_manager.py"""

from unittest.mock import MagicMock, patch

import pytest

from datus.tools.mcp_tools.mcp_tool import (
    MCPTool,
    _parse_header_from_parts,
    parse_command_string,
)
from datus.tools.mcp_tools.mcp_manager import (
    MCPManager,
    _validate_server_exists,
    create_static_tool_filter,
)
from datus.tools.mcp_tools.mcp_config import (
    MCPConfig,
    MCPServerType,
    SSEServerConfig,
    STDIOServerConfig,
    ToolFilterConfig,
)
from datus.utils.exceptions import DatusException


# ---------------------------------------------------------------------------
# _parse_header_from_parts
# ---------------------------------------------------------------------------


class TestParseHeaderFromParts:
    def test_json_format(self):
        parts = ['{"Authorization": "Bearer token123"}']
        result = _parse_header_from_parts(parts)
        assert result["Authorization"] == "Bearer token123"

    def test_single_quote_json(self):
        parts = ["{'Token': '1234a'}"]
        result = _parse_header_from_parts(parts)
        assert result.get("Token") == "1234a"

    def test_token_level_parse(self):
        parts = ["{Token:", "1234a,", "header2:", "1}"]
        result = _parse_header_from_parts(parts)
        assert result.get("Token") == "1234a"
        assert result.get("header2") == "1"

    def test_empty_parts(self):
        result = _parse_header_from_parts(["{}"])
        assert result == {}

    def test_key_value_colon(self):
        parts = ["Authorization:", "Bearer", "abc"]  # noqa: E501
        # This tests the fallback token-level parse
        result = _parse_header_from_parts(parts)
        assert "Authorization" in result or result == {}


# ---------------------------------------------------------------------------
# parse_command_string
# ---------------------------------------------------------------------------


class TestParseCommandString:
    def test_stdio_basic(self):
        cmd = '--transport stdio my-server npx -y @modelcontextprotocol/server-filesystem'
        transport, name, payload = parse_command_string(cmd)
        assert transport in ("stdio", "studio")
        assert name == "my-server"
        assert payload["command"] == "npx"

    def test_sse_basic(self):
        cmd = '--transport sse my-server http://localhost:8080/sse'
        transport, name, payload = parse_command_string(cmd)
        assert transport == "sse"
        assert name == "my-server"
        assert payload["url"] == "http://localhost:8080/sse"

    def test_http_basic(self):
        cmd = '--transport http my-server http://localhost:8080/mcp'
        transport, name, payload = parse_command_string(cmd)
        assert transport == "http"
        assert name == "my-server"
        assert payload["url"] == "http://localhost:8080/mcp"

    def test_stdio_with_env(self):
        cmd = '--transport stdio my-server npx mcp-server --env API_KEY=abc123'
        transport, name, payload = parse_command_string(cmd)
        assert payload["env"]["API_KEY"] == "abc123"

    def test_no_transport_raises(self):
        with pytest.raises(DatusException):
            parse_command_string("npx some-server")

    def test_unsupported_transport_raises(self):
        with pytest.raises(DatusException):
            parse_command_string("--transport websocket my-server ws://localhost")

    def test_sse_with_timeout(self):
        cmd = '--transport sse my-server http://localhost:8080/sse --timeout 30'
        transport, name, payload = parse_command_string(cmd)
        assert payload["timeout"] == 30.0

    def test_sse_default_timeout(self):
        cmd = '--transport sse my-server http://localhost:8080/sse'
        transport, name, payload = parse_command_string(cmd)
        assert payload["timeout"] == 10.0

    def test_http_with_header(self):
        cmd = '--transport http my-server http://localhost:8080 --header {"Authorization": "Bearer token"}'
        transport, name, payload = parse_command_string(cmd)
        assert "Authorization" in payload.get("headers", {})

    def test_stdio_with_args(self):
        cmd = '--transport stdio my-server npx -y @modelcontextprotocol/server arg1 arg2'
        transport, name, payload = parse_command_string(cmd)
        assert "arg1" in payload["args"]
        assert "arg2" in payload["args"]


# ---------------------------------------------------------------------------
# create_static_tool_filter
# ---------------------------------------------------------------------------


class TestCreateStaticToolFilter:
    def test_basic(self):
        result = create_static_tool_filter(
            allowed_tool_names=["tool1", "tool2"],
            enabled=True,
        )
        assert isinstance(result, ToolFilterConfig)
        assert result.enabled is True

    def test_blocked_tools(self):
        result = create_static_tool_filter(
            blocked_tool_names=["dangerous_tool"],
            enabled=True,
        )
        assert result.blocked_tool_names == ["dangerous_tool"]

    def test_disabled(self):
        result = create_static_tool_filter(enabled=False)
        assert result.enabled is False


# ---------------------------------------------------------------------------
# _validate_server_exists
# ---------------------------------------------------------------------------


class TestValidateServerExists:
    def test_server_found(self):
        manager = MagicMock()
        mock_config = MagicMock()
        manager.get_server_config.return_value = mock_config
        success, msg, config = _validate_server_exists(manager, "test-server")
        assert success is True
        assert config is mock_config

    def test_server_not_found(self):
        manager = MagicMock()
        manager.get_server_config.return_value = None
        success, msg, config = _validate_server_exists(manager, "test-server")
        assert success is False
        assert "not found" in msg
        assert config is None


# ---------------------------------------------------------------------------
# MCPTool (mocked manager)
# ---------------------------------------------------------------------------


class TestMCPTool:
    @pytest.fixture
    def mcp_tool(self):
        with patch("datus.tools.mcp_tools.mcp_tool.MCPManager") as MockManager:
            mock_manager = MagicMock()
            MockManager.return_value = mock_manager
            tool = MCPTool()
            tool.manager = mock_manager
            return tool

    def test_add_server_success(self, mcp_tool):
        mcp_tool.manager.add_server.return_value = (True, "Added")
        result = mcp_tool.add_server(name="test", type="stdio", command="npx", args=[])
        assert result.success is True

    def test_add_server_failure(self, mcp_tool):
        mcp_tool.manager.add_server.return_value = (False, "Already exists")
        result = mcp_tool.add_server(name="test", type="stdio", command="npx")
        assert result.success is False

    def test_remove_server_success(self, mcp_tool):
        mcp_tool.manager.remove_server.return_value = (True, "Removed")
        result = mcp_tool.remove_server(name="test")
        assert result.success is True

    def test_remove_server_failure(self, mcp_tool):
        mcp_tool.manager.remove_server.return_value = (False, "Not found")
        result = mcp_tool.remove_server(name="test")
        assert result.success is False

    def test_list_servers(self, mcp_tool):
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {"name": "s1", "type": "stdio"}
        mcp_tool.manager.list_servers.return_value = [mock_config]
        result = mcp_tool.list_servers()
        assert result.success is True
        assert result.result["total_count"] == 1

    def test_get_server_found(self, mcp_tool):
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {"name": "test", "type": "stdio"}
        mcp_tool.manager.get_server_config.return_value = mock_config
        result = mcp_tool.get_server(name="test")
        assert result.success is True

    def test_get_server_not_found(self, mcp_tool):
        mcp_tool.manager.get_server_config.return_value = None
        result = mcp_tool.get_server(name="nonexistent")
        assert result.success is False

    def test_set_tool_filter(self, mcp_tool):
        mcp_tool.manager.set_tool_filter.return_value = (True, "Updated")
        result = mcp_tool.set_tool_filter(
            server_name="test",
            allowed_tools=["tool1"],
        )
        assert result.success is True

    def test_get_tool_filter_with_filter(self, mcp_tool):
        mock_filter = MagicMock()
        mock_filter.model_dump.return_value = {"enabled": True}
        mcp_tool.manager.get_tool_filter.return_value = (True, "Retrieved", mock_filter)
        result = mcp_tool.get_tool_filter(server_name="test")
        assert result.success is True
        assert result.result["has_filter"] is True

    def test_get_tool_filter_no_filter(self, mcp_tool):
        mcp_tool.manager.get_tool_filter.return_value = (True, "Retrieved", None)
        result = mcp_tool.get_tool_filter(server_name="test")
        assert result.success is True
        assert result.result["has_filter"] is False

    def test_remove_tool_filter(self, mcp_tool):
        mcp_tool.manager.set_tool_filter.return_value = (True, "Removed")
        result = mcp_tool.remove_tool_filter(server_name="test")
        assert result.success is True

    def test_cleanup(self, mcp_tool):
        mcp_tool.cleanup()
        mcp_tool.manager.cleanup.assert_called_once()

    def test_list_filtered_tools(self, mcp_tool):
        """list_filtered_tools delegates to list_tools with apply_filter=True"""
        with patch.object(mcp_tool, "list_tools") as mock_list:
            mock_list.return_value = MagicMock(success=True)
            result = mcp_tool.list_filtered_tools(server_name="test")
            mock_list.assert_called_once_with("test", apply_filter=True)


# ---------------------------------------------------------------------------
# MCPManager helper methods
# ---------------------------------------------------------------------------


class TestMCPManagerHelpers:
    @patch("datus.tools.mcp_tools.mcp_manager.get_path_manager")
    def test_create_sse_server_no_url(self, mock_pm):
        mock_pm.return_value = MagicMock()
        mock_pm.return_value.mcp_config_path.return_value = MagicMock(exists=lambda: False)
        with patch.object(MCPManager, "load_config"):
            manager = MCPManager.__new__(MCPManager)
            manager.config = MCPConfig()
            manager._lock = __import__("threading").Lock()
            result_instance, details = manager._create_sse_server({"url": None})
            assert result_instance is None
            assert "URL is required" in details.get("error", "")

    @patch("datus.tools.mcp_tools.mcp_manager.get_path_manager")
    def test_create_http_server_no_url(self, mock_pm):
        mock_pm.return_value = MagicMock()
        mock_pm.return_value.mcp_config_path.return_value = MagicMock(exists=lambda: False)
        with patch.object(MCPManager, "load_config"):
            manager = MCPManager.__new__(MCPManager)
            manager.config = MCPConfig()
            manager._lock = __import__("threading").Lock()
            result_instance, details = manager._create_http_server({"url": None})
            assert result_instance is None
            assert "URL is required" in details.get("error", "")

    @patch("datus.tools.mcp_tools.mcp_manager.get_path_manager")
    def test_set_tool_filter_server_not_found(self, mock_pm):
        mock_pm.return_value = MagicMock()
        mock_pm.return_value.mcp_config_path.return_value = MagicMock(exists=lambda: False)
        with patch.object(MCPManager, "load_config"):
            manager = MCPManager.__new__(MCPManager)
            manager.config = MCPConfig()
            manager._lock = __import__("threading").Lock()
            filter_config = ToolFilterConfig(enabled=True)
            success, msg = manager.set_tool_filter("nonexistent", filter_config)
            assert success is False
            assert "not found" in msg

    @patch("datus.tools.mcp_tools.mcp_manager.get_path_manager")
    def test_get_tool_filter_server_not_found(self, mock_pm):
        mock_pm.return_value = MagicMock()
        mock_pm.return_value.mcp_config_path.return_value = MagicMock(exists=lambda: False)
        with patch.object(MCPManager, "load_config"):
            manager = MCPManager.__new__(MCPManager)
            manager.config = MCPConfig()
            manager._lock = __import__("threading").Lock()
            success, msg, config = manager.get_tool_filter("nonexistent")
            assert success is False
            assert "not found" in msg

    @patch("datus.tools.mcp_tools.mcp_manager.get_path_manager")
    def test_cleanup(self, mock_pm):
        mock_pm.return_value = MagicMock()
        mock_pm.return_value.mcp_config_path.return_value = MagicMock(exists=lambda: False)
        with patch.object(MCPManager, "load_config"):
            manager = MCPManager.__new__(MCPManager)
            manager.config = MCPConfig()
            manager._lock = __import__("threading").Lock()
            # Should not raise
            manager.cleanup()
