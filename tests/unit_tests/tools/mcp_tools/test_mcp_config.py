# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for mcp_config.py – 41% coverage target."""

import os
from unittest.mock import patch

import pytest

from datus.tools.mcp_tools.mcp_config import (
    HTTPServerConfig,
    MCPConfig,
    MCPServerConfig,
    MCPServerType,
    SSEServerConfig,
    STDIOServerConfig,
    ToolFilterConfig,
    expand_config_env_vars,
    expand_env_vars,
)


# ---------------------------------------------------------------------------
# ToolFilterConfig
# ---------------------------------------------------------------------------


class TestToolFilterConfig:
    def test_all_allowed_when_disabled(self):
        f = ToolFilterConfig(enabled=False, blocked_tool_names=["x"])
        assert f.is_tool_allowed("x") is True

    def test_allowlist_only(self):
        f = ToolFilterConfig(allowed_tool_names=["a", "b"])
        assert f.is_tool_allowed("a") is True
        assert f.is_tool_allowed("c") is False

    def test_blocklist_only(self):
        f = ToolFilterConfig(blocked_tool_names=["x"])
        assert f.is_tool_allowed("x") is False
        assert f.is_tool_allowed("y") is True

    def test_allowlist_and_blocklist(self):
        f = ToolFilterConfig(allowed_tool_names=["a", "b"], blocked_tool_names=["b"])
        assert f.is_tool_allowed("a") is True
        assert f.is_tool_allowed("b") is False
        assert f.is_tool_allowed("c") is False

    def test_no_filter(self):
        f = ToolFilterConfig()
        assert f.is_tool_allowed("anything") is True


# ---------------------------------------------------------------------------
# expand_env_vars / expand_config_env_vars
# ---------------------------------------------------------------------------


class TestExpandEnvVars:
    def test_simple_var(self):
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            assert expand_env_vars("${MY_VAR}") == "hello"

    def test_var_with_default(self):
        # Ensure the var is not set
        env = dict(os.environ)
        env.pop("MISSING_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            assert expand_env_vars("${MISSING_VAR:-default_val}") == "default_val"

    def test_var_not_found_keeps_original(self):
        env = dict(os.environ)
        env.pop("UNDEFINED_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            assert expand_env_vars("${UNDEFINED_VAR}") == "${UNDEFINED_VAR}"

    def test_no_vars(self):
        assert expand_env_vars("plain text") == "plain text"

    def test_mixed(self):
        with patch.dict(os.environ, {"FOO": "bar"}):
            assert expand_env_vars("prefix-${FOO}-suffix") == "prefix-bar-suffix"


class TestExpandConfigEnvVars:
    def test_strings(self):
        with patch.dict(os.environ, {"KEY": "value"}):
            result = expand_config_env_vars({"cmd": "${KEY}"})
            assert result["cmd"] == "value"

    def test_nested_dict(self):
        with patch.dict(os.environ, {"TOKEN": "abc"}):
            result = expand_config_env_vars({"headers": {"Authorization": "Bearer ${TOKEN}"}})
            assert result["headers"]["Authorization"] == "Bearer abc"

    def test_list(self):
        with patch.dict(os.environ, {"ARG": "hello"}):
            result = expand_config_env_vars({"args": ["${ARG}", "world"]})
            assert result["args"] == ["hello", "world"]

    def test_non_string_values(self):
        result = expand_config_env_vars({"count": 42, "flag": True})
        assert result["count"] == 42
        assert result["flag"] is True

    def test_nested_dict_non_string(self):
        result = expand_config_env_vars({"headers": {"timeout": 30}})
        assert result["headers"]["timeout"] == 30

    def test_list_non_string(self):
        result = expand_config_env_vars({"args": [1, 2, 3]})
        assert result["args"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# MCPServerType
# ---------------------------------------------------------------------------


class TestMCPServerType:
    def test_values(self):
        assert MCPServerType.STDIO == "stdio"
        assert MCPServerType.SSE == "sse"
        assert MCPServerType.HTTP == "http"


# ---------------------------------------------------------------------------
# MCPServerConfig.from_config_format
# ---------------------------------------------------------------------------


class TestMCPServerConfigFromConfig:
    def test_stdio(self):
        config = MCPServerConfig.from_config_format("test", {"type": "stdio", "command": "npx", "args": ["-y"]})
        assert isinstance(config, STDIOServerConfig)
        assert config.name == "test"
        assert config.command == "npx"
        assert config.args == ["-y"]

    def test_sse(self):
        config = MCPServerConfig.from_config_format(
            "test", {"type": "sse", "url": "http://localhost:3000/sse", "timeout": 60.0}
        )
        assert isinstance(config, SSEServerConfig)
        assert config.url == "http://localhost:3000/sse"
        assert config.timeout == 60.0

    def test_http(self):
        config = MCPServerConfig.from_config_format(
            "test",
            {"type": "http", "url": "http://localhost:3000", "headers": {"X-Key": "val"}},
        )
        assert isinstance(config, HTTPServerConfig)
        assert config.url == "http://localhost:3000"
        assert config.headers["X-Key"] == "val"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown server type"):
            MCPServerConfig.from_config_format("test", {"type": "grpc"})

    def test_with_tool_filter(self):
        config = MCPServerConfig.from_config_format(
            "test",
            {
                "type": "stdio",
                "command": "cmd",
                "tool_filter": {"allowed_tool_names": ["a", "b"], "enabled": True},
            },
        )
        assert config.tool_filter is not None
        assert config.tool_filter.allowed_tool_names == ["a", "b"]

    def test_with_env_expansion(self):
        with patch.dict(os.environ, {"CMD_PATH": "/usr/bin/test"}):
            config = MCPServerConfig.from_config_format("test", {"type": "stdio", "command": "${CMD_PATH}"})
            assert config.command == "/usr/bin/test"

    def test_default_type_is_stdio(self):
        config = MCPServerConfig.from_config_format("test", {"command": "npx"})
        assert isinstance(config, STDIOServerConfig)


# ---------------------------------------------------------------------------
# STDIOServerConfig / SSEServerConfig / HTTPServerConfig
# ---------------------------------------------------------------------------


class TestServerConfigs:
    def test_stdio_get_connection_info(self):
        config = STDIOServerConfig(name="s", command="cmd", args=["a", "b"], env={"K": "V"})
        info = config.get_connection_info()
        assert info["type"] == "stdio"
        assert info["command"] == "cmd"
        assert info["args"] == ["a", "b"]
        assert info["env"] == {"K": "V"}

    def test_stdio_defaults(self):
        config = STDIOServerConfig(name="s", command="cmd")
        info = config.get_connection_info()
        assert info["args"] == []
        assert info["env"] == {}

    def test_sse_get_connection_info(self):
        config = SSEServerConfig(name="s", url="http://localhost", headers={"A": "B"}, timeout=10.0)
        info = config.get_connection_info()
        assert info["type"] == "sse"
        assert info["url"] == "http://localhost"
        assert info["timeout"] == 10.0

    def test_sse_timeout_validation(self):
        with pytest.raises(ValueError):
            SSEServerConfig(name="s", url="http://localhost", timeout=-1.0)

    def test_http_get_connection_info(self):
        config = HTTPServerConfig(name="s", url="http://localhost")
        info = config.get_connection_info()
        assert info["type"] == "http"
        assert info["url"] == "http://localhost"

    def test_http_timeout_validation(self):
        with pytest.raises(ValueError):
            HTTPServerConfig(name="s", url="http://localhost", timeout=0)

    def test_type_validator_string(self):
        config = STDIOServerConfig(name="s", type="stdio", command="cmd")
        assert config.type == MCPServerType.STDIO

    def test_type_validator_invalid(self):
        with pytest.raises(ValueError):
            STDIOServerConfig(name="s", type="invalid", command="cmd")


# ---------------------------------------------------------------------------
# MCPConfig
# ---------------------------------------------------------------------------


class TestMCPConfig:
    def test_add_and_get(self):
        cfg = MCPConfig()
        server = STDIOServerConfig(name="test", command="cmd")
        cfg.add_server(server)
        assert cfg.get_server("test") is not None
        assert cfg.get_server("nonexistent") is None

    def test_remove(self):
        cfg = MCPConfig()
        server = STDIOServerConfig(name="test", command="cmd")
        cfg.add_server(server)
        assert cfg.remove_server("test") is True
        assert cfg.get_server("test") is None
        assert cfg.remove_server("test") is False

    def test_list_servers_all(self):
        cfg = MCPConfig()
        cfg.add_server(STDIOServerConfig(name="s1", command="cmd1"))
        cfg.add_server(SSEServerConfig(name="s2", url="http://localhost"))
        assert len(cfg.list_servers()) == 2

    def test_list_servers_filtered(self):
        cfg = MCPConfig()
        cfg.add_server(STDIOServerConfig(name="s1", command="cmd1"))
        cfg.add_server(SSEServerConfig(name="s2", url="http://localhost"))
        stdio_servers = cfg.list_servers(server_type=MCPServerType.STDIO)
        assert len(stdio_servers) == 1
        assert stdio_servers[0].name == "s1"

    def test_from_config_format(self):
        raw = {
            "mcpServers": {
                "server1": {"type": "stdio", "command": "cmd"},
                "server2": {"type": "sse", "url": "http://localhost"},
            }
        }
        cfg = MCPConfig.from_config_format(raw)
        assert len(cfg.servers) == 2
        assert isinstance(cfg.get_server("server1"), STDIOServerConfig)
        assert isinstance(cfg.get_server("server2"), SSEServerConfig)

    def test_to_config_format(self):
        cfg = MCPConfig()
        cfg.add_server(STDIOServerConfig(name="s1", command="cmd", args=["a"]))
        cfg.add_server(SSEServerConfig(name="s2", url="http://localhost", timeout=30.0))
        cfg.add_server(HTTPServerConfig(name="s3", url="http://localhost/api", headers={"X": "Y"}))

        result = cfg.to_config_format()
        assert "mcpServers" in result
        assert result["mcpServers"]["s1"]["command"] == "cmd"
        assert result["mcpServers"]["s2"]["url"] == "http://localhost"
        assert result["mcpServers"]["s3"]["headers"] == {"X": "Y"}

    def test_to_config_format_with_filter(self):
        cfg = MCPConfig()
        cfg.add_server(
            STDIOServerConfig(
                name="s1",
                command="cmd",
                tool_filter=ToolFilterConfig(allowed_tool_names=["t1"]),
            )
        )
        result = cfg.to_config_format()
        assert "tool_filter" in result["mcpServers"]["s1"]
        assert result["mcpServers"]["s1"]["tool_filter"]["allowed_tool_names"] == ["t1"]

    def test_roundtrip(self):
        original = {
            "mcpServers": {
                "my_server": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "server"],
                    "env": {"KEY": "VAL"},
                }
            }
        }
        cfg = MCPConfig.from_config_format(original)
        exported = cfg.to_config_format()
        assert exported["mcpServers"]["my_server"]["command"] == "npx"
        assert exported["mcpServers"]["my_server"]["args"] == ["-y", "server"]
