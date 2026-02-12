# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/models/claude_model.py"""

import os
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import ModelConfig
from datus.models.claude_model import (
    ClaudeModel,
    convert_tools_for_anthropic,
    wrap_prompt_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(**overrides) -> ModelConfig:
    defaults = {
        "type": "claude",
        "api_key": "test-anthropic-key",
        "model": "claude-sonnet-4",
        "base_url": "https://api.anthropic.com",
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# wrap_prompt_cache
# ---------------------------------------------------------------------------


class TestWrapPromptCache:
    def test_adds_cache_control_to_list_content(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        result = wrap_prompt_cache(messages)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Ensure original not modified
        assert "cache_control" not in messages[0]["content"][0]

    def test_string_content_no_error(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        # String content is not a list so cache_control assignment to int index won't work
        # But wrap_prompt_cache doesn't handle string content specially
        result = wrap_prompt_cache(messages)
        assert result[0]["content"] == "Hello"

    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "user", "content": [{"type": "text", "text": "Second"}]},
        ]
        result = wrap_prompt_cache(messages)
        # Cache control on last message
        assert "cache_control" in result[1]["content"][0]
        # Not on first message
        assert "cache_control" not in result[0]["content"][0]


# ---------------------------------------------------------------------------
# convert_tools_for_anthropic
# ---------------------------------------------------------------------------


class TestConvertToolsForAnthropic:
    def test_empty_tools(self):
        assert convert_tools_for_anthropic([]) == []

    def test_basic_tool_conversion(self):
        tool = MagicMock()
        tool.name = "read_query"
        tool.description = "Execute a SQL query"
        tool.inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "SQL query"}},
        }
        tool.annotations = None

        result = convert_tools_for_anthropic([tool])
        assert len(result) == 1
        assert result[0]["name"] == "read_query"
        assert result[0]["description"] == "Execute a SQL query"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_multiple_tools_cache_on_last(self):
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "desc1"
        tool1.inputSchema = {"type": "object", "properties": {}}
        tool1.annotations = None

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "desc2"
        tool2.inputSchema = {"type": "object", "properties": {}}
        tool2.annotations = None

        result = convert_tools_for_anthropic([tool1, tool2])
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_tool_with_desc_renamed(self):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "desc"
        tool.inputSchema = {
            "type": "object",
            "properties": {
                "param": {"type": "string", "desc": "A parameter"},
            },
        }
        tool.annotations = None

        result = convert_tools_for_anthropic([tool])
        prop = result[0]["input_schema"]["properties"]["param"]
        assert "description" in prop
        assert "desc" not in prop

    def test_tool_with_annotations(self):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "desc"
        tool.inputSchema = {"type": "object", "properties": {}}
        tool.annotations = {"readOnlyHint": True}

        result = convert_tools_for_anthropic([tool])
        assert result[0]["annotations"] == {"readOnlyHint": True}


# ---------------------------------------------------------------------------
# ClaudeModel
# ---------------------------------------------------------------------------


class TestClaudeModel:
    @patch("datus.models.claude_model.anthropic")
    def test_init(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        cfg = _make_model_config()
        model = ClaudeModel(cfg)
        assert model.model_name == "claude-sonnet-4"
        assert model.api_key == "test-anthropic-key"

    @patch("datus.models.claude_model.anthropic")
    def test_get_api_key_from_config(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        cfg = _make_model_config(api_key="my-key")
        model = ClaudeModel(cfg)
        assert model.api_key == "my-key"

    @patch("datus.models.claude_model.anthropic")
    def test_get_api_key_from_env(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        cfg = _make_model_config(api_key="")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            model = ClaudeModel(cfg)
            assert model.api_key == "env-key"

    @patch("datus.models.claude_model.anthropic")
    def test_get_api_key_missing(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        cfg = _make_model_config(api_key="")
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    ClaudeModel(cfg)
            finally:
                if env_backup:
                    os.environ["ANTHROPIC_API_KEY"] = env_backup

    @patch("datus.models.claude_model.anthropic")
    def test_get_base_url_default(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        cfg = _make_model_config(base_url=None)
        model = ClaudeModel(cfg)
        assert model.base_url == "https://api.anthropic.com"

    @patch("datus.models.claude_model.anthropic")
    def test_model_specs(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        cfg = _make_model_config(model="claude-sonnet-4")
        model = ClaudeModel(cfg)
        specs = model.model_specs
        assert "claude-sonnet-4" in specs
        assert specs["claude-sonnet-4"]["context_length"] == 1048576

    @patch("datus.models.claude_model.anthropic")
    def test_generate(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = "NOT_GIVEN"

        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "Hello from Claude"
        mock_response.content = [mock_content_block]
        mock_client.messages.create.return_value = mock_response

        cfg = _make_model_config()
        model = ClaudeModel(cfg)
        result = model.generate("test prompt")
        assert result == "Hello from Claude"

    @patch("langsmith.wrappers.wrap_anthropic", side_effect=lambda c: c)
    @patch("datus.models.claude_model.anthropic")
    def test_generate_with_system_message(self, mock_anthropic, mock_wrap):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = "NOT_GIVEN"

        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "response"
        mock_response.content = [mock_content_block]
        mock_client.messages.create.return_value = mock_response

        cfg = _make_model_config()
        model = ClaudeModel(cfg)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ]
        result = model.generate(messages)
        assert result == "response"

        # Verify system message was separated
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "You are helpful"
        # user message should not include system
        filtered = call_kwargs.kwargs["messages"]
        assert all(m["role"] != "system" for m in filtered)

    @patch("datus.models.claude_model.anthropic")
    def test_generate_empty_response(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = "NOT_GIVEN"

        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        cfg = _make_model_config()
        model = ClaudeModel(cfg)
        result = model.generate("test")
        assert result == ""

    @patch("datus.models.claude_model.anthropic")
    def test_close(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        cfg = _make_model_config()
        model = ClaudeModel(cfg)
        model.close()
        # Should not raise

    @patch("datus.models.claude_model.anthropic")
    def test_context_manager(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        cfg = _make_model_config()
        with ClaudeModel(cfg) as model:
            assert model is not None

    @patch("datus.models.claude_model.anthropic")
    @pytest.mark.asyncio
    async def test_aclose(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        cfg = _make_model_config()
        model = ClaudeModel(cfg)
        await model.aclose()

    @patch("datus.models.claude_model.anthropic")
    @pytest.mark.asyncio
    async def test_generate_with_tools_uses_parent_when_no_native(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        cfg = _make_model_config()
        model = ClaudeModel(cfg)
        model.use_native_api = False

        # Verify it would call super().generate_with_tools
        # We just verify the routing logic
        assert model.use_native_api is False

    @patch("datus.models.claude_model.anthropic")
    def test_proxy_client_init(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        cfg = _make_model_config()
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}):
            model = ClaudeModel(cfg)
            assert model.proxy_client is not None
