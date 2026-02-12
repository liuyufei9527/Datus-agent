# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Round 2 unit tests for AgenticNode base class (datus/agent/node/agentic_node.py).

Tests cover:
- get_node_name() — automatic name derivation from class name
- _generate_session_id()
- _get_or_create_session() — session creation and summary handling
- _count_session_tokens() / _add_session_tokens()
- _manual_compact() — session summarization
- _auto_compact() — threshold-based compaction
- _parse_node_config() — config parsing from agent.yml
- execute() — synchronous wrapper
- _get_result_class() — result class lookup
- setup_input() / update_context() — workflow integration
- clear_session() / delete_session() / get_session_info()
- _resolve_workspace_root()
- _get_system_prompt() — template loading
- _get_tool_category() — tool categorization
- set_permission_callback()
- _get_available_skills_context()
"""

import asyncio
from typing import AsyncGenerator, Optional
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SQLContext, SqlTask
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


class ConcreteAgenticNode(AgenticNode):
    """Concrete implementation of AgenticNode for testing."""

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        ahm = action_history_manager or ActionHistoryManager()
        action = ActionHistory(
            action_id="test_action",
            role=ActionRole.ASSISTANT,
            messages="test response",
            action_type="response",
            input={},
            output={"raw_output": "test", "success": True},
            status=ActionStatus.SUCCESS,
        )
        ahm.add_action(action)
        yield action


def _make_concrete_node(agent_config=None):
    return ConcreteAgenticNode(
        node_id="test_agentic",
        description="Test agentic node",
        node_type="test_type",
        agent_config=agent_config,
    )


class TestGetNodeName:
    """Tests for AgenticNode.get_node_name()."""

    def test_default_name_extraction(self, real_agent_config, mock_llm_create):
        """Default get_node_name extracts from class name minus 'AgenticNode'."""
        node = _make_concrete_node(real_agent_config)
        # ConcreteAgenticNode -> "concrete"
        assert node.get_node_name() == "concrete"

    def test_name_without_agentic_suffix(self, real_agent_config, mock_llm_create):
        """Class name without 'AgenticNode' suffix uses full class name."""

        class TestNode(AgenticNode):
            async def execute_stream(self, ahm=None):
                yield  # pragma: no cover

        node = TestNode.__new__(TestNode)
        name = node.get_node_name()
        assert name == "testnode"


class TestGenerateSessionId:
    """Tests for AgenticNode._generate_session_id()."""

    def test_session_id_format(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        session_id = node._generate_session_id()
        assert session_id.startswith("concrete_session_")
        assert len(session_id) > len("concrete_session_")


class TestGetOrCreateSession:
    """Tests for AgenticNode._get_or_create_session()."""

    def test_creates_session(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        session, summary = node._get_or_create_session()
        assert session is not None
        assert node.session_id is not None
        assert summary is None

    def test_reuses_existing_session(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        session1, _ = node._get_or_create_session()
        session2, _ = node._get_or_create_session()
        assert session1 is session2

    def test_returns_summary_from_compact(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.last_summary = "Previous conversation summary"
        session, summary = node._get_or_create_session()
        assert summary == "Previous conversation summary"
        # Summary should be cleared after use
        assert node.last_summary is None


class TestSessionTokens:
    """Tests for _count_session_tokens / _add_session_tokens."""

    def test_initial_token_count(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        assert node._count_session_tokens() == 0

    def test_add_tokens(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node._add_session_tokens(100)
        assert node._count_session_tokens() == 100

    def test_add_zero_tokens(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node._add_session_tokens(0)
        assert node._count_session_tokens() == 0

    def test_add_negative_tokens(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node._add_session_tokens(-5)
        assert node._count_session_tokens() == 0

    def test_exceed_context_length(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.context_length = 100
        node._add_session_tokens(50)
        assert node._count_session_tokens() == 50
        # Try to exceed
        node._add_session_tokens(60)  # 50+60=110 > 100
        # Should not add, stays at 50
        assert node._count_session_tokens() == 50


class TestManualCompact:
    """Tests for AgenticNode._manual_compact()."""

    @pytest.mark.asyncio
    async def test_compact_no_model(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.model = None
        result = await node._manual_compact()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_compact_no_session(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node._session = None
        result = await node._manual_compact()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_compact_success(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        # Create a session first
        node._get_or_create_session()
        assert node._session is not None

        mock_llm_create.reset(
            responses=[MockLLMResponse(content="Summary of conversation.")]
        )

        result = await node._manual_compact()
        assert result["success"] is True
        assert result["summary"] == "Summary of conversation."
        # Session should be cleared
        assert node._session is None
        assert node.session_id is None
        assert node._session_tokens == 0
        # Summary should be stored for next session
        assert node.last_summary == "Summary of conversation."


class TestAutoCompact:
    """Tests for AgenticNode._auto_compact()."""

    @pytest.mark.asyncio
    async def test_auto_compact_no_model(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.model = None
        result = await node._auto_compact()
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_compact_no_context_length(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.context_length = None
        result = await node._auto_compact()
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_compact_below_threshold(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.context_length = 10000
        node._session_tokens = 1000  # 10% - below 90%
        result = await node._auto_compact()
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_compact_above_threshold(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.context_length = 10000
        node._session_tokens = 9500  # 95% - above 90%
        node._get_or_create_session()

        mock_llm_create.reset(
            responses=[MockLLMResponse(content="Summary.")]
        )

        result = await node._auto_compact()
        # Returns the compact result dict (truthy)
        assert result


class TestParseNodeConfig:
    """Tests for AgenticNode._parse_node_config()."""

    def test_parse_with_no_config(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(None)
        config = node._parse_node_config(None, "test")
        assert config == {}

    def test_parse_nonexistent_node(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        config = node._parse_node_config(real_agent_config, "nonexistent_node")
        assert config == {}

    def test_parse_existing_node(self, real_agent_config, mock_llm_create):
        config = AgenticNode._parse_node_config(None, real_agent_config, "chat")
        assert "system_prompt" in config
        assert config["system_prompt"] == "chat"
        assert "tools" in config
        assert "max_turns" in config
        assert config["max_turns"] == 5

    def test_parse_node_with_rules_dict(self, real_agent_config, mock_llm_create):
        """Rules with dict items should be normalized to strings."""
        real_agent_config.agentic_nodes["test_rules"] = {
            "system_prompt": "test",
            "rules": [{"key": "value"}, "simple rule"],
        }
        config = AgenticNode._parse_node_config(None, real_agent_config, "test_rules")
        assert "rules" in config
        assert config["rules"][0] == "key: value"
        assert config["rules"][1] == "simple rule"
        # Cleanup
        del real_agent_config.agentic_nodes["test_rules"]


class TestExecuteSync:
    """Tests for AgenticNode.execute() — sync wrapper."""

    def test_execute_basic(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        result = node.execute()
        assert result is not None
        assert isinstance(result, BaseResult)

    def test_execute_with_result(self, real_agent_config, mock_llm_create):
        """execute() wraps execute_stream and extracts result."""
        node = _make_concrete_node(real_agent_config)
        result = node.execute()
        # Should have a result stored
        assert node.result is not None


class TestGetResultClass:
    """Tests for AgenticNode._get_result_class()."""

    def test_unknown_class(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        assert node._get_result_class() is None


class TestSetupInput:
    """Tests for AgenticNode.setup_input()."""

    def test_setup_input_default(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        task = SqlTask(task="Test task", database_name="california_schools")
        wf = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input is not None

    def test_setup_input_populates_fields(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        task = SqlTask(
            task="Test task",
            database_name="california_schools",
            catalog_name="test_catalog",
            schema_name="test_schema",
        )
        wf = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(wf)
        assert result["success"] is True


class TestUpdateContext:
    """Tests for AgenticNode.update_context()."""

    def test_update_context_no_result(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.result = None
        task = SqlTask(task="Test", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)

        result = node.update_context(wf)
        assert result["success"] is False

    def test_update_context_with_sql(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.result = MagicMock()
        node.result.sql = "SELECT 1"
        node.result.response = "explanation"
        node.result.explanation = "explanation"

        task = SqlTask(task="Test", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)

        result = node.update_context(wf)
        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 1


class TestSessionManagement:
    """Tests for clear_session, delete_session, get_session_info."""

    def test_clear_session(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node._get_or_create_session()
        assert node._session is not None
        original_session_id = node.session_id

        node.clear_session()
        assert node._session is None
        assert node._session_tokens == 0
        # session_id should still be set
        assert node.session_id == original_session_id

    def test_delete_session(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node._get_or_create_session()
        assert node._session is not None

        node.delete_session()
        assert node._session is None
        assert node.session_id is None
        assert node._session_tokens == 0

    def test_get_session_info_no_session(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        info = node.get_session_info()
        assert info["session_id"] is None
        assert info["active"] is False

    def test_get_session_info_with_session(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node._get_or_create_session()
        node._add_session_tokens(100)

        info = node.get_session_info()
        assert info["session_id"] is not None
        assert info["active"] is True
        assert info["token_count"] == 100
        assert info["context_length"] == 128000


class TestResolveWorkspaceRoot:
    """Tests for AgenticNode._resolve_workspace_root()."""

    def test_default_workspace_root(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        root = node._resolve_workspace_root()
        assert root is not None

    def test_node_specific_workspace(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.node_config["workspace_root"] = "/custom/path"
        root = node._resolve_workspace_root()
        assert root == "/custom/path"

    def test_tilde_expansion(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.node_config["workspace_root"] = "~/my_workspace"
        root = node._resolve_workspace_root()
        assert "~" not in root
        assert "my_workspace" in root


class TestGetToolCategory:
    """Tests for AgenticNode._get_tool_category()."""

    def test_skill_tools(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        assert node._get_tool_category("load_skill") == "skills"
        assert node._get_tool_category("skill_execute") == "skills"

    def test_db_tools(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        assert node._get_tool_category("list_tables") == "db_tools"
        assert node._get_tool_category("describe_table") == "db_tools"
        assert node._get_tool_category("execute_sql") == "db_tools"
        assert node._get_tool_category("get_sample_data") == "db_tools"
        assert node._get_tool_category("db_query") == "db_tools"

    def test_generic_tools(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        assert node._get_tool_category("some_random_tool") == "tools"

    def test_mcp_tools(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.mcp_servers = {"my_server": MagicMock()}
        assert node._get_tool_category("my_server_query") == "mcp"


class TestSetPermissionCallback:
    """Tests for set_permission_callback."""

    def test_set_callback(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)

        async def mock_callback(cat, name, ctx):
            return True

        node.set_permission_callback(mock_callback)
        assert node._permission_callback is mock_callback


class TestGetAvailableSkillsContext:
    """Tests for _get_available_skills_context."""

    def test_no_skill_manager(self, real_agent_config, mock_llm_create):
        node = _make_concrete_node(real_agent_config)
        node.skill_manager = None
        result = node._get_available_skills_context()
        assert result == ""
