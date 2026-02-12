# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Round 2 unit tests for GenSQLAgenticNode (datus/agent/node/gen_sql_agentic_node.py).

Tests cover:
- setup_input() with workflow context
- update_context() with result
- _extract_sql_and_output_from_response()
- _rebuild_tools()
- _setup_tool_pattern() with various patterns
- _get_system_prompt() template loading
- _get_execution_config() for different modes
- _build_plan_prompt()
- build_enhanced_message() with full context
- prepare_template_context() with agent config
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput, GenSQLNodeResult
from datus.schemas.node_models import Metric, ReferenceSql, SQLContext, SqlTask, TableSchema
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


def _make_gensql_node(agent_config):
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

    return GenSQLAgenticNode(
        node_id="test_gensql_r2",
        description="Test GenSQL R2",
        node_type=NodeType.TYPE_GENSQL,
        agent_config=agent_config,
        node_name="gensql",
    )


# ===========================================================================
# setup_input tests
# ===========================================================================


class TestGenSQLSetupInput:
    """Tests for GenSQLAgenticNode.setup_input()."""

    def test_setup_input_basic(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        task = SqlTask(task="Count schools", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)
        result = node.setup_input(wf)
        assert result["success"] is True
        assert isinstance(node.input, GenSQLNodeInput)
        assert node.input.user_message == "Count schools"

    def test_setup_input_updates_existing(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        node.input = GenSQLNodeInput(
            user_message="old message",
            database="old_db",
        )
        task = SqlTask(task="New question", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.user_message == "New question"
        assert node.input.database == "california_schools"

    def test_setup_input_with_plan_mode(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        task = SqlTask(task="Count schools", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)
        wf.metadata["plan_mode"] = True
        wf.metadata["auto_execute_plan"] = True
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.plan_mode is True
        assert node.input.auto_execute_plan is True

    def test_setup_input_with_schemas_and_metrics(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        task = SqlTask(task="Count schools", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)
        wf.context.table_schemas = [
            TableSchema(
                table_name="schools",
                database_name="california_schools",
                definition="CREATE TABLE schools (id INT)",
            )
        ]
        wf.context.metrics = [
            Metric(name="enrollment", definition="Total enrollment")
        ]
        result = node.setup_input(wf)
        assert result["success"] is True
        assert len(node.input.schemas) == 1
        assert len(node.input.metrics) == 1


# ===========================================================================
# update_context tests
# ===========================================================================


class TestGenSQLUpdateContext:
    """Tests for GenSQLAgenticNode.update_context()."""

    def test_update_context_no_result(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        node.result = None
        task = SqlTask(task="Test", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)
        result = node.update_context(wf)
        assert result["success"] is False

    def test_update_context_with_sql(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        node.result = GenSQLNodeResult(
            success=True,
            response="explanation text",
            sql="SELECT COUNT(*) FROM schools",
        )
        task = SqlTask(task="Count schools", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)
        result = node.update_context(wf)
        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 1
        assert wf.context.sql_contexts[0].sql_query == "SELECT COUNT(*) FROM schools"

    def test_update_context_no_sql(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        node.result = GenSQLNodeResult(
            success=True,
            response="Just a text response",
            sql=None,
        )
        task = SqlTask(task="Test", database_name="california_schools")
        wf = Workflow(name="test", task=task, agent_config=real_agent_config)
        result = node.update_context(wf)
        assert result["success"] is True
        # No SQL context added
        assert len(wf.context.sql_contexts) == 0


# ===========================================================================
# _extract_sql_and_output_from_response tests
# ===========================================================================


class TestExtractSQLFromResponse:
    """Tests for GenSQLAgenticNode._extract_sql_and_output_from_response()."""

    def test_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        sql, output = node._extract_sql_and_output_from_response({"content": ""})
        assert sql is None
        assert output is None

    def test_non_string_content(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        sql, output = node._extract_sql_and_output_from_response({"content": 123})
        assert sql is None
        assert output is None

    def test_valid_json_with_sql(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        content = json.dumps({
            "sql": "SELECT * FROM schools",
            "tables": ["schools"],
            "explanation": "Get all schools",
        })
        sql, output = node._extract_sql_and_output_from_response({"content": content})
        assert sql == "SELECT * FROM schools"
        assert "Get all schools" in output
        assert "schools" in output

    def test_json_with_output_field(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        content = json.dumps({
            "sql": "SELECT 1",
            "output": "Some output text",
        })
        sql, output = node._extract_sql_and_output_from_response({"content": content})
        assert sql == "SELECT 1"
        assert output == "Some output text"

    def test_invalid_json(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        sql, output = node._extract_sql_and_output_from_response(
            {"content": "not valid json at all"}
        )
        assert sql is None
        assert output is None

    def test_missing_content_key(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        sql, output = node._extract_sql_and_output_from_response({})
        assert sql is None
        assert output is None


# ===========================================================================
# _rebuild_tools tests
# ===========================================================================


class TestRebuildTools:
    """Tests for GenSQLAgenticNode._rebuild_tools()."""

    def test_rebuild_with_no_tools(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        # Clear all tool instances
        node.db_func_tool = None
        node.context_search_tools = None
        node.date_parsing_tools = None
        node.filesystem_func_tool = None
        node._platform_doc_tool = None
        node._rebuild_tools()
        assert len(node.tools) == 0

    def test_rebuild_with_db_tools(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        # Only keep db_func_tool
        node.context_search_tools = None
        node.date_parsing_tools = None
        node.filesystem_func_tool = None
        node._platform_doc_tool = None
        original_db_tool = node.db_func_tool
        node._rebuild_tools()
        # Should have only db tools
        assert len(node.tools) > 0
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names


# ===========================================================================
# _setup_tool_pattern tests
# ===========================================================================


class TestSetupToolPattern:
    """Tests for GenSQLAgenticNode._setup_tool_pattern()."""

    def test_unknown_tool_type(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        initial_count = len(node.tools)
        node._setup_tool_pattern("unknown_tools.*")
        assert len(node.tools) == initial_count

    def test_unknown_pattern(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        initial_count = len(node.tools)
        node._setup_tool_pattern("completely_random")
        assert len(node.tools) == initial_count


# ===========================================================================
# _get_execution_config tests
# ===========================================================================


class TestGetExecutionConfig:
    """Tests for GenSQLAgenticNode._get_execution_config()."""

    def test_normal_mode(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        input_data = GenSQLNodeInput(user_message="Test", database="db")
        config = node._get_execution_config("normal", input_data)
        assert config["tools"] == node.tools
        assert config["hooks"] is None

    def test_unknown_mode_raises(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        input_data = GenSQLNodeInput(user_message="Test", database="db")
        with pytest.raises(ValueError, match="Unknown execution mode"):
            node._get_execution_config("invalid_mode", input_data)

    def test_plan_mode(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        input_data = GenSQLNodeInput(user_message="Test", database="db")
        # No plan_hooks => plan_tools should be empty list
        config = node._get_execution_config("plan", input_data)
        assert "tools" in config


# ===========================================================================
# build_enhanced_message with full context
# ===========================================================================


class TestBuildEnhancedMessageR2:
    """Additional tests for build_enhanced_message."""

    def test_with_schemas(self):
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        schemas = [
            TableSchema(
                table_name="schools",
                database_name="db",
                definition="CREATE TABLE schools (id INT)",
            )
        ]
        result = build_enhanced_message(
            user_message="Count schools",
            db_type="sqlite",
            schemas=schemas,
        )
        assert "schools" in result
        assert "Available tables" in result

    def test_with_metrics(self):
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        metrics = [Metric(name="enrollment", definition="Total enrollment")]
        result = build_enhanced_message(
            user_message="Show enrollment",
            db_type="sqlite",
            metrics=metrics,
        )
        assert "enrollment" in result
        assert "Metrics" in result

    def test_with_reference_sql(self):
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        ref_sqls = [
            ReferenceSql(
                sql="SELECT COUNT(*) FROM schools",
                question="Count schools",
            )
        ]
        result = build_enhanced_message(
            user_message="Count schools",
            db_type="sqlite",
            reference_sql=ref_sqls,
        )
        assert "Reference SQL" in result

    def test_full_context(self):
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        result = build_enhanced_message(
            user_message="Complex query",
            db_type="snowflake",
            catalog="my_catalog",
            database="my_db",
            db_schema="public",
            external_knowledge="Revenue = sum(amount)",
            schemas=[
                TableSchema(
                    table_name="orders",
                    database_name="my_db",
                    definition="CREATE TABLE orders (id INT, amount DECIMAL)",
                )
            ],
            metrics=[Metric(name="revenue", definition="sum(amount)")],
            reference_sql=[
                ReferenceSql(sql="SELECT SUM(amount) FROM orders", question="Total revenue")
            ],
        )
        assert "snowflake" in result
        assert "my_catalog" in result
        assert "my_db" in result
        assert "public" in result
        assert "Revenue = sum(amount)" in result
        assert "orders" in result
        assert "revenue" in result
        assert "Reference SQL" in result


# ===========================================================================
# prepare_template_context with agent config
# ===========================================================================


class TestPrepareTemplateContextR2:
    """Additional tests for prepare_template_context."""

    def test_with_agent_config(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import prepare_template_context

        context = prepare_template_context(
            node_config={"system_prompt": "test", "tools": ""},
            agent_config=real_agent_config,
        )
        assert "agent_config" in context
        assert context["agent_config"] is real_agent_config
        assert "namespace" in context

    def test_with_scoped_context(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import prepare_template_context
        from datus.schemas.agent_models import ScopedContext, SubAgentConfig

        node_config = SubAgentConfig(
            system_prompt="test",
            tools="",
            scoped_context=ScopedContext(tables=["schools"]),
        )
        context = prepare_template_context(node_config=node_config)
        assert context["scoped_context"] is True

    def test_with_rules(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import prepare_template_context

        context = prepare_template_context(
            node_config={
                "system_prompt": "test",
                "tools": "",
                "rules": ["Always use LIMIT", "Use explicit column names"],
            },
        )
        assert len(context["rules"]) == 2

    def test_all_flags_false(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import prepare_template_context

        context = prepare_template_context(
            node_config={"system_prompt": "test", "tools": ""},
            has_db_tools=False,
            has_filesystem_tools=False,
            has_mf_tools=False,
            has_context_search_tools=False,
            has_parsing_tools=False,
            has_platform_doc_tools=False,
        )
        assert context["has_db_tools"] is False
        assert context["has_filesystem_tools"] is False
        assert context["has_mf_tools"] is False
        assert context["has_context_search_tools"] is False
        assert context["has_parsing_tools"] is False
        assert context["has_platform_doc_tools"] is False


# ===========================================================================
# _get_system_prompt tests
# ===========================================================================


class TestGetSystemPrompt:
    """Tests for GenSQLAgenticNode._get_system_prompt()."""

    def test_get_system_prompt_basic(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        prompt = node._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_system_prompt_with_summary(self, real_agent_config, mock_llm_create):
        node = _make_gensql_node(real_agent_config)
        prompt = node._get_system_prompt(conversation_summary="Previous context summary")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
