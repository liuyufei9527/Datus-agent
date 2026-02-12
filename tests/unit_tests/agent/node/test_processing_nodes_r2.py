# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Round 2 unit tests for processing nodes:
- ExecuteSQLNode
- OutputNode
- FixNode
- CompareNode
- DateParserNode
- DocSearchNode

Tests cover execute(), execute_stream(), setup_input(), update_context()
and edge cases for each node type.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext, SqlTask
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


def _make_workflow(agent_config, sql_contexts=None, task_text="Count schools"):
    task = SqlTask(task=task_text, database_name="california_schools")
    wf = Workflow(name="test_wf", task=task, agent_config=agent_config)
    if sql_contexts:
        wf.context.sql_contexts = sql_contexts
    return wf


# ===========================================================================
# ExecuteSQLNode Tests
# ===========================================================================


class TestExecuteSQLNode:
    """Tests for ExecuteSQLNode."""

    def _make_node(self, agent_config):
        from datus.agent.node.execute_sql_node import ExecuteSQLNode

        return ExecuteSQLNode(
            node_id="test_exec_sql",
            description="Test execute SQL",
            node_type=NodeType.TYPE_EXECUTE_SQL,
            agent_config=agent_config,
        )

    def test_strip_sql_markdown(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        result = node._strip_sql_markdown("```sql\nSELECT 1\n```")
        assert result == "SELECT 1"

    def test_strip_sql_markdown_no_markers(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        result = node._strip_sql_markdown("SELECT 1")
        assert result == "SELECT 1"

    def test_strip_sql_markdown_non_string(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        result = node._strip_sql_markdown(123)
        assert result == 123

    def test_setup_input(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.sql_query == "SELECT COUNT(*) FROM schools"

    def test_execute_sql_success(self, real_agent_config, mock_llm_create):
        from datus.schemas.node_models import ExecuteSQLInput

        node = self._make_node(real_agent_config)
        node.input = ExecuteSQLInput(
            sql_query="SELECT COUNT(*) FROM satscores",
            database_name="california_schools",
        )
        node.execute()
        assert node.result is not None
        assert node.result.success is True
        assert node.result.row_count >= 0

    def test_execute_sql_invalid_query(self, real_agent_config, mock_llm_create):
        from datus.schemas.node_models import ExecuteSQLInput

        node = self._make_node(real_agent_config)
        node.input = ExecuteSQLInput(
            sql_query="SELECT * FROM nonexistent_table_xyz",
            database_name="california_schools",
        )
        node.execute()
        assert node.result is not None
        assert node.result.success is False

    def test_update_context(self, real_agent_config, mock_llm_create):
        from datus.schemas.node_models import ExecuteSQLResult

        node = self._make_node(real_agent_config)
        node.result = ExecuteSQLResult(
            success=True,
            sql_return="100",
            row_count=1,
        )
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        result = node.update_context(wf)
        assert result["success"] is True
        assert wf.context.sql_contexts[-1].sql_return == "100"

    def test_update_context_no_contexts(self, real_agent_config, mock_llm_create):
        from datus.schemas.node_models import ExecuteSQLResult

        node = self._make_node(real_agent_config)
        node.result = ExecuteSQLResult(success=True, sql_return="100", row_count=1)
        wf = _make_workflow(real_agent_config, sql_contexts=[])
        result = node.update_context(wf)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_stream(self, real_agent_config, mock_llm_create):
        from datus.schemas.node_models import ExecuteSQLInput

        node = self._make_node(real_agent_config)
        node.input = ExecuteSQLInput(
            sql_query="SELECT COUNT(*) FROM satscores",
            database_name="california_schools",
        )
        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)
        # Should have connection and execution actions
        assert len(actions) >= 2
        # Last action should have results
        assert actions[-1].output is not None

    @pytest.mark.asyncio
    async def test_execute_stream_no_connector(self, real_agent_config, mock_llm_create):
        from datus.schemas.node_models import ExecuteSQLInput

        node = self._make_node(real_agent_config)
        node.input = ExecuteSQLInput(
            sql_query="SELECT 1",
            database_name="nonexistent_db_xyz",
        )
        node.agent_config = None  # Remove config to prevent connector creation
        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)
        # Should handle gracefully with FAILED status
        assert any(a.status == ActionStatus.FAILED for a in actions)


# ===========================================================================
# OutputNode Tests
# ===========================================================================


class TestOutputNode:
    """Tests for OutputNode."""

    def _make_node(self, agent_config):
        from datus.agent.node.output_node import OutputNode

        return OutputNode(
            node_id="test_output",
            description="Test output",
            node_type=NodeType.TYPE_OUTPUT,
            agent_config=agent_config,
        )

    def test_setup_input(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT 1", sql_return="1", row_count=1)
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.gen_sql == "SELECT 1"
        assert node.input.sql_result == "1"

    def test_update_context(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        wf = _make_workflow(real_agent_config)
        result = node.update_context(wf)
        assert result["success"] is True

    def test_execute(self, real_agent_config, mock_llm_create):
        from datus.schemas.node_models import OutputInput

        node = self._make_node(real_agent_config)
        node.input = OutputInput(
            finished=True,
            task_id="test_1",
            task="Count schools",
            database_name="california_schools",
            output_dir="/tmp/test_output",
            gen_sql="SELECT COUNT(*) FROM schools",
            sql_result="100",
            row_count=1,
        )
        node.execute()
        assert node.result is not None

    @pytest.mark.asyncio
    async def test_execute_stream(self, real_agent_config, mock_llm_create):
        from datus.schemas.node_models import OutputInput

        node = self._make_node(real_agent_config)
        node.input = OutputInput(
            finished=True,
            task_id="test_1",
            task="Count schools",
            database_name="california_schools",
            output_dir="/tmp/test_output",
            gen_sql="SELECT COUNT(*) FROM schools",
            sql_result="100",
            row_count=1,
        )
        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)
        assert len(actions) >= 1


# ===========================================================================
# FixNode Tests
# ===========================================================================


class TestFixNode:
    """Tests for FixNode."""

    def _make_node(self, agent_config):
        from datus.agent.node.fix_node import FixNode

        return FixNode(
            node_id="test_fix",
            description="Test fix",
            node_type=NodeType.TYPE_FIX,
            agent_config=agent_config,
        )

    def test_setup_input(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT * FROM tbl", sql_error="no such table: tbl")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.sql_context.sql_query == "SELECT * FROM tbl"

    def test_execute_no_model(self, real_agent_config, mock_llm_create):
        from datus.schemas.fix_node_models import FixInput

        node = self._make_node(real_agent_config)
        node.model = None
        sql_ctx = SQLContext(sql_query="SELECT 1", sql_error="error")
        node.input = FixInput(
            sql_task=SqlTask(task="Test", database_name="db"),
            sql_context=sql_ctx,
        )
        node.execute()
        assert node.result.success is False
        assert "not provided" in node.result.error

    def test_execute_with_model(self, real_agent_config, mock_llm_create):
        from datus.schemas.fix_node_models import FixInput

        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content='{"sql": "SELECT COUNT(*) FROM schools", "explanation": "Fixed table name"}'
                ),
            ]
        )
        node = self._make_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM tbl", sql_error="no such table")
        node.input = FixInput(
            sql_task=SqlTask(task="Count schools", database_name="california_schools"),
            sql_context=sql_ctx,
        )
        node.execute()
        assert node.result is not None
        # autofix_sql may produce various results, we just check it ran

    def test_update_context(self, real_agent_config, mock_llm_create):
        from datus.schemas.fix_node_models import FixResult

        node = self._make_node(real_agent_config)
        node.result = FixResult(
            success=True,
            sql_query="SELECT COUNT(*) FROM schools",
            explanation="Fixed table name",
        )
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM tbl")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        result = node.update_context(wf)
        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 2

    @pytest.mark.asyncio
    async def test_execute_stream_no_model(self, real_agent_config, mock_llm_create):
        from datus.schemas.fix_node_models import FixInput

        node = self._make_node(real_agent_config)
        node.model = None
        sql_ctx = SQLContext(sql_query="SELECT 1", sql_error="error")
        node.input = FixInput(
            sql_task=SqlTask(task="Test", database_name="db"),
            sql_context=sql_ctx,
        )
        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)
        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_execute_stream_with_model(self, real_agent_config, mock_llm_create):
        from datus.schemas.fix_node_models import FixInput

        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content='{"sql": "SELECT 1", "explanation": "Fixed"}'
                ),
            ]
        )
        node = self._make_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELEC 1", sql_error="syntax error")
        node.input = FixInput(
            sql_task=SqlTask(task="Test", database_name="california_schools"),
            sql_context=sql_ctx,
        )
        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)
        assert len(actions) >= 1


# ===========================================================================
# CompareNode Tests
# ===========================================================================


class TestCompareNode:
    """Tests for CompareNode."""

    def _make_node(self, agent_config):
        from datus.agent.node.compare_node import CompareNode

        return CompareNode(
            node_id="test_compare",
            description="Test compare",
            node_type=NodeType.TYPE_COMPARE,
            agent_config=agent_config,
        )

    def test_setup_input(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools", sql_return="100")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.sql_context.sql_query == "SELECT COUNT(*) FROM schools"

    def test_setup_input_with_expectation(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        node.input = "expected result should be 100"
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.expectation == "expected result should be 100"

    def test_execute_compare_no_model(self, real_agent_config, mock_llm_create):
        from datus.schemas.compare_node_models import CompareInput

        node = self._make_node(real_agent_config)
        node.model = None
        sql_ctx = SQLContext(sql_query="SELECT 1")
        node.input = CompareInput(
            sql_task=SqlTask(task="Test", database_name="db"),
            sql_context=sql_ctx,
        )
        from datus.utils.exceptions import DatusException

        with pytest.raises(DatusException):
            node.execute()

    def test_execute_compare_success(self, real_agent_config, mock_llm_create):
        from datus.schemas.compare_node_models import CompareInput

        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content='{"explanation": "Query looks correct", "suggest": "No changes needed"}'
                ),
            ]
        )
        node = self._make_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools", sql_return="100")
        node.input = CompareInput(
            sql_task=SqlTask(task="Count schools", database_name="california_schools"),
            sql_context=sql_ctx,
        )
        node.execute()
        assert node.result is not None
        assert node.result.success is True

    def test_update_context(self, real_agent_config, mock_llm_create):
        from datus.schemas.compare_node_models import CompareInput, CompareResult

        node = self._make_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools")
        node.input = CompareInput(
            sql_task=SqlTask(task="Count", database_name="db"),
            sql_context=sql_ctx,
        )
        node.result = CompareResult(
            success=True,
            explanation="Query is correct",
            suggest="No changes needed",
        )
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        result = node.update_context(wf)
        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 2

    @pytest.mark.asyncio
    async def test_execute_stream_no_model(self, real_agent_config, mock_llm_create):
        from datus.schemas.compare_node_models import CompareInput

        node = self._make_node(real_agent_config)
        node.model = None
        sql_ctx = SQLContext(sql_query="SELECT 1")
        node.input = CompareInput(
            sql_task=SqlTask(task="Test", database_name="db"),
            sql_context=sql_ctx,
        )
        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)
        assert len(actions) == 0


# ===========================================================================
# DateParserNode Tests
# ===========================================================================


class TestDateParserNode:
    """Tests for DateParserNode."""

    def _make_node(self, agent_config):
        from datus.agent.node.date_parser_node import DateParserNode

        return DateParserNode(
            node_id="test_date_parser",
            description="Test date parser",
            node_type=NodeType.TYPE_DATE_PARSER,
            agent_config=agent_config,
        )

    def test_get_language_setting_default(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        assert node._get_language_setting() == "en"

    def test_get_language_setting_no_config(self, real_agent_config, mock_llm_create):
        node = self._make_node(None)
        assert node._get_language_setting() == "en"

    def test_setup_input(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        wf = _make_workflow(real_agent_config, task_text="Show sales from last month")
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.sql_task.task == "Show sales from last month"

    def test_execute_date_parsing(self, real_agent_config, mock_llm_create):
        from datus.schemas.date_parser_node_models import DateParserInput

        mock_llm_create.reset(
            responses=[
                MockLLMResponse(content="[]"),
            ]
        )
        node = self._make_node(real_agent_config)
        node.input = DateParserInput(
            sql_task=SqlTask(task="Show sales from January 2025", database_name="california_schools")
        )
        node.execute()
        assert node.result is not None

    def test_update_context_success(self, real_agent_config, mock_llm_create):
        from datus.schemas.date_parser_node_models import DateParserResult

        node = self._make_node(real_agent_config)
        enriched_task = SqlTask(task="Show sales", database_name="db", date_ranges="2025-01-01 to 2025-01-31")
        node.result = DateParserResult(
            success=True,
            extracted_dates=[{"type": "absolute", "value": "2025-01-01"}],
            enriched_task=enriched_task,
            date_context="Date range: 2025-01-01 to 2025-01-31",
        )
        wf = _make_workflow(real_agent_config)
        result = node.update_context(wf)
        assert result["success"] is True
        assert wf.task == enriched_task

    def test_update_context_append_date_context(self, real_agent_config, mock_llm_create):
        from datus.schemas.date_parser_node_models import DateParserResult

        node = self._make_node(real_agent_config)
        enriched_task = SqlTask(task="Show sales", database_name="db")
        node.result = DateParserResult(
            success=True,
            extracted_dates=[{"type": "relative", "value": "last month"}],
            enriched_task=enriched_task,
            date_context="relative: last month",
        )
        wf = _make_workflow(real_agent_config)
        wf.date_context = "existing context"
        result = node.update_context(wf)
        assert result["success"] is True
        assert "existing context" in wf.date_context
        assert "relative: last month" in wf.date_context

    def test_update_context_failed_parsing(self, real_agent_config, mock_llm_create):
        from datus.schemas.date_parser_node_models import DateParserResult

        node = self._make_node(real_agent_config)
        node.result = DateParserResult(
            success=False,
            error="Parse error",
            extracted_dates=[],
            enriched_task=SqlTask(task="Test", database_name="db"),
            date_context="",
        )
        wf = _make_workflow(real_agent_config)
        result = node.update_context(wf)
        assert result["success"] is True
        assert "failed" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_stream(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        # DateParserNode's execute_stream yields a single None
        assert len(actions) == 1


# ===========================================================================
# DocSearchNode Tests
# ===========================================================================


class TestDocSearchNode:
    """Tests for DocSearchNode."""

    def _make_node(self, agent_config):
        from datus.agent.node.doc_search_node import DocSearchNode

        return DocSearchNode(
            node_id="test_doc_search",
            description="Test doc search",
            node_type=NodeType.TYPE_DOC_SEARCH,
            agent_config=agent_config,
        )

    def test_setup_input(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config)
        wf = _make_workflow(real_agent_config)
        wf.context.doc_search_keywords = ["school", "enrollment"]
        result = node.setup_input(wf)
        assert result["success"] is True
        assert node.input.keywords == ["school", "enrollment"]

    def test_update_context(self, real_agent_config, mock_llm_create):
        from datus.schemas.doc_search_node_models import DocSearchResult

        node = self._make_node(real_agent_config)
        node.result = DocSearchResult(success=True, docs=["doc1 content"])
        wf = _make_workflow(real_agent_config)
        result = node.update_context(wf)
        assert result["success"] is True
        assert wf.context.document_result is not None

    def test_execute(self, real_agent_config, mock_llm_create):
        from datus.schemas.doc_search_node_models import DocSearchInput

        node = self._make_node(real_agent_config)
        node.input = DocSearchInput(keywords=["test"], top_n=3)
        node.execute()
        assert node.result is not None

    @pytest.mark.asyncio
    async def test_execute_stream(self, real_agent_config, mock_llm_create):
        from datus.schemas.doc_search_node_models import DocSearchInput

        node = self._make_node(real_agent_config)
        node.input = DocSearchInput(keywords=["school"], top_n=3)
        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)
        assert len(actions) >= 1
