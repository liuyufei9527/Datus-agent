# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for ReflectNode (datus/agent/node/reflect_node.py).

Tests cover:
- setup_input() — input preparation from workflow
- execute() / _execute_reflect() — reflection analysis
- execute_stream() — streaming execution
- update_context() — context update with reflection results
- _execute_reflection_strategy() — strategy routing
- _execute_strategy() — strategy execution and workflow modification
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.reflect_node import ReflectNode
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import (
    ReflectionInput,
    ReflectionResult,
    SQLContext,
    SqlTask,
    StrategyType,
)
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


def _make_reflect_node(agent_config=None):
    return ReflectNode(
        node_id="test_reflect",
        description="Test reflection node",
        node_type=NodeType.TYPE_REFLECT,
        agent_config=agent_config,
    )


def _make_workflow(agent_config, sql_contexts=None):
    task = SqlTask(task="Count schools", database_name="california_schools")
    wf = Workflow(name="test_wf", task=task, agent_config=agent_config)
    if sql_contexts:
        wf.context.sql_contexts = sql_contexts
    return wf


class TestReflectNodeSetupInput:
    """Tests for ReflectNode.setup_input()."""

    def test_setup_input_basic(self, real_agent_config, mock_llm_create):
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])

        result = node.setup_input(wf)

        assert result["success"] is True
        assert node.input is not None
        assert isinstance(node.input, ReflectionInput)
        assert node.input.task_description is wf.task
        assert len(node.input.sql_context) == 1

    def test_setup_input_multiple_contexts(self, real_agent_config, mock_llm_create):
        node = _make_reflect_node(real_agent_config)
        ctx1 = SQLContext(sql_query="SELECT 1")
        ctx2 = SQLContext(sql_query="SELECT 2")
        wf = _make_workflow(real_agent_config, sql_contexts=[ctx1, ctx2])

        result = node.setup_input(wf)

        assert result["success"] is True
        assert len(node.input.sql_context) == 2


class TestReflectNodeExecute:
    """Tests for ReflectNode.execute() and _execute_reflect()."""

    def test_execute_no_model_raises(self, real_agent_config, mock_llm_create):
        """execute raises ValueError when model is None."""
        node = _make_reflect_node(real_agent_config)
        node.model = None
        node.input = ReflectionInput(
            task_description=SqlTask(task="Test", database_name="db"),
            sql_context=[SQLContext(sql_query="SELECT 1")],
        )
        with pytest.raises(ValueError, match="Model is required"):
            node.execute()

    def test_execute_empty_sql_context(self, real_agent_config, mock_llm_create):
        """execute returns failure when sql_context is empty."""
        node = _make_reflect_node(real_agent_config)
        node.model = mock_llm_create
        node.input = ReflectionInput(
            task_description=SqlTask(task="Test", database_name="db"),
            sql_context=[],
        )
        node.execute()
        assert node.result is not None
        assert node.result.success is False
        assert "No SQL context" in node.result.error

    def test_execute_success_strategy(self, real_agent_config, mock_llm_create):
        """execute with model returning SUCCESS classification."""
        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content='{"classification": "SUCCESS", "explanation": "Query is correct"}'
                ),
            ]
        )
        node = _make_reflect_node(real_agent_config)
        node.model = mock_llm_create
        node.input = ReflectionInput(
            task_description=SqlTask(task="Count schools", database_name="california_schools"),
            sql_context=[
                SQLContext(
                    sql_query="SELECT COUNT(*) FROM schools",
                    sql_return="1000",
                    row_count=1,
                )
            ],
        )
        node.execute()
        assert node.result is not None
        assert node.result.success is True
        assert node.result.strategy == "SUCCESS"

    def test_execute_unknown_strategy_fallback(self, real_agent_config, mock_llm_create):
        """execute with model returning invalid classification falls back to UNKNOWN."""
        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content='{"classification": "INVALID_STRATEGY", "explanation": "bad"}'
                ),
            ]
        )
        node = _make_reflect_node(real_agent_config)
        node.model = mock_llm_create
        node.input = ReflectionInput(
            task_description=SqlTask(task="Count schools", database_name="california_schools"),
            sql_context=[
                SQLContext(sql_query="SELECT COUNT(*) FROM schools", sql_return="1000", row_count=1)
            ],
        )
        node.execute()
        assert node.result.strategy == "UNKNOWN"

    def test_execute_model_error(self, real_agent_config, mock_llm_create):
        """execute handles model error gracefully."""
        mock_llm_create.reset(responses=[])  # No responses
        node = _make_reflect_node(real_agent_config)
        node.model = mock_llm_create
        node.input = ReflectionInput(
            task_description=SqlTask(task="Count schools", database_name="california_schools"),
            sql_context=[SQLContext(sql_query="SELECT 1", sql_return="1", row_count=1)],
        )
        node.execute()
        # generate_with_json_output returns {} when no responses => classification defaults to UNKNOWN
        assert node.result is not None


class TestReflectNodeExecuteStream:
    """Tests for ReflectNode.execute_stream()."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_model(self, real_agent_config, mock_llm_create):
        """execute_stream without model yields nothing."""
        node = _make_reflect_node(real_agent_config)
        node.model = None
        node.input = ReflectionInput(
            task_description=SqlTask(task="Test", database_name="db"),
            sql_context=[SQLContext(sql_query="SELECT 1")],
        )
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_execute_stream_success(self, real_agent_config, mock_llm_create):
        """execute_stream yields PROCESSING then SUCCESS actions."""
        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content='{"classification": "SUCCESS", "explanation": "Good query"}'
                ),
            ]
        )
        node = _make_reflect_node(real_agent_config)
        node.model = mock_llm_create
        node.input = ReflectionInput(
            task_description=SqlTask(task="Count schools", database_name="california_schools"),
            sql_context=[
                SQLContext(sql_query="SELECT COUNT(*) FROM schools", sql_return="100", row_count=1)
            ],
        )
        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert len(actions) == 2
        # First action: PROCESSING
        assert actions[0].status.value == "processing" or actions[0].status == "processing"
        # Second action: SUCCESS
        assert actions[1].output is not None
        assert actions[1].output["strategy"] == "SUCCESS"
        # Result should be stored on node
        assert node.result is not None
        assert node.result.strategy == "SUCCESS"


class TestReflectNodeUpdateContext:
    """Tests for ReflectNode.update_context()."""

    def test_update_context_success_strategy(self, real_agent_config, mock_llm_create):
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])

        node.input = ReflectionInput(
            task_description=wf.task,
            sql_context=[sql_ctx],
        )
        node.result = ReflectionResult(
            success=True,
            strategy="SUCCESS",
            details={"explanation": "Query is correct"},
        )

        result = node.update_context(wf)
        assert result["success"] is True

    def test_update_context_with_keywords(self, real_agent_config, mock_llm_create):
        """update_context extracts keywords from details."""
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM schools")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])

        node.input = ReflectionInput(
            task_description=wf.task,
            sql_context=[sql_ctx],
        )
        node.result = ReflectionResult(
            success=True,
            strategy="SUCCESS",
            details={"keywords": ["schools", "count"], "explanation": "ok"},
        )

        result = node.update_context(wf)
        assert result["success"] is True
        assert wf.context.doc_search_keywords == ["schools", "count"]

    def test_update_context_sql_mismatch(self, real_agent_config, mock_llm_create):
        """update_context fails when SQL context mismatch."""
        node = _make_reflect_node(real_agent_config)
        sql_ctx_wf = SQLContext(sql_query="SELECT 1")
        sql_ctx_input = SQLContext(sql_query="SELECT 2")  # Different
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx_wf])

        node.input = ReflectionInput(
            task_description=wf.task,
            sql_context=[sql_ctx_input],
        )
        node.result = ReflectionResult(
            success=True,
            strategy="SUCCESS",
            details={"explanation": "ok"},
        )

        result = node.update_context(wf)
        assert result["success"] is False
        assert "mismatch" in result["message"]

    def test_update_context_unknown_strategy(self, real_agent_config, mock_llm_create):
        """update_context fails with unknown strategy."""
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT 1")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])

        node.input = ReflectionInput(
            task_description=wf.task,
            sql_context=[sql_ctx],
        )
        node.result = ReflectionResult(
            success=True,
            strategy="TOTALLY_INVALID",
            details={"explanation": "bad"},
        )

        result = node.update_context(wf)
        assert result["success"] is False
        assert "Unknown" in result["message"]

    def test_update_context_exception_handling(self, real_agent_config, mock_llm_create):
        """update_context handles exceptions gracefully."""
        node = _make_reflect_node(real_agent_config)
        # No input set causes AttributeError
        node.result = ReflectionResult(
            success=True,
            strategy="SUCCESS",
            details={},
        )
        wf = _make_workflow(real_agent_config, sql_contexts=[SQLContext(sql_query="SELECT 1")])
        # node.input is not set
        result = node.update_context(wf)
        assert result["success"] is False
        assert "failed" in result["message"].lower()


class TestReflectNodeExecuteStrategy:
    """Tests for ReflectNode._execute_reflection_strategy()."""

    def test_success_strategy(self, real_agent_config, mock_llm_create):
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT 1")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        wf.reflection_round = 0

        result = node._execute_reflection_strategy("SUCCESS", {}, wf)
        assert result["success"] is True
        assert "go on to output" in result["message"]

    def test_max_round_exceeded(self, real_agent_config, mock_llm_create):
        """When reflection_round exceeds max, should exit."""
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT 1")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])

        with patch("datus.agent.node.reflect_node.get_env_int", return_value=3):
            wf.reflection_round = 4  # > max_round (3)
            result = node._execute_reflection_strategy("DOC_SEARCH", {}, wf)
            assert result["success"] is True
            assert "exceeded" in result["message"]

    def test_unknown_strategy_type(self, real_agent_config, mock_llm_create):
        """Unknown strategy returns failure."""
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT 1")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        wf.reflection_round = 0

        with patch("datus.agent.node.reflect_node.get_env_int", return_value=3):
            result = node._execute_reflection_strategy("TOTALLY_UNKNOWN", {}, wf)
            assert result["success"] is False
            assert "Unknown strategy" in result["message"]


class TestReflectNodeExecuteStrategyImpl:
    """Tests for ReflectNode._execute_strategy()."""

    def test_simple_regenerate_with_sql(self, real_agent_config, mock_llm_create):
        """SIMPLE_REGENERATE with sql in details appends to sql_contexts."""
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT 1")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        wf.reflection_round = 0

        node.result = ReflectionResult(
            success=True,
            strategy="SIMPLE_REGENERATE",
            details={"sql": "SELECT COUNT(*) FROM schools", "explanation": "Fixed query"},
        )

        result = node._execute_strategy(
            node.result.details, wf, StrategyType.SIMPLE_REGENERATE
        )
        assert result["success"] is True
        # Should have added a new SQL context
        assert len(wf.context.sql_contexts) == 2

    def test_simple_regenerate_without_sql(self, real_agent_config, mock_llm_create):
        """SIMPLE_REGENERATE without sql in details — logs warning but continues."""
        node = _make_reflect_node(real_agent_config)
        sql_ctx = SQLContext(sql_query="SELECT 1")
        wf = _make_workflow(real_agent_config, sql_contexts=[sql_ctx])
        wf.reflection_round = 0

        node.result = ReflectionResult(
            success=True,
            strategy="SIMPLE_REGENERATE",
            details={"explanation": "Try again"},
        )

        result = node._execute_strategy(
            node.result.details, wf, StrategyType.SIMPLE_REGENERATE
        )
        assert result["success"] is True
        # No new SQL added
        assert len(wf.context.sql_contexts) == 1
