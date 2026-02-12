# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for ParallelNode (datus/agent/node/parallel_node.py).

Tests cover:
- execute() with no child nodes
- execute() with valid child nodes
- setup_input() validation
- update_context() with results
- _create_child_nodes() node creation logic
- _execute_child_node() single child execution
"""

import pytest

from datus.agent.node.parallel_node import ParallelNode
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SqlTask
from datus.schemas.parallel_node_models import ParallelInput, ParallelResult
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


class TestParallelNodeExecute:
    """Tests for ParallelNode.execute()."""

    def _make_parallel_node(self, agent_config=None, input_data=None):
        return ParallelNode(
            node_id="test_parallel",
            description="Test parallel node",
            node_type=NodeType.TYPE_PARALLEL,
            input_data=input_data,
            agent_config=agent_config,
        )

    def test_execute_no_input(self, real_agent_config, mock_llm_create):
        """execute() with no input returns failure."""
        node = self._make_parallel_node(agent_config=real_agent_config)
        node.input = None

        result = node.execute()

        assert isinstance(result, ParallelResult)
        assert result.success is False
        assert "No child nodes" in result.error

    def test_execute_empty_child_nodes(self, real_agent_config, mock_llm_create):
        """execute() with empty child_nodes returns failure."""
        node = self._make_parallel_node(agent_config=real_agent_config)
        node.input = ParallelInput(child_nodes=[])

        result = node.execute()

        assert result.success is False
        assert "No child nodes" in result.error

    def test_execute_no_workflow(self, real_agent_config, mock_llm_create):
        """execute() without workflow context fails."""
        node = self._make_parallel_node(agent_config=real_agent_config)
        node.input = ParallelInput(child_nodes=[NodeType.TYPE_GENERATE_SQL])
        # No workflow set on the node

        result = node.execute()

        assert result.success is False
        assert "Workflow context not available" in result.error

    def test_execute_with_workflow_and_child_nodes(self, real_agent_config, mock_llm_create):
        """execute() with a valid workflow and child nodes runs them."""
        mock_llm_create.reset(
            responses=[
                MockLLMResponse(content='{"sql": "SELECT 1", "explanation": "test"}'),
                MockLLMResponse(content='{"sql": "SELECT 2", "explanation": "test2"}'),
            ]
        )

        node = self._make_parallel_node(agent_config=real_agent_config)
        node.input = ParallelInput(child_nodes=[NodeType.TYPE_GENERATE_SQL])

        task = SqlTask(task="Test parallel", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)
        node.workflow = workflow

        result = node.execute()

        # Result should have child_results
        assert isinstance(result, ParallelResult)
        assert isinstance(result.child_results, dict)
        assert len(result.execution_order) > 0


class TestParallelNodeSetupInput:
    """Tests for ParallelNode.setup_input()."""

    def test_setup_input_invalid_type(self, real_agent_config, mock_llm_create):
        """setup_input() fails with invalid input type."""
        node = ParallelNode(
            node_id="test_parallel",
            description="Test parallel node",
            node_type=NodeType.TYPE_PARALLEL,
            agent_config=real_agent_config,
        )
        node.input = None

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)
        assert result["success"] is False
        assert "Invalid input type" in result["message"]

    def test_setup_input_valid_action_types(self, real_agent_config, mock_llm_create):
        """setup_input() succeeds with valid action node types."""
        node = ParallelNode(
            node_id="test_parallel",
            description="Test parallel node",
            node_type=NodeType.TYPE_PARALLEL,
            input_data=ParallelInput(
                child_nodes=[NodeType.TYPE_GENERATE_SQL, NodeType.TYPE_REASONING]
            ),
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)
        assert result["success"] is True

    def test_setup_input_unsupported_child_type(self, real_agent_config, mock_llm_create):
        """setup_input() fails with unsupported child node type."""
        node = ParallelNode(
            node_id="test_parallel",
            description="Test parallel node",
            node_type=NodeType.TYPE_PARALLEL,
            input_data=ParallelInput(child_nodes=["invalid_type_xyz"]),
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)
        assert result["success"] is False
        assert "Unsupported child entry" in result["message"]


class TestParallelNodeUpdateContext:
    """Tests for ParallelNode.update_context()."""

    def test_update_context_with_results(self, real_agent_config, mock_llm_create):
        """update_context() stores parallel results in workflow context."""
        node = ParallelNode(
            node_id="test_parallel",
            description="Test parallel node",
            node_type=NodeType.TYPE_PARALLEL,
            agent_config=real_agent_config,
        )

        child_results = {
            "child_1": {"success": True, "result": {"sql": "SELECT 1"}},
            "child_2": {"success": True, "result": {"sql": "SELECT 2"}},
        }
        node.result = ParallelResult(
            success=True,
            child_results=child_results,
            execution_order=["child_1", "child_2"],
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.update_context(workflow)
        assert result["success"] is True

        # Verify context was updated
        assert workflow.context.parallel_results is not None

    def test_update_context_no_result(self, real_agent_config, mock_llm_create):
        """update_context() with no result doesn't crash."""
        node = ParallelNode(
            node_id="test_parallel",
            description="Test parallel node",
            node_type=NodeType.TYPE_PARALLEL,
            agent_config=real_agent_config,
        )
        node.result = None

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.update_context(workflow)
        assert result["success"] is True


class TestParallelNodeCreateChildNodes:
    """Tests for ParallelNode._create_child_nodes()."""

    def test_create_child_nodes_action_types(self, real_agent_config, mock_llm_create):
        """_create_child_nodes() creates correct child node instances."""
        node = ParallelNode(
            node_id="test_parallel",
            description="Test parallel node",
            node_type=NodeType.TYPE_PARALLEL,
            input_data=ParallelInput(child_nodes=[NodeType.TYPE_GENERATE_SQL]),
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)
        node.workflow = workflow

        child_nodes = node._create_child_nodes(workflow)

        assert len(child_nodes) == 1
        assert child_nodes[0].type == NodeType.TYPE_GENERATE_SQL
        assert "test_parallel_child_0" in child_nodes[0].id


class TestParallelNodeExecuteStream:
    """Tests for ParallelNode.execute_stream()."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_action_manager(self, real_agent_config, mock_llm_create):
        """execute_stream without action_history_manager still executes."""
        node = ParallelNode(
            node_id="test_parallel",
            description="Test parallel node",
            node_type=NodeType.TYPE_PARALLEL,
            agent_config=real_agent_config,
        )
        node.input = None

        actions = []
        async for action in node.execute_stream(action_history_manager=None):
            actions.append(action)

        # Without action_history_manager, no actions are yielded
        assert len(actions) == 0
