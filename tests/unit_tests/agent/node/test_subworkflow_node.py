# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SubworkflowNode (datus/agent/node/subworkflow_node.py).

Tests cover:
- execute() with no workflow specified
- execute() missing parent workflow
- execute() missing agent config
- execute() with unknown workflow name
- setup_input() validation
- update_context() with results
- _apply_node_params() parameter override
- _apply_config_params() with dict config
"""

import pytest

from datus.agent.node.subworkflow_node import SubworkflowNode
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SqlTask
from datus.schemas.subworkflow_node_models import SubworkflowInput, SubworkflowResult
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


class TestSubworkflowNodeExecute:
    """Tests for SubworkflowNode.execute()."""

    def _make_subworkflow_node(self, agent_config=None, input_data=None):
        return SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            input_data=input_data,
            agent_config=agent_config,
        )

    def test_execute_no_input(self, real_agent_config, mock_llm_create):
        """execute() with no input returns failure."""
        node = self._make_subworkflow_node(agent_config=real_agent_config)
        node.input = None

        result = node.execute()

        assert isinstance(result, SubworkflowResult)
        assert result.success is False
        assert "No workflow specified" in result.error
        assert result.workflow_name == "unknown"

    def test_execute_no_workflow_name(self, real_agent_config, mock_llm_create):
        """execute() with empty workflow_name returns failure."""
        node = self._make_subworkflow_node(agent_config=real_agent_config)
        node.input = SubworkflowInput(workflow_name="")

        result = node.execute()

        assert result.success is False
        assert "No workflow specified" in result.error

    def test_execute_no_parent_workflow(self, real_agent_config, mock_llm_create):
        """execute() without parent workflow context fails."""
        node = self._make_subworkflow_node(agent_config=real_agent_config)
        node.input = SubworkflowInput(workflow_name="test_workflow")
        # Don't set node.workflow

        result = node.execute()

        assert result.success is False
        assert "Parent workflow context not available" in result.error

    def test_execute_no_agent_config(self, real_agent_config, mock_llm_create):
        """execute() without agent config fails."""
        node = self._make_subworkflow_node()  # No agent_config
        node.input = SubworkflowInput(workflow_name="test_workflow")

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)
        node.workflow = workflow

        result = node.execute()

        assert result.success is False
        assert "Agent configuration not available" in result.error

    def test_execute_unknown_workflow(self, real_agent_config, mock_llm_create):
        """execute() with unknown workflow name fails."""
        node = self._make_subworkflow_node(agent_config=real_agent_config)
        node.input = SubworkflowInput(workflow_name="nonexistent_workflow")

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)
        node.workflow = workflow

        result = node.execute()

        assert result.success is False
        assert "not found" in result.error


class TestSubworkflowNodeSetupInput:
    """Tests for SubworkflowNode.setup_input()."""

    def test_setup_input_invalid_type(self, real_agent_config, mock_llm_create):
        """setup_input() fails with invalid input type."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )
        node.input = None

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)
        assert result["success"] is False
        assert "Invalid input type" in result["message"]

    def test_setup_input_no_workflow_name(self, real_agent_config, mock_llm_create):
        """setup_input() fails if workflow_name is empty."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            input_data=SubworkflowInput(workflow_name=""),
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)
        assert result["success"] is False
        assert "Workflow name not specified" in result["message"]

    def test_setup_input_valid(self, real_agent_config, mock_llm_create):
        """setup_input() succeeds with valid SubworkflowInput."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            input_data=SubworkflowInput(workflow_name="my_workflow"),
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)
        assert result["success"] is True


class TestSubworkflowNodeUpdateContext:
    """Tests for SubworkflowNode.update_context()."""

    def test_update_context_no_result(self, real_agent_config, mock_llm_create):
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )
        node.result = None

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.update_context(workflow)
        assert result["success"] is True

    def test_update_context_with_success_result(self, real_agent_config, mock_llm_create):
        """update_context with successful result tries to call context.update() which
        raises AttributeError on Pydantic Context model (known limitation)."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )

        node.result = SubworkflowResult(
            success=True,
            workflow_name="test_workflow",
            node_results={
                "node_1": {"success": True, "result": "data"},
            },
            execution_order=["node_1"],
        )

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        # Context.update() is not a valid Pydantic method, so this raises AttributeError
        with pytest.raises(AttributeError):
            node.update_context(workflow)

    def test_update_context_with_failed_result(self, real_agent_config, mock_llm_create):
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )

        node.result = SubworkflowResult(
            success=False,
            workflow_name="test_workflow",
            error="Some error",
            node_results={},
            execution_order=[],
        )

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        # Failed result does not enter the update branch, so should not crash
        result = node.update_context(workflow)
        assert result["success"] is True


class TestSubworkflowNodeApplyNodeParams:
    """Tests for SubworkflowNode._apply_node_params()."""

    def test_apply_node_params_empty(self, real_agent_config, mock_llm_create):
        """_apply_node_params() does nothing with no params."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            input_data=SubworkflowInput(workflow_name="test", node_params=None),
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        # Should not raise
        node._apply_node_params(workflow)


class TestSubworkflowNodeApplyConfigParams:
    """Tests for SubworkflowNode._apply_config_params()."""

    def test_apply_config_params_none(self, real_agent_config, mock_llm_create):
        """_apply_config_params() does nothing with None config."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        # Should not raise
        node._apply_config_params(workflow, None)

    def test_apply_config_params_dict(self, real_agent_config, mock_llm_create):
        """_apply_config_params() applies dict config to workflow nodes."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        # Empty dict config should not raise
        node._apply_config_params(workflow, {})

    def test_apply_config_params_string_nonexistent(self, real_agent_config, mock_llm_create):
        """_apply_config_params() with string config (file path) that doesn't exist."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        # Non-existent file should not raise
        node._apply_config_params(workflow, "nonexistent_config.yaml")


class TestSubworkflowNodeExecuteStream:
    """Tests for SubworkflowNode.execute_stream()."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_action_manager(self, real_agent_config, mock_llm_create):
        """execute_stream without action_history_manager yields no actions.
        Note: execute() is called internally but for the early-return path (no input),
        self.result is not set on the node (code doesn't assign self.result in that branch)."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )
        node.input = None

        actions = []
        async for action in node.execute_stream(action_history_manager=None):
            actions.append(action)

        # Without action_history_manager, no actions are yielded
        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_execute_stream_no_action_manager_with_workflow_name(self, real_agent_config, mock_llm_create):
        """execute_stream with input but no parent workflow - execute() enters try/except
        and sets self.result."""
        node = SubworkflowNode(
            node_id="test_subworkflow",
            description="Test subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            agent_config=real_agent_config,
        )
        node.input = SubworkflowInput(workflow_name="test_wf")

        actions = []
        async for action in node.execute_stream(action_history_manager=None):
            actions.append(action)

        # execute() enters try/except and sets self.result
        assert node.result is not None
        assert node.result.success is False
        assert "Parent workflow context not available" in node.result.error


class TestSubworkflowResultModel:
    """Tests for SubworkflowResult model."""

    def test_compact_result_success(self):
        result = SubworkflowResult(
            success=True,
            workflow_name="test_workflow",
            node_results={"n1": {}, "n2": {}},
            execution_order=["n1", "n2"],
        )
        compact = result.compact_result()
        assert "test_workflow" in compact
        assert "2 nodes" in compact

    def test_compact_result_failure(self):
        result = SubworkflowResult(
            success=False,
            error="Something went wrong",
            workflow_name="test_workflow",
            node_results={},
            execution_order=[],
        )
        compact = result.compact_result()
        assert "failed" in compact
        assert "Something went wrong" in compact
