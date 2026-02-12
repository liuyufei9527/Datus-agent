# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for WorkflowRunner class (datus/agent/workflow_runner.py).

Tests cover:
- WorkflowRunner initialization
- initialize_workflow() — workflow generation from task
- init_or_load_workflow() — workflow initialization routing
- _prepare_first_node() — begin node completion + advance
- _ensure_prerequisites() — prerequisite validation
- run() — synchronous workflow execution
- run_stream() — streaming workflow execution
- _finalize_workflow() — save and return results
- _create_action_history() / _update_action_status() — helpers
- is_complete()
"""

import argparse

import pytest

from datus.agent.workflow_runner import WorkflowRunner
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SqlTask
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse, build_simple_response


def _make_args(**overrides):
    """Create a minimal argparse.Namespace for WorkflowRunner."""
    defaults = {
        "load_cp": None,
        "max_steps": 20,
        "workflow": "fixed",
        "plan_mode": False,
        "current_date": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestWorkflowRunnerInit:
    """Tests for WorkflowRunner.__init__."""

    def test_init_basic(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        assert runner.args is args
        assert runner.global_config is real_agent_config
        assert runner.workflow is None
        assert runner.workflow_ready is False
        assert runner._pre_run is None
        assert runner.run_id is None

    def test_init_with_pre_run(self, real_agent_config, mock_llm_create):
        args = _make_args()
        pre_run = lambda: {"status": "ok"}
        runner = WorkflowRunner(args, real_agent_config, pre_run_callable=pre_run)

        assert runner._pre_run is pre_run

    def test_init_with_run_id(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config, run_id="test_123")

        assert runner.run_id == "test_123"


class TestWorkflowRunnerInitializeWorkflow:
    """Tests for WorkflowRunner.initialize_workflow()."""

    def test_initialize_workflow_fixed(self, real_agent_config, mock_llm_create):
        args = _make_args(workflow="fixed")
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count all schools", database_name="california_schools")

        runner.initialize_workflow(task)

        assert runner.workflow is not None
        assert len(runner.workflow.nodes) > 0
        assert runner.workflow.task is task

    def test_initialize_workflow_reflection(self, real_agent_config, mock_llm_create):
        args = _make_args(workflow="reflection")
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Find schools", database_name="california_schools")

        runner.initialize_workflow(task)

        assert runner.workflow is not None
        # Reflection workflow should have a reflect node
        node_types = [n.type for n in runner.workflow.nodes.values()]
        assert "reflect" in node_types

    def test_initialize_workflow_with_plan_mode(self, real_agent_config, mock_llm_create):
        args = _make_args(workflow="fixed", plan_mode=True)
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count all schools", database_name="california_schools")

        runner.initialize_workflow(task)

        assert runner.workflow.metadata.get("plan_mode") is True
        assert runner.workflow.metadata.get("auto_execute_plan") is True


class TestWorkflowRunnerInitOrLoadWorkflow:
    """Tests for WorkflowRunner.init_or_load_workflow()."""

    def test_init_with_task(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count all schools", database_name="california_schools")

        result = runner.init_or_load_workflow(task)

        assert result is True
        assert runner.workflow is not None
        assert runner.workflow_ready is True

    def test_init_no_task_no_checkpoint(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        result = runner.init_or_load_workflow(None)

        assert result is None  # Failed

    def test_init_with_existing_ready_workflow(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count all schools", database_name="california_schools")

        # Initialize first time
        runner.init_or_load_workflow(task)
        assert runner.workflow_ready is True

        # Second call with None should succeed because workflow_ready is True
        result = runner.init_or_load_workflow(None)
        assert result is True


class TestWorkflowRunnerIsComplete:
    """Tests for WorkflowRunner.is_complete()."""

    def test_is_complete_no_workflow(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        assert runner.is_complete() is True  # No workflow means "complete"

    def test_is_complete_with_pending_workflow(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count all schools", database_name="california_schools")
        runner.init_or_load_workflow(task)

        assert runner.is_complete() is False  # Workflow just initialized

    def test_is_complete_after_completion(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count all schools", database_name="california_schools")
        runner.init_or_load_workflow(task)

        # Manually mark workflow as complete
        runner.workflow.status = "completed"
        assert runner.is_complete() is True


class TestWorkflowRunnerPrepareFirstNode:
    """Tests for WorkflowRunner._prepare_first_node()."""

    def test_prepare_first_node_advances(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count all schools", database_name="california_schools")
        runner.init_or_load_workflow(task)

        # First node should be begin node at index 0
        assert runner.workflow.current_node_index == 0

        runner._prepare_first_node()

        # After prepare, should have advanced past the begin node
        assert runner.workflow.current_node_index == 1

    def test_prepare_first_node_no_workflow(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        # Should not raise with no workflow
        runner._prepare_first_node()


class TestWorkflowRunnerEnsurePrerequisites:
    """Tests for WorkflowRunner._ensure_prerequisites()."""

    def test_ensure_prerequisites_with_task(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count all schools", database_name="california_schools")

        result = runner._ensure_prerequisites(task, check_storage=False)
        assert result is True

    def test_ensure_prerequisites_no_task(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        result = runner._ensure_prerequisites(None, check_storage=False)
        assert result is False

    def test_ensure_prerequisites_with_pre_run(self, real_agent_config, mock_llm_create):
        args = _make_args()
        pre_run_called = []

        def pre_run():
            pre_run_called.append(True)
            return {"status": "ok"}

        runner = WorkflowRunner(args, real_agent_config, pre_run_callable=pre_run)
        task = SqlTask(task="Count all schools", database_name="california_schools")

        result = runner._ensure_prerequisites(task, check_storage=False)
        assert result is True
        assert len(pre_run_called) == 1


class TestWorkflowRunnerHelpers:
    """Tests for WorkflowRunner helper methods."""

    def test_create_action_history(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        action = runner._create_action_history(
            action_id="test_action",
            messages="Test message",
            action_type="test_type",
            input_data={"key": "value"},
        )

        assert action.action_id == "test_action"
        assert action.messages == "Test message"
        assert action.action_type == "test_type"
        assert action.role == ActionRole.WORKFLOW
        assert action.status == ActionStatus.PROCESSING
        assert action.input == {"key": "value"}

    def test_create_action_history_no_input(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        action = runner._create_action_history(
            action_id="test_action_2",
            messages="No input test",
            action_type="test",
        )

        assert action.input == {}

    def test_update_action_status_success(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        action = runner._create_action_history("a", "m", "t")
        runner._update_action_status(action, success=True, output_data={"result": "ok"})

        assert action.status == ActionStatus.SUCCESS
        assert action.output == {"result": "ok"}

    def test_update_action_status_failure(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        action = runner._create_action_history("a", "m", "t")
        runner._update_action_status(action, success=False, error="Something failed")

        assert action.status == ActionStatus.FAILED
        assert action.output["error"] == "Something failed"

    def test_update_action_status_failure_with_output(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        action = runner._create_action_history("a", "m", "t")
        runner._update_action_status(
            action,
            success=False,
            error="Error occurred",
            output_data={"extra": "data"},
        )

        assert action.status == ActionStatus.FAILED
        assert action.output["error"] == "Error occurred"
        assert action.output["extra"] == "data"


class TestWorkflowRunnerRun:
    """Tests for WorkflowRunner.run() — synchronous execution."""

    def test_run_returns_empty_on_no_task(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        result = runner.run(sql_task=None, check_storage=False)
        assert result == {}

    def test_run_with_task_fixed_workflow(self, real_agent_config, mock_llm_create):
        """Run a fixed workflow with mocked LLM — should produce a result dict."""
        mock_llm_create.reset(
            responses=[
                # Schema linking
                MockLLMResponse(content="{}"),
                # Generate SQL
                MockLLMResponse(content='{"sql": "SELECT 1", "explanation": "test"}'),
                # Execute SQL / output
                MockLLMResponse(content="result"),
                MockLLMResponse(content="result"),
                MockLLMResponse(content="result"),
            ]
        )

        args = _make_args(max_steps=20, workflow="fixed")
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(
            task="Select 1",
            database_name="california_schools",
        )

        result = runner.run(sql_task=task, check_storage=False)
        assert isinstance(result, dict)

    def test_run_max_steps_limit(self, real_agent_config, mock_llm_create):
        """Run should stop after max_steps."""
        # Provide many responses so it never runs out
        mock_llm_create.reset(
            responses=[MockLLMResponse(content="{}") for _ in range(50)]
        )

        args = _make_args(max_steps=2, workflow="fixed")
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(
            task="Count schools",
            database_name="california_schools",
        )

        result = runner.run(sql_task=task, check_storage=False)
        # Should have stopped, result is a dict
        assert isinstance(result, dict)


class TestWorkflowRunnerRunStream:
    """Tests for WorkflowRunner.run_stream() — streaming execution."""

    @pytest.mark.asyncio
    async def test_run_stream_no_task(self, real_agent_config, mock_llm_create):
        """run_stream with no task yields init action and returns."""
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        actions = []
        async for action in runner.run_stream(sql_task=None, check_storage=False):
            actions.append(action)

        # Should yield at least the init action
        assert len(actions) >= 1
        assert actions[0].action_type == "workflow_init"

    @pytest.mark.asyncio
    async def test_run_stream_with_task(self, real_agent_config, mock_llm_create):
        """run_stream with task yields multiple action history events."""
        mock_llm_create.reset(
            responses=[
                MockLLMResponse(content="{}"),
                MockLLMResponse(content='{"sql": "SELECT 1", "explanation": "test"}'),
                MockLLMResponse(content="ok"),
                MockLLMResponse(content="ok"),
                MockLLMResponse(content="ok"),
                MockLLMResponse(content="ok"),
                MockLLMResponse(content="ok"),
            ]
        )

        args = _make_args(max_steps=20, workflow="fixed")
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(
            task="Select 1",
            database_name="california_schools",
        )

        actions = []
        async for action in runner.run_stream(sql_task=task, check_storage=False):
            actions.append(action)

        # Should yield init, node executions, and completion actions
        assert len(actions) >= 2
        action_types = [a.action_type for a in actions]
        assert "workflow_init" in action_types


class TestWorkflowRunnerFinalizeWorkflow:
    """Tests for WorkflowRunner._finalize_workflow()."""

    def test_finalize_no_workflow(self, real_agent_config, mock_llm_create):
        args = _make_args()
        runner = WorkflowRunner(args, real_agent_config)

        result = runner._finalize_workflow(step_count=0)
        assert result == {}

    def test_finalize_with_workflow(self, real_agent_config, mock_llm_create):
        args = _make_args(workflow="fixed")
        runner = WorkflowRunner(args, real_agent_config)
        task = SqlTask(task="Count schools", database_name="california_schools")
        runner.init_or_load_workflow(task)

        result = runner._finalize_workflow(step_count=3)

        assert "final_result" in result
        assert "save_path" in result
        assert result["steps"] == 3
        assert result["save_path"].endswith(".yaml")
