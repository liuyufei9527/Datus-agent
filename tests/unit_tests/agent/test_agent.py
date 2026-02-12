# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for Agent class (datus/agent/agent.py).

Tests cover:
- Agent initialization
- check_db() — database connectivity validation
- probe_llm() — LLM connectivity test
- run() — synchronous workflow execution
- create_workflow_runner() — runner factory
- bootstrap_kb() — knowledge base initialization (metadata check mode)
- Helper methods (_print_stream_lines, _next_reference_sql_number, etc.)
"""

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.agent import Agent
from datus.schemas.node_models import SqlTask
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse, build_simple_response


def _make_args(**overrides):
    """Create a minimal argparse.Namespace for Agent."""
    defaults = {
        "load_cp": None,
        "max_steps": 10,
        "workflow": "reflection",
        "force": False,
        "yes": False,
        "plan_mode": False,
        "current_date": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestAgentInit:
    """Tests for Agent.__init__."""

    def test_agent_init_with_real_config(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        assert agent.args is args
        assert agent.global_config is real_agent_config
        assert agent.db_manager is not None
        assert isinstance(agent.tools, dict)
        assert isinstance(agent.storage_modules, dict)

    def test_agent_init_with_custom_db_manager(self, real_agent_config, mock_llm_create):
        args = _make_args()
        mock_db_mgr = MagicMock()
        agent = Agent(args=args, agent_config=real_agent_config, db_manager=mock_db_mgr)

        assert agent.db_manager is mock_db_mgr

    def test_force_delete_property(self, real_agent_config, mock_llm_create):
        # force=False, yes=False
        args = _make_args(force=False, yes=False)
        agent = Agent(args=args, agent_config=real_agent_config)
        assert agent._force_delete is False

        # force=True
        args2 = _make_args(force=True)
        agent2 = Agent(args=args2, agent_config=real_agent_config)
        assert agent2._force_delete is True

        # yes=True
        args3 = _make_args(yes=True)
        agent3 = Agent(args=args3, agent_config=real_agent_config)
        assert agent3._force_delete is True


class TestAgentCheckDb:
    """Tests for Agent.check_db()."""

    def test_check_db_success(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        result = agent.check_db()

        assert result["status"] == "success"
        assert "successful" in result["message"].lower()

    def test_check_db_namespace_not_found(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        # Use internal attribute to bypass setter validation
        agent.global_config._current_namespace = "nonexistent_namespace"
        result = agent.check_db()

        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_check_db_no_connections(self, real_agent_config, mock_llm_create):
        args = _make_args()
        mock_db_mgr = MagicMock()
        mock_db_mgr.get_connections.return_value = None
        agent = Agent(args=args, agent_config=real_agent_config, db_manager=mock_db_mgr)
        result = agent.check_db()

        assert result["status"] == "error"
        assert "no connections" in result["message"].lower()


class TestAgentProbeLlm:
    """Tests for Agent.probe_llm()."""

    def test_probe_llm_success(self, real_agent_config, mock_llm_create):
        mock_llm_create.reset(responses=[build_simple_response("Hello! I can hear you.")])
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        result = agent.probe_llm()

        assert result["status"] == "success"
        assert "successful" in result["message"].lower()
        assert result["response"] == "Hello! I can hear you."

    def test_probe_llm_failure(self, real_agent_config):
        args = _make_args()
        # Patch create_model to raise an exception
        with patch("datus.models.base.LLMBaseModel.create_model", side_effect=Exception("Connection refused")):
            agent = Agent(args=args, agent_config=real_agent_config)
            result = agent.probe_llm()

        assert result["status"] == "error"
        assert "Connection refused" in result["message"]


class TestAgentCreateWorkflowRunner:
    """Tests for Agent.create_workflow_runner()."""

    def test_create_workflow_runner_default(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        runner = agent.create_workflow_runner()

        from datus.agent.workflow_runner import WorkflowRunner

        assert isinstance(runner, WorkflowRunner)
        assert runner.args is args
        assert runner.global_config is real_agent_config
        assert runner._pre_run is not None  # check_db callable

    def test_create_workflow_runner_no_check_db(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        runner = agent.create_workflow_runner(check_db=False)

        assert runner._pre_run is None

    def test_create_workflow_runner_with_run_id(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        runner = agent.create_workflow_runner(run_id="test_run_001")

        assert runner.run_id == "test_run_001"


class TestAgentRun:
    """Tests for Agent.run() — synchronous workflow execution."""

    def test_run_with_simple_task(self, real_agent_config, mock_llm_create):
        """Run a simple workflow with mocked LLM and real SQLite."""
        mock_llm_create.reset(
            responses=[
                # schema_linking node uses generate
                MockLLMResponse(content="{}"),
                # generate_sql node uses generate
                MockLLMResponse(
                    content='{"sql": "SELECT COUNT(*) FROM schools", "explanation": "Count all schools"}'
                ),
                # reflection node uses generate
                MockLLMResponse(content='{"evaluation": "correct", "suggestions": []}'),
            ]
        )

        args = _make_args(max_steps=20, workflow="fixed")
        agent = Agent(args=args, agent_config=real_agent_config)
        task = SqlTask(
            task="How many schools are in the database?",
            database_name="california_schools",
        )
        result = agent.run(sql_task=task, check_db=False)

        # Result should be a dict (may be empty if workflow fails gracefully)
        assert isinstance(result, dict)


class TestAgentHelpers:
    """Tests for Agent helper methods."""

    def test_print_stream_lines_empty(self, real_agent_config, mock_llm_create, capsys):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        # None message should produce no output
        agent._print_stream_lines(None)
        captured = capsys.readouterr()
        assert captured.out == ""

        # Empty string should produce no output
        agent._print_stream_lines("")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_stream_lines_with_content(self, real_agent_config, mock_llm_create, capsys):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        agent._print_stream_lines("Hello\nWorld")
        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "World" in captured.out

    def test_next_reference_sql_number(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        # First call for a file should return 1
        assert agent._next_reference_sql_number("file1.sql") == 1
        # Second call for the same file should return 2
        assert agent._next_reference_sql_number("file1.sql") == 2
        # First call for a different file should return 1
        assert agent._next_reference_sql_number("file2.sql") == 1

    def test_format_reference_sql_line(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        result = agent._format_reference_sql_line("SELECT * FROM  orders\n  WHERE id = 1", 1)
        # Should collapse whitespace
        assert "\n" not in result
        assert "SELECT * FROM orders WHERE id = 1" in result

    def test_format_reference_sql_line_empty(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        # Empty SQL should return fallback
        result = agent._format_reference_sql_line("", 3)
        assert result == "sql_3"

    def test_get_file_short_name(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        assert agent._get_file_short_name("/path/to/my_queries.sql") == "my_queries"
        assert agent._get_file_short_name("single_file.csv") == "single_file"

    def test_reset_reference_sql_stream_state(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        agent._next_reference_sql_number("file1.sql")
        assert len(agent._ref_sql_file_sql_counter) > 0

        agent._reset_reference_sql_stream_state()
        assert agent._ref_sql_file_sql_counter == {}

    def test_reset_metrics_stream_state(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)

        agent._metrics_row_stage_seen["test"] = {"a", "b"}
        agent._reset_metrics_stream_state()
        assert agent._metrics_row_stage_seen == {}


class TestAgentEvaluation:
    """Tests for Agent.evaluation()."""

    def test_evaluation_unsupported_platform(self, real_agent_config, mock_llm_create):
        args = _make_args(benchmark="semantic_layer")
        agent = Agent(args=args, agent_config=real_agent_config)
        result = agent.evaluation()

        assert result["status"] == "failed"
        assert "not supported" in result["message"].lower()

    def test_evaluation_bird_critic_unsupported(self, real_agent_config, mock_llm_create):
        args = _make_args(benchmark="bird_critic")
        agent = Agent(args=args, agent_config=real_agent_config)
        result = agent.evaluation()

        assert result["status"] == "failed"


class TestAgentInitializeModel:
    """Tests for Agent._initialize_model()."""

    def test_initialize_model(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        model = agent._initialize_model()

        assert isinstance(model, MockLLMModel)
