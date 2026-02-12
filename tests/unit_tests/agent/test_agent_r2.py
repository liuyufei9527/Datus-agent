# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Round 2 unit tests for Agent class (datus/agent/agent.py).

Targets uncovered lines: run_stream, _emit_reference_sql_event,
_emit_metrics_event, benchmark*, evaluation, _refresh_scoped_agents,
bootstrap_kb (more branches), _check_benchmark_file, _cleanup_benchmark_output_paths,
generate_dataset, bootstrap_platform_doc.
"""

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.agent import Agent, bootstrap_platform_doc
from datus.schemas.batch_events import BatchEvent, BatchStage
from datus.schemas.node_models import SqlTask
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse, build_simple_response


def _make_args(**overrides):
    defaults = {
        "load_cp": None,
        "max_steps": 10,
        "workflow": "fixed",
        "force": False,
        "yes": False,
        "plan_mode": False,
        "current_date": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ===========================================================================
# run_stream tests
# ===========================================================================


class TestAgentRunStream:
    """Tests for Agent.run_stream() — async streaming execution."""

    @pytest.mark.asyncio
    async def test_run_stream_basic(self, real_agent_config, mock_llm_create):
        """run_stream yields ActionHistory objects for a simple task."""
        mock_llm_create.reset(
            responses=[
                MockLLMResponse(content="{}"),
                MockLLMResponse(content='{"sql": "SELECT 1", "explanation": "test"}'),
                MockLLMResponse(content="ok"),
                MockLLMResponse(content="ok"),
                MockLLMResponse(content="ok"),
            ]
        )

        args = _make_args(max_steps=20, workflow="fixed")
        agent = Agent(args=args, agent_config=real_agent_config)
        task = SqlTask(task="Select 1", database_name="california_schools")

        actions = []
        async for action in agent.run_stream(sql_task=task, run_id="test_stream"):
            actions.append(action)

        assert len(actions) >= 1
        # Actions should have action_type attributes
        for a in actions:
            assert hasattr(a, "action_type")


# ===========================================================================
# _emit_reference_sql_event tests
# ===========================================================================


class TestEmitReferenceSqlEvent:
    """Tests for Agent._emit_reference_sql_event()."""

    def _make_agent(self, real_agent_config, mock_llm_create):
        args = _make_args()
        return Agent(args=args, agent_config=real_agent_config)

    def test_group_started(self, real_agent_config, mock_llm_create, capsys):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="reference_sql_init",
            stage=BatchStage.GROUP_STARTED,
            group_id="/path/to/queries.sql",
        )
        agent._emit_reference_sql_event(event)
        output = capsys.readouterr().out
        assert "queries" in output
        assert "start processing" in output

    def test_group_completed(self, real_agent_config, mock_llm_create, capsys):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="reference_sql_init",
            stage=BatchStage.GROUP_COMPLETED,
            group_id="/path/to/queries.sql",
        )
        agent._emit_reference_sql_event(event)
        output = capsys.readouterr().out
        assert "completed" in output

    def test_item_started(self, real_agent_config, mock_llm_create, capsys):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="reference_sql_init",
            stage=BatchStage.ITEM_STARTED,
            group_id="/path/to/queries.sql",
            payload={"sql": "SELECT * FROM orders WHERE id = 1"},
        )
        agent._emit_reference_sql_event(event)
        output = capsys.readouterr().out
        assert "#1" in output
        assert "SELECT" in output

    def test_item_processing(self, real_agent_config, mock_llm_create, capsys):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="reference_sql_init",
            stage=BatchStage.ITEM_PROCESSING,
            group_id="/path/to/queries.sql",
            payload={"output": {"raw_output": "Processing summary"}},
        )
        agent._emit_reference_sql_event(event)
        output = capsys.readouterr().out
        assert "Processing summary" in output

    def test_item_failed(self, real_agent_config, mock_llm_create, capsys):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="reference_sql_init",
            stage=BatchStage.ITEM_FAILED,
            group_id="/path/to/queries.sql",
            error="SQL parse error",
        )
        agent._emit_reference_sql_event(event)
        output = capsys.readouterr().out
        assert "SQL parse error" in output

    def test_item_failed_no_error(self, real_agent_config, mock_llm_create, capsys):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="reference_sql_init",
            stage=BatchStage.ITEM_FAILED,
            group_id="/path/to/queries.sql",
        )
        agent._emit_reference_sql_event(event)
        # No error, nothing printed for the error line
        output = capsys.readouterr().out
        assert output == ""  # empty error means nothing printed


# ===========================================================================
# _emit_metrics_event tests
# ===========================================================================


class TestEmitMetricsEvent:
    """Tests for Agent._emit_metrics_event()."""

    def _make_agent(self, real_agent_config, mock_llm_create):
        args = _make_args()
        return Agent(args=args, agent_config=real_agent_config)

    def test_task_started(self, real_agent_config, mock_llm_create):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="metrics_init",
            stage=BatchStage.TASK_STARTED,
        )
        # Should not raise
        agent._emit_metrics_event(event)

    def test_task_completed(self, real_agent_config, mock_llm_create):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="metrics_init",
            stage=BatchStage.TASK_COMPLETED,
        )
        # Should not raise
        agent._emit_metrics_event(event)

    def test_item_processing_with_raw_output(self, real_agent_config, mock_llm_create, capsys):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="metrics_init",
            stage=BatchStage.ITEM_PROCESSING,
            action_name="gen_metrics",
            payload={"output": {"raw_output": "metric generated"}},
        )
        agent._emit_metrics_event(event)
        output = capsys.readouterr().out
        assert "gen_metrics" in output
        assert "metric generated" in output

    def test_item_processing_with_semantic_model(self, real_agent_config, mock_llm_create, capsys):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="metrics_init",
            stage=BatchStage.ITEM_PROCESSING,
            action_name="gen_semantic_model",
            payload={"output": {"semantic_model": "/path/to/model.yaml"}},
        )
        agent._emit_metrics_event(event)
        output = capsys.readouterr().out
        assert "semantic_model" in output
        assert "/path/to/model.yaml" in output

    def test_item_processing_deduplication(self, real_agent_config, mock_llm_create, capsys):
        """Same action_name should only print header once."""
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event1 = BatchEvent(
            biz_name="metrics_init",
            stage=BatchStage.ITEM_PROCESSING,
            action_name="gen_metrics",
            payload={"output": {"raw_output": "line1"}},
        )
        event2 = BatchEvent(
            biz_name="metrics_init",
            stage=BatchStage.ITEM_PROCESSING,
            action_name="gen_metrics",
            payload={"output": {"raw_output": "line2"}},
        )
        agent._emit_metrics_event(event1)
        agent._emit_metrics_event(event2)
        output = capsys.readouterr().out
        # "gen_metrics:" should appear exactly once as header
        assert output.count("gen_metrics:") == 1

    def test_item_completed(self, real_agent_config, mock_llm_create):
        agent = self._make_agent(real_agent_config, mock_llm_create)
        event = BatchEvent(
            biz_name="metrics_init",
            stage=BatchStage.ITEM_COMPLETED,
        )
        # Should not raise
        agent._emit_metrics_event(event)


# ===========================================================================
# _refresh_scoped_agents tests
# ===========================================================================


class TestRefreshScopedAgents:
    """Tests for Agent._refresh_scoped_agents()."""

    def _make_agent(self, real_agent_config, mock_llm_create):
        args = _make_args()
        return Agent(args=args, agent_config=real_agent_config)

    def test_unsupported_component(self, real_agent_config, mock_llm_create):
        """Components not in SUB_AGENT_COMPONENTS should return early."""
        agent = self._make_agent(real_agent_config, mock_llm_create)
        # Should not raise
        agent._refresh_scoped_agents("nonexistent_component", "overwrite")

    def test_unsupported_strategy(self, real_agent_config, mock_llm_create):
        """Strategy not in {overwrite, incremental} should return early."""
        agent = self._make_agent(real_agent_config, mock_llm_create)
        agent._refresh_scoped_agents("metadata", "check")

    def test_no_agentic_nodes(self, real_agent_config, mock_llm_create):
        """If agentic_nodes is empty/None should return early."""
        agent = self._make_agent(real_agent_config, mock_llm_create)
        original = real_agent_config.agentic_nodes
        real_agent_config.agentic_nodes = {}
        agent._refresh_scoped_agents("metadata", "overwrite")
        real_agent_config.agentic_nodes = original


# ===========================================================================
# _check_benchmark_file tests
# ===========================================================================


class TestCheckBenchmarkFile:
    """Tests for Agent._check_benchmark_file()."""

    def test_file_not_found(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        with pytest.raises(FileNotFoundError, match="Benchmarking task file not found"):
            agent._check_benchmark_file("/nonexistent/path.csv")


# ===========================================================================
# _cleanup_benchmark_output_paths tests
# ===========================================================================


class TestCleanupBenchmarkOutputPaths:
    """Tests for Agent._cleanup_benchmark_output_paths()."""

    def test_cleanup_nonexistent_paths(self, real_agent_config, mock_llm_create):
        """Should not raise when paths don't exist."""
        args = _make_args(force=True)
        agent = Agent(args=args, agent_config=real_agent_config)
        # This should not raise even when paths don't exist
        agent._cleanup_benchmark_output_paths("/nonexistent/benchmark/path")


# ===========================================================================
# check_db with dict connections
# ===========================================================================


class TestCheckDbDictConnections:
    """Tests for check_db with dict-style connections."""

    def test_check_db_with_dict_connections(self, real_agent_config, mock_llm_create):
        """check_db handles dict connections properly."""
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        # The default test config should have dict connections
        result = agent.check_db()
        assert result["status"] == "success"

    def test_check_db_with_single_connection(self, real_agent_config, mock_llm_create):
        """check_db handles non-dict connection return."""
        args = _make_args()
        mock_conn = MagicMock()
        mock_conn.test_connection.return_value = True
        mock_db_mgr = MagicMock()
        mock_db_mgr.get_connections.return_value = mock_conn
        agent = Agent(args=args, agent_config=real_agent_config, db_manager=mock_db_mgr)
        result = agent.check_db()
        assert result["status"] == "success"


# ===========================================================================
# evaluation tests
# ===========================================================================


class TestAgentEvaluationR2:
    """Additional evaluation tests."""

    def test_evaluation_with_valid_benchmark(self, real_agent_config, mock_llm_create):
        """evaluation with valid benchmark platform calls evaluate_benchmark_and_report."""
        args = _make_args(
            benchmark="bird_dev",
            task_ids=None,
            output_file=None,
            run_id=None,
            summary_report_file=None,
        )
        agent = Agent(args=args, agent_config=real_agent_config)

        with patch(
            "datus.agent.agent.evaluate_benchmark_and_report",
            return_value={"status": "success", "generated_time": "2025-01-01", "error": None},
        ):
            result = agent.evaluation()
            assert result["status"] == "success"


# ===========================================================================
# benchmark_bird_critic tests
# ===========================================================================


class TestBenchmarkBirdCritic:
    """Test benchmark_bird_critic method."""

    def test_benchmark_bird_critic_returns_none(self, real_agent_config, mock_llm_create):
        args = _make_args()
        agent = Agent(args=args, agent_config=real_agent_config)
        # Currently a pass/no-op
        result = agent.benchmark_bird_critic()
        assert result is None


# ===========================================================================
# bootstrap_platform_doc tests
# ===========================================================================


class TestBootstrapPlatformDoc:
    """Tests for the standalone bootstrap_platform_doc function."""

    def test_no_source(self, real_agent_config, mock_llm_create):
        """When no source is provided, returns success with skip message."""
        args = _make_args(update_strategy="check", pool_size=4, platform="default")
        result = bootstrap_platform_doc(args, real_agent_config)
        assert result["status"] == "success"
        assert "skipped" in result["message"]


# ===========================================================================
# Agent with current_date
# ===========================================================================


class TestAgentWithCurrentDate:
    """Tests Agent with current_date argument."""

    def test_agent_init_with_current_date(self, real_agent_config, mock_llm_create):
        args = _make_args(current_date="2025-01-15")
        agent = Agent(args=args, agent_config=real_agent_config)
        assert agent.args.current_date == "2025-01-15"
