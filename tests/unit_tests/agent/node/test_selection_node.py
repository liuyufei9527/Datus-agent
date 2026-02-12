# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SelectionNode (datus/agent/node/selection_node.py).

Tests cover:
- execute() with no candidates
- execute() with a single candidate
- execute() with multiple candidates (rule-based fallback)
- execute() with LLM-based selection
- setup_input() from workflow context
- update_context() on successful selection
"""

import json
from unittest.mock import MagicMock

import pytest

from datus.agent.node.selection_node import SelectionNode
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SqlTask
from datus.schemas.parallel_node_models import SelectionInput, SelectionResult
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


class TestSelectionNodeExecute:
    """Tests for SelectionNode.execute()."""

    def _make_selection_node(self, agent_config=None, input_data=None):
        return SelectionNode(
            node_id="test_selection",
            description="Test selection node",
            node_type=NodeType.TYPE_SELECTION,
            input_data=input_data,
            agent_config=agent_config,
        )

    def test_execute_no_candidates(self, real_agent_config, mock_llm_create):
        """execute() with no input returns failure."""
        node = self._make_selection_node(agent_config=real_agent_config)
        node.input = None

        result = node.execute()

        assert isinstance(result, SelectionResult)
        assert result.success is False
        assert "No candidate results" in result.error

    def test_execute_empty_candidates(self, real_agent_config, mock_llm_create):
        """execute() with empty candidates returns failure."""
        node = self._make_selection_node(agent_config=real_agent_config)
        node.input = SelectionInput(candidate_results={})

        result = node.execute()

        assert result.success is False

    def test_execute_single_candidate(self, real_agent_config, mock_llm_create):
        """execute() with a single candidate selects it directly."""
        candidate = {"success": True, "result": {"sql_query": "SELECT 1"}}
        node = self._make_selection_node(agent_config=real_agent_config)
        node.input = SelectionInput(
            candidate_results={"candidate_a": candidate},
        )

        result = node.execute()

        assert result.success is True
        assert result.selected_source == "candidate_a"
        assert result.selected_result == candidate
        assert "Only one candidate" in result.selection_reason

    def test_execute_multiple_candidates_rule_based(self, real_agent_config, mock_llm_create):
        """execute() with multiple candidates and no model uses rule-based selection."""
        candidates = {
            "candidate_a": {"success": True, "result": {"sql_query": "SELECT 1"}},
            "candidate_b": {"success": False, "error": "Failed"},
        }
        node = self._make_selection_node(agent_config=real_agent_config)
        node.input = SelectionInput(candidate_results=candidates)
        node.model = None  # No LLM model, force rule-based selection

        result = node.execute()

        assert result.success is True
        # Rule-based should prefer the successful candidate
        assert result.selected_source == "candidate_a"
        assert "Rule-based" in result.selection_reason

    def test_execute_multiple_candidates_llm_based(self, real_agent_config, mock_llm_create):
        """execute() with LLM model uses LLM-based selection."""
        candidates = {
            "candidate_a": {"success": True, "result": {"sql_query": "SELECT 1"}},
            "candidate_b": {"success": True, "result": {"sql_query": "SELECT 2"}},
        }

        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content=json.dumps(
                        {
                            "best_candidate": "candidate_b",
                            "reason": "Better query structure",
                            "score_analysis": {
                                "candidate_a": {"score": 7},
                                "candidate_b": {"score": 9},
                            },
                        }
                    )
                )
            ]
        )

        node = self._make_selection_node(agent_config=real_agent_config)
        node.input = SelectionInput(candidate_results=candidates)
        node.model = mock_llm_create

        result = node.execute()

        assert result.success is True
        assert result.selected_source == "candidate_b"
        assert "Better query structure" in result.selection_reason

    def test_execute_llm_returns_invalid_candidate_fallback(self, real_agent_config, mock_llm_create):
        """If LLM returns invalid candidate ID, fall back to rule-based."""
        candidates = {
            "candidate_a": {"success": True, "result": {"sql_query": "SELECT 1"}},
            "candidate_b": {"success": True, "result": {"sql_query": "SELECT 2"}},
        }

        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content=json.dumps(
                        {
                            "best_candidate": "nonexistent_candidate",
                            "reason": "Invalid",
                        }
                    )
                )
            ]
        )

        node = self._make_selection_node(agent_config=real_agent_config)
        node.input = SelectionInput(candidate_results=candidates)
        node.model = mock_llm_create

        result = node.execute()

        assert result.success is True
        assert "Rule-based" in result.selection_reason

    def test_execute_llm_exception_fallback(self, real_agent_config, mock_llm_create):
        """If LLM raises exception, fall back to rule-based."""
        candidates = {
            "candidate_a": {"success": True, "result": {"sql_query": "SELECT 1"}},
            "candidate_b": {"success": True, "result": {"sql_query": "SELECT 2"}},
        }

        # LLM model that raises an exception
        mock_model = MagicMock()
        mock_model.generate_with_json_output.side_effect = Exception("LLM error")

        node = self._make_selection_node(agent_config=real_agent_config)
        node.input = SelectionInput(candidate_results=candidates)
        node.model = mock_model

        result = node.execute()

        assert result.success is True
        assert "Rule-based" in result.selection_reason


class TestSelectionNodeRuleBasedSelection:
    """Tests for SelectionNode._rule_based_selection()."""

    def _make_selection_node(self, agent_config=None):
        return SelectionNode(
            node_id="test_selection",
            description="Test selection node",
            node_type=NodeType.TYPE_SELECTION,
            agent_config=agent_config,
        )

    def test_rule_based_prefers_success(self, real_agent_config, mock_llm_create):
        """Rule-based selection prefers candidates with success=True."""
        candidates = {
            "failed_one": {"success": False, "error": "Error"},
            "success_one": {"success": True, "result": {"data": "ok"}},
        }

        node = self._make_selection_node(agent_config=real_agent_config)
        result = node._rule_based_selection(candidates)

        assert result.success is True
        assert result.selected_source == "success_one"

    def test_rule_based_prefers_result_content(self, real_agent_config, mock_llm_create):
        """Rule-based selection prefers candidates with result content."""
        candidates = {
            "no_result": {"success": True},
            "with_result": {"success": True, "result": {"data": "some data"}},
        }

        node = self._make_selection_node(agent_config=real_agent_config)
        result = node._rule_based_selection(candidates)

        assert result.success is True
        assert result.selected_source == "with_result"

    def test_rule_based_all_failed(self, real_agent_config, mock_llm_create):
        """Rule-based selection still returns a result even if all candidates failed."""
        candidates = {
            "failed_a": {"success": False, "error": "Error A"},
            "failed_b": {"success": False, "error": "Error B"},
        }

        node = self._make_selection_node(agent_config=real_agent_config)
        result = node._rule_based_selection(candidates)

        # Still succeeds in selecting one (even though candidates all failed)
        assert result.success is True
        assert result.selected_source in candidates


class TestSelectionNodeSetupInput:
    """Tests for SelectionNode.setup_input()."""

    def test_setup_input_from_parallel_results(self, real_agent_config, mock_llm_create):
        """setup_input() should populate candidates from workflow context."""
        node = SelectionNode(
            node_id="test_selection",
            description="Test selection node",
            node_type=NodeType.TYPE_SELECTION,
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        # Add parallel results to workflow context
        parallel_results = {
            "child_1": {"success": True, "result": {"sql": "SELECT 1"}},
            "child_2": {"success": True, "result": {"sql": "SELECT 2"}},
        }
        workflow.context.update_parallel_results(parallel_results)

        result = node.setup_input(workflow)

        assert result["success"] is True
        assert isinstance(node.input, SelectionInput)
        assert len(node.input.candidate_results) == 2

    def test_setup_input_no_parallel_results_no_reasoning(self, real_agent_config, mock_llm_create):
        """setup_input() fails if no parallel results and no reasoning node."""
        node = SelectionNode(
            node_id="test_selection",
            description="Test selection node",
            node_type=NodeType.TYPE_SELECTION,
            agent_config=real_agent_config,
        )

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)

        assert result["success"] is False


class TestSelectionNodeUpdateContext:
    """Tests for SelectionNode.update_context()."""

    def test_update_context_no_result(self, real_agent_config, mock_llm_create):
        node = SelectionNode(
            node_id="test_selection",
            description="Test selection node",
            node_type=NodeType.TYPE_SELECTION,
            agent_config=real_agent_config,
        )
        node.result = None

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.update_context(workflow)
        assert result["success"] is True

    def test_update_context_with_successful_selection(self, real_agent_config, mock_llm_create):
        node = SelectionNode(
            node_id="test_selection",
            description="Test selection node",
            node_type=NodeType.TYPE_SELECTION,
            agent_config=real_agent_config,
        )

        selected = {"success": True, "result": {"sql_query": "SELECT 1"}}
        node.result = SelectionResult(
            success=True,
            selected_result=selected,
            selected_source="candidate_a",
            selection_reason="Best quality",
            all_candidates={"candidate_a": selected},
            score_analysis={"candidate_a": {"score": 10}},
        )

        task = SqlTask(task="Test task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)

        result = node.update_context(workflow)
        assert result["success"] is True


class TestSelectionNodeExecuteStream:
    """Tests for SelectionNode.execute_stream()."""

    @pytest.mark.asyncio
    async def test_execute_stream_single_candidate(self, real_agent_config, mock_llm_create):
        """execute_stream yields action history for single candidate selection."""
        node = SelectionNode(
            node_id="test_selection",
            description="Test selection node",
            node_type=NodeType.TYPE_SELECTION,
            agent_config=real_agent_config,
        )

        candidate = {"success": True, "result": {"sql_query": "SELECT 1"}}
        node.input = SelectionInput(
            candidate_results={"candidate_a": candidate},
        )

        # execute_stream requires an ActionHistoryManager-like object
        # The node's execute_stream checks for action_history_manager
        actions = []
        async for action in node.execute_stream(action_history_manager=None):
            actions.append(action)

        # Without action_history_manager, no actions are yielded
        assert len(actions) == 0
        # But the result should be set
        assert node.result is not None
        assert node.result.success is True
