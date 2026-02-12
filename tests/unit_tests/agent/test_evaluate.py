# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for evaluate.py (datus/agent/evaluate.py) and extended plan.py coverage.

Tests cover:
- setup_node_input() — node input setup dispatch
- evaluate_result() — evaluation + context update + next node setup
- update_context_from_node() — context update dispatch
- load_builtin_workflow_config()
- create_nodes_from_config() with parallel and selection configs
"""

import pytest

from datus.agent.evaluate import evaluate_result, setup_node_input, update_context_from_node
from datus.agent.node import Node
from datus.agent.plan import (
    create_nodes_from_config,
    generate_workflow,
    load_builtin_workflow_config,
)
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import BaseResult, SqlTask
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


class TestLoadBuiltinWorkflowConfig:
    """Tests for load_builtin_workflow_config()."""

    def test_load_config_returns_dict(self):
        config = load_builtin_workflow_config()
        assert isinstance(config, dict)
        assert "workflow" in config

    def test_load_config_has_fixed_and_reflection(self):
        config = load_builtin_workflow_config()
        workflows = config["workflow"]
        assert "fixed" in workflows
        assert "reflection" in workflows


class TestCreateNodesFromConfig:
    """Tests for create_nodes_from_config()."""

    def test_create_simple_nodes(self, real_agent_config, mock_llm_create):
        config = ["schema_linking", "generate_sql", "execute_sql", "output"]
        task = SqlTask(task="Count schools", database_name="california_schools")

        nodes = create_nodes_from_config(config, task, agent_config=real_agent_config)

        # Should include begin + 4 nodes = 5
        assert len(nodes) == 5
        assert nodes[0].type == NodeType.TYPE_BEGIN

    def test_create_nodes_with_parallel_dict(self, real_agent_config, mock_llm_create):
        config = [
            "schema_linking",
            {"parallel": [NodeType.TYPE_GENERATE_SQL, NodeType.TYPE_REASONING]},
            "output",
        ]
        task = SqlTask(task="Count schools", database_name="california_schools")

        nodes = create_nodes_from_config(config, task, agent_config=real_agent_config)

        node_types = [n.type for n in nodes]
        assert NodeType.TYPE_PARALLEL in node_types

    def test_create_nodes_with_selection_dict(self, real_agent_config, mock_llm_create):
        config = [
            "schema_linking",
            {"selection": "best_quality"},
            "output",
        ]
        task = SqlTask(task="Count schools", database_name="california_schools")

        nodes = create_nodes_from_config(config, task, agent_config=real_agent_config)

        node_types = [n.type for n in nodes]
        assert NodeType.TYPE_SELECTION in node_types

    def test_create_nodes_with_selection_string(self, real_agent_config, mock_llm_create):
        config = [
            "schema_linking",
            "selection",
            "output",
        ]
        task = SqlTask(task="Count schools", database_name="california_schools")

        nodes = create_nodes_from_config(config, task, agent_config=real_agent_config)

        node_types = [n.type for n in nodes]
        assert NodeType.TYPE_SELECTION in node_types

    def test_create_nodes_with_reflection_alias(self, real_agent_config, mock_llm_create):
        config = ["schema_linking", "reflection", "output"]
        task = SqlTask(task="Count schools", database_name="california_schools")

        nodes = create_nodes_from_config(config, task, agent_config=real_agent_config)

        node_types = [n.type for n in nodes]
        assert NodeType.TYPE_REFLECT in node_types

    def test_create_nodes_with_reasoning_alias(self, real_agent_config, mock_llm_create):
        config = ["schema_linking", "reason_sql", "output"]
        task = SqlTask(task="Count schools", database_name="california_schools")

        nodes = create_nodes_from_config(config, task, agent_config=real_agent_config)

        node_types = [n.type for n in nodes]
        assert NodeType.TYPE_REASONING in node_types

    def test_create_nodes_with_execute_alias(self, real_agent_config, mock_llm_create):
        config = ["schema_linking", "execute", "output"]
        task = SqlTask(task="Count schools", database_name="california_schools")

        nodes = create_nodes_from_config(config, task, agent_config=real_agent_config)

        node_types = [n.type for n in nodes]
        assert NodeType.TYPE_EXECUTE_SQL in node_types


class TestGenerateWorkflowExtended:
    """Extended tests for generate_workflow()."""

    def test_generate_fixed_has_output_node(self, real_agent_config, mock_llm_create):
        task = SqlTask(task="Count schools", database_name="california_schools")
        workflow = generate_workflow(task, "fixed", agent_config=real_agent_config)

        node_types = [n.type for n in workflow.nodes.values()]
        assert NodeType.TYPE_OUTPUT in node_types

    def test_generate_workflow_invalid_plan(self, real_agent_config, mock_llm_create):
        task = SqlTask(task="Count schools", database_name="california_schools")

        with pytest.raises(ValueError, match="Invalid plan type"):
            generate_workflow(task, "totally_invalid_plan", agent_config=real_agent_config)

    def test_generate_workflow_default_plan(self, real_agent_config, mock_llm_create):
        """When plan_type is empty, should fallback."""
        task = SqlTask(task="Count schools", database_name="california_schools")
        # Set a default workflow plan in config
        real_agent_config.workflow_plan = "fixed"

        workflow = generate_workflow(task, "", agent_config=real_agent_config)
        assert workflow is not None


class TestSetupNodeInput:
    """Tests for setup_node_input()."""

    def test_setup_input_for_schema_linking(self, real_agent_config, mock_llm_create):
        task = SqlTask(task="Count schools", database_name="california_schools")
        workflow = generate_workflow(task, "fixed", agent_config=real_agent_config)

        # Find schema linking node
        sl_node = None
        for node in workflow.nodes.values():
            if node.type == NodeType.TYPE_SCHEMA_LINKING:
                sl_node = node
                break

        if sl_node:
            result = setup_node_input(sl_node, workflow)
            assert result.get("success") is True

    def test_setup_input_for_begin_node(self, real_agent_config, mock_llm_create):
        task = SqlTask(task="Count schools", database_name="california_schools")
        workflow = generate_workflow(task, "fixed", agent_config=real_agent_config)

        begin_node = workflow.get_current_node()
        assert begin_node.type == NodeType.TYPE_BEGIN

        result = setup_node_input(begin_node, workflow)
        assert result.get("success") is True


class TestEvaluateResult:
    """Tests for evaluate_result()."""

    def test_evaluate_result_begin_node(self, real_agent_config, mock_llm_create):
        task = SqlTask(task="Count schools", database_name="california_schools")
        workflow = generate_workflow(task, "fixed", agent_config=real_agent_config)

        begin_node = workflow.get_current_node()
        begin_node.start()
        begin_node.complete(BaseResult(success=True))

        result = evaluate_result(begin_node, workflow)
        assert result["success"] is True

    def test_evaluate_result_last_node(self, real_agent_config, mock_llm_create):
        task = SqlTask(task="Count schools", database_name="california_schools")
        workflow = generate_workflow(task, "fixed", agent_config=real_agent_config)

        # Move to the last node
        while workflow.current_node_index < len(workflow.node_order) - 1:
            node = workflow.get_current_node()
            node.start()
            node.complete(BaseResult(success=True))
            workflow.advance_to_next_node()

        # Now at the last node
        last_node = workflow.get_current_node()
        last_node.start()
        last_node.complete(BaseResult(success=True))

        result = evaluate_result(last_node, workflow)
        assert result["success"] is True
        assert "finished" in result.get("message", "").lower()


class TestUpdateContextFromNode:
    """Tests for update_context_from_node()."""

    def test_update_context_begin_node_unknown_type(self, real_agent_config, mock_llm_create):
        """Begin node type ('start') is not handled by update_context_from_node,
        so it returns failure with 'Unknown node type'."""
        task = SqlTask(task="Count schools", database_name="california_schools")
        workflow = generate_workflow(task, "fixed", agent_config=real_agent_config)

        begin_node = workflow.get_current_node()
        begin_node.start()
        begin_node.complete(BaseResult(success=True))

        result = update_context_from_node(begin_node, workflow)
        # 'start' type is NOT in the handled list in update_context_from_node
        assert result["success"] is False
        assert "Unknown node type" in result["message"]

    def test_update_context_schema_linking_node(self, real_agent_config, mock_llm_create):
        """Schema linking node is an ACTION_TYPE, update_context dispatches to it.
        With a BaseResult instead of SchemaLinkingResult, it fails gracefully."""
        from datus.schemas.schema_linking_node_models import SchemaLinkingResult

        task = SqlTask(task="Count schools", database_name="california_schools")
        workflow = generate_workflow(task, "fixed", agent_config=real_agent_config)

        # Find schema linking node
        sl_node = None
        for node in workflow.nodes.values():
            if node.type == NodeType.TYPE_SCHEMA_LINKING:
                sl_node = node
                break

        if sl_node:
            # Use a proper SchemaLinkingResult with required fields
            sl_result = SchemaLinkingResult(
                success=True,
                table_schemas=[],
                table_values=[],
                schema_count=0,
                value_count=0,
            )
            sl_node.start()
            sl_node.complete(sl_result)
            result = update_context_from_node(sl_node, workflow)
            assert result["success"] is True

    def test_update_context_output_node(self, real_agent_config, mock_llm_create):
        task = SqlTask(task="Count schools", database_name="california_schools")
        workflow = generate_workflow(task, "fixed", agent_config=real_agent_config)

        # Find output node
        output_node = None
        for node in workflow.nodes.values():
            if node.type == NodeType.TYPE_OUTPUT:
                output_node = node
                break

        if output_node:
            output_node.start()
            output_node.complete(BaseResult(success=True))
            result = update_context_from_node(output_node, workflow)
            assert result["success"] is True
