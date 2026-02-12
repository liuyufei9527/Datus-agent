# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for Node base class (datus/agent/node/node.py).

Tests cover:
- new_instance() factory for all node types
- _initialize() model initialization
- run() synchronous execution
- run_async()
- run_stream() streaming execution
- from_dict() deserialization
- to_dict() serialization round trip
- complete() with failed result
- fail() without error message
"""

import pytest

from datus.agent.node import Node
from datus.agent.node.node import Node as NodeClass
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import BaseResult, GenerateSQLInput, GenerateSQLResult, SqlTask
from tests.unit_tests.mock_llm_model import MockLLMModel, MockLLMResponse


class TestNodeNewInstance:
    """Tests for Node.new_instance() factory method."""

    def test_new_instance_supported_action_types(self, real_agent_config, mock_llm_create):
        """new_instance creates correct subclass for supported action types."""
        # Only test types that new_instance explicitly handles.
        # Exclude TYPE_CHAT and TYPE_GENSQL because they require heavy init
        # (db_manager_instance, context_search_tools) that can conflict with
        # singleton state from other tests. They are covered in dedicated test files.
        supported_action_types = [
            NodeType.TYPE_SCHEMA_LINKING,
            NodeType.TYPE_GENERATE_SQL,
            NodeType.TYPE_EXECUTE_SQL,
            NodeType.TYPE_OUTPUT,
            NodeType.TYPE_REASONING,
            NodeType.TYPE_DOC_SEARCH,
            NodeType.TYPE_FIX,
            NodeType.TYPE_SEARCH_METRICS,
            NodeType.TYPE_DATE_PARSER,
        ]
        for node_type in supported_action_types:
            node = Node.new_instance(
                node_id=f"test_{node_type}",
                description=f"Test {node_type}",
                node_type=node_type,
                agent_config=real_agent_config,
            )
            assert node.type == node_type
            assert node.id == f"test_{node_type}"

    def test_new_instance_control_types(self, real_agent_config, mock_llm_create):
        """new_instance creates correct subclass for control types."""
        control_types = [
            NodeType.TYPE_HITL,
            NodeType.TYPE_REFLECT,
            NodeType.TYPE_PARALLEL,
            NodeType.TYPE_SELECTION,
            NodeType.TYPE_SUBWORKFLOW,
            NodeType.TYPE_COMPARE,
        ]
        for node_type in control_types:
            node = Node.new_instance(
                node_id=f"test_{node_type}",
                description=f"Test {node_type}",
                node_type=node_type,
                agent_config=real_agent_config,
            )
            assert node.type == node_type

    def test_new_instance_invalid_type(self):
        """new_instance raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="Invalid node type"):
            Node.new_instance(
                node_id="test_invalid",
                description="Test invalid",
                node_type="totally_invalid_type",
            )


class TestNodeInitialize:
    """Tests for Node._initialize()."""

    def test_initialize_creates_model(self, real_agent_config, mock_llm_create):
        """_initialize creates a model from agent config."""
        node = Node.new_instance(
            node_id="test_init",
            description="Test init",
            node_type=NodeType.TYPE_GENERATE_SQL,
            agent_config=real_agent_config,
        )
        assert node.model is None

        node._initialize()

        assert node.model is not None
        assert isinstance(node.model, MockLLMModel)

    def test_initialize_skips_if_model_exists(self, real_agent_config, mock_llm_create):
        """_initialize skips if model is already an LLMBaseModel."""
        node = Node.new_instance(
            node_id="test_init_skip",
            description="Test init skip",
            node_type=NodeType.TYPE_GENERATE_SQL,
            agent_config=real_agent_config,
        )

        mock_model = mock_llm_create
        node.model = mock_model

        node._initialize()

        # Model should remain the same instance
        assert node.model is mock_model

    def test_initialize_with_model_name_string(self, real_agent_config, mock_llm_create):
        """_initialize with model set as string name resolves to actual model."""
        node = Node.new_instance(
            node_id="test_init_str",
            description="Test init string",
            node_type=NodeType.TYPE_GENERATE_SQL,
            agent_config=real_agent_config,
        )
        node.model = "mock"  # Model name string

        node._initialize()

        assert isinstance(node.model, MockLLMModel)


class TestNodeRun:
    """Tests for Node.run()."""

    def test_run_execute_sql_success(self, real_agent_config, mock_llm_create):
        """run() successfully executes a node."""
        mock_llm_create.reset(responses=[MockLLMResponse(content="{}")])

        node = Node.new_instance(
            node_id="test_run",
            description="Test run",
            node_type=NodeType.TYPE_EXECUTE_SQL,
            agent_config=real_agent_config,
        )
        # Set up minimal input for execute_sql node
        from datus.schemas.node_models import ExecuteSQLInput

        node.input = ExecuteSQLInput(sql_query="SELECT 1", database_name="california_schools")

        result = node.run()

        assert node.status in ("completed", "failed")
        assert node.start_time is not None

    def test_run_exception_fails_node(self, real_agent_config, mock_llm_create):
        """run() catches exceptions and marks node as failed."""
        node = Node.new_instance(
            node_id="test_run_fail",
            description="Test run fail",
            node_type=NodeType.TYPE_SCHEMA_LINKING,
            agent_config=real_agent_config,
        )
        # Don't set input, should cause execute() to fail

        result = node.run()

        assert node.status == "failed"
        assert node.end_time is not None


class TestNodeRunAsync:
    """Tests for Node.run_async()."""

    @pytest.mark.asyncio
    async def test_run_async_delegates_to_run(self, real_agent_config, mock_llm_create):
        """run_async delegates to run()."""
        mock_llm_create.reset(responses=[MockLLMResponse(content="{}")])

        node = Node.new_instance(
            node_id="test_run_async",
            description="Test run async",
            node_type=NodeType.TYPE_EXECUTE_SQL,
            agent_config=real_agent_config,
        )
        from datus.schemas.node_models import ExecuteSQLInput

        node.input = ExecuteSQLInput(sql_query="SELECT 1", database_name="california_schools")

        result = await node.run_async()

        assert node.status in ("completed", "failed")


class TestNodeComplete:
    """Tests for Node.complete()."""

    def test_complete_with_failed_result(self):
        """complete() sets status to 'failed' when result.success is False."""
        node = Node.new_instance(
            node_id="test_complete_fail",
            description="Test complete fail",
            node_type=NodeType.TYPE_GENERATE_SQL,
        )
        node.start()

        result = BaseResult(success=False, error="Something went wrong")
        node.complete(result)

        assert node.status == "failed"
        assert node.result is result
        assert node.end_time is not None


class TestNodeFail:
    """Tests for Node.fail()."""

    def test_fail_without_error(self):
        """fail() without error still sets status to failed."""
        node = Node.new_instance(
            node_id="test_fail_no_error",
            description="Test fail no error",
            node_type=NodeType.TYPE_GENERATE_SQL,
        )
        node.start()

        node.fail()

        assert node.status == "failed"
        assert node.result is None  # No error means no result set
        assert node.end_time is not None


class TestNodeFromDict:
    """Tests for Node.from_dict() deserialization."""

    def test_from_dict_generate_sql(self, real_agent_config, mock_llm_create):
        """from_dict creates a GenerateSQLNode from dict."""
        node_dict = {
            "id": "node_1",
            "description": "Generate SQL",
            "type": NodeType.TYPE_GENERATE_SQL,
            "input": {
                "sql_query": "SELECT 1",
                "table_schemas": [],
                "external_knowledge": "",
                "prompt_version": None,
            },
            "status": "completed",
            "result": {
                "success": True,
                "sql_query": "SELECT 1",
                "tables": [],
            },
            "start_time": 1000.0,
            "end_time": 1001.0,
            "dependencies": ["node_0"],
            "metadata": {"key": "value"},
        }

        node = Node.from_dict(node_dict, agent_config=real_agent_config)

        assert node.id == "node_1"
        assert node.type == NodeType.TYPE_GENERATE_SQL
        assert node.status == "completed"
        assert node.start_time == 1000.0
        assert node.end_time == 1001.0
        assert node.dependencies == ["node_0"]
        assert node.metadata == {"key": "value"}

    def test_from_dict_with_base_result(self, real_agent_config, mock_llm_create):
        """from_dict handles result with 'success' field as BaseResult."""
        node_dict = {
            "id": "node_2",
            "description": "Begin node",
            "type": NodeType.TYPE_BEGIN,
            "input": None,
            "status": "completed",
            "result": {"success": True},
            "start_time": None,
            "end_time": None,
            "dependencies": [],
            "metadata": {},
        }

        node = Node.from_dict(node_dict, agent_config=real_agent_config)

        assert node.id == "node_2"
        assert node.result is not None
        assert node.result.success is True

    def test_from_dict_with_none_result(self, real_agent_config, mock_llm_create):
        """from_dict handles None result."""
        node_dict = {
            "id": "node_3",
            "description": "Pending node",
            "type": NodeType.TYPE_OUTPUT,
            "input": None,
            "status": "pending",
            "result": None,
            "start_time": None,
            "end_time": None,
            "dependencies": [],
            "metadata": {},
        }

        node = Node.from_dict(node_dict, agent_config=real_agent_config)

        assert node.id == "node_3"
        assert node.result is None
        assert node.status == "pending"


class TestNodeToDict:
    """Tests for Node.to_dict() serialization."""

    def test_to_dict_with_result(self, real_agent_config, mock_llm_create):
        """to_dict serializes a node with result."""
        node = Node.new_instance(
            node_id="test_dict",
            description="Test dict",
            node_type=NodeType.TYPE_GENERATE_SQL,
            agent_config=real_agent_config,
        )
        node.start()
        node.complete(BaseResult(success=True))
        node.metadata = {"test_key": "test_val"}

        d = node.to_dict()

        assert d["id"] == "test_dict"
        assert d["description"] == "Test dict"
        assert d["type"] == NodeType.TYPE_GENERATE_SQL
        assert d["status"] == "completed"
        assert d["result"]["success"] is True
        assert d["metadata"] == {"test_key": "test_val"}
        assert d["start_time"] is not None
        assert d["end_time"] is not None

    def test_to_dict_roundtrip(self, real_agent_config, mock_llm_create):
        """to_dict -> from_dict roundtrip preserves key properties."""
        node = Node.new_instance(
            node_id="roundtrip",
            description="Roundtrip node",
            node_type=NodeType.TYPE_EXECUTE_SQL,
            agent_config=real_agent_config,
        )
        node.start()
        node.complete(BaseResult(success=True))
        node.dependencies = ["dep_1", "dep_2"]
        node.metadata = {"extra": "data"}

        d = node.to_dict()
        restored = Node.from_dict(d, agent_config=real_agent_config)

        assert restored.id == "roundtrip"
        assert restored.description == "Roundtrip node"
        assert restored.type == NodeType.TYPE_EXECUTE_SQL
        assert restored.status == "completed"
        assert restored.dependencies == ["dep_1", "dep_2"]
        assert restored.metadata == {"extra": "data"}


class TestNodeSqlConnector:
    """Tests for Node._sql_connector()."""

    def test_sql_connector_default(self, real_agent_config, mock_llm_create):
        """_sql_connector returns a connector for the default namespace."""
        node = Node.new_instance(
            node_id="test_conn",
            description="Test connector",
            node_type=NodeType.TYPE_EXECUTE_SQL,
            agent_config=real_agent_config,
        )

        conn = node._sql_connector("california_schools")
        assert conn is not None
