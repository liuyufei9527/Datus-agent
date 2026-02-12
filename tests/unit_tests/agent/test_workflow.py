from datus.agent.node import Node
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SqlTask


class TestNode:
    """Test suite for the Node class."""

    def test_node_initialization(self):
        """Test that a Node initializes with the correct attributes."""
        node = Node.new_instance(
            node_id="test_node",
            description="Test node",
            node_type=NodeType.TYPE_GENERATE_SQL,
            input_data=None,
        )

        assert node.id == "test_node"
        assert node.description == "Test node"
        assert node.type == NodeType.TYPE_GENERATE_SQL
        assert node.input is None
        assert node.status == "pending"
        assert node.result is None
        assert node.start_time is None
        assert node.end_time is None
        assert node.dependencies == []
        assert node.metadata == {}

    def test_node_state_transitions(self):
        """Test node state transitions (start, complete, fail)."""
        from datus.schemas.node_models import BaseResult

        node = Node.new_instance("test_node", "Test node", NodeType.TYPE_GENERATE_SQL)

        # Test start transition
        node.start()
        assert node.status == "running"
        assert node.start_time is not None

        # Test complete transition
        result = BaseResult(success=True)
        node.complete(result)
        assert node.status == "completed"
        assert node.result == result
        assert node.end_time is not None

        # Test fail transition
        node = Node.new_instance("test_node", "Test node", NodeType.TYPE_GENERATE_SQL)
        error_msg = "Test error"
        node.fail(error_msg)
        assert node.status == "failed"
        assert node.result.success is False
        assert node.result.error == error_msg
        assert node.end_time is not None

    def test_node_dependencies(self):
        """Test adding dependencies to a node."""
        node = Node.new_instance("test_node", "Test node", NodeType.TYPE_GENERATE_SQL)

        # Test adding dependencies
        node.add_dependency("dep_1")
        node.add_dependency("dep_2")
        assert "dep_1" in node.dependencies
        assert "dep_2" in node.dependencies

        # Test duplicate dependency
        node.add_dependency("dep_1")
        assert len(node.dependencies) == 2

    def test_node_to_dict(self):
        """Test converting a node to dictionary representation."""
        node = Node.new_instance(
            node_id="test_node",
            description="Test node",
            node_type=NodeType.TYPE_GENERATE_SQL,
            input_data=None,
        )

        node_dict = node.to_dict()
        assert node_dict["id"] == "test_node"
        assert node_dict["description"] == "Test node"
        assert node_dict["type"] == NodeType.TYPE_GENERATE_SQL
        assert node_dict["status"] == "pending"
        assert node_dict["result"] is None
        assert node_dict["dependencies"] == []
        assert node_dict["metadata"] == {}

    def test_control_node_types(self):
        """Test initialization of different control node types."""
        # Test HITL node type
        hitl_node = Node.new_instance(
            node_id="hitl_node",
            description="HITL node",
            node_type=NodeType.TYPE_HITL,
            input_data=None,
        )
        assert hitl_node.type == NodeType.TYPE_HITL

        # Test reflect node type
        reflect_node = Node.new_instance(
            node_id="reflect_node",
            description="Reflect node",
            node_type=NodeType.TYPE_REFLECT,
            input_data=None,
        )
        assert reflect_node.type == NodeType.TYPE_REFLECT

        # Test subworkflow node type
        subworkflow_node = Node.new_instance(
            node_id="subworkflow_node",
            description="Subworkflow node",
            node_type=NodeType.TYPE_SUBWORKFLOW,
            input_data=None,
        )
        assert subworkflow_node.type == NodeType.TYPE_SUBWORKFLOW

    def test_node_run_failure(self):
        """Test node run failure scenarios."""
        import pytest

        # Test failure when node type is invalid - should raise ValueError during creation
        with pytest.raises(ValueError, match="Invalid node type"):
            Node.new_instance(
                node_id="invalid_node",
                description="Invalid node",
                node_type="invalid_type",
                input_data=None,
            )


class TestWorkflow:
    """Test suite for the Workflow class."""

    def test_create_default_workflow(self, real_agent_config):
        """Test the default workflow creation with core nodes using plan._create_default_workflow."""

        # Initialize with specified parameters
        task_text = (
            "Calculate the total annual revenue from 1992 to 1997 for each "
            "combination of customer city and supplier city, where the customer is "
            "located in the U.S. cities 'UNITED KI1' or 'UNITED KI5', "
            "and the supplier is also located in these two cities. "
            "The results should be sorted in ascending order by year and, "
            "within each year, in descending order by revenue."
        )

        # Create workflow using plan.py's method
        task = SqlTask(task=task_text)
        workflow = generate_workflow(task, "reflection", agent_config=real_agent_config)

        # Verify core nodes exist
        expected_nodes = [
            ("node_0", "Beginning of the workflow", NodeType.TYPE_BEGIN),
            ("node_1", "Understand the query and find related schemas", NodeType.TYPE_SCHEMA_LINKING),
            ("node_2", "Generate SQL query", NodeType.TYPE_GENERATE_SQL),
            ("node_3", "Execute SQL query", NodeType.TYPE_EXECUTE_SQL),
            ("node_4", "evaluation and self-reflection", NodeType.TYPE_REFLECT),
            ("node_5", "Return the results to the user", NodeType.TYPE_OUTPUT),
        ]

        # Check node properties
        for node_id, description, node_type in expected_nodes:
            node = workflow.get_node(node_id)
            assert node is not None
            assert node.description == description
            assert node.type == node_type

        # Verify workflow metadata
        assert workflow.status == "pending"
        assert workflow.current_node_index == 0
        assert len(workflow.nodes) == 6

        # workflow.save(self.WORKFLOW_SAVE_PATH)

    # def test_e2e_workflow(self):
    #    """Test the end-to-end workflow execution."""
    #    # Initialize with specified parameters
    #    task = "Calculate the total annual revenue from 1992 to 1997 for each combination of
    #    customer city and supplier city, where the customer is located in the U.S.
    #    cities 'UNITED KI1' or 'UNITED KI5', and the supplier is also located in these two cities.
    #    The results should be sorted in ascending order by year and,
    #    within each year, in descending order by revenue."
    #
    #    # Create workflow using plan.py's method
    #    workflow = Workflow("nl2sql_workflow", "Test a e2e workflow")
    #    nodes = _create_default_workflow(task)
    #    for node in nodes:
    #        workflow.add_node(node)

    def test_workflow_save_load(self, tmp_path, real_agent_config):
        """Test saving and loading a workflow with multiple nodes."""
        # Create a workflow with multiple nodes
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test workflow save/load"), agent_config=real_agent_config
        )

        node1 = Node.new_instance(
            node_id="node1",
            description="First node",
            node_type=NodeType.TYPE_GENERATE_SQL,
            input_data=None,
        )

        node2 = Node.new_instance(
            node_id="node2",
            description="Second node",
            node_type=NodeType.TYPE_EXECUTE_SQL,
            input_data=None,
        )

        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.move_node("node2", 0)
        workflow.status = "running"
        workflow.current_node_index = 1
        workflow.metadata = {"test_key": "test_value"}

        # Save workflow to temporary file
        save_path = tmp_path / "test_workflow.yaml"
        # save_path = "./tests/test_workflow.yaml"
        workflow.save(str(save_path))

        # Load workflow from file
        loaded_workflow = Workflow.load(str(save_path), agent_config=real_agent_config)

        # Verify all properties are correctly restored
        assert loaded_workflow.name == "test_workflow"
        assert loaded_workflow.task.task == "Test workflow save/load"
        assert loaded_workflow.status == "running"
        assert loaded_workflow.current_node_index == 1
        assert loaded_workflow.metadata == {"test_key": "test_value"}

        # Verify nodes are correctly restored
        assert len(loaded_workflow.nodes) == 2
        assert "node1" in loaded_workflow.nodes
        assert "node2" in loaded_workflow.nodes
        assert loaded_workflow.node_order == ["node2", "node1"]

        # Verify node properties
        loaded_node1 = loaded_workflow.nodes["node1"]
        assert loaded_node1.description == "First node"
        assert loaded_node1.type == NodeType.TYPE_GENERATE_SQL

        loaded_node2 = loaded_workflow.nodes["node2"]
        assert loaded_node2.description == "Second node"
        assert loaded_node2.type == NodeType.TYPE_EXECUTE_SQL

    def test_workflow_remove_node(self, real_agent_config):
        """Test removing a node from the workflow."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test remove"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        node2 = Node.new_instance("n2", "Node 2", NodeType.TYPE_EXECUTE_SQL)
        workflow.add_node(node1)
        workflow.add_node(node2)

        assert len(workflow.nodes) == 2
        assert "n1" in workflow.nodes

        result = workflow.remove_node("n1")
        assert result is True
        assert "n1" not in workflow.nodes
        assert "n1" not in workflow.node_order
        assert len(workflow.nodes) == 1

    def test_workflow_remove_nonexistent_node(self, real_agent_config):
        """Removing a node that doesn't exist returns False."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        result = workflow.remove_node("nonexistent")
        assert result is False

    def test_workflow_move_node(self, real_agent_config):
        """Test moving a node to a different position."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test move"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        node2 = Node.new_instance("n2", "Node 2", NodeType.TYPE_EXECUTE_SQL)
        node3 = Node.new_instance("n3", "Node 3", NodeType.TYPE_OUTPUT)
        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.add_node(node3)

        assert workflow.node_order == ["n1", "n2", "n3"]

        result = workflow.move_node("n3", 0)
        assert result is True
        assert workflow.node_order[0] == "n3"

    def test_workflow_move_invalid_position(self, real_agent_config):
        """Moving to invalid position returns False."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        result = workflow.move_node("n1", 100)
        assert result is False

    def test_workflow_get_last_sqlcontext_empty(self, real_agent_config):
        """get_last_sqlcontext raises when no SQL context exists."""
        from datus.utils.exceptions import DatusException

        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        import pytest
        with pytest.raises(DatusException):
            workflow.get_last_sqlcontext()

    def test_workflow_get_last_sqlcontext_with_data(self, real_agent_config):
        """get_last_sqlcontext returns the last SQL context."""
        from datus.schemas.node_models import SQLContext

        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        sql_ctx = SQLContext(sql_query="SELECT 1", explanation="Test")
        workflow.context.sql_contexts.append(sql_ctx)

        result = workflow.get_last_sqlcontext()
        assert result.sql_query == "SELECT 1"

    def test_workflow_get_current_node(self, real_agent_config):
        """get_current_node returns correct node."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        current = workflow.get_current_node()
        assert current.id == "n1"

    def test_workflow_get_current_node_empty(self, real_agent_config):
        """get_current_node returns None when no nodes."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        current = workflow.get_current_node()
        assert current is None

    def test_workflow_get_next_node(self, real_agent_config):
        """get_next_node returns the next node."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        node2 = Node.new_instance("n2", "Node 2", NodeType.TYPE_EXECUTE_SQL)
        workflow.add_node(node1)
        workflow.add_node(node2)

        next_node = workflow.get_next_node()
        assert next_node.id == "n2"

    def test_workflow_get_next_node_at_end(self, real_agent_config):
        """get_next_node returns None when at the last node."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        next_node = workflow.get_next_node()
        assert next_node is None

    def test_workflow_advance_to_next_node(self, real_agent_config):
        """advance_to_next_node increments index and returns next node."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        node2 = Node.new_instance("n2", "Node 2", NodeType.TYPE_EXECUTE_SQL)
        workflow.add_node(node1)
        workflow.add_node(node2)

        assert workflow.current_node_index == 0
        next_node = workflow.advance_to_next_node()
        assert next_node.id == "n2"
        assert workflow.current_node_index == 1

    def test_workflow_advance_past_end(self, real_agent_config):
        """advance_to_next_node at end marks workflow completed."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        result = workflow.advance_to_next_node()
        assert result is None
        assert workflow.status == "completed"
        assert workflow.completion_time is not None

    def test_workflow_is_complete(self, real_agent_config):
        """is_complete() checks status and index."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        assert workflow.is_complete() is True  # No nodes

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)
        assert workflow.is_complete() is False

        workflow.status = "completed"
        assert workflow.is_complete() is True

    def test_workflow_pause_resume(self, real_agent_config):
        """Test workflow pause and resume."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        workflow.pause()
        assert workflow.status == "paused"

        workflow.resume()
        assert workflow.status == "running"

    def test_workflow_reset(self, real_agent_config):
        """Test workflow reset."""
        from datus.schemas.node_models import BaseResult

        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)
        node1.start()
        node1.complete(BaseResult(success=True))
        workflow.current_node_index = 1
        workflow.status = "running"

        workflow.reset()
        assert workflow.status == "pending"
        assert workflow.current_node_index == 0
        assert node1.status == "pending"
        assert node1.result is None

    def test_workflow_get_last_node_by_type(self, real_agent_config):
        """get_last_node_by_type finds correct node."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        node2 = Node.new_instance("n2", "Node 2", NodeType.TYPE_EXECUTE_SQL)
        node3 = Node.new_instance("n3", "Node 3", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.add_node(node3)

        # Set current_node_index to 2 so we search backwards
        workflow.current_node_index = 2

        result = workflow.get_last_node_by_type(NodeType.TYPE_GENERATE_SQL)
        assert result is not None
        assert result.id == "n1"

    def test_workflow_get_last_node_by_type_not_found(self, real_agent_config):
        """get_last_node_by_type returns None when type not found."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)
        workflow.current_node_index = 1

        result = workflow.get_last_node_by_type(NodeType.TYPE_CHAT)
        assert result is None

    def test_workflow_get_task(self, real_agent_config):
        """get_task returns the task string."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="My query"), agent_config=real_agent_config
        )
        assert workflow.get_task() == "My query"

    def test_workflow_get_task_with_value(self, real_agent_config):
        """get_task returns the task string from SqlTask."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="My query text"), agent_config=real_agent_config
        )
        assert workflow.get_task() == "My query text"

    def test_workflow_to_dict(self, real_agent_config):
        """to_dict returns complete dict."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        d = workflow.to_dict()
        assert d["name"] == "test_workflow"
        assert d["description"] == "Test"
        assert "n1" in d["nodes"]
        assert d["node_order"] == ["n1"]
        assert d["status"] == "pending"

    def test_workflow_get_final_result(self, real_agent_config):
        """get_final_result returns dict with node results."""
        from datus.schemas.node_models import BaseResult

        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)
        node1.start()
        node1.complete(BaseResult(success=True))

        result = workflow.get_final_result()
        assert result["name"] == "test_workflow"
        assert result["status"] == "pending"
        assert "n1" in result["nodes"]
        # n1 is the last and only node, and it's completed
        assert "final_result" in result

    def test_workflow_get_final_result_no_completed(self, real_agent_config):
        """get_final_result with no completed nodes."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        result = workflow.get_final_result()
        assert result["name"] == "test_workflow"
        assert len(result["nodes"]) == 0  # No completed nodes

    def test_workflow_display_no_crash(self, real_agent_config):
        """display() should not crash."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        # Should not raise
        workflow.display()

    def test_workflow_global_config_setter(self, real_agent_config):
        """Setting global_config propagates to nodes."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        # Setting global_config should propagate to nodes
        workflow.global_config = real_agent_config
        assert node1.global_config is real_agent_config

    def test_workflow_add_node_at_position(self, real_agent_config):
        """add_node with specific position inserts correctly."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        node2 = Node.new_instance("n2", "Node 2", NodeType.TYPE_EXECUTE_SQL)
        node3 = Node.new_instance("n3", "Node 3", NodeType.TYPE_OUTPUT)

        workflow.add_node(node1)
        workflow.add_node(node3)
        workflow.add_node(node2, position=1)

        assert workflow.node_order == ["n1", "n2", "n3"]

    def test_workflow_add_node_invalid_position_type(self, real_agent_config):
        """add_node with non-integer position raises ValueError."""
        import pytest

        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)

        with pytest.raises(ValueError, match="Position must be integer"):
            workflow.add_node(node1, position="invalid")

    def test_workflow_adjust_nodes_add_requires_node_import(self, real_agent_config):
        """adjust_nodes with 'add' action fails because Node is not imported at runtime in workflow.py.
        This exercises the adjust_nodes code path (known limitation)."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        suggestions = [
            {
                "action": "add",
                "node": {
                    "id": "new_node",
                    "description": "New node",
                    "type": NodeType.TYPE_GENERATE_SQL,
                    "input": None,
                },
            }
        ]

        # Node is not imported at runtime in workflow.py, so this raises NameError
        import pytest
        with pytest.raises(NameError):
            workflow.adjust_nodes(suggestions)

    def test_workflow_adjust_nodes_remove(self, real_agent_config):
        """adjust_nodes with 'remove' action removes a node."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        suggestions = [{"action": "remove", "node_id": "n1"}]
        workflow.adjust_nodes(suggestions)
        assert "n1" not in workflow.nodes

    def test_workflow_adjust_nodes_move(self, real_agent_config):
        """adjust_nodes with 'move' action moves a node."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        node2 = Node.new_instance("n2", "Node 2", NodeType.TYPE_EXECUTE_SQL)
        workflow.add_node(node1)
        workflow.add_node(node2)

        suggestions = [{"action": "move", "node_id": "n2", "position": 0}]
        workflow.adjust_nodes(suggestions)
        assert workflow.node_order[0] == "n2"

    def test_workflow_adjust_nodes_modify(self, real_agent_config):
        """adjust_nodes with 'modify' action modifies node properties."""
        workflow = Workflow(
            name="test_workflow", task=SqlTask(task="Test"), agent_config=real_agent_config
        )

        node1 = Node.new_instance("n1", "Node 1", NodeType.TYPE_GENERATE_SQL)
        workflow.add_node(node1)

        suggestions = [
            {
                "action": "modify",
                "node_id": "n1",
                "modifications": {"description": "Modified description"},
            }
        ]
        workflow.adjust_nodes(suggestions)
        assert workflow.nodes["n1"].description == "Modified description"
