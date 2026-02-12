# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for plan_tools.py"""

import json
from unittest.mock import MagicMock

import pytest

from datus.tools.func_tool.plan_tools import (
    PlanTool,
    SessionTodoStorage,
    TodoItem,
    TodoList,
    TodoStatus,
)


# ---------------------------------------------------------------------------
# TodoStatus enum
# ---------------------------------------------------------------------------


class TestTodoStatus:
    def test_values(self):
        assert TodoStatus.PENDING == "pending"
        assert TodoStatus.COMPLETED == "completed"
        assert TodoStatus.FAILED == "failed"

    def test_from_string(self):
        assert TodoStatus("pending") == TodoStatus.PENDING
        assert TodoStatus("completed") == TodoStatus.COMPLETED
        assert TodoStatus("failed") == TodoStatus.FAILED


# ---------------------------------------------------------------------------
# TodoItem
# ---------------------------------------------------------------------------


class TestTodoItem:
    def test_default_id(self):
        item = TodoItem(content="test task")
        assert item.id is not None
        assert len(item.id) > 0

    def test_default_status(self):
        item = TodoItem(content="test task")
        assert item.status == TodoStatus.PENDING

    def test_custom_status(self):
        item = TodoItem(content="done task", status=TodoStatus.COMPLETED)
        assert item.status == TodoStatus.COMPLETED


# ---------------------------------------------------------------------------
# TodoList
# ---------------------------------------------------------------------------


class TestTodoList:
    def test_add_item(self):
        todo_list = TodoList()
        item = todo_list.add_item("Write tests")
        assert item.content == "Write tests"
        assert item.status == TodoStatus.PENDING
        assert len(todo_list.items) == 1

    def test_get_item_found(self):
        todo_list = TodoList()
        item = todo_list.add_item("Write tests")
        found = todo_list.get_item(item.id)
        assert found is not None
        assert found.content == "Write tests"

    def test_get_item_not_found(self):
        todo_list = TodoList()
        found = todo_list.get_item("nonexistent-id")
        assert found is None

    def test_update_item_status_success(self):
        todo_list = TodoList()
        item = todo_list.add_item("Do something")
        result = todo_list.update_item_status(item.id, TodoStatus.COMPLETED)
        assert result is True
        assert item.status == TodoStatus.COMPLETED

    def test_update_item_status_not_found(self):
        todo_list = TodoList()
        result = todo_list.update_item_status("nonexistent-id", TodoStatus.COMPLETED)
        assert result is False

    def test_get_completed_items(self):
        todo_list = TodoList()
        item1 = todo_list.add_item("Task 1")
        todo_list.add_item("Task 2")
        item1.status = TodoStatus.COMPLETED
        completed = todo_list.get_completed_items()
        assert len(completed) == 1
        assert completed[0].content == "Task 1"

    def test_get_completed_items_empty(self):
        todo_list = TodoList()
        todo_list.add_item("Pending task")
        completed = todo_list.get_completed_items()
        assert len(completed) == 0


# ---------------------------------------------------------------------------
# SessionTodoStorage
# ---------------------------------------------------------------------------


class TestSessionTodoStorage:
    @pytest.fixture
    def storage(self):
        mock_session = MagicMock()
        return SessionTodoStorage(mock_session)

    def test_save_list(self, storage):
        todo_list = TodoList()
        todo_list.add_item("Item 1")
        result = storage.save_list(todo_list)
        assert result is True

    def test_get_todo_list_after_save(self, storage):
        todo_list = TodoList()
        todo_list.add_item("Item 1")
        storage.save_list(todo_list)
        retrieved = storage.get_todo_list()
        assert retrieved is not None
        assert len(retrieved.items) == 1

    def test_get_todo_list_empty(self, storage):
        retrieved = storage.get_todo_list()
        assert retrieved is None

    def test_clear_all(self, storage):
        todo_list = TodoList()
        todo_list.add_item("Item 1")
        storage.save_list(todo_list)
        storage.clear_all()
        assert storage.get_todo_list() is None

    def test_has_todo_list_false(self, storage):
        assert storage.has_todo_list() is False

    def test_has_todo_list_true(self, storage):
        todo_list = TodoList()
        storage.save_list(todo_list)
        assert storage.has_todo_list() is True


# ---------------------------------------------------------------------------
# PlanTool
# ---------------------------------------------------------------------------


class TestPlanTool:
    @pytest.fixture
    def plan_tool(self):
        mock_session = MagicMock()
        return PlanTool(mock_session)

    def test_available_tools(self, plan_tool):
        tools = plan_tool.available_tools()
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "todo_read" in tool_names
        assert "todo_write" in tool_names
        assert "todo_update" in tool_names

    def test_todo_read_empty(self, plan_tool):
        result = plan_tool.todo_read()
        assert result.result["total_lists"] == 0
        assert result.result["lists"] == []

    def test_todo_write_valid_json(self, plan_tool):
        todos_json = json.dumps([
            {"content": "Task 1", "status": "pending"},
            {"content": "Task 2", "status": "completed"},
        ])
        result = plan_tool.todo_write(todos_json)
        assert result.success == 1
        assert "2 items" in result.result["message"]

    def test_todo_write_invalid_json(self, plan_tool):
        result = plan_tool.todo_write("not valid json")
        assert result.success == 0
        assert "Invalid JSON" in result.error

    def test_todo_write_empty_list(self, plan_tool):
        result = plan_tool.todo_write("[]")
        assert result.success == 0
        assert "no todo items" in result.error

    def test_todo_write_skips_empty_content(self, plan_tool):
        todos_json = json.dumps([
            {"content": "", "status": "pending"},
            {"content": "Real task", "status": "pending"},
        ])
        result = plan_tool.todo_write(todos_json)
        assert result.success == 1
        todo_list = result.result["todo_list"]
        assert len(todo_list["items"]) == 1

    def test_todo_read_after_write(self, plan_tool):
        todos_json = json.dumps([{"content": "Task 1", "status": "pending"}])
        plan_tool.todo_write(todos_json)
        result = plan_tool.todo_read()
        assert result.result["total_lists"] == 1
        assert len(result.result["lists"]) == 1

    def test_todo_update_status(self, plan_tool):
        todos_json = json.dumps([{"content": "Task 1", "status": "pending"}])
        write_result = plan_tool.todo_write(todos_json)
        todo_id = write_result.result["todo_list"]["items"][0]["id"]

        update_result = plan_tool.todo_update(todo_id, "completed")
        assert update_result.success == 1
        assert update_result.result["updated_item"]["status"] == "completed"

    def test_todo_update_invalid_status(self, plan_tool):
        todos_json = json.dumps([{"content": "Task 1", "status": "pending"}])
        write_result = plan_tool.todo_write(todos_json)
        todo_id = write_result.result["todo_list"]["items"][0]["id"]

        update_result = plan_tool.todo_update(todo_id, "invalid_status")
        assert update_result.success == 0
        assert "Invalid status" in update_result.error

    def test_todo_update_no_list(self, plan_tool):
        result = plan_tool.todo_update("some-id", "completed")
        assert result.success == 0
        assert "No todo list found" in result.error

    def test_todo_update_item_not_found(self, plan_tool):
        todos_json = json.dumps([{"content": "Task 1", "status": "pending"}])
        plan_tool.todo_write(todos_json)

        result = plan_tool.todo_update("nonexistent-id", "completed")
        assert result.success == 0
        assert "not found" in result.error

    def test_todo_update_to_failed(self, plan_tool):
        todos_json = json.dumps([{"content": "Task 1", "status": "pending"}])
        write_result = plan_tool.todo_write(todos_json)
        todo_id = write_result.result["todo_list"]["items"][0]["id"]

        update_result = plan_tool.todo_update(todo_id, "failed")
        assert update_result.success == 1
        assert update_result.result["updated_item"]["status"] == "failed"

    def test_todo_write_with_completed_items(self, plan_tool):
        todos_json = json.dumps([
            {"content": "Done step", "status": "completed"},
            {"content": "Next step", "status": "pending"},
        ])
        result = plan_tool.todo_write(todos_json)
        assert result.success == 1
        assert "1 already completed" in result.result["message"]
