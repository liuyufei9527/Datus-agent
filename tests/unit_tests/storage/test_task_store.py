"""Tests for datus/storage/task/store.py"""

import pytest

from datus.storage.task.store import TaskStore
from datus.utils.exceptions import DatusException


@pytest.fixture
def task_store(tmp_path):
    return TaskStore(str(tmp_path / "task_db"))


class TestTaskStoreInit:
    def test_creates_directory(self, tmp_path):
        db_path = str(tmp_path / "new_db")
        store = TaskStore(db_path)
        assert (tmp_path / "new_db").exists()
        assert (tmp_path / "new_db" / "task.db").exists()

    def test_table_created(self, task_store):
        import sqlite3

        conn = sqlite3.connect(task_store.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        assert cursor.fetchone() is not None
        conn.close()


class TestCreateTask:
    def test_create_basic_task(self, task_store):
        result = task_store.create_task("task-001", "What is the total revenue?")
        assert result["task_id"] == "task-001"
        assert result["task_query"] == "What is the total revenue?"
        assert result["status"] == "running"
        assert result["sql_query"] == ""
        assert result["sql_result"] == ""
        assert result["user_feedback"] == ""

    def test_create_task_replaces_existing(self, task_store):
        task_store.create_task("task-001", "Query 1")
        result = task_store.create_task("task-001", "Query 2")
        assert result["task_query"] == "Query 2"


class TestGetTask:
    def test_get_existing_task(self, task_store):
        task_store.create_task("task-001", "Test query")
        result = task_store.get_task("task-001")
        assert result is not None
        assert result["task_id"] == "task-001"
        assert result["task_query"] == "Test query"

    def test_get_nonexistent_task(self, task_store):
        result = task_store.get_task("nonexistent")
        assert result is None


class TestUpdateTask:
    def test_update_sql_query(self, task_store):
        task_store.create_task("task-001", "Test")
        result = task_store.update_task("task-001", sql_query="SELECT 1")
        assert result is True
        task = task_store.get_task("task-001")
        assert task["sql_query"] == "SELECT 1"

    def test_update_sql_result(self, task_store):
        task_store.create_task("task-001", "Test")
        result = task_store.update_task("task-001", sql_result="[{\"value\": 42}]")
        assert result is True
        task = task_store.get_task("task-001")
        assert "42" in task["sql_result"]

    def test_update_status(self, task_store):
        task_store.create_task("task-001", "Test")
        result = task_store.update_task("task-001", status="completed")
        assert result is True
        task = task_store.get_task("task-001")
        assert task["status"] == "completed"

    def test_update_multiple_fields(self, task_store):
        task_store.create_task("task-001", "Test")
        result = task_store.update_task("task-001", sql_query="SELECT 1", status="completed")
        assert result is True
        task = task_store.get_task("task-001")
        assert task["sql_query"] == "SELECT 1"
        assert task["status"] == "completed"

    def test_update_nonexistent_task(self, task_store):
        result = task_store.update_task("nonexistent", status="completed")
        assert result is False


class TestDeleteTask:
    def test_delete_existing_task(self, task_store):
        task_store.create_task("task-001", "Test")
        result = task_store.delete_task("task-001")
        assert result is True
        assert task_store.get_task("task-001") is None

    def test_delete_nonexistent_task(self, task_store):
        result = task_store.delete_task("nonexistent")
        assert result is False


class TestRecordFeedback:
    def test_record_feedback_success(self, task_store):
        task_store.create_task("task-001", "Test")
        result = task_store.record_feedback("task-001", "success")
        assert result["task_id"] == "task-001"
        assert result["user_feedback"] == "success"

    def test_record_feedback_failed(self, task_store):
        task_store.create_task("task-001", "Test")
        result = task_store.record_feedback("task-001", "failed")
        assert result["user_feedback"] == "failed"

    def test_record_feedback_nonexistent_task(self, task_store):
        with pytest.raises(Exception):
            task_store.record_feedback("nonexistent", "success")


class TestGetFeedback:
    def test_get_feedback_with_feedback(self, task_store):
        task_store.create_task("task-001", "Test")
        task_store.record_feedback("task-001", "success")
        result = task_store.get_feedback("task-001")
        assert result is not None
        assert result["user_feedback"] == "success"

    def test_get_feedback_without_feedback(self, task_store):
        task_store.create_task("task-001", "Test")
        result = task_store.get_feedback("task-001")
        assert result is None

    def test_get_feedback_nonexistent_task(self, task_store):
        result = task_store.get_feedback("nonexistent")
        assert result is None


class TestGetAllFeedback:
    def test_get_all_feedback(self, task_store):
        task_store.create_task("task-001", "Test 1")
        task_store.create_task("task-002", "Test 2")
        task_store.record_feedback("task-001", "success")
        task_store.record_feedback("task-002", "failed")
        results = task_store.get_all_feedback()
        assert len(results) == 2

    def test_get_all_feedback_empty(self, task_store):
        results = task_store.get_all_feedback()
        assert results == []


class TestDeleteFeedback:
    def test_delete_existing_feedback(self, task_store):
        task_store.create_task("task-001", "Test")
        task_store.record_feedback("task-001", "success")
        result = task_store.delete_feedback("task-001")
        assert result is True
        assert task_store.get_feedback("task-001") is None

    def test_delete_nonexistent_feedback(self, task_store):
        result = task_store.delete_feedback("nonexistent")
        assert result is False


class TestCleanupOldTasks:
    def test_cleanup_removes_old_tasks(self, task_store):
        import sqlite3
        from datetime import datetime, timedelta, timezone

        # Create a task and manually set its created_at to be old
        task_store.create_task("old-task", "Old query")
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat().replace("+00:00", "Z")
        conn = sqlite3.connect(task_store.db_file)
        conn.execute("UPDATE tasks SET created_at = ? WHERE task_id = ?", (old_time, "old-task"))
        conn.commit()
        conn.close()

        task_store.create_task("new-task", "New query")
        deleted = task_store.cleanup_old_tasks(hours=24)
        assert deleted == 1
        assert task_store.get_task("old-task") is None
        assert task_store.get_task("new-task") is not None

    def test_cleanup_no_old_tasks(self, task_store):
        task_store.create_task("task-001", "Test")
        deleted = task_store.cleanup_old_tasks(hours=24)
        assert deleted == 0
