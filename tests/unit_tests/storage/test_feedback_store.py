"""Tests for datus/storage/feedback/store.py"""

import pytest

from datus.storage.feedback.store import FeedbackStore
from datus.utils.exceptions import DatusException


@pytest.fixture
def feedback_store(tmp_path):
    return FeedbackStore(str(tmp_path / "feedback_db"))


class TestFeedbackStoreInit:
    def test_creates_directory(self, tmp_path):
        db_path = str(tmp_path / "new_db")
        store = FeedbackStore(db_path)
        assert (tmp_path / "new_db").exists()
        assert (tmp_path / "new_db" / "feedback.db").exists()

    def test_table_created(self, feedback_store):
        import sqlite3

        conn = sqlite3.connect(feedback_store.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_index_created(self, feedback_store):
        import sqlite3

        conn = sqlite3.connect(feedback_store.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_task_id'")
        assert cursor.fetchone() is not None
        conn.close()


class TestRecordFeedback:
    def test_record_success_feedback(self, feedback_store):
        result = feedback_store.record_feedback("task-001", "success")
        assert result["task_id"] == "task-001"
        assert result["status"] == "success"
        assert "recorded_at" in result

    def test_record_failed_feedback(self, feedback_store):
        result = feedback_store.record_feedback("task-001", "failed")
        assert result["task_id"] == "task-001"
        assert result["status"] == "failed"

    def test_record_replaces_existing(self, feedback_store):
        feedback_store.record_feedback("task-001", "success")
        result = feedback_store.record_feedback("task-001", "failed")
        assert result["status"] == "failed"
        # Should only have one record
        all_feedback = feedback_store.get_all_feedback()
        assert len(all_feedback) == 1


class TestGetFeedback:
    def test_get_existing_feedback(self, feedback_store):
        feedback_store.record_feedback("task-001", "success")
        result = feedback_store.get_feedback("task-001")
        assert result is not None
        assert result["task_id"] == "task-001"
        assert result["status"] == "success"

    def test_get_nonexistent_feedback(self, feedback_store):
        result = feedback_store.get_feedback("nonexistent")
        assert result is None


class TestGetAllFeedback:
    def test_get_all_with_multiple(self, feedback_store):
        feedback_store.record_feedback("task-001", "success")
        feedback_store.record_feedback("task-002", "failed")
        feedback_store.record_feedback("task-003", "success")
        results = feedback_store.get_all_feedback()
        assert len(results) == 3

    def test_get_all_empty(self, feedback_store):
        results = feedback_store.get_all_feedback()
        assert results == []

    def test_get_all_ordered_by_created_at_desc(self, feedback_store):
        feedback_store.record_feedback("task-001", "success")
        feedback_store.record_feedback("task-002", "failed")
        results = feedback_store.get_all_feedback()
        # Most recent should be first
        assert results[0]["task_id"] == "task-002"


class TestDeleteFeedback:
    def test_delete_existing_feedback(self, feedback_store):
        feedback_store.record_feedback("task-001", "success")
        result = feedback_store.delete_feedback("task-001")
        assert result is True
        assert feedback_store.get_feedback("task-001") is None

    def test_delete_nonexistent_feedback(self, feedback_store):
        result = feedback_store.delete_feedback("nonexistent")
        assert result is False

    def test_delete_does_not_affect_others(self, feedback_store):
        feedback_store.record_feedback("task-001", "success")
        feedback_store.record_feedback("task-002", "failed")
        feedback_store.delete_feedback("task-001")
        assert feedback_store.get_feedback("task-002") is not None
