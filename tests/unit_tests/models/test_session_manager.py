# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/models/session_manager.py using real SQLite databases."""

import json
import os
import sqlite3
from unittest.mock import patch

import pytest

from datus.models.session_manager import ExtendedSQLiteSession, SessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def session_manager(tmp_path):
    """Create a SessionManager using tmp_path for storage."""
    with patch("datus.utils.path_manager.get_path_manager") as mock_pm:
        mock_pm.return_value.sessions_dir = str(tmp_path / "sessions")
        manager = SessionManager()
        return manager


@pytest.fixture
def populated_session_manager(session_manager):
    """Session manager with pre-created sessions."""
    s1 = session_manager.get_session("session_1")
    s2 = session_manager.get_session("session_2")
    return session_manager


# ---------------------------------------------------------------------------
# ExtendedSQLiteSession
# ---------------------------------------------------------------------------


class TestExtendedSQLiteSession:
    def test_init_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        session = ExtendedSQLiteSession("test_session", db_path=db_path)

        # Verify the database was created with the correct schema
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Check agent_sessions table has total_tokens column
            cursor.execute("PRAGMA table_info(agent_sessions)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "total_tokens" in columns
            assert "session_id" in columns
            assert "created_at" in columns
            assert "updated_at" in columns

            # Check agent_messages table exists
            cursor.execute("PRAGMA table_info(agent_messages)")
            msg_columns = {row[1] for row in cursor.fetchall()}
            assert "session_id" in msg_columns
            assert "message_data" in msg_columns


# ---------------------------------------------------------------------------
# SessionManager - get_session / create_session
# ---------------------------------------------------------------------------


class TestSessionManagerBasic:
    def test_get_session_creates_new(self, session_manager):
        session = session_manager.get_session("test_session")
        assert session is not None
        assert "test_session" in session_manager._sessions

    def test_get_session_returns_existing(self, session_manager):
        s1 = session_manager.get_session("test_session")
        s2 = session_manager.get_session("test_session")
        assert s1 is s2

    def test_create_session_alias(self, session_manager):
        session = session_manager.create_session("my_session")
        assert session is not None
        # create_session is just an alias for get_session
        same_session = session_manager.get_session("my_session")
        assert session is same_session


# ---------------------------------------------------------------------------
# SessionManager - list_sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_empty_sessions(self, session_manager):
        sessions = session_manager.list_sessions()
        assert sessions == []

    def test_list_existing_sessions(self, session_manager):
        session_manager.get_session("session_a")
        session_manager.get_session("session_b")
        sessions = session_manager.list_sessions()
        assert len(sessions) >= 2
        assert "session_a" in sessions
        assert "session_b" in sessions

    def test_list_with_limit(self, session_manager):
        for i in range(5):
            session_manager.get_session(f"session_{i}")
        sessions = session_manager.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_list_sorted_by_modified(self, session_manager):
        session_manager.get_session("old_session")
        import time

        time.sleep(0.1)
        session_manager.get_session("new_session")
        sessions = session_manager.list_sessions(sort_by_modified=True)
        assert len(sessions) >= 2
        # Newest first
        assert sessions[0] == "new_session"

    def test_list_sorted_with_limit(self, session_manager):
        for i in range(5):
            session_manager.get_session(f"session_{i}")
            import time

            time.sleep(0.05)
        sessions = session_manager.list_sessions(limit=2, sort_by_modified=True)
        assert len(sessions) == 2


# ---------------------------------------------------------------------------
# SessionManager - clear_session
# ---------------------------------------------------------------------------


class TestClearSession:
    def test_clear_existing_session(self, session_manager):
        session_manager.get_session("test_session")
        session_manager.clear_session("test_session")
        # Should not raise

    def test_clear_nonexistent_session(self, session_manager):
        # Should just log a warning, not raise
        session_manager.clear_session("nonexistent")


# ---------------------------------------------------------------------------
# SessionManager - delete_session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_existing_session(self, session_manager):
        session_manager.get_session("test_session")
        db_path = os.path.join(session_manager.session_dir, "test_session.db")
        assert os.path.exists(db_path)

        session_manager.delete_session("test_session")
        assert "test_session" not in session_manager._sessions
        assert not os.path.exists(db_path)

    def test_delete_nonexistent_session(self, session_manager):
        # Should just log a warning, not raise
        session_manager.delete_session("nonexistent")


# ---------------------------------------------------------------------------
# SessionManager - session_exists
# ---------------------------------------------------------------------------


class TestSessionExists:
    def test_nonexistent_session(self, session_manager):
        assert session_manager.session_exists("no_such_session") is False

    def test_empty_session_file(self, session_manager):
        session_manager.get_session("empty_session")
        # Session file exists but may not have data yet
        # The result depends on whether the SDK creates initial records
        result = session_manager.session_exists("empty_session")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# SessionManager - get_session_info
# ---------------------------------------------------------------------------


class TestGetSessionInfo:
    def test_nonexistent_session(self, session_manager):
        info = session_manager.get_session_info("no_such_session")
        assert info["exists"] is False

    def test_existing_session_info(self, session_manager):
        session_manager.get_session("test_session")
        info = session_manager.get_session_info("test_session")
        assert info["exists"] is True
        assert info["session_id"] == "test_session"
        assert "db_path" in info


# ---------------------------------------------------------------------------
# SessionManager - close_all_sessions
# ---------------------------------------------------------------------------


class TestCloseAllSessions:
    def test_close_all(self, session_manager):
        session_manager.get_session("s1")
        session_manager.get_session("s2")
        session_manager.get_session("s3")
        assert len(session_manager._sessions) == 3

        session_manager.close_all_sessions()
        assert len(session_manager._sessions) == 0


# ---------------------------------------------------------------------------
# SessionManager - update_session_tokens
# ---------------------------------------------------------------------------


class TestUpdateSessionTokens:
    def test_update_nonexistent_session(self, session_manager):
        # Should log warning, not raise
        session_manager.update_session_tokens("nonexistent", 100)

    def test_update_existing_session_tokens(self, session_manager):
        session_manager.get_session("test_session")
        db_path = os.path.join(session_manager.session_dir, "test_session.db")

        # Insert an initial session record for session_exists to return True
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO agent_sessions (session_id, total_tokens) VALUES (?, ?)",
                ("test_session", 0),
            )
            conn.commit()

        session_manager.update_session_tokens("test_session", 500)

        # Verify the update
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT total_tokens FROM agent_sessions WHERE session_id = ?", ("test_session",))
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == 500


# ---------------------------------------------------------------------------
# SessionManager - get_session_info (detailed)
# ---------------------------------------------------------------------------


class TestGetSessionInfoDetailed:
    def test_session_info_with_messages(self, session_manager):
        session_manager.get_session("detailed_session")
        db_path = os.path.join(session_manager.session_dir, "detailed_session.db")

        # Insert session record and messages
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO agent_sessions (session_id, total_tokens) VALUES (?, ?)",
                ("detailed_session", 1000),
            )
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, datetime('now'))",
                ("detailed_session", json.dumps({"role": "user", "content": "Hello!"})),
            )
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, datetime('now'))",
                ("detailed_session", json.dumps({"role": "assistant", "content": "Hi there!"})),
            )
            conn.commit()

        info = session_manager.get_session_info("detailed_session")
        assert info["exists"] is True
        assert info["session_id"] == "detailed_session"
        assert info["total_tokens"] == 1000
        assert info["message_count"] == 2
        assert info["latest_user_message"] == "Hello!"

    def test_session_info_no_messages(self, session_manager):
        session_manager.get_session("empty_msg_session")
        db_path = os.path.join(session_manager.session_dir, "empty_msg_session.db")

        # Insert session record only, no messages
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO agent_sessions (session_id, total_tokens) VALUES (?, ?)",
                ("empty_msg_session", 0),
            )
            conn.commit()

        info = session_manager.get_session_info("empty_msg_session")
        assert info["exists"] is True
        assert info["total_tokens"] == 0
        assert info.get("latest_user_message") is None

    def test_session_info_malformed_message(self, session_manager):
        session_manager.get_session("malformed_session")
        db_path = os.path.join(session_manager.session_dir, "malformed_session.db")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO agent_sessions (session_id, total_tokens) VALUES (?, ?)",
                ("malformed_session", 0),
            )
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, datetime('now'))",
                ("malformed_session", "not valid json"),
            )
            conn.commit()

        info = session_manager.get_session_info("malformed_session")
        assert info["exists"] is True
        # Malformed message should be skipped gracefully
        assert info.get("latest_user_message") is None


# ---------------------------------------------------------------------------
# SessionManager - session_exists (detailed)
# ---------------------------------------------------------------------------


class TestSessionExistsDetailed:
    def test_exists_with_messages(self, session_manager):
        session_manager.get_session("msg_session")
        db_path = os.path.join(session_manager.session_dir, "msg_session.db")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, datetime('now'))",
                ("msg_session", json.dumps({"role": "user", "content": "test"})),
            )
            conn.commit()

        assert session_manager.session_exists("msg_session") is True

    def test_exists_with_session_record_only(self, session_manager):
        session_manager.get_session("record_session")
        db_path = os.path.join(session_manager.session_dir, "record_session.db")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO agent_sessions (session_id, total_tokens) VALUES (?, ?)",
                ("record_session", 0),
            )
            conn.commit()

        assert session_manager.session_exists("record_session") is True

    def test_not_exists_empty_db(self, session_manager):
        session_manager.get_session("empty_db_session")
        # No data inserted, just the db file exists with schema
        assert session_manager.session_exists("empty_db_session") is False
