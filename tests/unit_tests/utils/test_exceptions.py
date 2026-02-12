"""
Tests for datus/utils/exceptions.py.
Integration tests for SQLAlchemy connector exception handling.
Tests real database scenarios with SQLite.
"""

import os
import sys
import tempfile

import pytest

from datus.tools.db_tools import SQLiteConnector
from datus.tools.db_tools.config import SQLiteConfig
from datus.utils.exceptions import DatusException, ErrorCode, setup_exception_handler


class TestIntegrationExceptions:
    """Integration tests with real SQLite database."""

    def test_sqlite_connection_failure(self):
        """Test connection failure with invalid SQLite path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "nonexistent", "database.db")
            config = SQLiteConfig(db_path=f"sqlite:///{invalid_path}")
            connector = SQLiteConnector(config)

            with pytest.raises(DatusException) as exc_info:
                connector.test_connection()
            # SQLite connection errors should be mapped to DB_CONNECTION_FAILED
            assert exc_info.value.code == ErrorCode.DB_CONNECTION_FAILED

    def test_sqlite_table_not_found(self):
        """Test actual table not found error."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        result = connector.execute_query("SELECT * FROM nonexistent_table")
        assert not result.success
        assert ErrorCode.DB_TABLE_NOT_EXISTS.code in result.error

    def test_sqlite_column_not_found(self):
        """Test actual column not found error."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create a table
        connector.execute_ddl("CREATE TABLE test_table (id INTEGER, name TEXT)")

        result = connector.execute_query("SELECT nonexistent_column FROM test_table")

        assert not result.success
        assert ErrorCode.DB_EXECUTION_ERROR.code in result.error

    def test_sqlite_syntax_error(self):
        """Test actual SQL syntax error."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        result = connector.execute_query("SELEC * FROM test_table")
        assert not result.success
        assert ErrorCode.DB_EXECUTION_SYNTAX_ERROR.code in result.error

    def test_sqlite_primary_key_violation(self):
        """Test actual primary key violation."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table with primary key
        connector.execute_ddl("CREATE TABLE test_pk (id INTEGER PRIMARY KEY)")
        connector.execute_insert("INSERT INTO test_pk (id) VALUES (1)")

        res = connector.execute_insert("INSERT INTO test_pk (id) VALUES (1)")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_unique_constraint_violation(self):
        """Test actual unique constraint violation."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table with unique constraint
        connector.execute_ddl("CREATE TABLE test_unique (email TEXT UNIQUE)")
        connector.execute_insert("INSERT INTO test_unique (email) VALUES ('test@example.com')")

        res = connector.execute_insert("INSERT INTO test_unique (email) VALUES ('test@example.com')")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_not_null_violation(self):
        """Test actual not null constraint violation."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table with not null constraint
        connector.execute_ddl("CREATE TABLE test_notnull (name TEXT NOT NULL)")

        res = connector.execute_insert("INSERT INTO test_notnull (name) VALUES (NULL)")
        assert res.success is False
        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_sqlite_foreign_key_violation(self):
        """Test actual foreign key violation."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Enable foreign key constraints
        connector.execute_ddl("PRAGMA foreign_keys = ON")

        # Create tables with foreign key
        connector.execute_ddl("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connector.execute_ddl("CREATE TABLE child (parent_id INTEGER, FOREIGN KEY (parent_id) REFERENCES parent(id))")

        res = connector.execute_insert("INSERT INTO child (parent_id) VALUES (999)")
        assert res.success is False

        assert ErrorCode.DB_CONSTRAINT_VIOLATION.code in res.error

    def test_successful_operations_do_not_raise_exceptions(self):
        """Test that successful operations don't raise exceptions."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table
        connector.execute_ddl("CREATE TABLE test_success (id INTEGER, name TEXT)")

        # Insert data
        result = connector.execute_insert("INSERT INTO test_success (id, name) VALUES (1, 'test')")
        assert result.sql_return == "1"  # rowcount should be 1

        # Query data
        df = connector.execute_pandas("SELECT * FROM test_success").sql_return
        assert len(df) == 1
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["name"] == "test"

    def test_update_operations(self):
        """Test update operations with exception handling."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table and insert data
        connector.execute_ddl("CREATE TABLE test_update (id INTEGER, value INTEGER)")
        connector.execute_insert("INSERT INTO test_update (id, value) VALUES (1, 100)")

        # Successful update
        res = connector.execute_update("UPDATE test_update SET value = 200 WHERE id = 1")
        assert res.row_count == 1

        # Update non-existent record (should succeed but return 0 rows)
        res = connector.execute_update("UPDATE test_update SET value = 300 WHERE id = 999")
        assert res.row_count == 0

    def test_delete_operations(self):
        """Test delete operations with exception handling."""
        config = SQLiteConfig(db_path="sqlite:///:memory:")
        connector = SQLiteConnector(config)

        # Create table and insert data
        connector.execute_ddl("CREATE TABLE test_delete (id INTEGER)")
        connector.execute_insert("INSERT INTO test_delete (id) VALUES (1)")

        # Successful delete
        res = connector.execute_delete("DELETE FROM test_delete WHERE id = 1")
        assert res.row_count == 1

        # Delete non-existent record (should succeed but return 0 rows)
        res = connector.execute_delete("DELETE FROM test_delete WHERE id = 999")
        assert res.row_count == 0


# ===========================================================================
# Unit tests for ErrorCode and DatusException
# ===========================================================================


class TestErrorCode:
    def test_code_attribute(self):
        assert ErrorCode.COMMON_UNKNOWN.code == "1000000"

    def test_desc_attribute(self):
        assert "Unknown error" in ErrorCode.COMMON_UNKNOWN.desc

    def test_field_invalid_has_template(self):
        assert "{field_name}" in ErrorCode.COMMON_FIELD_INVALID.desc

    def test_all_codes_have_str_code(self):
        for ec in ErrorCode:
            assert isinstance(ec.code, str)
            assert isinstance(ec.desc, str)


class TestDatusException:
    def test_basic_exception(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert "1000000" in str(ex)
        assert "Unknown error" in str(ex)

    def test_custom_message(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN, message="Custom error msg")
        assert "Custom error msg" in str(ex)
        assert "1000000" in str(ex)

    def test_message_args(self):
        ex = DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message_args={"field_name": "age", "except_values": "1-100", "your_value": "abc"},
        )
        assert "age" in str(ex)
        assert "abc" in str(ex)

    def test_is_exception(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert isinstance(ex, Exception)

    def test_build_msg_no_args_no_message(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert ex.message == f"error_code=1000000, error_message={ErrorCode.COMMON_UNKNOWN.desc}"

    def test_str_representation(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN, message="test msg")
        assert str(ex) == ex.message

    def test_raises_and_catches(self):
        with pytest.raises(DatusException):
            raise DatusException(ErrorCode.DB_FAILED, message_args={"error_message": "connection lost"})


class TestSetupExceptionHandler:
    def test_sets_excepthook(self):
        original_hook = sys.excepthook
        try:
            setup_exception_handler()
            assert sys.excepthook != original_hook
        finally:
            sys.excepthook = original_hook

    def test_handler_with_console_logger(self, tmp_path):
        original_hook = sys.excepthook
        try:
            from datus.utils.loggings import configure_logging

            configure_logging(log_dir=str(tmp_path / "logs"))

            console_log_calls = []

            def mock_console_logger(msg):
                console_log_calls.append(msg)

            setup_exception_handler(console_logger=mock_console_logger)

            try:
                raise DatusException(ErrorCode.COMMON_UNKNOWN, message="handler test")
            except DatusException:
                exc_type, exc_val, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_val, exc_tb)

            assert len(console_log_calls) > 0
            assert "handler test" in console_log_calls[0]
        finally:
            sys.excepthook = original_hook

    def test_handler_with_regular_exception(self, tmp_path):
        original_hook = sys.excepthook
        try:
            from datus.utils.loggings import configure_logging

            configure_logging(log_dir=str(tmp_path / "logs"))

            console_log_calls = []

            def mock_console_logger(msg):
                console_log_calls.append(msg)

            setup_exception_handler(console_logger=mock_console_logger)

            try:
                raise ValueError("regular error")
            except ValueError:
                exc_type, exc_val, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_val, exc_tb)

            assert len(console_log_calls) > 0
        finally:
            sys.excepthook = original_hook

    def test_handler_with_prefix_wrap_func(self, tmp_path):
        original_hook = sys.excepthook
        try:
            from datus.utils.loggings import configure_logging

            configure_logging(log_dir=str(tmp_path / "logs"))

            console_log_calls = []

            def mock_console_logger(msg):
                console_log_calls.append(msg)

            def prefix_wrapper(prefix):
                return f"[WRAPPED]{prefix}"

            setup_exception_handler(console_logger=mock_console_logger, prefix_wrap_func=prefix_wrapper)

            try:
                raise ValueError("wrap test")
            except ValueError:
                exc_type, exc_val, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_val, exc_tb)

            assert len(console_log_calls) > 0
            assert "[WRAPPED]" in console_log_calls[0]
        finally:
            sys.excepthook = original_hook

    def test_handler_no_console_logger(self, tmp_path):
        original_hook = sys.excepthook
        try:
            from datus.utils.loggings import configure_logging

            configure_logging(log_dir=str(tmp_path / "logs"))
            setup_exception_handler()

            try:
                raise ValueError("no console test")
            except ValueError:
                exc_type, exc_val, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_val, exc_tb)
        finally:
            sys.excepthook = original_hook
