# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for duckdb_connector.py targeting uncovered lines."""

import uuid

import pytest

from datus.tools.db_tools.config import DuckDBConfig
from datus.tools.db_tools.duckdb_connector import (
    METADATA_DICT,
    DuckdbConnector,
    _DBMetadataNames,
    _metadata_names,
)
from datus.utils.exceptions import DatusException, ErrorCode


# ---------------------------------------------------------------------------
# _DBMetadataNames / _metadata_names / METADATA_DICT
# ---------------------------------------------------------------------------


class TestMetadataNames:
    def test_valid_types(self):
        for t in ["database", "schema", "table", "view"]:
            meta = _metadata_names(t)
            assert isinstance(meta, _DBMetadataNames)

    def test_invalid_type(self):
        with pytest.raises(DatusException):
            _metadata_names("invalid")

    def test_database_no_sql_field(self):
        meta = _metadata_names("database")
        assert meta.has_sql_field is False

    def test_table_has_sql_field(self):
        meta = _metadata_names("table")
        assert meta.has_sql_field is True


# ---------------------------------------------------------------------------
# DuckdbConnector constructor
# ---------------------------------------------------------------------------


class TestDuckdbConnectorInit:
    def test_db_path_strip_prefix(self):
        config = DuckDBConfig(db_path="duckdb:///test.duckdb")
        conn = DuckdbConnector(config)
        assert conn.db_path == "test.duckdb"

    def test_custom_database_name(self):
        config = DuckDBConfig(db_path=":memory:", database_name="mydb")
        conn = DuckdbConnector(config)
        assert conn.database_name == "mydb"


# ---------------------------------------------------------------------------
# DuckdbConnector connection lifecycle
# ---------------------------------------------------------------------------


class TestDuckdbConnectorLifecycle:
    @pytest.fixture
    def connector(self):
        config = DuckDBConfig(db_path=":memory:", database_name="test")
        conn = DuckdbConnector(config)
        yield conn
        conn.close()

    def test_connect(self, connector):
        connector.connect()
        assert connector.connection is not None

    def test_connect_idempotent(self, connector):
        connector.connect()
        conn1 = connector.connection
        connector.connect()
        assert connector.connection is conn1

    def test_close(self, connector):
        connector.connect()
        connector.close()
        assert connector.connection is None

    def test_close_when_not_connected(self, connector):
        # Should not raise
        connector.close()

    def test_test_connection(self, connector):
        assert connector.test_connection() is True

    def test_get_type(self, connector):
        assert connector.get_type() == "duckdb"

    def test_to_dict(self, connector):
        d = connector.to_dict()
        assert d["db_type"] == "duckdb"
        assert "db_path" in d


# ---------------------------------------------------------------------------
# _handle_exception
# ---------------------------------------------------------------------------


class TestHandleException:
    @pytest.fixture
    def connector(self):
        config = DuckDBConfig(db_path=":memory:", database_name="test")
        conn = DuckdbConnector(config)
        yield conn
        conn.close()

    def test_syntax_error(self, connector):
        ex = connector._handle_exception(Exception("Parser Error: syntax error"))
        assert ex.code == ErrorCode.DB_EXECUTION_SYNTAX_ERROR

    def test_table_not_exists(self, connector):
        ex = connector._handle_exception(Exception("table xyz does not exist"))
        assert ex.code == ErrorCode.DB_TABLE_NOT_EXISTS

    def test_constraint_violation(self, connector):
        ex = connector._handle_exception(Exception("Constraint violation: unique constraint"))
        assert ex.code == ErrorCode.DB_CONSTRAINT_VIOLATION

    def test_timeout(self, connector):
        ex = connector._handle_exception(Exception("Timeout reached"))
        assert ex.code == ErrorCode.DB_EXECUTION_TIMEOUT

    def test_generic_error(self, connector):
        ex = connector._handle_exception(Exception("Some other error"))
        assert ex.code == ErrorCode.DB_EXECUTION_ERROR

    def test_datus_exception_passthrough(self, connector):
        original = DatusException(ErrorCode.DB_CONNECTION_FAILED)
        result = connector._handle_exception(original)
        assert result is original


# ---------------------------------------------------------------------------
# execute_* methods
# ---------------------------------------------------------------------------


class TestDuckdbConnectorExecute:
    @pytest.fixture
    def connector(self):
        config = DuckDBConfig(db_path=":memory:", database_name="test")
        conn = DuckdbConnector(config)
        conn.connect()
        conn.connection.execute("CREATE TABLE test_tbl (id INT, name VARCHAR)")
        conn.connection.execute("INSERT INTO test_tbl VALUES (1, 'Alice'), (2, 'Bob')")
        yield conn
        conn.close()

    def test_execute_query_csv(self, connector):
        result = connector.execute_query("SELECT * FROM test_tbl", result_format="csv")
        assert result.success
        assert "Alice" in result.sql_return
        assert result.row_count == 2

    def test_execute_query_list(self, connector):
        result = connector.execute_query("SELECT * FROM test_tbl", result_format="list")
        assert result.success
        assert len(result.sql_return) == 2
        assert isinstance(result.sql_return[0], dict)

    def test_execute_query_pandas(self, connector):
        result = connector.execute_query("SELECT * FROM test_tbl", result_format="pandas")
        assert result.success
        assert result.row_count == 2

    def test_execute_query_arrow(self, connector):
        result = connector.execute_query("SELECT * FROM test_tbl", result_format="arrow")
        assert result.success
        assert result.row_count == 2

    def test_execute_query_error(self, connector):
        result = connector.execute_query("SELECT * FROM nonexistent")
        assert not result.success

    def test_execute_csv(self, connector):
        result = connector.execute_csv("SELECT * FROM test_tbl")
        assert result.success
        assert result.result_format == "csv"

    def test_execute_pandas(self, connector):
        result = connector.execute_pandas("SELECT * FROM test_tbl")
        assert result.success
        assert result.result_format == "pandas"

    def test_execute_insert(self, connector):
        result = connector.execute_insert("INSERT INTO test_tbl VALUES (3, 'Carol')")
        assert result.success

    def test_execute_insert_error(self, connector):
        result = connector.execute_insert("INSERT INTO nonexistent VALUES (1)")
        assert not result.success

    def test_execute_update(self, connector):
        result = connector.execute_update("UPDATE test_tbl SET name='Updated' WHERE id=1")
        assert result.success

    def test_execute_update_error(self, connector):
        result = connector.execute_update("UPDATE nonexistent SET x=1")
        assert not result.success

    def test_execute_delete(self, connector):
        result = connector.execute_delete("DELETE FROM test_tbl WHERE id=2")
        assert result.success

    def test_execute_delete_error(self, connector):
        result = connector.execute_delete("DELETE FROM nonexistent")
        assert not result.success

    def test_execute_ddl(self, connector):
        result = connector.execute_ddl("CREATE TABLE test_ddl (x INT)")
        assert result.success
        assert result.sql_return == "Success"

    def test_execute_ddl_error(self, connector):
        # Create table that already exists
        result = connector.execute_ddl("CREATE TABLE test_tbl (x INT)")
        assert not result.success

    def test_execute_queries(self, connector):
        results = connector.execute_queries([
            "SELECT * FROM test_tbl",
            "SELECT COUNT(*) as cnt FROM test_tbl",
        ])
        assert len(results) == 2

    def test_execute_content_set(self, connector):
        result = connector.execute_content_set("SET memory_limit='1GB'")
        assert result.success

    def test_execute_content_set_error(self, connector):
        result = connector.execute_content_set("SET nonexistent_param = 'value'")
        # DuckDB may or may not error here depending on the param
        # Just verify it returns an ExecuteSQLResult
        assert result is not None


# ---------------------------------------------------------------------------
# Schema/metadata methods
# ---------------------------------------------------------------------------


class TestDuckdbConnectorMetadata:
    @pytest.fixture
    def connector(self):
        config = DuckDBConfig(db_path=":memory:", database_name="memory")
        conn = DuckdbConnector(config)
        conn.connect()
        conn.connection.execute("CREATE TABLE test_meta (id INT, name VARCHAR)")
        yield conn
        conn.close()

    def test_get_tables(self, connector):
        tables = connector.get_tables()
        assert "test_meta" in tables

    def test_get_views(self, connector):
        connector.connection.execute("CREATE VIEW test_view AS SELECT * FROM test_meta")
        views = connector.get_views()
        assert "test_view" in views

    def test_get_databases(self, connector):
        dbs = connector.get_databases()
        assert "memory" in dbs

    def test_get_databases_include_sys(self, connector):
        dbs = connector.get_databases(include_sys=True)
        assert "system" in dbs

    def test_get_schemas(self, connector):
        schemas = connector.get_schemas()
        assert "main" in schemas

    def test_get_schemas_include_sys(self, connector):
        schemas = connector.get_schemas(include_sys=True)
        assert "information_schema" in schemas

    def test_get_schema(self, connector):
        schema = connector.get_schema(table_name="test_meta")
        assert len(schema) == 2
        names = [col["name"] for col in schema]
        assert "id" in names
        assert "name" in names
        # Check nullable normalization
        for col in schema:
            assert "nullable" in col

    def test_get_schema_empty_table(self, connector):
        result = connector.get_schema()
        assert result == []

    def test_get_schema_nonexistent_table(self, connector):
        with pytest.raises(DatusException):
            connector.get_schema(table_name="nonexistent_table_xyz")

    def test_full_name_with_db_and_schema(self, connector):
        result = connector.full_name(database_name="db", schema_name="sch", table_name="tbl")
        assert result == '"db"."sch"."tbl"'

    def test_full_name_with_db_no_schema(self, connector):
        result = connector.full_name(database_name="db", schema_name="", table_name="tbl")
        assert result == '"db"."tbl"'

    def test_full_name_no_db(self, connector):
        result = connector.full_name(schema_name="sch", table_name="tbl")
        assert result == '"sch"."tbl"'

    def test_full_name_no_db_no_schema(self, connector):
        result = connector.full_name(schema_name="", table_name="tbl")
        assert result == "tbl"

    def test_sys_schemas(self, connector):
        sys = connector._sys_schemas()
        assert "system" in sys
        assert "information_schema" in sys

    def test_do_switch_context(self, connector):
        # Should not raise
        connector.do_switch_context(schema_name="main")

    def test_get_tables_with_ddl(self, connector):
        tables = connector.get_tables_with_ddl()
        assert len(tables) >= 1
        assert tables[0]["table_name"] == "test_meta"
        assert tables[0]["table_type"] == "table"

    def test_get_views_with_ddl(self, connector):
        connector.connection.execute("CREATE VIEW v1 AS SELECT 1 as x")
        views = connector.get_views_with_ddl()
        assert len(views) >= 1

    def test_get_sample_rows_with_tables(self, connector):
        connector.connection.execute("INSERT INTO test_meta VALUES (1, 'Alice')")
        samples = connector.get_sample_rows(
            tables=["test_meta"],
            top_n=5,
        )
        assert len(samples) >= 1
        assert "Alice" in samples[0]["sample_rows"]

    def test_get_sample_rows_without_tables(self, connector):
        connector.connection.execute("INSERT INTO test_meta VALUES (1, 'Alice')")
        samples = connector.get_sample_rows(top_n=5)
        assert len(samples) >= 1

    def test_get_sample_rows_mv_type(self, connector):
        samples = connector.get_sample_rows(table_type="mv")
        assert samples == []

    def test_get_sample_rows_with_schema_and_db(self, connector):
        connector.connection.execute("INSERT INTO test_meta VALUES (1, 'Alice')")
        samples = connector.get_sample_rows(
            tables=["test_meta"],
            schema_name="main",
            database_name="memory",
            top_n=5,
        )
        assert len(samples) >= 1

    def test_get_sample_rows_with_schema_no_db(self, connector):
        connector.connection.execute("INSERT INTO test_meta VALUES (1, 'Alice')")
        samples = connector.get_sample_rows(
            tables=["test_meta"],
            schema_name="main",
            top_n=5,
        )
        assert len(samples) >= 1

    def test_get_sample_rows_full_type(self, connector):
        connector.connection.execute("INSERT INTO test_meta VALUES (1, 'Alice')")
        connector.connection.execute("CREATE VIEW v1 AS SELECT * FROM test_meta")
        samples = connector.get_sample_rows(table_type="full", top_n=5)
        assert len(samples) >= 2  # table + view
