"""Tests for datus/storage/schema_metadata/local_init.py."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from datus.storage.schema_metadata.local_init import (
    _fill_sample_rows,
    init_duckdb_schema,
    init_local_schema,
    init_mysql_schema,
    init_other_three_level_schema,
    init_sqlite_schema,
    init_starrocks_schema,
    store_tables,
)
from datus.utils.constants import DBType


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.after_init = MagicMock()
    return store


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.current_namespace = "default"
    return config


@pytest.fixture
def mock_connector():
    conn = MagicMock()
    conn.identifier.return_value = "db.tbl"
    conn.get_tables_with_ddl.return_value = [
        {
            "identifier": "db.tbl",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
            "table_name": "tbl",
            "definition": "CREATE TABLE tbl (id INT)",
        }
    ]
    conn.get_sample_rows.return_value = [{"identifier": "db.tbl", "sample_rows": "1,2,3"}]
    return conn


@pytest.fixture
def mock_db_manager(mock_connector):
    manager = MagicMock()
    manager.get_conn.return_value = mock_connector
    return manager


# ── store_tables ──


class TestStoreTables:
    def test_empty_tables(self, mock_store, mock_connector):
        store_tables(mock_store, "db", {}, set(), [], "table", mock_connector)
        mock_store.store_batch.assert_not_called()

    def test_new_table(self, mock_store, mock_connector):
        tables = [
            {
                "identifier": "db.t1",
                "catalog_name": "",
                "database_name": "db",
                "schema_name": "",
                "table_name": "t1",
                "definition": "CREATE TABLE t1(id INT)",
            }
        ]
        store_tables(mock_store, "db", {}, set(), tables, "table", mock_connector)
        mock_store.store_batch.assert_called_once()

    def test_existing_table_same_definition(self, mock_store, mock_connector):
        tables = [
            {
                "identifier": "db.t1",
                "catalog_name": "",
                "database_name": "db",
                "schema_name": "",
                "table_name": "t1",
                "definition": "CREATE TABLE t1(id INT)",
            }
        ]
        exists = {"db.t1": "CREATE TABLE t1(id INT)"}
        store_tables(mock_store, "db", exists, {"db.t1"}, tables, "table", mock_connector)
        mock_store.store_batch.assert_not_called()

    def test_existing_table_different_definition(self, mock_store, mock_connector):
        tables = [
            {
                "identifier": "db.t1",
                "catalog_name": "",
                "database_name": "db",
                "schema_name": "",
                "table_name": "t1",
                "definition": "CREATE TABLE t1(id INT, name TEXT)",
            }
        ]
        exists = {"db.t1": "CREATE TABLE t1(id INT)"}
        store_tables(mock_store, "db", exists, set(), tables, "table", mock_connector)
        mock_store.remove_data.assert_called_once()
        mock_store.store_batch.assert_called_once()

    def test_fill_missing_identifier(self, mock_store, mock_connector):
        tables = [
            {
                "catalog_name": "cat",
                "database_name": "db",
                "schema_name": "sch",
                "table_name": "t1",
                "definition": "CREATE TABLE t1(id INT)",
            }
        ]
        store_tables(mock_store, "db", {}, set(), tables, "table", mock_connector)
        mock_connector.identifier.assert_called()

    def test_existing_schema_missing_value(self, mock_store, mock_connector):
        """If schema exists but value does not, add sample rows only."""
        tables = [
            {
                "identifier": "db.t1",
                "catalog_name": "",
                "database_name": "db",
                "schema_name": "",
                "table_name": "t1",
                "definition": "CREATE TABLE t1(id INT)",
            }
        ]
        exists_schemas = {"db.t1": "CREATE TABLE t1(id INT)"}
        exists_values = set()  # no values exist
        store_tables(mock_store, "db", exists_schemas, exists_values, tables, "table", mock_connector)
        mock_store.store_batch.assert_called_once()


# ── _fill_sample_rows ──


class TestFillSampleRows:
    def test_fill_sample_rows_success(self, mock_connector):
        new_values = []
        table_data = {
            "table_name": "t1",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
        }
        _fill_sample_rows(new_values, "db.t1", table_data, mock_connector)
        assert len(new_values) == 1

    def test_fill_sample_rows_empty(self, mock_connector):
        mock_connector.get_sample_rows.return_value = []
        new_values = []
        table_data = {"table_name": "t1", "catalog_name": "", "database_name": "db", "schema_name": ""}
        _fill_sample_rows(new_values, "db.t1", table_data, mock_connector)
        assert len(new_values) == 0

    def test_fill_sample_rows_exception(self, mock_connector):
        mock_connector.get_sample_rows.side_effect = RuntimeError("db error")
        new_values = []
        table_data = {"table_name": "t1", "catalog_name": "", "database_name": "db", "schema_name": ""}
        # Should not raise
        _fill_sample_rows(new_values, "db.t1", table_data, mock_connector)
        assert len(new_values) == 0

    def test_fill_missing_identifier_in_rows(self, mock_connector):
        mock_connector.get_sample_rows.return_value = [{"sample_rows": "data"}]
        new_values = []
        table_data = {"table_name": "t1", "catalog_name": "", "database_name": "db", "schema_name": ""}
        _fill_sample_rows(new_values, "db.t1", table_data, mock_connector)
        assert new_values[0]["identifier"] == "db.t1"


# ── init_local_schema dispatching ──


class TestInitLocalSchema:
    def _make_db_config(self, db_type):
        cfg = MagicMock()
        cfg.type = db_type
        cfg.database = "testdb"
        return cfg

    @patch("datus.storage.schema_metadata.local_init.init_sqlite_schema")
    def test_dispatch_sqlite(self, mock_init, mock_agent_config, mock_store, mock_db_manager):
        db_cfg = self._make_db_config(DBType.SQLITE)
        mock_agent_config.namespaces = {"default": {"db1": db_cfg}}
        init_local_schema(mock_store, mock_agent_config, mock_db_manager)
        mock_init.assert_called_once()

    @patch("datus.storage.schema_metadata.local_init.init_duckdb_schema")
    def test_dispatch_duckdb(self, mock_init, mock_agent_config, mock_store, mock_db_manager):
        db_cfg = self._make_db_config(DBType.DUCKDB)
        mock_agent_config.namespaces = {"default": {"db1": db_cfg}}
        init_local_schema(mock_store, mock_agent_config, mock_db_manager)
        mock_init.assert_called_once()

    @patch("datus.storage.schema_metadata.local_init.init_mysql_schema")
    def test_dispatch_mysql(self, mock_init, mock_agent_config, mock_store, mock_db_manager):
        db_cfg = self._make_db_config(DBType.MYSQL)
        mock_agent_config.namespaces = {"default": {"db1": db_cfg}}
        init_local_schema(mock_store, mock_agent_config, mock_db_manager)
        mock_init.assert_called_once()

    @patch("datus.storage.schema_metadata.local_init.init_starrocks_schema")
    def test_dispatch_starrocks(self, mock_init, mock_agent_config, mock_store, mock_db_manager):
        db_cfg = self._make_db_config(DBType.STARROCKS)
        mock_agent_config.namespaces = {"default": {"db1": db_cfg}}
        init_local_schema(mock_store, mock_agent_config, mock_db_manager)
        mock_init.assert_called_once()

    @patch("datus.storage.schema_metadata.local_init.init_other_three_level_schema")
    def test_dispatch_other(self, mock_init, mock_agent_config, mock_store, mock_db_manager):
        db_cfg = self._make_db_config(DBType.POSTGRESQL)
        mock_agent_config.namespaces = {"default": {"db1": db_cfg}}
        init_local_schema(mock_store, mock_agent_config, mock_db_manager)
        mock_init.assert_called_once()

    def test_multiple_databases(self, mock_agent_config, mock_store, mock_db_manager):
        cfg1 = self._make_db_config(DBType.SQLITE)
        cfg2 = self._make_db_config(DBType.SQLITE)
        mock_agent_config.namespaces = {"default": {"db1": cfg1, "db2": cfg2}}

        with patch("datus.storage.schema_metadata.local_init.init_sqlite_schema") as mock_init:
            init_local_schema(mock_store, mock_agent_config, mock_db_manager)
            assert mock_init.call_count == 2

    def test_empty_db_configs(self, mock_agent_config, mock_store, mock_db_manager):
        mock_agent_config.namespaces = {"default": {}}
        # Should handle empty config gracefully
        init_local_schema(mock_store, mock_agent_config, mock_db_manager)
        mock_store.after_init.assert_called_once()

    def test_filter_by_database_name(self, mock_agent_config, mock_store, mock_db_manager):
        cfg1 = self._make_db_config(DBType.SQLITE)
        cfg2 = self._make_db_config(DBType.SQLITE)
        mock_agent_config.namespaces = {"default": {"db1": cfg1, "db2": cfg2}}

        with patch("datus.storage.schema_metadata.local_init.init_sqlite_schema") as mock_init:
            init_local_schema(mock_store, mock_agent_config, mock_db_manager, init_database_name="db1")
            assert mock_init.call_count == 1


# ── init_sqlite_schema ──


class TestInitSqliteSchema:
    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_basic_call(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.database = "testdb"
        init_sqlite_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager)
        mock_store_tables.assert_called()

    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_full_type_includes_views(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.database = "testdb"
        conn = mock_db_manager.get_conn.return_value
        conn.get_views_with_ddl.return_value = []
        init_sqlite_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager, table_type="full")
        # Should call store_tables at least twice (tables and views)
        assert mock_store_tables.call_count >= 2


# ── init_duckdb_schema ──


class TestInitDuckdbSchema:
    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_basic_call(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.database = "duckdb"
        db_cfg.schema = "main"
        init_duckdb_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager)
        mock_store_tables.assert_called()

    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_fills_missing_database_name(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.database = "duckdb"
        db_cfg.schema = ""
        conn = mock_db_manager.get_conn.return_value
        tables = [{"table_name": "t1", "definition": "DDL"}]
        conn.get_tables_with_ddl.return_value = tables
        init_duckdb_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager)
        # Tables should have database_name filled
        assert tables[0].get("database_name") == "duckdb"


# ── init_mysql_schema ──


class TestInitMysqlSchema:
    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_basic_call(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.database = "mydb"
        init_mysql_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager)
        mock_store_tables.assert_called()

    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_full_includes_views(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.database = "mydb"
        conn = mock_db_manager.get_conn.return_value
        conn.get_views_with_ddl.return_value = []
        init_mysql_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager, table_type="full")
        assert mock_store_tables.call_count >= 2


# ── init_starrocks_schema ──


class TestInitStarrocksSchema:
    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_basic_call(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.database = "srdb"
        db_cfg.catalog = "default_catalog"
        conn = mock_db_manager.get_conn.return_value
        conn.catalog_name = "default_catalog"
        conn.database_name = "srdb"
        init_starrocks_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager)
        mock_store_tables.assert_called()

    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_full_includes_views_and_mvs(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.database = "srdb"
        db_cfg.catalog = "cat"
        conn = mock_db_manager.get_conn.return_value
        conn.catalog_name = "cat"
        conn.database_name = "srdb"
        conn.get_views_with_ddl.return_value = []
        conn.get_materialized_views_with_ddl.return_value = []
        init_starrocks_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager, table_type="full")
        # tables, views, mvs
        assert mock_store_tables.call_count >= 3


# ── init_other_three_level_schema ──


class TestInitOtherThreeLevelSchema:
    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_postgresql(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.type = DBType.POSTGRESQL
        db_cfg.database = "pgdb"
        db_cfg.schema = "public"
        db_cfg.catalog = ""
        conn = mock_db_manager.get_conn.return_value
        conn.database_name = "pgdb"
        conn.schema_name = "public"
        init_other_three_level_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager)
        mock_store_tables.assert_called()

    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_snowflake(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.type = DBType.SNOWFLAKE
        db_cfg.database = "sfdb"
        db_cfg.schema = "PUBLIC"
        db_cfg.catalog = ""
        conn = mock_db_manager.get_conn.return_value
        conn.database_name = "sfdb"
        conn.schema_name = "PUBLIC"
        init_other_three_level_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager)
        mock_store_tables.assert_called()

    @patch("datus.storage.schema_metadata.local_init.store_tables")
    @patch("datus.storage.schema_metadata.local_init.exists_table_value", return_value=({}, set()))
    def test_fallback_without_get_tables_with_ddl(self, mock_exists, mock_store_tables, mock_store, mock_agent_config, mock_db_manager):
        db_cfg = MagicMock()
        db_cfg.type = "other"
        db_cfg.database = "odb"
        db_cfg.schema = ""
        db_cfg.catalog = ""
        conn = mock_db_manager.get_conn.return_value
        conn.database_name = "odb"
        # Remove get_tables_with_ddl to trigger fallback
        del conn.get_tables_with_ddl
        conn.get_tables.return_value = ["t1"]
        conn.identifier.return_value = "odb.t1"
        init_other_three_level_schema(mock_store, mock_agent_config, db_cfg, mock_db_manager)
        mock_store_tables.assert_called()
