# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for db_manager.py"""

from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import DbConfig
from datus.tools.db_tools.db_manager import (
    DBManager,
    _clean_str,
    _extract_schema_from_pg_options,
    _normalize_dialect_name,
    _port_or_none,
    _value_or_none,
    db_config_name,
    gen_uri,
    get_connection,
)
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException


# ---------------------------------------------------------------------------
# _normalize_dialect_name
# ---------------------------------------------------------------------------


class TestNormalizeDialectName:
    def test_postgres_alias(self):
        assert _normalize_dialect_name("postgres") == "postgresql"

    def test_sqlserver_alias(self):
        assert _normalize_dialect_name("sqlserver") == "mssql"

    def test_regular_type(self):
        assert _normalize_dialect_name("mysql") == "mysql"

    def test_dbtype_enum(self):
        assert _normalize_dialect_name(DBType.POSTGRESQL) == "postgresql"

    def test_none(self):
        assert _normalize_dialect_name(None) == ""

    def test_empty_string(self):
        assert _normalize_dialect_name("") == ""

    def test_whitespace(self):
        assert _normalize_dialect_name("  mysql  ") == "mysql"


# ---------------------------------------------------------------------------
# _clean_str
# ---------------------------------------------------------------------------


class TestCleanStr:
    def test_none(self):
        assert _clean_str(None) == ""

    def test_string(self):
        assert _clean_str("  hello  ") == "hello"

    def test_int(self):
        assert _clean_str(5432) == "5432"

    def test_list_with_items(self):
        assert _clean_str(["first", "second"]) == "first"

    def test_empty_list(self):
        assert _clean_str([]) == ""

    def test_tuple(self):
        assert _clean_str(("val",)) == "val"

    def test_set_with_items(self):
        result = _clean_str({"item"})
        assert result == "item"


# ---------------------------------------------------------------------------
# _extract_schema_from_pg_options
# ---------------------------------------------------------------------------


class TestExtractSchemaFromPgOptions:
    def test_empty(self):
        assert _extract_schema_from_pg_options("") == ""

    def test_with_search_path(self):
        assert _extract_schema_from_pg_options("-c search_path=myschema") == "myschema"

    def test_with_multiple_schemas(self):
        result = _extract_schema_from_pg_options("-c search_path=schema1,schema2")
        assert result == "schema1"

    def test_no_search_path(self):
        assert _extract_schema_from_pg_options("-c timezone=UTC") == ""


# ---------------------------------------------------------------------------
# _value_or_none / _port_or_none
# ---------------------------------------------------------------------------


class TestValueOrNone:
    def test_none(self):
        assert _value_or_none(None) is None

    def test_empty(self):
        assert _value_or_none("") is None

    def test_value(self):
        assert _value_or_none("host") == "host"


class TestPortOrNone:
    def test_none(self):
        assert _port_or_none(None) is None

    def test_empty(self):
        assert _port_or_none("") is None

    def test_valid_int(self):
        assert _port_or_none("5432") == 5432

    def test_valid_int_type(self):
        assert _port_or_none(5432) == 5432

    def test_invalid(self):
        assert _port_or_none("abc") is None


# ---------------------------------------------------------------------------
# get_connection
# ---------------------------------------------------------------------------


class TestGetConnection:
    def test_single_connector(self):
        mock_conn = MagicMock()
        result = get_connection(mock_conn)
        assert result is mock_conn

    def test_dict_single_entry(self):
        mock_conn = MagicMock()
        result = get_connection({"db1": mock_conn})
        assert result is mock_conn

    def test_dict_with_name(self):
        conn1 = MagicMock()
        conn2 = MagicMock()
        result = get_connection({"db1": conn1, "db2": conn2}, "db2")
        assert result is conn2

    def test_dict_no_name(self):
        conn1 = MagicMock()
        conn2 = MagicMock()
        result = get_connection({"db1": conn1, "db2": conn2})
        assert result is conn1

    def test_dict_name_not_found(self):
        conn1 = MagicMock()
        with pytest.raises(DatusException):
            get_connection({"db1": conn1}, "nonexistent")


# ---------------------------------------------------------------------------
# gen_uri
# ---------------------------------------------------------------------------


class TestGenUri:
    def test_with_existing_uri(self):
        config = DbConfig(type="postgresql", uri="postgresql://user:pass@host/db")
        assert gen_uri(config) == "postgresql://user:pass@host/db"

    def test_postgresql(self):
        config = DbConfig(
            type="postgresql",
            username="user",
            password="pass",
            host="localhost",
            port="5432",
            database="testdb",
        )
        uri = gen_uri(config)
        assert "postgresql+psycopg" in uri
        assert "user" in uri

    def test_clickhouse(self):
        config = DbConfig(
            type="clickhouse",
            username="default",
            host="localhost",
            database="testdb",
        )
        uri = gen_uri(config)
        assert "clickhouse" in uri

    def test_mssql(self):
        config = DbConfig(
            type="mssql",
            username="sa",
            password="pass",
            host="localhost",
            database="testdb",
        )
        uri = gen_uri(config)
        assert "mssql+pyodbc" in uri

    def test_bigquery_missing_fields(self):
        config = DbConfig(type="bigquery")
        with pytest.raises(DatusException):
            gen_uri(config)

    def test_bigquery_valid(self):
        config = DbConfig(
            type="bigquery",
            catalog="project-id",
            database="dataset",
        )
        uri = gen_uri(config)
        assert "bigquery" in uri

    def test_oracle(self):
        config = DbConfig(
            type="oracle",
            username="system",
            password="pass",
            host="localhost",
            database="ORCL",
        )
        uri = gen_uri(config)
        assert "oracle" in uri

    def test_generic_type(self):
        config = DbConfig(
            type="mysql",
            username="root",
            host="localhost",
            database="testdb",
        )
        uri = gen_uri(config)
        assert "mysql" in uri


# ---------------------------------------------------------------------------
# db_config_name
# ---------------------------------------------------------------------------


class TestDbConfigName:
    def test_sqlite(self):
        result = db_config_name("ns1", DBType.SQLITE, "mydb")
        assert result == "ns1::mydb"

    def test_duckdb(self):
        result = db_config_name("ns1", DBType.DUCKDB, "mydb")
        assert result == "ns1::mydb"

    def test_other(self):
        result = db_config_name("ns1", "postgresql")
        assert result == "ns1::ns1"


# ---------------------------------------------------------------------------
# DBManager
# ---------------------------------------------------------------------------


class TestDBManager:
    def test_init_empty(self):
        manager = DBManager({})
        assert manager._db_configs == {}

    def test_current_db_configs(self):
        configs = {"ns1": {"db1": MagicMock()}}
        manager = DBManager(configs)
        assert manager.current_db_configs("ns1") == configs["ns1"]

    def test_get_db_uris(self):
        mock_config = MagicMock()
        mock_config.uri = "sqlite:///test.db"
        configs = {"ns1": {"db1": mock_config}}
        manager = DBManager(configs)
        uris = manager.get_db_uris("ns1")
        assert uris == {"db1": "sqlite:///test.db"}

    def test_get_db_uris_no_namespace(self):
        manager = DBManager({})
        uris = manager.get_db_uris("nonexistent")
        assert uris == {}

    def test_init_connections_namespace_not_found(self):
        manager = DBManager({})
        with pytest.raises(DatusException):
            manager._init_connections("nonexistent")

    def test_context_manager(self):
        manager = DBManager({})
        with manager as m:
            assert m is manager

    @patch("datus.tools.db_tools.db_manager.connector_registry")
    def test_init_conn_sqlite(self, mock_registry):
        mock_connector = MagicMock()
        mock_registry.create_connector.return_value = mock_connector

        db_config = DbConfig(type="sqlite", uri="sqlite:///test.db")
        configs = {"ns1": {"db1": db_config}}
        manager = DBManager(configs)
        conn = manager.get_conn("ns1")
        assert conn is mock_connector

    @patch("datus.tools.db_tools.db_manager.connector_registry")
    def test_init_conn_duckdb(self, mock_registry):
        mock_connector = MagicMock()
        mock_registry.create_connector.return_value = mock_connector

        db_config = DbConfig(type="duckdb", uri="duckdb:///test.duckdb")
        configs = {"ns1": {"db1": db_config}}
        manager = DBManager(configs)
        conn = manager.get_conn("ns1")
        assert conn is mock_connector

    @patch("datus.tools.db_tools.db_manager.connector_registry")
    def test_multi_db_init(self, mock_registry):
        mock_conn1 = MagicMock()
        mock_conn2 = MagicMock()
        mock_registry.create_connector.side_effect = [mock_conn1, mock_conn2]

        db_config1 = DbConfig(type="sqlite", uri="sqlite:///test1.db")
        db_config2 = DbConfig(type="sqlite", uri="sqlite:///test2.db")
        configs = {"ns1": {"db1": db_config1, "db2": db_config2}}
        manager = DBManager(configs)
        conn = manager.get_conn("ns1", "db1")
        assert conn is mock_conn1

    @patch("datus.tools.db_tools.db_manager.connector_registry")
    def test_first_conn(self, mock_registry):
        mock_connector = MagicMock()
        mock_registry.create_connector.return_value = mock_connector

        db_config = DbConfig(type="sqlite", uri="sqlite:///test.db")
        configs = {"ns1": {"db1": db_config}}
        manager = DBManager(configs)
        conn = manager.first_conn("ns1")
        assert conn is mock_connector

    @patch("datus.tools.db_tools.db_manager.connector_registry")
    def test_first_conn_with_name(self, mock_registry):
        mock_connector = MagicMock()
        mock_registry.create_connector.return_value = mock_connector

        db_config = DbConfig(type="sqlite", uri="sqlite:///test.db")
        db_config.logic_name = "main_db"
        configs = {"ns1": {"db1": db_config}}
        manager = DBManager(configs)
        name, conn = manager.first_conn_with_name("ns1")
        assert conn is mock_connector

    @patch("datus.tools.db_tools.db_manager.connector_registry")
    def test_first_conn_with_name_multi_db(self, mock_registry):
        mock_conn1 = MagicMock()
        mock_conn2 = MagicMock()
        mock_registry.create_connector.side_effect = [mock_conn1, mock_conn2]

        db_config1 = DbConfig(type="sqlite", uri="sqlite:///test1.db")
        db_config2 = DbConfig(type="sqlite", uri="sqlite:///test2.db")
        configs = {"ns1": {"db1": db_config1, "db2": db_config2}}
        manager = DBManager(configs)
        name, conn = manager.first_conn_with_name("ns1")
        assert name == "db1"
        assert conn is mock_conn1

    @patch("datus.tools.db_tools.db_manager.connector_registry")
    def test_close(self, mock_registry):
        mock_connector = MagicMock()
        mock_registry.create_connector.return_value = mock_connector

        db_config = DbConfig(type="sqlite", uri="sqlite:///test.db")
        configs = {"ns1": {"db1": db_config}}
        manager = DBManager(configs)
        manager.get_conn("ns1")
        manager.close()
        mock_connector.close.assert_called_once()

    def test_db_config_to_connection_config_generic(self):
        db_config = DbConfig(
            type="postgresql",
            uri="postgresql://user:pass@localhost/db",
            host="localhost",
            port="5432",
            username="user",
        )
        manager = DBManager({})
        result = manager._db_config_to_connection_config(db_config)
        assert isinstance(result, dict)
        assert result["port"] == 5432

    def test_db_config_to_connection_config_with_extra(self):
        db_config = DbConfig(
            type="postgresql",
            uri="postgresql://user:pass@localhost/db",
            host="localhost",
            extra={"ssl_mode": "require"},
        )
        manager = DBManager({})
        result = manager._db_config_to_connection_config(db_config)
        assert isinstance(result, dict)
        assert result.get("ssl_mode") == "require"

    @patch("datus.tools.db_tools.db_manager.connector_registry")
    def test_get_connections(self, mock_registry):
        mock_connector = MagicMock()
        mock_registry.create_connector.return_value = mock_connector

        db_config = DbConfig(type="sqlite", uri="sqlite:///test.db")
        configs = {"ns1": {"db1": db_config}}
        manager = DBManager(configs)
        conns = manager.get_connections("ns1")
        assert conns is mock_connector
