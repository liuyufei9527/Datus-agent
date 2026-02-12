"""Tests for datus/storage/schema_metadata/benchmark_init_bird.py."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from datus.storage.schema_metadata.benchmark_init_bird import (
    generate_sql_by_desc_file,
    init_db,
    init_dev_schema,
    init_dev_schema_by_db,
    load_table_keys,
)


# ── load_table_keys ──


class TestLoadTableKeys:
    def test_load_table_keys(self, tmp_path):
        dev_tables = [
            {
                "db_id": "testdb",
                "table_names_original": ["users", "orders"],
                "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [1, "order_id"], [1, "user_id"]],
                "primary_keys": [1, 3],
                "foreign_keys": [[4, 1]],
            }
        ]
        json_path = tmp_path / "dev_tables.json"
        json_path.write_text(json.dumps(dev_tables))

        result = load_table_keys(str(json_path))
        assert "testdb" in result
        assert "users" in result["testdb"]
        assert "orders" in result["testdb"]
        assert result["testdb"]["users"]["primary_keys"] == ["`id`"]
        assert result["testdb"]["orders"]["primary_keys"] == ["`order_id`"]
        # Foreign key: orders.user_id -> users.id
        assert "foreign_keys" in result["testdb"]["orders"]
        fk = result["testdb"]["orders"]["foreign_keys"][0]
        assert fk["column"] == "`user_id`"
        assert fk["target_table"] == "`users`"
        assert fk["target_column"] == "`id`"

    def test_composite_primary_key(self, tmp_path):
        dev_tables = [
            {
                "db_id": "testdb",
                "table_names_original": ["t1"],
                "column_names_original": [[-1, "*"], [0, "a"], [0, "b"]],
                "primary_keys": [[1, 2]],
                "foreign_keys": [],
            }
        ]
        json_path = tmp_path / "dev_tables.json"
        json_path.write_text(json.dumps(dev_tables))

        result = load_table_keys(str(json_path))
        assert result["testdb"]["t1"]["primary_keys"] == ["`a`", "`b`"]


# ── generate_sql_by_desc_file ──


class TestGenerateSqlByDescFile:
    def test_basic_generation(self, tmp_path):
        csv_content = (
            "original_column_name,data_format,column_description,value_description\n"
            "id,integer,User ID,\n"
            "name,text,User name,Full name\n"
            "score,real,Score value,Between 0-100\n"
        )
        csv_path = tmp_path / "users.csv"
        csv_path.write_text(csv_content)

        table_infos = {
            "users": {
                "primary_keys": ["`id`"],
                "foreign_keys": [],
            }
        }

        result = generate_sql_by_desc_file("testdb", table_infos, str(csv_path))
        assert result["table_name"] == "users"
        assert result["database_name"] == "testdb"
        assert "CREATE TABLE" in result["definition"]
        assert "`id` INTEGER" in result["definition"]
        assert "`name` TEXT" in result["definition"]
        assert "`score` REAL" in result["definition"]
        assert "PRIMARY KEY" in result["definition"]
        assert "User ID" in result["definition"]

    def test_with_foreign_keys(self, tmp_path):
        csv_content = (
            "original_column_name,data_format,column_description,value_description\n"
            "order_id,integer,Order ID,\n"
            "user_id,integer,User reference,\n"
        )
        csv_path = tmp_path / "orders.csv"
        csv_path.write_text(csv_content)

        table_infos = {
            "orders": {
                "primary_keys": ["`order_id`"],
                "foreign_keys": [
                    {"column": "`user_id`", "target_table": "`users`", "target_column": "`id`"},
                ],
            }
        }

        result = generate_sql_by_desc_file("testdb", table_infos, str(csv_path))
        assert "FOREIGN KEY" in result["definition"]
        assert "REFERENCES" in result["definition"]

    def test_no_constraints(self, tmp_path):
        csv_content = (
            "original_column_name,data_format,column_description,value_description\n" "col1,text,Test column,\n"
        )
        csv_path = tmp_path / "simple.csv"
        csv_path.write_text(csv_content)

        table_infos = {"simple": {}}

        result = generate_sql_by_desc_file("testdb", table_infos, str(csv_path))
        assert "PRIMARY KEY" not in result["definition"]
        assert "FOREIGN KEY" not in result["definition"]


# ── init_db ──


class TestInitDb:
    def test_nonexistent_path(self, tmp_path):
        mock_manager = MagicMock()
        result = init_db(
            db_manager=mock_manager,
            namespace="ns",
            database_name="nonexist",
            table_keys={},
            databases_path=str(tmp_path / "nosuchdir"),
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        assert result == ([], [])

    def test_init_db_with_data(self, tmp_path):
        # Create dir structure
        db_dir = tmp_path / "testdb"
        desc_dir = db_dir / "database_description"
        desc_dir.mkdir(parents=True)

        csv_content = (
            "original_column_name,data_format,column_description,value_description\n" "id,integer,ID,\n"
        )
        (desc_dir / "users.csv").write_text(csv_content)

        table_keys = {"users": {"primary_keys": ["`id`"]}}

        mock_conn = MagicMock()
        mock_conn.identifier.return_value = "testdb.users"
        mock_conn.get_sample_rows.return_value = [{"sample_rows": "1,Alice"}]

        mock_manager = MagicMock()
        mock_manager.get_conn.return_value = mock_conn

        schema_result, value_result = init_db(
            db_manager=mock_manager,
            namespace="ns",
            database_name="testdb",
            table_keys=table_keys,
            databases_path=str(tmp_path),
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        assert len(schema_result) == 1
        assert schema_result[0]["table_name"] == "users"
        assert len(value_result) == 1

    def test_skip_existing_tables(self, tmp_path):
        db_dir = tmp_path / "testdb"
        desc_dir = db_dir / "database_description"
        desc_dir.mkdir(parents=True)

        csv_content = (
            "original_column_name,data_format,column_description,value_description\n" "id,integer,ID,\n"
        )
        (desc_dir / "users.csv").write_text(csv_content)

        table_keys = {"users": {"primary_keys": ["`id`"]}}

        mock_conn = MagicMock()
        mock_conn.identifier.return_value = "testdb.users"

        mock_manager = MagicMock()
        mock_manager.get_conn.return_value = mock_conn

        # Already in schema tables
        schema_result, value_result = init_db(
            db_manager=mock_manager,
            namespace="ns",
            database_name="testdb",
            table_keys=table_keys,
            databases_path=str(tmp_path),
            all_schema_tables={"testdb.users"},
            all_value_tables={"testdb.users"},
        )
        assert len(schema_result) == 0

    def test_skip_non_csv_files(self, tmp_path):
        db_dir = tmp_path / "testdb"
        desc_dir = db_dir / "database_description"
        desc_dir.mkdir(parents=True)
        (desc_dir / "readme.txt").write_text("not a csv")

        mock_manager = MagicMock()
        mock_manager.get_conn.return_value = MagicMock()

        schema_result, value_result = init_db(
            db_manager=mock_manager,
            namespace="ns",
            database_name="testdb",
            table_keys={},
            databases_path=str(tmp_path),
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        assert schema_result == []
        assert value_result == []


# ── init_dev_schema_by_db ──


class TestInitDevSchemaByDb:
    def test_skip_ds_store(self):
        mock_rag = MagicMock()
        init_dev_schema_by_db(
            rag=mock_rag,
            db_manager=MagicMock(),
            namespace="ns",
            database_name=".DS_Store",
            table_keys={},
            databases_path="",
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        mock_rag.store_batch.assert_not_called()


# ── init_dev_schema ──


class TestInitDevSchema:
    def test_init_dev_schema(self, tmp_path):
        # Create minimal bird structure
        bird_path = tmp_path / "bird"
        (bird_path / "dev_databases" / "testdb" / "database_description").mkdir(parents=True)

        dev_tables = [
            {
                "db_id": "testdb",
                "table_names_original": ["t1"],
                "column_names_original": [[-1, "*"], [0, "col1"]],
                "primary_keys": [1],
                "foreign_keys": [],
            }
        ]
        (bird_path / "dev_tables.json").write_text(json.dumps(dev_tables))

        csv_content = (
            "original_column_name,data_format,column_description,value_description\n" "col1,text,Column 1,\n"
        )
        (bird_path / "dev_databases" / "testdb" / "database_description" / "t1.csv").write_text(csv_content)

        mock_conn = MagicMock()
        mock_conn.identifier.return_value = "testdb.t1"
        mock_conn.get_sample_rows.return_value = []

        mock_manager = MagicMock()
        mock_manager.get_conn.return_value = mock_conn

        mock_rag = MagicMock()

        with patch(
            "datus.storage.schema_metadata.benchmark_init_bird.exists_table_value",
            return_value=({}, set()),
        ):
            init_dev_schema(
                rag=mock_rag,
                db_manager=mock_manager,
                namespace="ns",
                bird_path=str(bird_path),
                pool_size=1,
            )
        mock_rag.store_batch.assert_called()
        mock_rag.after_init.assert_called_once()

    def test_filter_by_database_names(self, tmp_path):
        bird_path = tmp_path / "bird"
        (bird_path / "dev_databases" / "db1" / "database_description").mkdir(parents=True)
        (bird_path / "dev_databases" / "db2" / "database_description").mkdir(parents=True)

        dev_tables = [
            {
                "db_id": "db1",
                "table_names_original": [],
                "column_names_original": [[-1, "*"]],
                "primary_keys": [],
                "foreign_keys": [],
            },
            {
                "db_id": "db2",
                "table_names_original": [],
                "column_names_original": [[-1, "*"]],
                "primary_keys": [],
                "foreign_keys": [],
            },
        ]
        (bird_path / "dev_tables.json").write_text(json.dumps(dev_tables))

        mock_rag = MagicMock()

        with patch(
            "datus.storage.schema_metadata.benchmark_init_bird.exists_table_value",
            return_value=({}, set()),
        ):
            init_dev_schema(
                rag=mock_rag,
                db_manager=MagicMock(),
                namespace="ns",
                bird_path=str(bird_path),
                pool_size=1,
                database_names=["db1"],
            )
        mock_rag.after_init.assert_called_once()
