# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for llms_tools/match_schema.py – 21.7% coverage target."""

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.tools.llms_tools.match_schema import (
    MatchSchemaTool,
    gen_all_table_dict,
    parse_matched_tables,
)
from datus.utils.exceptions import DatusException


# ---------------------------------------------------------------------------
# gen_all_table_dict
# ---------------------------------------------------------------------------


class TestGenAllTableDict:
    def test_basic(self):
        table = pa.table(
            {
                "identifier": ["db.schema.t1", "db.schema.t2"],
                "table_name": ["t1", "t2"],
            }
        )
        result = gen_all_table_dict(table)
        assert "db.schema.t1" in result
        assert "db.schema.t2" in result

    def test_empty(self):
        table = pa.table({"identifier": [], "table_name": []})
        result = gen_all_table_dict(table)
        assert len(result) == 0

    def test_large_batch(self):
        ids = [f"db.schema.t{i}" for i in range(2500)]
        names = [f"t{i}" for i in range(2500)]
        table = pa.table({"identifier": ids, "table_name": names})
        result = gen_all_table_dict(table)
        assert len(result) == 2500


# ---------------------------------------------------------------------------
# parse_matched_tables
# ---------------------------------------------------------------------------


class TestParseMatchedTables:
    def test_basic(self):
        all_table_dict = {
            "mydb.public.orders": {"schema_text": "CREATE TABLE orders(...)"},
        }
        tables = {
            "table": "public.orders",
            "score": 0.95,
            "reasons": ["Has order data"],
        }
        result = parse_matched_tables("mydb", tables, all_table_dict)
        assert len(result) == 1
        assert result[0]["table_name"] == "orders"
        assert result[0]["score"] == 0.95

    def test_list_of_tables(self):
        all_table_dict = {
            "mydb.public.orders": {"schema_text": "CREATE TABLE orders(...)"},
            "mydb.public.items": {"schema_text": "CREATE TABLE items(...)"},
        }
        tables = {
            "table": ["public.orders", "public.items"],
            "score": 0.9,
            "reasons": ["related"],
        }
        result = parse_matched_tables("mydb", tables, all_table_dict)
        assert len(result) == 2

    def test_table_not_found(self):
        all_table_dict = {}
        tables = {
            "table": "public.missing",
            "score": 0.5,
            "reasons": [],
        }
        result = parse_matched_tables("mydb", tables, all_table_dict)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# MatchSchemaTool
# ---------------------------------------------------------------------------


class TestMatchSchemaTool:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.generate.return_value = '[{"table": "public.orders", "score": 0.95, "reasons": ["order data"]}]'
        model.token_count.return_value = 100
        model.max_tokens.return_value = 10000
        return model

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        return storage

    @pytest.fixture
    def tool(self, mock_model, mock_storage):
        return MatchSchemaTool(model=mock_model, storage=mock_storage)

    def test_validate_input(self, tool):
        inp = SchemaLinkingInput(
            input_text="find orders",
            database_name="mydb",
            database_type="postgresql",
        )
        assert tool.validate_input(inp) is True
        assert tool.validate_input("not valid") is False

    def test_execute_no_metadata(self, tool, mock_storage):
        mock_storage.search_all.return_value = pa.table(
            {"identifier": [], "table_name": [], "schema_name": [], "catalog_name": [],
             "definition": [], "table_type": [], "schema_text": []}
        )
        inp = SchemaLinkingInput(
            input_text="find orders",
            database_name="mydb",
            database_type="postgresql",
        )
        result = tool.execute(inp)
        assert result.success is False
        assert "No table metadata" in result.error

    def test_count_task_size_single(self, tool, mock_model):
        mock_model.token_count.return_value = 100
        mock_model.max_tokens.return_value = 10000
        assert tool.count_task_size(mock_model, "test", "prompt") == 1

    def test_count_task_size_multiple(self, tool, mock_model):
        mock_model.token_count.return_value = 30000
        mock_model.max_tokens.return_value = 10000
        assert tool.count_task_size(mock_model, "test", "prompt") == 3

    def test_process_match_result_empty(self, tool):
        inp = SchemaLinkingInput(
            input_text="test",
            database_name="mydb",
            database_type="postgresql",
        )
        result = tool._process_match_result(inp, None, "mydb", {})
        assert result.success is False

    def test_process_match_result_with_data(self, tool):
        inp = SchemaLinkingInput(
            input_text="test",
            database_name="mydb",
            database_type="postgresql",
        )
        all_tables = {
            "mydb.public.orders": {
                "identifier": "mydb.public.orders",
                "catalog_name": "",
                "schema_name": "public",
                "table_name": "orders",
                "definition": "CREATE TABLE orders...",
                "table_type": "TABLE",
            }
        }
        match_result = [{"table": "public.orders", "score": 0.95, "reasons": ["match"]}]
        result = tool._process_match_result(inp, match_result, "mydb", all_tables)
        assert result.success is True
        assert result.schema_count == 1

    def test_process_match_result_table_not_found(self, tool):
        inp = SchemaLinkingInput(
            input_text="test",
            database_name="mydb",
            database_type="postgresql",
        )
        match_result = [{"table": "public.missing", "score": 0.5, "reasons": []}]
        result = tool._process_match_result(inp, match_result, "mydb", {})
        assert result.success is True
        assert result.schema_count == 0

    def test_process_match_result_with_identifier(self, tool):
        inp = SchemaLinkingInput(
            input_text="test",
            database_name="mydb",
            database_type="postgresql",
        )
        all_tables = {
            "mydb.public.orders": {
                "identifier": "custom_identifier",
                "catalog_name": "",
                "schema_name": "public",
                "table_name": "orders",
                "definition": "CREATE TABLE...",
                "table_type": "TABLE",
            }
        }
        match_result = [{"table": "public.orders", "score": 0.9, "reasons": []}]
        result = tool._process_match_result(inp, match_result, "mydb", all_tables)
        assert result.table_schemas[0].identifier == "custom_identifier"

    def test_process_match_result_list_table(self, tool):
        inp = SchemaLinkingInput(
            input_text="test",
            database_name="mydb",
            database_type="postgresql",
        )
        all_tables = {
            "mydb.public.t1": {
                "catalog_name": "",
                "schema_name": "public",
                "table_name": "t1",
                "definition": "...",
                "table_type": "TABLE",
            },
            "mydb.public.t2": {
                "catalog_name": "",
                "schema_name": "public",
                "table_name": "t2",
                "definition": "...",
                "table_type": "TABLE",
            },
        }
        match_result = [{"table": ["public.t1", "public.t2"], "score": 0.9, "reasons": []}]
        result = tool._process_match_result(inp, match_result, "mydb", all_tables)
        assert result.schema_count == 2
