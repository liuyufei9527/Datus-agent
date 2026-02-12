# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/models/mcp_result_extractors.py"""

from unittest.mock import MagicMock

import pytest

from datus.models.mcp_result_extractors import (
    DB_QUERY_FUNCTIONS,
    extract_sql_contexts,
    get_function_call_names,
)
from datus.utils.constants import DBType


# =====================================================================
# get_function_call_names
# =====================================================================


class TestGetFunctionCallNames:
    def test_snowflake(self):
        names = get_function_call_names(DBType.SNOWFLAKE)
        assert "read_query" in names
        assert "list_tables" in names

    def test_sqlite(self):
        names = get_function_call_names(DBType.SQLITE)
        assert "read_query" in names
        assert "write_query" in names

    def test_starrocks(self):
        names = get_function_call_names(DBType.STARROCKS)
        assert "read_query" in names
        assert "table_overview" in names

    def test_duckdb(self):
        names = get_function_call_names(DBType.DUCKDB)
        assert "query" in names

    def test_unknown_db_type(self):
        names = get_function_call_names("unknown_db")
        assert names == set()


# =====================================================================
# extract_sql_contexts
# =====================================================================


class TestExtractSqlContexts:
    def test_empty_result(self):
        result = MagicMock()
        result.to_input_list.return_value = []
        contexts = extract_sql_contexts(result)
        assert contexts == []

    def test_no_to_input_list(self):
        result = MagicMock(spec=[])
        contexts = extract_sql_contexts(result)
        assert contexts == []

    def test_single_function_call_with_output(self):
        result = MagicMock()
        result.to_input_list.return_value = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "call_1",
                "arguments": '{"query": "SELECT * FROM users"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '[{"id": 1, "name": "Alice"}]',
            },
        ]
        contexts = extract_sql_contexts(result)
        assert len(contexts) == 1
        assert "read_query" in contexts[0].sql_query
        assert contexts[0].sql_return == '[{"id": 1, "name": "Alice"}]'

    def test_function_call_with_reflection(self):
        result = MagicMock()
        result.to_input_list.return_value = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "call_1",
                "arguments": '{"query": "SELECT 1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "1",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"text": "The query returned 1 row."}],
            },
        ]
        contexts = extract_sql_contexts(result)
        assert len(contexts) == 1
        assert contexts[0].reflection_explanation == "The query returned 1 row."

    def test_non_query_function_ignored(self):
        result = MagicMock()
        result.to_input_list.return_value = [
            {
                "type": "function_call",
                "name": "some_other_tool",
                "call_id": "call_1",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result",
            },
        ]
        contexts = extract_sql_contexts(result)
        assert len(contexts) == 0

    def test_multiple_function_calls(self):
        result = MagicMock()
        result.to_input_list.return_value = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "call_1",
                "arguments": '{"query": "SELECT 1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "1",
            },
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "call_2",
                "arguments": '{"query": "SELECT 2"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "2",
            },
        ]
        contexts = extract_sql_contexts(result)
        assert len(contexts) == 2

    def test_function_call_without_output(self):
        result = MagicMock()
        result.to_input_list.return_value = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "call_1",
                "arguments": '{"query": "SELECT 1"}',
            },
        ]
        contexts = extract_sql_contexts(result)
        assert len(contexts) == 1
        assert contexts[0].sql_return is None

    def test_mismatched_call_ids(self):
        result = MagicMock()
        result.to_input_list.return_value = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "call_1",
                "arguments": '{"query": "SELECT 1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_999",
                "output": "wrong",
            },
        ]
        contexts = extract_sql_contexts(result)
        assert len(contexts) == 1
        assert contexts[0].sql_return is None

    def test_reflection_blocked_by_next_function_call(self):
        result = MagicMock()
        result.to_input_list.return_value = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "call_1",
                "arguments": '{"query": "SELECT 1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "1",
            },
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "call_2",
                "arguments": '{"query": "SELECT 2"}',
            },
        ]
        contexts = extract_sql_contexts(result)
        assert len(contexts) == 2
        # First context should not have reflection since next item is a function call
        assert contexts[0].reflection_explanation is None

    def test_duckdb_query_function(self):
        result = MagicMock()
        result.to_input_list.return_value = [
            {
                "type": "function_call",
                "name": "query",
                "call_id": "call_1",
                "arguments": '{"sql": "SELECT * FROM t"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result",
            },
        ]
        contexts = extract_sql_contexts(result, db_type=DBType.DUCKDB)
        assert len(contexts) == 1
