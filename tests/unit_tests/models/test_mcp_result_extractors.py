# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.mcp_result_extractors``."""

from datus.models.mcp_result_extractors import (
    DB_QUERY_FUNCTIONS,
    extract_sql_contexts,
    get_function_call_names,
)
from datus.models.result import NewItem, RunResult


class TestGetFunctionCallNames:
    def test_known_db_type_returns_set(self):
        names = get_function_call_names("snowflake")
        assert "read_query" in names

    def test_unknown_db_type_returns_empty(self):
        assert get_function_call_names("nonexistent") == set()

    def test_registry_has_all_main_dialects(self):
        for db_type in ("snowflake", "starrocks"):
            assert db_type in DB_QUERY_FUNCTIONS


class TestExtractSqlContexts:
    def test_call_paired_with_output(self):
        result = RunResult(
            new_items=[
                NewItem(
                    type="tool_call",
                    data={"call_id": "c1", "name": "read_query", "arguments": '{"sql":"SELECT 1"}'},
                ),
                NewItem(type="tool_result", data={"call_id": "c1", "output": "[]"}),
            ]
        )
        contexts = extract_sql_contexts(result, db_type="snowflake")
        assert len(contexts) == 1
        assert contexts[0].sql_return == "[]"
        assert "read_query" in contexts[0].sql_query

    def test_unknown_function_name_skipped(self):
        result = RunResult(
            new_items=[
                NewItem(type="tool_call", data={"call_id": "c1", "name": "noop", "arguments": "{}"}),
                NewItem(type="tool_result", data={"call_id": "c1", "output": ""}),
            ]
        )
        assert extract_sql_contexts(result, db_type="snowflake") == []

    def test_reflection_picked_from_following_assistant_message(self):
        result = RunResult(
            new_items=[
                NewItem(
                    type="tool_call",
                    data={"call_id": "c1", "name": "read_query", "arguments": '{"sql":"SELECT 1"}'},
                ),
                NewItem(type="tool_result", data={"call_id": "c1", "output": "[]"}),
                NewItem(type="message", data={"content": "Reflecting on result"}),
            ]
        )
        contexts = extract_sql_contexts(result, db_type="snowflake")
        assert contexts[0].reflection_explanation == "Reflecting on result"

    def test_orphan_tool_call_without_output(self):
        result = RunResult(
            new_items=[
                NewItem(
                    type="tool_call",
                    data={"call_id": "c1", "name": "read_query", "arguments": "{}"},
                )
            ]
        )
        contexts = extract_sql_contexts(result, db_type="snowflake")
        # Call without output still produces an entry — sql_return is None.
        assert len(contexts) == 1
        assert contexts[0].sql_return is None
