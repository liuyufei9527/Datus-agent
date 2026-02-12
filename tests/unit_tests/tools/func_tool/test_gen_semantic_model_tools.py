# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for gen_semantic_model_tools.py – 13.6% coverage target."""

from unittest.mock import MagicMock, patch

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.gen_semantic_model_tools import GenSemanticModelTools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_tool():
    tool = MagicMock()
    tool.agent_config = MagicMock()
    tool.sub_agent_name = "test_agent"
    return tool


@pytest.fixture
def gen_tools(mock_db_tool):
    return GenSemanticModelTools(mock_db_tool)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    def test_init(self, gen_tools, mock_db_tool):
        assert gen_tools.db_tool is mock_db_tool
        assert gen_tools.agent_config is mock_db_tool.agent_config
        assert gen_tools.sub_agent_name == "test_agent"


# ---------------------------------------------------------------------------
# available_tools
# ---------------------------------------------------------------------------


class TestAvailableTools:
    def test_returns_three_tools(self, gen_tools):
        tools = gen_tools.available_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "analyze_table_relationships" in names
        assert "get_multiple_tables_ddl" in names
        assert "analyze_column_usage_patterns" in names


# ---------------------------------------------------------------------------
# _deduplicate_relationships
# ---------------------------------------------------------------------------


class TestDeduplicateRelationships:
    def test_empty(self, gen_tools):
        assert gen_tools._deduplicate_relationships([]) == []

    def test_no_duplicates(self, gen_tools):
        rels = [
            {"source_table": "a", "source_column": "id", "target_table": "b", "target_column": "a_id", "confidence": "high"},
        ]
        result = gen_tools._deduplicate_relationships(rels)
        assert len(result) == 1

    def test_duplicates_keep_higher_confidence(self, gen_tools):
        rels = [
            {"source_table": "a", "source_column": "id", "target_table": "b", "target_column": "a_id", "confidence": "low"},
            {"source_table": "a", "source_column": "id", "target_table": "b", "target_column": "a_id", "confidence": "high"},
        ]
        result = gen_tools._deduplicate_relationships(rels)
        assert len(result) == 1
        assert result[0]["confidence"] == "high"

    def test_sort_order(self, gen_tools):
        rels = [
            {"source_table": "a", "source_column": "c1", "target_table": "b", "target_column": "c2", "confidence": "low"},
            {"source_table": "c", "source_column": "c3", "target_table": "d", "target_column": "c4", "confidence": "high"},
            {"source_table": "e", "source_column": "c5", "target_table": "f", "target_column": "c6", "confidence": "medium"},
        ]
        result = gen_tools._deduplicate_relationships(rels)
        assert len(result) == 3
        assert result[0]["confidence"] == "high"
        assert result[1]["confidence"] == "medium"
        assert result[2]["confidence"] == "low"


# ---------------------------------------------------------------------------
# _extract_foreign_keys_from_ddl
# ---------------------------------------------------------------------------


class TestExtractForeignKeysFromDdl:
    def test_basic(self, gen_tools, mock_db_tool):
        mock_db_tool.get_table_ddl.return_value = FuncToolResult(
            result={"definition": "CREATE TABLE orders (id INT, customer_id INT, FOREIGN KEY (customer_id) REFERENCES customers(id))"}
        )
        result = gen_tools._extract_foreign_keys_from_ddl(["orders"], "", "db", "public")
        assert len(result) == 1
        assert result[0]["source_table"] == "orders"
        assert result[0]["source_column"] == "customer_id"
        assert result[0]["target_table"] == "customers"
        assert result[0]["target_column"] == "id"
        assert result[0]["confidence"] == "high"

    def test_no_fk(self, gen_tools, mock_db_tool):
        mock_db_tool.get_table_ddl.return_value = FuncToolResult(
            result={"definition": "CREATE TABLE orders (id INT, name VARCHAR(100))"}
        )
        result = gen_tools._extract_foreign_keys_from_ddl(["orders"], "", "db", "")
        assert len(result) == 0

    def test_ddl_failure(self, gen_tools, mock_db_tool):
        mock_db_tool.get_table_ddl.return_value = FuncToolResult(success=0, error="fail")
        result = gen_tools._extract_foreign_keys_from_ddl(["orders"], "", "", "")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _infer_from_column_names
# ---------------------------------------------------------------------------


class TestInferFromColumnNames:
    def test_infer(self, gen_tools, mock_db_tool):
        mock_db_tool.describe_table.side_effect = [
            FuncToolResult(result={"columns": [{"name": "id"}, {"name": "order_id"}]}),
            FuncToolResult(result={"columns": [{"name": "id"}, {"name": "name"}]}),
        ]
        result = gen_tools._infer_from_column_names(["orders", "order"], "", "db", "public")
        assert len(result) == 1
        assert result[0]["source_table"] == "orders"
        assert result[0]["source_column"] == "order_id"
        assert result[0]["confidence"] == "low"

    def test_no_match(self, gen_tools, mock_db_tool):
        mock_db_tool.describe_table.return_value = FuncToolResult(
            result={"columns": [{"name": "id"}, {"name": "name"}]}
        )
        result = gen_tools._infer_from_column_names(["orders"], "", "", "")
        assert len(result) == 0

    def test_describe_failure(self, gen_tools, mock_db_tool):
        mock_db_tool.describe_table.return_value = FuncToolResult(success=0, error="fail")
        result = gen_tools._infer_from_column_names(["orders"], "", "", "")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _analyze_join_patterns_from_history
# ---------------------------------------------------------------------------


class TestAnalyzeJoinPatterns:
    _REF_SQL_RAG_PATH = "datus.storage.reference_sql.store.ReferenceSqlRAG"

    def test_no_agent_config(self, gen_tools):
        gen_tools.agent_config = None
        result = gen_tools._analyze_join_patterns_from_history(["orders"], 10)
        assert len(result) == 0

    def test_with_join_pattern(self, gen_tools):
        # pre-import module so patch target resolves
        import datus.storage.reference_sql.store  # noqa: F401

        mock_rag = MagicMock()
        # Use full table names (not aliases) in the join condition so the regex matches
        mock_rag.search_reference_sql.return_value = [
            {"sql": "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id"},
        ]
        with patch(self._REF_SQL_RAG_PATH, return_value=mock_rag):
            result = gen_tools._analyze_join_patterns_from_history(["orders", "customers"], 10)
            assert len(result) >= 1
            assert result[0]["confidence"] == "medium"

    def test_search_error(self, gen_tools):
        import datus.storage.reference_sql.store  # noqa: F401

        mock_rag = MagicMock()
        mock_rag.search_reference_sql.side_effect = Exception("fail")
        with patch(self._REF_SQL_RAG_PATH, return_value=mock_rag):
            result = gen_tools._analyze_join_patterns_from_history(["orders"], 10)
            assert len(result) == 0


# ---------------------------------------------------------------------------
# analyze_table_relationships
# ---------------------------------------------------------------------------


class TestAnalyzeTableRelationships:
    _REF_SQL_RAG_PATH = "datus.storage.reference_sql.store.ReferenceSqlRAG"

    def test_fk_relationships(self, gen_tools, mock_db_tool):
        import datus.storage.reference_sql.store  # noqa: F401

        mock_db_tool.get_table_ddl.return_value = FuncToolResult(
            result={"definition": "FOREIGN KEY (cust_id) REFERENCES customers(id)"}
        )
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = []
        with patch(self._REF_SQL_RAG_PATH, return_value=mock_rag):
            result = gen_tools.analyze_table_relationships(["orders", "customers"])
            assert result.success == 1
            assert len(result.result["relationships"]) >= 1

    def test_fallback_to_column_names(self, gen_tools, mock_db_tool):
        # No FK constraints, no join patterns -> fall back to column names
        mock_db_tool.get_table_ddl.return_value = FuncToolResult(
            result={"definition": "CREATE TABLE orders (id INT)"}
        )
        mock_db_tool.describe_table.side_effect = [
            FuncToolResult(result={"columns": [{"name": "id"}, {"name": "customer_id"}]}),
            FuncToolResult(result={"columns": [{"name": "id"}]}),
        ]
        gen_tools.agent_config = None  # No SQL history
        result = gen_tools.analyze_table_relationships(["orders", "customer"])
        assert result.success == 1

    def test_error(self, gen_tools, mock_db_tool):
        mock_db_tool.get_table_ddl.side_effect = Exception("db error")
        result = gen_tools.analyze_table_relationships(["orders"])
        assert result.success == 0


# ---------------------------------------------------------------------------
# get_multiple_tables_ddl
# ---------------------------------------------------------------------------


class TestGetMultipleTablesDdl:
    def test_success(self, gen_tools, mock_db_tool):
        mock_db_tool.get_table_ddl.return_value = FuncToolResult(
            result={"definition": "CREATE TABLE orders ..."}
        )
        result = gen_tools.get_multiple_tables_ddl(["orders", "customers"])
        assert result.success == 1
        assert len(result.result) == 2

    def test_partial_failure(self, gen_tools, mock_db_tool):
        mock_db_tool.get_table_ddl.side_effect = [
            FuncToolResult(result={"definition": "CREATE TABLE orders ..."}),
            FuncToolResult(success=0, error="not found"),
        ]
        result = gen_tools.get_multiple_tables_ddl(["orders", "missing"])
        assert result.success == 1
        assert len(result.result) == 2
        assert "error" in result.result[1]

    def test_error(self, gen_tools, mock_db_tool):
        mock_db_tool.get_table_ddl.side_effect = Exception("db error")
        result = gen_tools.get_multiple_tables_ddl(["orders"])
        assert result.success == 0


# ---------------------------------------------------------------------------
# analyze_column_usage_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeColumnUsagePatterns:
    _REF_SQL_RAG_PATH = "datus.storage.reference_sql.store.ReferenceSqlRAG"

    def test_no_agent_config(self, gen_tools):
        gen_tools.agent_config = None
        result = gen_tools.analyze_column_usage_patterns("orders")
        assert result.success == 0

    def test_schema_failure(self, gen_tools, mock_db_tool):
        mock_db_tool.describe_table.return_value = FuncToolResult(success=0, error="fail")
        result = gen_tools.analyze_column_usage_patterns("orders")
        assert result.success == 0

    def test_with_patterns(self, gen_tools, mock_db_tool):
        import datus.storage.reference_sql.store  # noqa: F401

        mock_db_tool.describe_table.return_value = FuncToolResult(
            result={"columns": [{"name": "status"}, {"name": "name"}]}
        )
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = [
            {"sql": "SELECT * FROM orders WHERE status = 1 AND name LIKE '%test%'"},
            {"sql": "SELECT * FROM orders WHERE status IN (1,2,3)"},
        ]
        with patch(self._REF_SQL_RAG_PATH, return_value=mock_rag):
            result = gen_tools.analyze_column_usage_patterns("orders")
            assert result.success == 1
            patterns = result.result["column_patterns"]
            assert "status" in patterns
            assert "=" in patterns["status"]["operators"]

    def test_with_specific_columns(self, gen_tools, mock_db_tool):
        import datus.storage.reference_sql.store  # noqa: F401

        mock_db_tool.describe_table.return_value = FuncToolResult(
            result={"columns": [{"name": "status"}, {"name": "name"}, {"name": "tags"}]}
        )
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = [
            {"sql": "SELECT * FROM orders WHERE FIND_IN_SET('vip', tags)"},
        ]
        with patch(self._REF_SQL_RAG_PATH, return_value=mock_rag):
            result = gen_tools.analyze_column_usage_patterns("orders", columns=["tags"])
            assert result.success == 1
            patterns = result.result["column_patterns"]
            assert "tags" in patterns
            assert "FIND_IN_SET" in patterns["tags"]["functions"]

    def test_no_matching_sql(self, gen_tools, mock_db_tool):
        import datus.storage.reference_sql.store  # noqa: F401

        mock_db_tool.describe_table.return_value = FuncToolResult(
            result={"columns": [{"name": "id"}]}
        )
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = [
            {"sql": "SELECT * FROM other_table WHERE x = 1"},
        ]
        with patch(self._REF_SQL_RAG_PATH, return_value=mock_rag):
            result = gen_tools.analyze_column_usage_patterns("orders")
            assert result.success == 1
            assert len(result.result["column_patterns"]) == 0

    def test_error(self, gen_tools, mock_db_tool):
        mock_db_tool.describe_table.side_effect = Exception("error")
        result = gen_tools.analyze_column_usage_patterns("orders")
        assert result.success == 0
