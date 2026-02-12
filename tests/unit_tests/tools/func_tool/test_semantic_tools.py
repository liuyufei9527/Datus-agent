# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for func_tool/semantic_tools.py"""

from unittest.mock import MagicMock, patch

import pytest

from datus.tools.func_tool.semantic_tools import (
    SemanticTools,
    _normalize_null,
)


# ---------------------------------------------------------------------------
# _normalize_null
# ---------------------------------------------------------------------------


class TestNormalizeNull:
    def test_null_string(self):
        assert _normalize_null("null") is None

    def test_none_string(self):
        assert _normalize_null("None") is None

    def test_regular_string(self):
        assert _normalize_null("hello") == "hello"

    def test_none_value(self):
        assert _normalize_null(None) is None

    def test_list_value(self):
        val = ["a", "b"]
        assert _normalize_null(val) == ["a", "b"]


# ---------------------------------------------------------------------------
# SemanticTools
# ---------------------------------------------------------------------------


class TestSemanticTools:
    @pytest.fixture
    def mock_agent_config(self):
        config = MagicMock()
        config.current_namespace = "test_ns"
        return config

    @pytest.fixture
    def semantic_tools(self, mock_agent_config):
        with patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"), \
             patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
            return SemanticTools(agent_config=mock_agent_config)

    def test_all_tools_name(self):
        names = SemanticTools.all_tools_name()
        assert "search_metrics" in names
        assert "list_metrics" in names
        assert "get_dimensions" in names
        assert "query_metrics" in names
        assert "validate_semantic" in names
        assert "attribution_analyze" in names

    def test_init(self, semantic_tools):
        assert semantic_tools._adapter is None
        assert semantic_tools._attribution_tool is None
        assert semantic_tools.adapter_type is None

    def test_adapter_no_type(self, semantic_tools):
        """When adapter_type is None, adapter should remain None."""
        assert semantic_tools.adapter is None

    def test_attribution_tool_no_adapter(self, semantic_tools):
        """When no adapter, attribution_tool should be None."""
        assert semantic_tools.attribution_tool is None

    def test_search_metrics_no_results(self, semantic_tools):
        semantic_tools.metric_rag.search_metrics.return_value = []
        result = semantic_tools.search_metrics(query_text="revenue")
        assert result.success == 0
        assert "No metrics found" in result.error

    def test_search_metrics_with_results(self, semantic_tools):
        semantic_tools.metric_rag.search_metrics.return_value = [
            {
                "name": "revenue",
                "description": "Total revenue",
                "metric_type": "sum",
                "dimensions": ["region"],
                "base_measures": ["amount"],
                "subject_path": ["Finance"],
            }
        ]
        result = semantic_tools.search_metrics(query_text="revenue")
        assert result.success == 1
        assert len(result.result) == 1
        assert result.result[0]["name"] == "revenue"

    def test_search_metrics_exception(self, semantic_tools):
        semantic_tools.metric_rag.search_metrics.side_effect = RuntimeError("DB error")
        result = semantic_tools.search_metrics(query_text="revenue")
        assert result.success == 0
        assert "Failed to search" in result.error

    def test_search_metrics_with_null_subject_path(self, semantic_tools):
        """Test that 'null' subject_path is normalized to None."""
        semantic_tools.metric_rag.search_metrics.return_value = []
        result = semantic_tools.search_metrics(query_text="revenue", subject_path="null")
        # It should not crash; normalize_null converts "null" to None
        assert result.success == 0

    def test_list_metrics_with_results(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = [
            {
                "name": "revenue",
                "description": "Total revenue",
                "metric_type": "sum",
                "dimensions": [],
                "base_measures": [],
                "subject_path": ["Finance"],
            }
        ]
        result = semantic_tools.list_metrics()
        assert result.success == 1
        assert len(result.result) == 1

    def test_list_metrics_empty_no_adapter(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = []
        result = semantic_tools.list_metrics()
        assert result.success == 1
        assert result.result == []

    def test_list_metrics_with_path_filter(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = [
            {"name": "m1", "subject_path": ["Finance", "Revenue"]},
            {"name": "m2", "subject_path": ["Marketing"]},
        ]
        result = semantic_tools.list_metrics(path=["Finance"])
        assert result.success == 1
        assert len(result.result) == 1

    def test_list_metrics_pagination(self, semantic_tools):
        metrics = [{"name": f"m{i}", "subject_path": []} for i in range(10)]
        semantic_tools.metric_rag.search_all_metrics.return_value = metrics
        result = semantic_tools.list_metrics(limit=3, offset=2)
        assert result.success == 1
        assert len(result.result) == 3

    def test_list_metrics_exception(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.side_effect = RuntimeError("fail")
        result = semantic_tools.list_metrics()
        assert result.success == 0
        assert "Failed to list" in result.error

    def test_get_dimensions_no_adapter_with_storage(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = [
            {"name": "revenue", "dimensions": ["region", "product"]}
        ]
        result = semantic_tools.get_dimensions(metric_name="revenue")
        assert result.success == 1
        assert result.result == ["region", "product"]

    def test_get_dimensions_not_found(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = []
        result = semantic_tools.get_dimensions(metric_name="unknown")
        assert result.success == 0
        assert "not found" in result.error

    def test_get_dimensions_exception(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.side_effect = RuntimeError("err")
        result = semantic_tools.get_dimensions(metric_name="revenue")
        assert result.success == 0
        assert "Failed to get dimensions" in result.error

    def test_get_dimensions_with_path(self, semantic_tools):
        semantic_tools.metric_rag.storage = MagicMock()
        semantic_tools.metric_rag.storage.search_all_metrics.return_value = [
            {"name": "revenue", "dimensions": ["region"]}
        ]
        result = semantic_tools.get_dimensions(metric_name="revenue", path=["Finance"])
        assert result.success == 1

    def test_query_metrics_no_adapter(self, semantic_tools):
        result = semantic_tools.query_metrics(metrics=["revenue"])
        assert result.success == 0
        assert "No semantic adapter" in result.error

    def test_validate_semantic_no_adapter(self, semantic_tools):
        result = semantic_tools.validate_semantic()
        assert result.success == 0
        assert "No semantic adapter" in result.error

    def test_attribution_analyze_no_tool(self, semantic_tools):
        result = semantic_tools.attribution_analyze(
            metric_name="revenue",
            candidate_dimensions=["region"],
            baseline_start="2026-01-01",
            baseline_end="2026-01-07",
            current_start="2026-01-08",
            current_end="2026-01-14",
        )
        assert result.success == 0
        assert "not available" in result.error

    def test_available_tools_no_adapter(self, semantic_tools):
        tools = semantic_tools.available_tools()
        tool_names = [t.name for t in tools]
        assert "search_metrics" in tool_names
        assert "list_metrics" in tool_names
        assert "get_dimensions" in tool_names
        assert "query_metrics" in tool_names
        # validate_semantic requires adapter
        assert "validate_semantic" not in tool_names

    def test_reload_adapter_no_type(self, semantic_tools):
        result = semantic_tools._reload_adapter()
        assert result is False
