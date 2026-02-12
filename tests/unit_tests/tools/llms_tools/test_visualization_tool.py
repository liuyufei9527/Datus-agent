# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for visualization_tool.py – 0% coverage target."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from datus.schemas.visualization import VisualizationInput, VisualizationOutput
from datus.tools.llms_tools.visualization_tool import VisualizationTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_no_model():
    """VisualizationTool without LLM model (heuristic only)."""
    return VisualizationTool(agent_config=None, model=None)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.generate_with_json_output.return_value = {
        "chart_type": "bar",
        "x_col": "category",
        "y_cols": ["amount"],
        "reason": "Categories vs amounts",
    }
    return model


@pytest.fixture
def tool_with_model(mock_model):
    return VisualizationTool(agent_config=None, model=mock_model)


# ---------------------------------------------------------------------------
# _convert_to_dataframe
# ---------------------------------------------------------------------------


class TestConvertToDataframe:
    def test_none_data(self, tool_no_model):
        assert tool_no_model._convert_to_dataframe(None) is None

    def test_dataframe(self, tool_no_model):
        df = pd.DataFrame({"a": [1, 2]})
        result = tool_no_model._convert_to_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a"]

    def test_pyarrow_table(self, tool_no_model):
        table = pa.table({"x": [1, 2, 3]})
        result = tool_no_model._convert_to_dataframe(table)
        assert isinstance(result, pd.DataFrame)
        assert "x" in result.columns

    def test_list_of_dicts(self, tool_no_model):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = tool_no_model._convert_to_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_empty_list(self, tool_no_model):
        result = tool_no_model._convert_to_dataframe([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_unsupported_type(self, tool_no_model):
        assert tool_no_model._convert_to_dataframe("string") is None
        assert tool_no_model._convert_to_dataframe(42) is None


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_execute_wrong_type(self, tool_no_model):
        with pytest.raises(TypeError):
            tool_no_model.execute("not a VisualizationInput")

    def test_execute_empty_dataframe(self, tool_no_model):
        inp = VisualizationInput(data=pd.DataFrame())
        result = tool_no_model.execute(inp)
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_execute_heuristic(self, tool_no_model):
        df = pd.DataFrame({"category": ["A", "B", "C"], "sales": [100, 200, 300]})
        inp = VisualizationInput(data=df)
        result = tool_no_model.execute(inp)
        assert result.success is True
        assert result.chart_type in ("Bar Chart", "Pie Chart")

    def test_execute_with_model(self, tool_with_model):
        df = pd.DataFrame({"category": ["A", "B"], "amount": [10, 20]})
        inp = VisualizationInput(data=df)
        result = tool_with_model.execute(inp)
        assert result.success is True
        assert result.chart_type == "Bar Chart"

    def test_execute_llm_fallback_on_error(self, tool_with_model, mock_model):
        mock_model.generate_with_json_output.side_effect = Exception("LLM failed")
        df = pd.DataFrame({"category": ["A", "B"], "sales": [10, 20]})
        inp = VisualizationInput(data=df)
        result = tool_with_model.execute(inp)
        # Should fall back to heuristics
        assert result.success is True


# ---------------------------------------------------------------------------
# _rule_based_recommendation
# ---------------------------------------------------------------------------


class TestRuleBasedRecommendation:
    def test_datetime_plus_numeric(self, tool_no_model):
        df = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=5), "value": [1, 2, 3, 4, 5]}
        )
        result = tool_no_model._rule_based_recommendation(df)
        assert result.chart_type == "Line Chart"
        assert result.x_col == "date"

    def test_categorical_plus_numeric_pie(self, tool_no_model):
        df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [10, 20, 30]})
        result = tool_no_model._rule_based_recommendation(df)
        assert result.chart_type == "Pie Chart"

    def test_categorical_plus_numeric_bar(self, tool_no_model):
        # More than max_pie_categories distinct values
        tool_no_model.max_pie_categories = 2
        df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [10, 20, 30]})
        result = tool_no_model._rule_based_recommendation(df)
        assert result.chart_type == "Bar Chart"

    def test_two_numeric_scatter(self, tool_no_model):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = tool_no_model._rule_based_recommendation(df)
        assert result.chart_type == "Scatter Plot"

    def test_unknown(self, tool_no_model):
        df = pd.DataFrame({"a": [True, False, True]})
        result = tool_no_model._rule_based_recommendation(df)
        # Bool columns are categorical but without numeric, may be Unknown
        assert result.chart_type in ("Unknown", "Bar Chart", "Pie Chart")


# ---------------------------------------------------------------------------
# LLM-based recommendation
# ---------------------------------------------------------------------------


class TestLLMBasedRecommendation:
    def test_llm_recommendation(self, tool_with_model, mock_model):
        df = pd.DataFrame({"category": ["A", "B"], "amount": [10, 20]})
        result = tool_with_model._llm_based_recommendation(df)
        assert result is not None
        assert result.chart_type == "Bar Chart"

    def test_llm_returns_non_dict(self, tool_with_model, mock_model):
        mock_model.generate_with_json_output.return_value = "not a dict"
        df = pd.DataFrame({"x": [1, 2]})
        result = tool_with_model._llm_based_recommendation(df)
        assert result is None

    def test_llm_y_cols_as_string(self, tool_with_model, mock_model):
        mock_model.generate_with_json_output.return_value = {
            "chart_type": "line",
            "x_col": "date",
            "y_cols": "value",
            "reason": "",
        }
        df = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=3), "value": [1, 2, 3]}
        )
        result = tool_with_model._llm_based_recommendation(df)
        assert isinstance(result.y_cols, list)

    def test_llm_invalid_x_col(self, tool_with_model, mock_model):
        mock_model.generate_with_json_output.return_value = {
            "chart_type": "bar",
            "x_col": "nonexistent_col",
            "y_cols": ["amount"],
            "reason": "",
        }
        df = pd.DataFrame({"category": ["A", "B"], "amount": [10, 20]})
        result = tool_with_model._llm_based_recommendation(df)
        # Should fall back to selecting dimension
        assert result.x_col in df.columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_normalize_chart_type(self, tool_no_model):
        assert tool_no_model._normalize_chart_type("bar") == "Bar Chart"
        assert tool_no_model._normalize_chart_type("Line Chart") == "Line Chart"
        assert tool_no_model._normalize_chart_type("SCATTER") == "Scatter Plot"
        assert tool_no_model._normalize_chart_type("pie chart") == "Pie Chart"
        assert tool_no_model._normalize_chart_type("") == "Unknown"
        assert tool_no_model._normalize_chart_type("weird") == "Unknown"

    def test_default_reason(self, tool_no_model):
        reason = tool_no_model._default_reason("Line Chart", "date", ["sales"])
        assert "line chart" in reason.lower()

        reason = tool_no_model._default_reason("Bar Chart", "category", ["count"])
        assert "bar chart" in reason.lower()

        reason = tool_no_model._default_reason("Scatter Plot", "x", ["y"])
        assert "scatter" in reason.lower()

        reason = tool_no_model._default_reason("Pie Chart", "cat", ["val"])
        assert "pie" in reason.lower()

        reason = tool_no_model._default_reason("Unknown", "", [])
        assert "unable" in reason.lower()

    def test_select_dimension(self, tool_no_model):
        df = pd.DataFrame({"cat": ["A", "B"], "num": [1, 2]})
        dim = tool_no_model._select_dimension(df)
        assert dim == "cat"

    def test_select_dimension_datetime(self, tool_no_model):
        df = pd.DataFrame({"dt": pd.date_range("2024-01-01", periods=2), "num": [1, 2]})
        dim = tool_no_model._select_dimension(df)
        assert dim == "dt"

    def test_select_dimension_exclude(self, tool_no_model):
        df = pd.DataFrame({"cat": ["A", "B"], "num": [1, 2]})
        dim = tool_no_model._select_dimension(df, exclude={"cat"})
        assert dim == "num"

    def test_select_dimension_empty(self, tool_no_model):
        df = pd.DataFrame({"a": [1]})
        dim = tool_no_model._select_dimension(df, exclude={"a"})
        assert dim == ""

    def test_sanitize_y_cols(self, tool_no_model):
        df = pd.DataFrame({"x": ["A"], "y1": [1], "y2": [2]})
        result = tool_no_model._sanitize_y_cols(df, ["y1"], exclude={"x"})
        assert result == ["y1"]

    def test_sanitize_y_cols_empty(self, tool_no_model):
        df = pd.DataFrame({"x": ["A"], "y1": [1]})
        result = tool_no_model._sanitize_y_cols(df, ["nonexistent"], exclude={"x"})
        assert "y1" in result

    def test_select_numeric_metrics_limit(self, tool_no_model):
        tool_no_model.max_y_cols = 2
        result = tool_no_model._select_numeric_metrics(["a", "b", "c", "d"])
        assert len(result) == 2

    def test_format_columns_info(self, tool_no_model):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        info = tool_no_model._format_columns_info(df)
        assert "a" in info
        assert "b" in info
        assert "unique=" in info

    def test_format_data_preview(self, tool_no_model):
        df = pd.DataFrame({"a": [1, 2, 3]})
        preview = tool_no_model._format_data_preview(df)
        assert "1" in preview

    def test_format_data_preview_truncation(self, tool_no_model):
        tool_no_model.max_preview_char = 10
        df = pd.DataFrame({"a": list(range(100))})
        preview = tool_no_model._format_data_preview(df)
        assert "truncated" in preview.lower()

    def test_serialize_value(self, tool_no_model):
        assert tool_no_model._serialize_value(np.int64(5)) == 5
        assert tool_no_model._serialize_value(b"hello") == "hello"
        assert tool_no_model._serialize_value([1, 2]) == [1, 2]
        ts = pd.Timestamp("2024-01-01")
        assert isinstance(tool_no_model._serialize_value(ts), str)

    def test_is_numeric(self):
        assert VisualizationTool._is_numeric(pd.Series([1, 2, 3])) is True
        assert VisualizationTool._is_numeric(pd.Series(["a", "b"])) is False

    def test_is_categorical(self):
        assert VisualizationTool._is_categorical(pd.Series(["a", "b"])) is True
        assert VisualizationTool._is_categorical(pd.Series([True, False])) is True
        assert VisualizationTool._is_categorical(pd.Series([1, 2])) is False

    def test_error_output(self, tool_no_model):
        result = tool_no_model._error_output("err msg", "Bar Chart", "some reason")
        assert result.success is False
        assert result.error == "err msg"
        assert result.reason == "some reason"

    def test_error_output_default_reason(self, tool_no_model):
        result = tool_no_model._error_output("err", "Unknown")
        assert result.reason == "err"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_init_no_config(self):
        tool = VisualizationTool()
        assert tool.model is None

    def test_init_with_model(self):
        mock = MagicMock()
        tool = VisualizationTool(model=mock)
        assert tool.model is mock

    def test_init_agent_config_failure(self):
        config = MagicMock()
        with patch("datus.tools.llms_tools.visualization_tool.LLMBaseModel.create_model", side_effect=Exception("fail")):
            tool = VisualizationTool(agent_config=config)
            assert tool.model is None
