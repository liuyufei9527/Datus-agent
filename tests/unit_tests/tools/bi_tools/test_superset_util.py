# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for superset_util.py – 28.2% coverage target."""

import json

import pytest

from datus.tools.bi_tools.superset.superset_util import (
    DTTM_ALIAS,
    LEGACY_VIZ_TYPES,
    ComparisonType,
    ContributionMode,
    DatasourceKey,
    DatasourceType,
    ExpressionType,
    QueryContext,
    QueryMode,
    QueryObject,
    RollingType,
    ensure_list,
    flatten_operator,
    get_column_label,
    get_metric_label,
    get_x_axis_column,
    get_x_axis_column_with_time_grain,
    get_x_axis_label,
    is_adhoc_column,
    is_physical_column,
    is_time_comparison,
    is_x_axis_set,
    normalize_orderby,
    normalize_time_column,
    pivot_operator,
    resample_operator,
    rolling_window_operator,
    time_compare_pivot_operator,
    uses_legacy_api,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_query_mode(self):
        assert QueryMode.AGGREGATE == "aggregate"
        assert QueryMode.RAW == "raw"

    def test_expression_type(self):
        assert ExpressionType.SIMPLE == "SIMPLE"
        assert ExpressionType.SQL == "SQL"

    def test_rolling_type(self):
        assert RollingType.CUMSUM == "cumsum"

    def test_comparison_type(self):
        assert ComparisonType.VALUES == "values"

    def test_contribution_mode(self):
        assert ContributionMode.ROW == "row"

    def test_datasource_type(self):
        assert DatasourceType.TABLE == "table"


# ---------------------------------------------------------------------------
# uses_legacy_api
# ---------------------------------------------------------------------------


class TestUsesLegacyApi:
    def test_legacy(self):
        assert uses_legacy_api("bubble") is True
        assert uses_legacy_api("world_map") is True

    def test_non_legacy(self):
        assert uses_legacy_api("echarts_timeseries_bar") is False
        assert uses_legacy_api("pie") is False


# ---------------------------------------------------------------------------
# QueryObject
# ---------------------------------------------------------------------------


class TestQueryObject:
    def test_to_dict_filters_empty(self):
        qo = QueryObject()
        d = qo.to_dict()
        assert "time_range" not in d
        assert "columns" not in d
        assert "metrics" not in d

    def test_to_dict_with_values(self):
        qo = QueryObject(
            time_range="last week",
            columns=["col1"],
            metrics=["count"],
            row_limit=100,
        )
        d = qo.to_dict()
        assert d["time_range"] == "last week"
        assert d["columns"] == ["col1"]
        assert d["row_limit"] == 100

    def test_to_dict_none_excluded(self):
        qo = QueryObject(granularity=None)
        d = qo.to_dict()
        assert "granularity" not in d


# ---------------------------------------------------------------------------
# QueryContext
# ---------------------------------------------------------------------------


class TestQueryContext:
    def test_to_dict(self):
        qo = QueryObject(columns=["c1"])
        qc = QueryContext(
            datasource={"id": 1, "type": "table"},
            force=False,
            queries=[qo],
            form_data={"viz_type": "bar"},
        )
        d = qc.to_dict()
        assert d["datasource"]["id"] == 1
        assert len(d["queries"]) == 1
        assert d["form_data"]["viz_type"] == "bar"


# ---------------------------------------------------------------------------
# DatasourceKey
# ---------------------------------------------------------------------------


class TestDatasourceKey:
    def test_from_string(self):
        dk = DatasourceKey.from_string("123__table")
        assert dk.id == 123
        assert dk.type == DatasourceType.TABLE

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            DatasourceKey.from_string("invalid")

    def test_from_dict(self):
        dk = DatasourceKey.from_string({"id": 5, "type": "table"})
        assert dk.id == 5

    def test_to_dict(self):
        dk = DatasourceKey(id=1, type=DatasourceType.TABLE)
        assert dk.to_dict() == {"id": 1, "type": "table"}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


class TestEnsureList:
    def test_none(self):
        assert ensure_list(None) == []

    def test_already_list(self):
        assert ensure_list([1, 2]) == [1, 2]

    def test_single(self):
        assert ensure_list("hello") == ["hello"]


class TestGetColumnLabel:
    def test_string(self):
        assert get_column_label("col1") == "col1"

    def test_dict_label(self):
        assert get_column_label({"label": "My Col"}) == "My Col"

    def test_dict_sql(self):
        assert get_column_label({"sqlExpression": "SUM(x)"}) == "SUM(x)"

    def test_dict_column_name(self):
        assert get_column_label({"column_name": "c"}) == "c"

    def test_other(self):
        assert get_column_label(42) == "42"


class TestGetMetricLabel:
    def test_string(self):
        assert get_metric_label("count") == "count"

    def test_dict_label(self):
        assert get_metric_label({"label": "My Metric"}) == "My Metric"

    def test_dict_aggregate(self):
        m = {"aggregate": "COUNT", "column": {"column_name": "id"}}
        assert get_metric_label(m) == "COUNT(id)"

    def test_dict_sql_expression(self):
        assert get_metric_label({"sqlExpression": "SUM(x)"}) == "SUM(x)"

    def test_other(self):
        assert get_metric_label(42) == "42"


class TestIsPhysicalColumn:
    def test_string(self):
        assert is_physical_column("col") is True

    def test_dict(self):
        assert is_physical_column({"sqlExpression": "x"}) is False


class TestIsXAxisSet:
    def test_string(self):
        assert is_x_axis_set({"x_axis": "date"}) is True

    def test_dict_with_sql(self):
        assert is_x_axis_set({"x_axis": {"sqlExpression": "date"}}) is True

    def test_none(self):
        assert is_x_axis_set({}) is False

    def test_empty_dict(self):
        assert is_x_axis_set({"x_axis": {}}) is False


class TestGetXAxisColumn:
    def test_explicit(self):
        assert get_x_axis_column({"x_axis": "date"}) == "date"

    def test_fallback_to_granularity(self):
        assert get_x_axis_column({"granularity_sqla": "ts"}) == "ts"

    def test_none(self):
        assert get_x_axis_column({}) is None


class TestGetXAxisLabel:
    def test_with_axis(self):
        assert get_x_axis_label({"x_axis": "date"}) == "date"

    def test_none(self):
        assert get_x_axis_label({}) is None


class TestIsAdhocColumn:
    def test_valid(self):
        col = {"sqlExpression": "SUM(x)", "label": "Total", "expressionType": "SQL"}
        assert is_adhoc_column(col) is True

    def test_missing_fields(self):
        assert is_adhoc_column({"sqlExpression": "x"}) is False

    def test_string(self):
        assert is_adhoc_column("col") is False


class TestGetXAxisColumnWithTimeGrain:
    def test_with_x_axis_and_time_grain(self):
        form_data = {"x_axis": "date", "time_grain_sqla": "P1D"}
        result = get_x_axis_column_with_time_grain(form_data)
        assert isinstance(result, dict)
        assert result["sqlExpression"] == "date"
        assert result["timeGrain"] == "P1D"

    def test_fallback_to_granularity(self):
        form_data = {"granularity_sqla": "ts", "time_grain_sqla": "P1D"}
        result = get_x_axis_column_with_time_grain(form_data)
        assert result is not None

    def test_no_axis(self):
        result = get_x_axis_column_with_time_grain({})
        assert result is None

    def test_dict_x_axis(self):
        form_data = {
            "x_axis": {"sqlExpression": "date_col", "label": "date_col", "expressionType": "SQL"},
            "time_grain_sqla": "P1D",
        }
        result = get_x_axis_column_with_time_grain(form_data)
        assert isinstance(result, dict)
        assert result.get("columnType") == "BASE_AXIS"


# ---------------------------------------------------------------------------
# normalize_time_column
# ---------------------------------------------------------------------------


class TestNormalizeTimeColumn:
    def test_physical_column(self):
        qo = QueryObject(columns=["date", "metric1"], extras={"time_grain_sqla": "P1D"})
        form_data = {"x_axis": "date", "time_grain_sqla": "P1D"}
        result = normalize_time_column(form_data, qo)
        assert isinstance(result.columns[0], dict)
        assert result.columns[0]["columnType"] == "BASE_AXIS"

    def test_no_x_axis(self):
        qo = QueryObject(columns=["col1"])
        result = normalize_time_column({}, qo)
        assert result.columns == ["col1"]

    def test_dict_column(self):
        col = {"sqlExpression": "date_col", "label": "date_col", "expressionType": "SQL"}
        qo = QueryObject(columns=[col, "metric1"])
        form_data = {"x_axis": col}
        result = normalize_time_column(form_data, qo)
        assert isinstance(result.columns[0], dict)
        assert result.columns[0].get("columnType") == "BASE_AXIS"


# ---------------------------------------------------------------------------
# is_time_comparison
# ---------------------------------------------------------------------------


class TestIsTimeComparison:
    def test_with_time_compare(self):
        assert is_time_comparison({"time_compare": ["1 week"]}, QueryObject()) is True

    def test_empty(self):
        assert is_time_comparison({}, QueryObject()) is False

    def test_none(self):
        assert is_time_comparison({"time_compare": None}, QueryObject()) is False


# ---------------------------------------------------------------------------
# normalize_orderby
# ---------------------------------------------------------------------------


class TestNormalizeOrderby:
    def test_existing_valid_orderby(self):
        qo = QueryObject(orderby=[("count", True)])
        result = normalize_orderby(qo)
        assert result == [("count", True)]

    def test_valid_orderby_with_json_string_mixed(self):
        """Valid orderby with first element as tuple, second as JSON string."""
        qo = QueryObject(orderby=[("count", True), json.dumps(["revenue", False])])
        result = normalize_orderby(qo)
        assert ("count", True) in result
        assert ("revenue", False) in result

    def test_from_series_limit_metric(self):
        qo = QueryObject(series_limit_metric="revenue", order_desc=True)
        result = normalize_orderby(qo)
        assert result == [("revenue", False)]

    def test_from_first_metric(self):
        qo = QueryObject(metrics=["count"], order_desc=False)
        result = normalize_orderby(qo)
        assert result == [("count", True)]

    def test_empty(self):
        qo = QueryObject()
        result = normalize_orderby(qo)
        assert result == []


# ---------------------------------------------------------------------------
# Post-Processing Operators
# ---------------------------------------------------------------------------


class TestPivotOperator:
    def test_basic(self):
        qo = QueryObject(metrics=["count"], columns=["category"])
        form_data = {"x_axis": "date"}
        result = pivot_operator(form_data, qo)
        assert result is not None
        assert result["operation"] == "pivot"
        assert "date" in result["options"]["index"]

    def test_no_axis(self):
        qo = QueryObject(metrics=["count"])
        result = pivot_operator({}, qo)
        assert result is None


class TestTimeComparePivotOperator:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "date", "time_compare": ["1 week"]}
        result = time_compare_pivot_operator(form_data, qo)
        assert result is not None
        assert "count__1 week" in result["options"]["aggregates"]

    def test_no_axis(self):
        result = time_compare_pivot_operator({}, QueryObject())
        assert result is None


class TestFlattenOperator:
    def test_basic(self):
        result = flatten_operator({}, QueryObject())
        assert result["operation"] == "flatten"


class TestRollingWindowOperator:
    def test_cumsum(self):
        form_data = {"rolling_type": "cumsum"}
        qo = QueryObject(metrics=["count"])
        result = rolling_window_operator(form_data, qo)
        assert result["operation"] == "cum"

    def test_mean(self):
        form_data = {"rolling_type": "mean", "rolling_periods": 7, "min_periods": 1}
        qo = QueryObject(metrics=["count"])
        result = rolling_window_operator(form_data, qo)
        assert result["operation"] == "rolling"
        assert result["options"]["window"] == 7

    def test_none(self):
        result = rolling_window_operator({}, QueryObject())
        assert result is None

    def test_none_type(self):
        result = rolling_window_operator({"rolling_type": "none"}, QueryObject())
        assert result is None


class TestResampleOperator:
    def test_basic(self):
        form_data = {"resample_rule": "1D", "resample_method": "mean", "resample_fill_value": 0}
        result = resample_operator(form_data, QueryObject())
        assert result["operation"] == "resample"
        assert result["options"]["rule"] == "1D"

    def test_none(self):
        result = resample_operator({}, QueryObject())
        assert result is None


# ---------------------------------------------------------------------------
# sort_operator, contribution_operator, etc.
# ---------------------------------------------------------------------------


from datus.tools.bi_tools.superset.superset_util import (
    boxplot_operator,
    build_default_query,
    build_query_object,
    contribution_operator,
    extract_extras,
    extract_query_fields,
    histogram_operator,
    process_filters,
    prophet_operator,
    rank_operator,
    rename_operator,
    sort_operator,
    time_compare_operator,
)


class TestSortOperator:
    def test_none_without_params(self):
        result = sort_operator({}, QueryObject())
        assert result is None

    def test_none_with_groupby(self):
        form_data = {"x_axis_sort": "date", "x_axis_sort_asc": True, "groupby": ["cat"]}
        result = sort_operator(form_data, QueryObject())
        assert result is None

    def test_sort_by_x_axis(self):
        form_data = {"x_axis_sort": "date", "x_axis_sort_asc": True, "x_axis": "date"}
        result = sort_operator(form_data, QueryObject())
        assert result["operation"] == "sort"
        assert result["options"]["is_sort_index"] is True

    def test_sort_by_other(self):
        form_data = {"x_axis_sort": "revenue", "x_axis_sort_asc": False, "x_axis": "date"}
        result = sort_operator(form_data, QueryObject())
        assert result["operation"] == "sort"
        assert result["options"]["by"] == "revenue"


class TestContributionOperator:
    def test_none(self):
        result = contribution_operator({}, QueryObject())
        assert result is None

    def test_with_mode(self):
        result = contribution_operator({"contributionMode": "row"}, QueryObject())
        assert result["operation"] == "contribution"
        assert result["options"]["orientation"] == "row"


class TestTimeCompareOperator:
    def test_none(self):
        result = time_compare_operator({}, QueryObject())
        assert result is None

    def test_with_compare(self):
        form_data = {"time_compare": ["1 week"], "comparison_type": "values"}
        result = time_compare_operator(form_data, QueryObject())
        assert result["operation"] == "compare"


class TestBoxplotOperator:
    def test_none(self):
        result = boxplot_operator({}, QueryObject())
        assert result is None

    def test_tukey(self):
        form_data = {"whiskerOptions": "Tukey", "groupby": ["cat"]}
        qo = QueryObject(metrics=["count"])
        result = boxplot_operator(form_data, qo)
        assert result["operation"] == "boxplot"
        assert result["options"]["whisker_type"] == "tukey"

    def test_min_max(self):
        form_data = {"whiskerOptions": "Min/max (no outliers)"}
        qo = QueryObject(metrics=["count"])
        result = boxplot_operator(form_data, qo)
        assert result["options"]["whisker_type"] == "min/max"

    def test_percentile(self):
        form_data = {"whiskerOptions": "10/90 percentiles"}
        qo = QueryObject(metrics=["count"])
        result = boxplot_operator(form_data, qo)
        assert result["options"]["whisker_type"] == "percentile"
        assert result["options"]["percentiles"] == [10, 90]


class TestRankOperator:
    def test_none(self):
        result = rank_operator({}, QueryObject())
        assert result is None

    def test_normalize_x(self):
        form_data = {"normalize_across": "x", "x_axis": "date"}
        qo = QueryObject(metrics=["count"])
        result = rank_operator(form_data, qo)
        assert result["operation"] == "rank"
        assert result["options"]["group_by"] == "date"

    def test_normalize_y(self):
        form_data = {"normalize_across": "y", "groupby": ["cat"]}
        qo = QueryObject(metrics=["count"])
        result = rank_operator(form_data, qo)
        assert result["options"]["group_by"] == "cat"


class TestHistogramOperator:
    def test_none(self):
        result = histogram_operator({}, QueryObject())
        assert result is None

    def test_basic(self):
        form_data = {"column": "age", "bins": 20, "cumulative": True}
        result = histogram_operator(form_data, QueryObject())
        assert result["operation"] == "histogram"
        assert result["options"]["column"] == "age"
        assert result["options"]["bins"] == 20
        assert result["options"]["cumulative"] is True

    def test_from_all_columns(self):
        form_data = {"all_columns": ["amount"], "bins": 10}
        result = histogram_operator(form_data, QueryObject())
        assert result["options"]["column"] == "amount"


class TestProphetOperator:
    def test_none_no_enable(self):
        result = prophet_operator({}, QueryObject())
        assert result is None

    def test_none_only_periods(self):
        form_data = {"forecastPeriods": 10, "x_axis": "date"}
        result = prophet_operator(form_data, QueryObject())
        assert result is None

    def test_enabled(self):
        form_data = {
            "forecastEnabled": True,
            "x_axis": "date",
            "forecastPeriods": 30,
            "forecastInterval": 0.95,
        }
        result = prophet_operator(form_data, QueryObject())
        assert result["operation"] == "prophet"
        assert result["options"]["periods"] == 30
        assert result["options"]["confidence_interval"] == 0.95


class TestRenameOperator:
    def test_none_without_params(self):
        result = rename_operator({}, QueryObject())
        assert result is None

    def test_rename_single_metric(self):
        form_data = {"truncate_metric": True, "metrics": ["count"], "x_axis": "date"}
        qo = QueryObject(metrics=["count"])
        result = rename_operator(form_data, qo)
        assert result["operation"] == "rename"

    def test_none_multi_metrics(self):
        form_data = {"truncate_metric": True, "metrics": ["count", "sum"], "x_axis": "date"}
        qo = QueryObject(metrics=["count", "sum"])
        result = rename_operator(form_data, qo)
        assert result is None


# ---------------------------------------------------------------------------
# Core query building
# ---------------------------------------------------------------------------


class TestExtractQueryFields:
    def test_aggregate_mode(self):
        form_data = {"groupby": ["cat"], "metrics": ["count"]}
        result = extract_query_fields(form_data)
        assert result["columns"] == ["cat"]
        assert result["metrics"] == ["count"]

    def test_raw_mode(self):
        form_data = {"query_mode": "raw", "groupby": ["cat"], "columns": ["col1"], "metrics": ["count"]}
        result = extract_query_fields(form_data)
        assert "cat" not in [str(c) for c in result["columns"]]

    def test_dedup_columns(self):
        form_data = {"groupby": ["cat", "cat"]}
        result = extract_query_fields(form_data)
        assert len(result["columns"]) == 1


class TestExtractExtras:
    def test_basic(self):
        result = extract_extras({"time_range": "last week"})
        assert result["time_range"] == "last week"

    def test_reserved_filters(self):
        form_data = {"extra_filters": [{"col": "__time_range", "val": "last month"}]}
        result = extract_extras(form_data)
        assert result["time_range"] == "last month"
        assert "__time_range" in result["applied_time_extras"]

    def test_non_reserved_filters(self):
        form_data = {"extra_filters": [{"col": "status", "val": 1, "op": "=="}]}
        result = extract_extras(form_data)
        assert len(result["filters"]) == 1

    def test_time_grain(self):
        form_data = {"extra_filters": [{"col": "__time_grain", "val": "P1D"}]}
        result = extract_extras(form_data)
        assert result["extras"]["time_grain_sqla"] == "P1D"


class TestProcessFilters:
    def test_empty(self):
        result = process_filters({})
        assert result["filters"] == []

    def test_simple_filter(self):
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "clause": "WHERE",
                    "subject": "status",
                    "operator": "==",
                    "comparator": 1,
                }
            ]
        }
        result = process_filters(form_data)
        assert len(result["filters"]) == 1
        assert result["filters"][0]["col"] == "status"

    def test_sql_filter(self):
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SQL",
                    "clause": "WHERE",
                    "sqlExpression": "status > 0",
                }
            ]
        }
        result = process_filters(form_data)
        assert "where" in result["extras"]

    def test_having_filter(self):
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SQL",
                    "clause": "HAVING",
                    "sqlExpression": "COUNT(*) > 10",
                }
            ]
        }
        result = process_filters(form_data)
        assert "having" in result["extras"]


class TestBuildQueryObject:
    def test_basic(self):
        form_data = {"groupby": ["cat"], "metrics": ["count"], "row_limit": 100}
        qo = build_query_object(form_data)
        assert isinstance(qo, QueryObject)
        assert qo.row_limit == 100

    def test_invalid_row_limit(self):
        form_data = {"row_limit": "invalid"}
        qo = build_query_object(form_data)
        assert qo.row_limit is None

    def test_with_time_range(self):
        form_data = {"time_range": "last week", "granularity_sqla": "ts"}
        qo = build_query_object(form_data)
        assert qo.time_range == "last week"

    def test_extra_form_data_override(self):
        form_data = {"groupby": ["cat"], "extra_form_data": {"row_limit": 50}}
        qo = build_query_object(form_data)
        assert qo.row_limit == 50


class TestBuildDefaultQuery:
    def test_basic(self):
        qo = QueryObject()
        result = build_default_query(qo, {})
        assert len(result) == 1
        assert result[0] is qo
