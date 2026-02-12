# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for superset_util.py – chart build query functions."""

import pytest

from datus.tools.bi_tools.superset.superset_util import (
    QueryObject,
    _add_spatial_null_filters,
    _get_spatial_columns,
    build_big_number_query,
    build_big_number_trendline_query,
    build_boxplot_query,
    build_bubble_query,
    build_deck_arc_query,
    build_funnel_query,
    build_gantt_query,
    build_gauge_query,
    build_graph_query,
    build_heatmap_query,
    build_histogram_query,
    build_mixed_timeseries_query,
    build_pie_query,
    build_pivot_table_query,
    build_pop_kpi_query,
    build_radar_query,
    build_sankey_query,
    build_sunburst_query,
    build_table_query,
    build_timeseries_query,
    build_tree_query,
    build_treemap_query,
    build_waterfall_query,
    build_word_cloud_query,
)


# ---------------------------------------------------------------------------
# build_timeseries_query
# ---------------------------------------------------------------------------


class TestBuildTimeseriesQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "date", "groupby": ["category"]}
        result = build_timeseries_query(qo, form_data)
        assert len(result) == 1
        q = result[0]
        assert len(q.columns) >= 1
        assert q.series_columns == ["category"]

    def test_no_x_axis_sets_timeseries(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"groupby": ["cat"]}
        result = build_timeseries_query(qo, form_data)
        assert result[0].is_timeseries is True

    def test_with_time_compare(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "x_axis": "date",
            "time_compare": ["1 week"],
            "comparison_type": "values",
        }
        result = build_timeseries_query(qo, form_data)
        assert len(result[0].time_offsets) == 1

    def test_with_rolling_window(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "date", "rolling_type": "mean", "rolling_periods": 7}
        result = build_timeseries_query(qo, form_data)
        pp = result[0].post_processing
        ops = [p["operation"] for p in pp]
        assert "rolling" in ops

    def test_with_contribution(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "date", "contributionMode": "row"}
        result = build_timeseries_query(qo, form_data)
        pp = result[0].post_processing
        ops = [p["operation"] for p in pp]
        assert "contribution" in ops

    def test_no_groupby(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "date"}
        result = build_timeseries_query(qo, form_data)
        assert result[0].series_columns is None


# ---------------------------------------------------------------------------
# build_boxplot_query
# ---------------------------------------------------------------------------


class TestBuildBoxplotQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "groupby": ["cat"],
            "columns": ["date"],
            "whiskerOptions": "Tukey",
        }
        result = build_boxplot_query(qo, form_data)
        assert len(result) == 1
        assert result[0].series_columns == ["cat"]

    def test_with_temporal_column(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "columns": ["ts"],
            "time_grain_sqla": "P1D",
            "temporal_columns_lookup": {"ts": True},
        }
        result = build_boxplot_query(qo, form_data)
        assert isinstance(result[0].columns[0], dict)
        assert result[0].columns[0]["columnType"] == "BASE_AXIS"

    def test_fallback_to_granularity(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "granularity_sqla": "ts",
        }
        result = build_boxplot_query(qo, form_data)
        assert "ts" in result[0].columns


# ---------------------------------------------------------------------------
# build_table_query
# ---------------------------------------------------------------------------


class TestBuildTableQuery:
    def test_aggregate_mode(self):
        qo = QueryObject(
            columns=["cat"],
            metrics=["count"],
        )
        form_data = {"query_mode": "aggregate"}
        result = build_table_query(qo, form_data)
        assert len(result) == 1

    def test_raw_mode_with_all_columns(self):
        qo = QueryObject(columns=["col1", "col2"])
        form_data = {"all_columns": ["col1", "col2"]}
        result = build_table_query(qo, form_data)
        assert len(result) == 1

    def test_with_percent_metrics(self):
        qo = QueryObject(
            columns=["cat"],
            metrics=["count"],
        )
        form_data = {
            "query_mode": "aggregate",
            "percent_metrics": ["count"],
        }
        result = build_table_query(qo, form_data)
        pp = result[0].post_processing
        ops = [p.get("operation") for p in pp]
        assert "contribution" in ops

    def test_with_time_comparison(self):
        qo = QueryObject(columns=["cat"], metrics=["count"])
        form_data = {
            "query_mode": "aggregate",
            "time_compare": ["1 week"],
            "comparison_type": "values",
        }
        result = build_table_query(qo, form_data)
        assert len(result[0].time_offsets) == 1

    def test_with_custom_time_compare(self):
        qo = QueryObject(columns=["cat"], metrics=["count"])
        form_data = {
            "query_mode": "aggregate",
            "time_compare": ["custom"],
            "start_date_offset": "2 weeks ago",
        }
        result = build_table_query(qo, form_data)
        assert "2 weeks ago" in result[0].time_offsets

    def test_with_inherit_time_compare(self):
        qo = QueryObject(columns=["cat"], metrics=["count"])
        form_data = {
            "query_mode": "aggregate",
            "time_compare": ["inherit"],
        }
        result = build_table_query(qo, form_data)
        assert "inherit" in result[0].time_offsets

    def test_with_temporal_columns(self):
        qo = QueryObject(columns=["ts", "cat"], metrics=["count"])
        form_data = {
            "query_mode": "aggregate",
            "time_grain_sqla": "P1D",
            "temporal_columns_lookup": {"ts": True},
        }
        result = build_table_query(qo, form_data)
        assert isinstance(result[0].columns[0], dict)


# ---------------------------------------------------------------------------
# build_big_number_query / build_big_number_trendline_query
# ---------------------------------------------------------------------------


class TestBuildBigNumberQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        result = build_big_number_query(qo, {})
        assert len(result) == 1
        assert result[0] is qo

    def test_trendline(self):
        qo = QueryObject(metrics=["count"])
        result = build_big_number_trendline_query(qo, {})
        assert result[0].is_timeseries is True
        pp = result[0].post_processing
        ops = [p["operation"] for p in pp]
        assert "pivot" in ops
        assert "flatten" in ops


# ---------------------------------------------------------------------------
# build_pie_query
# ---------------------------------------------------------------------------


class TestBuildPieQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"], columns=["cat"])
        result = build_pie_query(qo, {})
        pp = result[0].post_processing
        assert len(pp) == 1
        assert pp[0]["operation"] == "contribution"

    def test_with_sort_by_metric(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"sort_by_metric": True, "metric": "count"}
        result = build_pie_query(qo, form_data)
        assert result[0].orderby == [("count", False)]


# ---------------------------------------------------------------------------
# build_funnel_query
# ---------------------------------------------------------------------------


class TestBuildFunnelQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        result = build_funnel_query(qo, {})
        assert len(result) == 1

    def test_with_sort(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"sort_by_metric": True, "metric": "count"}
        result = build_funnel_query(qo, form_data)
        assert result[0].orderby == [("count", False)]


# ---------------------------------------------------------------------------
# build_gauge_query
# ---------------------------------------------------------------------------


class TestBuildGaugeQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        result = build_gauge_query(qo, {})
        assert len(result) == 1

    def test_with_sort(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"sort_by_metric": True}
        result = build_gauge_query(qo, form_data)
        assert result[0].orderby == [("count", False)]


# ---------------------------------------------------------------------------
# build_heatmap_query
# ---------------------------------------------------------------------------


class TestBuildHeatmapQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "date", "groupby": ["cat"]}
        result = build_heatmap_query(qo, form_data)
        assert len(result) == 1

    def test_with_sort_x(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "x_axis": "date",
            "groupby": ["cat"],
            "sort_x_axis": "value_asc",
        }
        result = build_heatmap_query(qo, form_data)
        assert len(result[0].orderby) >= 1

    def test_with_sort_y(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "x_axis": "date",
            "groupby": ["cat"],
            "sort_y_axis": "alpha_desc",
        }
        result = build_heatmap_query(qo, form_data)
        assert len(result[0].orderby) >= 1

    def test_with_normalize(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "x_axis": "date",
            "groupby": ["cat"],
            "normalize_across": "x",
        }
        result = build_heatmap_query(qo, form_data)
        pp = result[0].post_processing
        assert pp[0]["operation"] == "rank"


# ---------------------------------------------------------------------------
# build_histogram_query
# ---------------------------------------------------------------------------


class TestBuildHistogramQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"column": "age", "groupby": ["cat"]}
        result = build_histogram_query(qo, form_data)
        assert result[0].metrics == []
        assert "age" in result[0].columns

    def test_no_column(self):
        qo = QueryObject()
        result = build_histogram_query(qo, {})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# build_bubble_query
# ---------------------------------------------------------------------------


class TestBuildBubbleQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"entity": "company", "series": "region"}
        result = build_bubble_query(qo, form_data)
        assert "company" in result[0].columns
        assert "region" in result[0].columns

    def test_invert_orderby(self):
        qo = QueryObject(metrics=["count"], orderby=[("count", True)])
        result = build_bubble_query(qo, {})
        assert result[0].orderby == [("count", False)]


# ---------------------------------------------------------------------------
# build_waterfall_query
# ---------------------------------------------------------------------------


class TestBuildWaterfallQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "date", "groupby": ["cat"]}
        result = build_waterfall_query(qo, form_data)
        assert len(result) == 1
        assert len(result[0].orderby) >= 1

    def test_no_x_axis(self):
        qo = QueryObject(metrics=["count"])
        result = build_waterfall_query(qo, {})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# build_sankey_query
# ---------------------------------------------------------------------------


class TestBuildSankeyQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"source": "from_col", "target": "to_col"}
        result = build_sankey_query(qo, form_data)
        assert "from_col" in result[0].columns
        assert "to_col" in result[0].columns

    def test_with_sort(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "source": "from",
            "target": "to",
            "sort_by_metric": True,
        }
        result = build_sankey_query(qo, form_data)
        assert result[0].orderby == [("count", False)]


# ---------------------------------------------------------------------------
# build_sunburst_query / build_treemap_query / build_word_cloud_query
# ---------------------------------------------------------------------------


class TestBuildSunburstTreemapWordCloud:
    def test_sunburst_with_sort(self):
        qo = QueryObject(metrics=["count"])
        result = build_sunburst_query(qo, {"sort_by_metric": True})
        assert result[0].orderby == [("count", False)]

    def test_sunburst_no_sort(self):
        qo = QueryObject(metrics=["count"])
        result = build_sunburst_query(qo, {})
        assert len(result) == 1

    def test_treemap_with_sort(self):
        qo = QueryObject(metrics=["count"])
        result = build_treemap_query(qo, {"sort_by_metric": True})
        assert result[0].orderby == [("count", False)]

    def test_word_cloud_with_sort(self):
        qo = QueryObject(metrics=["count"])
        result = build_word_cloud_query(qo, {"sort_by_metric": True})
        assert result[0].orderby == [("count", False)]


# ---------------------------------------------------------------------------
# build_graph_query
# ---------------------------------------------------------------------------


class TestBuildGraphQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "source": "from",
            "target": "to",
            "source_category": "cat1",
            "target_category": "cat2",
        }
        result = build_graph_query(qo, form_data)
        assert len(result[0].columns) == 4

    def test_partial_columns(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"source": "from"}
        result = build_graph_query(qo, form_data)
        assert result[0].columns == ["from"]


# ---------------------------------------------------------------------------
# build_tree_query
# ---------------------------------------------------------------------------


class TestBuildTreeQuery:
    def test_basic(self):
        qo = QueryObject()
        form_data = {"id": "node_id", "parent": "parent_id", "name": "node_name"}
        result = build_tree_query(qo, form_data)
        assert len(result[0].columns) == 3

    def test_partial(self):
        qo = QueryObject()
        form_data = {"id": "node_id"}
        result = build_tree_query(qo, form_data)
        assert result[0].columns == ["node_id"]


# ---------------------------------------------------------------------------
# build_radar_query
# ---------------------------------------------------------------------------


class TestBuildRadarQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "category", "groupby": ["region"]}
        result = build_radar_query(qo, form_data)
        assert "category" in result[0].columns
        assert "region" in result[0].columns

    def test_no_dedup(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"groupby": ["cat"]}
        result = build_radar_query(qo, form_data)
        assert result[0].columns.count("cat") == 1


# ---------------------------------------------------------------------------
# build_pivot_table_query
# ---------------------------------------------------------------------------


class TestBuildPivotTableQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "groupbyColumns": ["col1"],
            "groupbyRows": ["row1"],
        }
        result = build_pivot_table_query(qo, form_data)
        assert "col1" in result[0].columns
        assert "row1" in result[0].columns

    def test_with_temporal(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "groupbyColumns": ["ts"],
            "groupbyRows": [],
            "time_grain_sqla": "P1D",
            "temporal_columns_lookup": {"ts": True},
        }
        result = build_pivot_table_query(qo, form_data)
        assert isinstance(result[0].columns[0], dict)

    def test_with_series_limit_metric(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "groupbyColumns": ["col1"],
            "series_limit_metric": "count",
            "order_desc": True,
        }
        result = build_pivot_table_query(qo, form_data)
        assert result[0].orderby == [("count", False)]


# ---------------------------------------------------------------------------
# build_mixed_timeseries_query
# ---------------------------------------------------------------------------


class TestBuildMixedTimeseriesQuery:
    def test_single_query(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"x_axis": "date"}
        result = build_mixed_timeseries_query(qo, form_data)
        assert len(result) >= 1

    def test_with_query_b(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "x_axis": "date",
            "metrics_b": ["revenue"],
            "groupby_b": ["region"],
        }
        result = build_mixed_timeseries_query(qo, form_data)
        assert len(result) == 2
        assert result[1].metrics == ["revenue"]


# ---------------------------------------------------------------------------
# build_gantt_query
# ---------------------------------------------------------------------------


class TestBuildGanttQuery:
    def test_basic(self):
        qo = QueryObject()
        form_data = {
            "start_time": "start_date",
            "end_time": "end_date",
            "y_axis": "task_name",
            "series": ["project"],
        }
        result = build_gantt_query(qo, form_data)
        assert "start_date" in result[0].columns
        assert "end_date" in result[0].columns

    def test_with_orderby(self):
        import json
        qo = QueryObject()
        form_data = {
            "start_time": "start_date",
            "order_by_cols": [json.dumps(["start_date", True])],
        }
        result = build_gantt_query(qo, form_data)
        assert len(result[0].orderby) == 1


# ---------------------------------------------------------------------------
# build_pop_kpi_query
# ---------------------------------------------------------------------------


class TestBuildPopKpiQuery:
    def test_basic(self):
        qo = QueryObject(metrics=["count"])
        form_data = {"cols": ["date"]}
        result = build_pop_kpi_query(qo, form_data)
        assert result[0].columns == ["date"]

    def test_with_time_compare(self):
        qo = QueryObject(metrics=["count"])
        form_data = {
            "cols": ["date"],
            "time_compare": ["1 week"],
            "comparison_type": "values",
        }
        result = build_pop_kpi_query(qo, form_data)
        assert len(result[0].time_offsets) >= 1


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------


class TestSpatialHelpers:
    def test_get_spatial_columns_latlong(self):
        spatial = {"type": "latlong", "latCol": "lat", "lonCol": "lon"}
        result = _get_spatial_columns(spatial)
        assert result == ["lat", "lon"]

    def test_get_spatial_columns_delimited(self):
        spatial = {"type": "delimited", "lonlatCol": "geom"}
        result = _get_spatial_columns(spatial)
        assert result == ["geom"]

    def test_get_spatial_columns_geohash(self):
        spatial = {"type": "geohash", "geohashCol": "hash"}
        result = _get_spatial_columns(spatial)
        assert result == ["hash"]

    def test_get_spatial_columns_none(self):
        result = _get_spatial_columns(None)
        assert result == []

    def test_add_spatial_null_filters(self):
        spatial = {"type": "latlong", "latCol": "lat", "lonCol": "lon"}
        result = _add_spatial_null_filters(spatial, [])
        assert len(result) == 2
        assert result[0]["op"] == "IS NOT NULL"

    def test_add_spatial_null_filters_none(self):
        result = _add_spatial_null_filters(None, [{"col": "x", "op": "==", "val": 1}])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# build_deck_arc_query
# ---------------------------------------------------------------------------


class TestBuildDeckArcQuery:
    def test_basic(self):
        qo = QueryObject()
        form_data = {
            "start_spatial": {"type": "latlong", "latCol": "start_lat", "lonCol": "start_lon"},
            "end_spatial": {"type": "latlong", "latCol": "end_lat", "lonCol": "end_lon"},
            "dimension": "route",
        }
        result = build_deck_arc_query(qo, form_data)
        assert "start_lat" in result[0].columns
        assert "route" in result[0].columns
        # IS NOT NULL filters for spatial columns
        assert len(result[0].filters) >= 4
