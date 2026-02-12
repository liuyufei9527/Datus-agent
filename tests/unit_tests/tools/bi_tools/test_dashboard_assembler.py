# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for dashboard_assembler.py – 0% coverage target."""

from unittest.mock import MagicMock, patch

import pytest

from datus.tools.bi_tools.base_adaptor import ChartInfo, DashboardInfo, DatasetInfo, QuerySpec
from datus.tools.bi_tools.dashboard_assembler import (
    ChartSelection,
    DashboardAssembler,
    DashboardExtraction,
    SelectedSqlCandidate,
    parts_match,
    split_table_parts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chart(
    chart_id=1,
    name="chart1",
    description="desc",
    chart_type="bar",
    sql=None,
    tables=None,
    payload=None,
    extra=None,
):
    query = None
    if sql is not None:
        query = QuerySpec(kind="sql", sql=sql, tables=tables or [], payload=payload or {})
    kwargs = {"id": chart_id, "name": name, "description": description, "chart_type": chart_type, "query": query}
    if extra is not None:
        kwargs["extra"] = extra
    return ChartInfo(**kwargs)


def _make_dataset(ds_id=1, name="ds1", dialect="mysql", tables=None, extra=None):
    kwargs = {"id": ds_id, "name": name, "dialect": dialect, "tables": tables or []}
    if extra is not None:
        kwargs["extra"] = extra
    return DatasetInfo(**kwargs)


def _make_dashboard(dash_id=1, name="Dashboard", chart_ids=None, description=""):
    return DashboardInfo(id=dash_id, name=name, description=description, chart_ids=chart_ids or [])


# ---------------------------------------------------------------------------
# SelectedSqlCandidate
# ---------------------------------------------------------------------------


class TestSelectedSqlCandidate:
    def test_question_with_both(self):
        c = SelectedSqlCandidate(chart_id=1, chart_name="Sales", description="Daily", sql="SELECT 1")
        assert c.question() == "Sales - Daily"

    def test_question_name_only(self):
        c = SelectedSqlCandidate(chart_id=1, chart_name="Sales", description=None, sql="SELECT 1")
        assert c.question() == "Sales"

    def test_question_desc_only(self):
        c = SelectedSqlCandidate(chart_id=1, chart_name="", description="Daily", sql="SELECT 1")
        assert c.question() == "Daily"

    def test_question_empty(self):
        c = SelectedSqlCandidate(chart_id=1, chart_name="", description=None, sql="SELECT 1")
        assert c.question() == ""

    def test_to_payload(self):
        c = SelectedSqlCandidate(chart_id=1, chart_name="Sales", description="Desc", sql="SELECT 1", index=2)
        payload = c.to_payload()
        assert payload["chart_id"] == 1
        assert payload["chart_name"] == "Sales"
        assert payload["index"] == 2
        assert payload["question"] == "Sales - Desc"


# ---------------------------------------------------------------------------
# DashboardExtraction
# ---------------------------------------------------------------------------


class TestDashboardExtraction:
    def test_chart_map(self):
        charts = [_make_chart(chart_id=1, name="c1"), _make_chart(chart_id=2, name="c2")]
        ext = DashboardExtraction(
            dashboard_id=1, dashboard=_make_dashboard(), charts=charts, datasets=[]
        )
        cmap = ext.chart_map()
        assert "1" in cmap
        assert "2" in cmap

    def test_dataset_map(self):
        datasets = [_make_dataset(ds_id=10, name="d1"), _make_dataset(ds_id=20, name="d2")]
        ext = DashboardExtraction(
            dashboard_id=1, dashboard=_make_dashboard(), charts=[], datasets=datasets
        )
        dmap = ext.dataset_map()
        assert "10" in dmap
        assert "20" in dmap


# ---------------------------------------------------------------------------
# split_table_parts / parts_match
# ---------------------------------------------------------------------------


class TestSplitTableParts:
    def test_simple(self):
        assert split_table_parts("db.schema.table") == ["db", "schema", "table"]

    def test_quoted(self):
        assert split_table_parts('`db`.`schema`.`table`') == ["db", "schema", "table"]

    def test_single_part(self):
        assert split_table_parts("orders") == ["orders"]

    def test_case_insensitive(self):
        parts = split_table_parts("DB.Schema.TABLE")
        assert parts == ["db", "schema", "table"]


class TestPartsMatch:
    def test_exact_match(self):
        assert parts_match(["db", "schema", "t"], ["db", "schema", "t"]) is True

    def test_suffix_match_left_shorter(self):
        assert parts_match(["schema", "t"], ["db", "schema", "t"]) is True

    def test_suffix_match_right_shorter(self):
        assert parts_match(["db", "schema", "t"], ["schema", "t"]) is True

    def test_no_match(self):
        assert parts_match(["db", "schema", "t1"], ["db", "schema", "t2"]) is False

    def test_empty_left(self):
        assert parts_match([], ["a"]) is False

    def test_empty_right(self):
        assert parts_match(["a"], []) is False

    def test_both_empty(self):
        assert parts_match([], []) is False


# ---------------------------------------------------------------------------
# DashboardAssembler
# ---------------------------------------------------------------------------


class TestDashboardAssembler:
    @pytest.fixture
    def mock_adaptor(self):
        adaptor = MagicMock()
        adaptor.parse_dashboard_id.return_value = 1
        adaptor.get_dashboard_info.return_value = _make_dashboard(chart_ids=[10, 20])
        adaptor.list_charts.return_value = [
            _make_chart(chart_id=10, name="Chart10"),
            _make_chart(chart_id=20, name="Chart20"),
        ]
        adaptor.get_chart.side_effect = lambda cid, did=None: _make_chart(
            chart_id=cid, name=f"Chart{cid}", sql=[f"SELECT * FROM table_{cid}"], tables=[f"table_{cid}"]
        )
        adaptor.list_datasets.return_value = [_make_dataset(ds_id=100, name="ds100")]
        adaptor.get_dataset.return_value = _make_dataset(ds_id=100, name="ds100")
        return adaptor

    @pytest.fixture
    def assembler(self, mock_adaptor):
        return DashboardAssembler(
            adaptor=mock_adaptor,
            default_dialect="mysql",
            default_database="testdb",
        )

    def test_extract_dashboard(self, assembler):
        result = assembler.extract_dashboard("http://example.com/dashboard/1/")
        assert isinstance(result, DashboardExtraction)
        assert result.dashboard_id == 1
        assert len(result.charts) == 2

    def test_extract_dashboard_not_found(self, assembler, mock_adaptor):
        mock_adaptor.get_dashboard_info.return_value = None
        with pytest.raises(ValueError, match="not found"):
            assembler.extract_dashboard("http://example.com/dashboard/999/")

    def test_hydrate_datasets(self, assembler, mock_adaptor):
        datasets = [_make_dataset(ds_id=100)]
        hydrated = assembler.hydrate_datasets(datasets, dashboard_id=1)
        assert len(hydrated) == 1
        mock_adaptor.get_dataset.assert_called_once()

    def test_hydrate_datasets_fallback(self, assembler, mock_adaptor):
        mock_adaptor.get_dataset.side_effect = Exception("fail")
        ds = _make_dataset(ds_id=100, name="orig")
        hydrated = assembler.hydrate_datasets([ds], dashboard_id=1)
        # Falls back to original
        assert hydrated[0].name == "orig"

    def test_assemble_basic(self, assembler):
        dashboard = _make_dashboard()
        chart = _make_chart(
            chart_id=1,
            name="c1",
            sql=["SELECT * FROM orders"],
            tables=["orders"],
        )
        selection = ChartSelection(chart=chart, sql_indices=[0])
        datasets = [_make_dataset(ds_id=1)]

        result = assembler.assemble(
            dashboard=dashboard,
            chart_selections_ref=[selection],
            chart_selections_metrics=[],
            datasets=datasets,
        )

        assert len(result.reference_sqls) == 1
        assert result.reference_sqls[0].sql == "SELECT * FROM orders"

    def test_assemble_deduplicates_tables(self, assembler):
        dashboard = _make_dashboard()
        chart1 = _make_chart(chart_id=1, sql=["SELECT * FROM t1"], tables=["t1"])
        chart2 = _make_chart(chart_id=2, sql=["SELECT * FROM t1"], tables=["t1"])
        sel1 = ChartSelection(chart=chart1, sql_indices=[0])
        sel2 = ChartSelection(chart=chart2, sql_indices=[0])

        result = assembler.assemble(
            dashboard=dashboard,
            chart_selections_ref=[sel1, sel2],
            chart_selections_metrics=[],
            datasets=[],
        )

        # Tables should be deduplicated
        table_names_lower = [t.lower() for t in result.tables]
        # There should be no exact duplicates after normalization
        assert len(table_names_lower) == len(set(table_names_lower))

    def test_load_charts_fallback(self, assembler, mock_adaptor):
        mock_adaptor.get_chart.side_effect = Exception("fail")
        dashboard = _make_dashboard(chart_ids=[10])
        mock_adaptor.list_charts.return_value = [_make_chart(chart_id=10, name="Fallback")]
        charts = assembler._load_charts(1, dashboard)
        assert len(charts) == 1
        assert charts[0].name == "Fallback"

    def test_load_charts_from_meta_when_no_chart_ids(self, assembler, mock_adaptor):
        dashboard = _make_dashboard(chart_ids=[])
        mock_adaptor.get_chart.side_effect = lambda cid, did=None: _make_chart(chart_id=cid)
        charts = assembler._load_charts(1, dashboard)
        # Should use chart IDs from list_charts
        assert len(charts) == 2

    def test_resolve_chart_dataset_from_query_payload(self, assembler):
        ds = _make_dataset(ds_id=5)
        chart = _make_chart(
            chart_id=1,
            sql=["SELECT 1"],
            payload={"datasource": {"id": 5, "type": "table"}},
        )
        result = assembler._resolve_chart_dataset(chart, {"5": ds}, {})
        assert result is not None
        assert result.id == 5

    def test_resolve_chart_dataset_from_extra(self, assembler):
        ds = _make_dataset(ds_id=5)
        chart = _make_chart(chart_id=1, extra={"datasource": {"id": 5}})
        result = assembler._resolve_chart_dataset(chart, {"5": ds}, {})
        assert result is not None

    def test_resolve_chart_dataset_by_name(self, assembler):
        ds = _make_dataset(ds_id=5, name="myds")
        chart = _make_chart(
            chart_id=1,
            sql=["SELECT 1"],
            payload={"datasource": {"name": "myds"}},
        )
        result = assembler._resolve_chart_dataset(chart, {}, {"myds": ds})
        assert result is not None
        assert result.name == "myds"

    def test_resolve_chart_dataset_none(self, assembler):
        chart = _make_chart(chart_id=1)
        result = assembler._resolve_chart_dataset(chart, {}, {})
        assert result is None

    def test_can_qualify_table_mysql(self, assembler):
        assert assembler._can_qualify_table("mysql", "", "mydb", "") is True
        assert assembler._can_qualify_table("mysql", "", "", "") is False

    def test_can_qualify_table_postgres(self, assembler):
        assert assembler._can_qualify_table("postgresql", "", "db", "public") is True
        assert assembler._can_qualify_table("postgresql", "", "db", "") is False

    def test_can_qualify_table_snowflake(self, assembler):
        assert assembler._can_qualify_table("snowflake", "", "db", "schema") is True
        assert assembler._can_qualify_table("snowflake", "", "", "schema") is False

    def test_can_qualify_table_unknown(self, assembler):
        # Unknown dialect: returns True if any of database or schema is set
        assert assembler._can_qualify_table("unknown", "", "db", "") is True
        assert assembler._can_qualify_table("unknown", "", "", "") is False

    def test_normalize_tables_with_dot(self, assembler):
        result = assembler._normalize_tables(["db.schema.table"])
        assert result == ["db.schema.table"]

    def test_normalize_tables_empty(self, assembler):
        result = assembler._normalize_tables(["", None, "  "])
        assert result == []

    def test_extract_database_name_dict(self, assembler):
        assert assembler._extract_database_name({"database_name": "mydb"}) == "mydb"
        assert assembler._extract_database_name({"name": "mydb"}) == "mydb"

    def test_extract_database_name_string(self, assembler):
        assert assembler._extract_database_name("mydb") == "mydb"

    def test_extract_database_name_other(self, assembler):
        assert assembler._extract_database_name(123) == ""

    def test_resolve_table_context_with_dataset_extra(self, assembler):
        ds = _make_dataset(ds_id=1, extra={"schema": "public", "catalog": "cat1", "database": "db2"})
        cat, db, schema = assembler._resolve_table_context(ds)
        assert schema == "public"
        assert cat == "cat1"
        assert db == "db2"

    def test_resolve_table_context_defaults(self, assembler):
        cat, db, schema = assembler._resolve_table_context(None)
        assert db == "testdb"
        assert cat == ""
        assert schema == ""

    def test_prefer_table_name(self, assembler):
        assert assembler._prefer_table_name("t", "db.schema.t") == "db.schema.t"
        assert assembler._prefer_table_name("db.schema.t", "t") == "db.schema.t"

    def test_dedupe_tables(self, assembler):
        tables = ["t1", "db.schema.t1", "t2", "", "  "]
        result = assembler._dedupe_tables(tables)
        assert len(result) == 2
        assert "db.schema.t1" in result
        assert "t2" in result
