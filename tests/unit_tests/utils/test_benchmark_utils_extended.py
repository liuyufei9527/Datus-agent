# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for datus/utils/benchmark_utils.py to improve coverage."""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from datus.utils.benchmark_utils import (
    AgentResultSqlProvider,
    ComparisonOutcome,
    ComparisonRecord,
    CsvColumnSqlProvider,
    CsvPerTaskResultProvider,
    DirectorySqlProvider,
    EvaluationReport,
    EvaluationReportBuilder,
    GoldArtifacts,
    JsonMappingSqlProvider,
    ResultData,
    SingleFileGoldProvider,
    SqlData,
    TableComparator,
    TaskEvaluation,
    TrajectoryParser,
    WorkflowAnalysis,
    WorkflowArtifacts,
    WorkflowOutput,
    _append_file_candidate,
    _append_unique,
    _clean_table_identifier_part,
    _collect_file_artifacts,
    _collect_metric_artifacts,
    _collect_reference_sql_artifacts,
    _collect_semantic_model_artifacts,
    _extract_artifacts_from_action_history,
    _extract_result_payload,
    _find_matching_candidates,
    _is_empty_table_identifier,
    _normalize_field_name,
    _normalize_text,
    _parse_table_identifier,
    _select_from_mapping,
    _split_expected_items,
    _tables_equivalent,
    _trim_trailing_non_empty_tables,
    _unique_preserve_order,
    collect_sql_tables,
    columns_match,
    compare_pandas_tables,
    compute_table_matches,
    csv_str_to_pands,
    preview_dataframe,
)


# ===========================================================================
# Simple utility functions
# ===========================================================================


class TestCsvStrToPands:
    def test_basic(self):
        df = csv_str_to_pands("a,b\n1,2\n3,4\n")
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]


class TestNormalizeFieldName:
    def test_basic(self):
        assert _normalize_field_name("Task ID") == "task_id"
        assert _normalize_field_name("  Name  ") == "name"

    def test_non_string(self):
        assert _normalize_field_name(None) == ""
        assert _normalize_field_name(123) == ""


class TestSelectFromMapping:
    def test_found(self):
        key, val = _select_from_mapping({"Task_ID": "abc"}, ["task_id"])
        assert val == "abc"

    def test_not_found(self):
        key, val = _select_from_mapping({"foo": "bar"}, ["baz"])
        assert key is None
        assert val is None

    def test_empty_value_skipped(self):
        key, val = _select_from_mapping({"task_id": "", "name": "abc"}, ["task_id", "name"])
        assert val == "abc"

    def test_non_mapping(self):
        key, val = _select_from_mapping("not a mapping", ["key"])
        assert key is None


class TestUniquePreserveOrder:
    def test_basic(self):
        result = _unique_preserve_order(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    def test_empty_items(self):
        result = _unique_preserve_order(["a", "", None, "b"])
        assert result == ["a", "b"]

    def test_whitespace(self):
        result = _unique_preserve_order(["  a  ", "a", "b"])
        assert result == ["a", "b"]


class TestParseTableIdentifier:
    def test_simple_name(self):
        norm, base, is_simple = _parse_table_identifier("users")
        assert norm == "users"
        assert base == "users"
        assert is_simple is True

    def test_qualified_name(self):
        norm, base, is_simple = _parse_table_identifier("schema.users")
        assert base == "users"
        assert is_simple is False

    def test_none(self):
        norm, base, is_simple = _parse_table_identifier(None)
        assert norm == ""
        assert base == ""

    def test_empty_string(self):
        norm, base, is_simple = _parse_table_identifier("")
        assert norm == ""

    def test_quoted_parts(self):
        norm, base, _ = _parse_table_identifier('"schema"."table"')
        assert base == "table"

    def test_leading_dot(self):
        norm, base, _ = _parse_table_identifier(".table")
        assert base == "table"


class TestIsEmptyTableIdentifier:
    def test_none(self):
        assert _is_empty_table_identifier(None) is True

    def test_empty_string(self):
        assert _is_empty_table_identifier("") is True
        assert _is_empty_table_identifier("  ") is True

    def test_non_empty(self):
        assert _is_empty_table_identifier("table") is False


class TestTrimTrailingNonEmptyTables:
    def test_basic(self):
        result = _trim_trailing_non_empty_tables(["a", "b", "", "c"])
        assert result == ["c"]

    def test_all_non_empty(self):
        result = _trim_trailing_non_empty_tables(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_all_empty(self):
        result = _trim_trailing_non_empty_tables(["", "", ""])
        assert result == []


class TestTablesEquivalent:
    def test_exact_match(self):
        assert _tables_equivalent("users", "users") is True

    def test_qualified_vs_simple(self):
        assert _tables_equivalent("schema.users", "users") is True

    def test_no_match(self):
        assert _tables_equivalent("users", "orders") is False

    def test_both_empty(self):
        assert _tables_equivalent("", "") is False


class TestCollectSqlTables:
    def test_basic(self):
        tables = collect_sql_tables("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
        assert len(tables) > 0

    def test_none(self):
        assert collect_sql_tables(None) == []

    def test_empty(self):
        assert collect_sql_tables("") == []


class TestNormalizeText:
    def test_basic(self):
        assert _normalize_text("  Hello   World  ") == "hello world"

    def test_none(self):
        assert _normalize_text(None) == ""


class TestFindMatchingCandidates:
    def test_exact_match(self):
        result = _find_matching_candidates("hello", ["hello", "world"])
        assert "hello" in result

    def test_substring_match(self):
        result = _find_matching_candidates("hello", ["hello world", "other"])
        assert "hello world" in result

    def test_empty_expected(self):
        assert _find_matching_candidates("", ["hello"]) == []

    def test_empty_candidates(self):
        assert _find_matching_candidates("hello", []) == []


class TestSplitExpectedItems:
    def test_semicolons(self):
        result = _split_expected_items("a;b;c")
        assert result == ["a", "b", "c"]

    def test_commas(self):
        result = _split_expected_items("x,y,z")
        assert result == ["x", "y", "z"]

    def test_newlines(self):
        result = _split_expected_items("a\nb\nc")
        assert result == ["a", "b", "c"]

    def test_empty(self):
        assert _split_expected_items("") == []
        assert _split_expected_items(None) == []


class TestAppendUnique:
    def test_string(self):
        container = []
        _append_unique(container, "a")
        _append_unique(container, "a")
        assert container == ["a"]

    def test_list(self):
        container = []
        _append_unique(container, ["a", "b"])
        assert container == ["a", "b"]

    def test_none(self):
        container = []
        _append_unique(container, None)
        assert container == []

    def test_mapping_ignored(self):
        container = []
        _append_unique(container, {"key": "value"})
        assert container == []

    def test_empty_string(self):
        container = []
        _append_unique(container, "")
        assert container == []


class TestExtractResultPayload:
    def test_raw_output_with_result(self):
        output = {"raw_output": {"result": "data"}}
        assert _extract_result_payload(output) == "data"

    def test_raw_output_without_result(self):
        output = {"raw_output": "direct_value"}
        assert _extract_result_payload(output) == "direct_value"

    def test_result_key(self):
        output = {"result": "fallback_data"}
        assert _extract_result_payload(output) == "fallback_data"

    def test_non_mapping(self):
        assert _extract_result_payload("string") == "string"


# ===========================================================================
# Artifact collection
# ===========================================================================


class TestAppendFileCandidate:
    def test_basic(self):
        artifacts = WorkflowArtifacts()
        _append_file_candidate(artifacts, "path/to/file.txt")
        assert "path/to/file.txt" in artifacts.files

    def test_empty(self):
        artifacts = WorkflowArtifacts()
        _append_file_candidate(artifacts, "")
        assert artifacts.files == []

    def test_file_written_pattern(self):
        artifacts = WorkflowArtifacts()
        _append_file_candidate(artifacts, "File written successfully: /tmp/output.csv")
        assert "/tmp/output.csv" in artifacts.files


class TestCollectFileArtifacts:
    def test_string(self):
        artifacts = WorkflowArtifacts()
        _collect_file_artifacts(artifacts, "file.txt")
        assert "file.txt" in artifacts.files

    def test_mapping(self):
        artifacts = WorkflowArtifacts()
        _collect_file_artifacts(artifacts, {"file1.txt": "data", "file2.txt": "data"})
        assert "file1.txt" in artifacts.files
        assert "file2.txt" in artifacts.files

    def test_list(self):
        artifacts = WorkflowArtifacts()
        _collect_file_artifacts(artifacts, ["file1.txt", "file2.txt"])
        assert "file1.txt" in artifacts.files


class TestCollectReferenceSqlArtifacts:
    def test_string(self):
        artifacts = WorkflowArtifacts()
        _collect_reference_sql_artifacts(artifacts, "SELECT 1")
        assert "SELECT 1" in artifacts.reference_sqls

    def test_mapping(self):
        artifacts = WorkflowArtifacts()
        _collect_reference_sql_artifacts(artifacts, {"sql": "SELECT 1", "name": "test_query"})
        assert "SELECT 1" in artifacts.reference_sqls
        assert "test_query" in artifacts.reference_sql_names

    def test_list_of_mappings(self):
        artifacts = WorkflowArtifacts()
        _collect_reference_sql_artifacts(artifacts, [{"sql": "SELECT 1"}, {"sql": "SELECT 2"}])
        assert len(artifacts.reference_sqls) == 2


class TestCollectSemanticModelArtifacts:
    def test_basic(self):
        artifacts = WorkflowArtifacts()
        payload = {"metadata": [{"semantic_model_name": "model1"}]}
        _collect_semantic_model_artifacts(artifacts, payload)
        assert "model1" in artifacts.semantic_models

    def test_description_fallback(self):
        artifacts = WorkflowArtifacts()
        payload = {"metadata": [{"description": "model_desc"}]}
        _collect_semantic_model_artifacts(artifacts, payload)
        assert "model_desc" in artifacts.semantic_models

    def test_no_metadata(self):
        artifacts = WorkflowArtifacts()
        _collect_semantic_model_artifacts(artifacts, {"other": "data"})
        assert artifacts.semantic_models == []


class TestCollectMetricArtifacts:
    def test_single_mapping(self):
        artifacts = WorkflowArtifacts()
        _collect_metric_artifacts(artifacts, {"name": "metric1", "description": "desc1"})
        assert "metric1" in artifacts.metrics_names
        assert "desc1" in artifacts.metrics_texts

    def test_list_of_mappings(self):
        artifacts = WorkflowArtifacts()
        _collect_metric_artifacts(artifacts, [{"name": "m1"}, {"name": "m2"}])
        assert len(artifacts.metrics_names) == 2


class TestExtractArtifactsFromActionHistory:
    def test_tool_role_entries(self):
        artifacts = WorkflowArtifacts()
        tool_calls = {}
        history = [
            {
                "role": "tool",
                "input": {"function_name": "write_file"},
                "output": {"raw_output": {"result": "output.txt"}},
            }
        ]
        _extract_artifacts_from_action_history(history, artifacts, tool_calls)
        assert "output.txt" in artifacts.files
        assert tool_calls.get("write_file") == 1

    def test_non_tool_role_skipped(self):
        artifacts = WorkflowArtifacts()
        history = [{"role": "assistant", "content": "hello"}]
        _extract_artifacts_from_action_history(history, artifacts)

    def test_empty_history(self):
        artifacts = WorkflowArtifacts()
        _extract_artifacts_from_action_history(None, artifacts)
        _extract_artifacts_from_action_history([], artifacts)

    def test_search_reference_sql(self):
        artifacts = WorkflowArtifacts()
        history = [
            {
                "role": "tool",
                "input": {"function_name": "search_reference_sql"},
                "output": {"raw_output": {"result": [{"sql": "SELECT 1", "name": "q1"}]}},
            }
        ]
        _extract_artifacts_from_action_history(history, artifacts)
        assert "SELECT 1" in artifacts.reference_sqls

    def test_search_metrics(self):
        artifacts = WorkflowArtifacts()
        history = [
            {
                "role": "tool",
                "input": {"function_name": "search_metrics"},
                "output": {"raw_output": {"result": {"name": "metric1", "description": "d1"}}},
            }
        ]
        _extract_artifacts_from_action_history(history, artifacts)
        assert "metric1" in artifacts.metrics_names

    def test_dotted_function_name(self):
        artifacts = WorkflowArtifacts()
        tool_calls = {}
        history = [
            {
                "role": "tool",
                "input": {"function_name": "module.write_file"},
                "output": {"raw_output": {"result": "file.txt"}},
            }
        ]
        _extract_artifacts_from_action_history(history, artifacts, tool_calls)
        assert tool_calls.get("write_file") == 1


# ===========================================================================
# Dataclass methods
# ===========================================================================


class TestWorkflowArtifacts:
    def test_to_dict(self):
        wa = WorkflowArtifacts(files=["f1"], reference_sqls=["s1"])
        d = wa.to_dict()
        assert d["files"] == ["f1"]
        assert d["reference_sqls"] == ["s1"]


class TestWorkflowAnalysis:
    def test_properties(self):
        outputs = [
            WorkflowOutput(node_id="1", success=True, status="success"),
            WorkflowOutput(node_id="2", success=False, status="failed"),
            WorkflowOutput(node_id="3", success=True, status="success"),
        ]
        analysis = WorkflowAnalysis(
            task_id="t1",
            completion_time="2025-01-01",
            status="completed",
            total_nodes=3,
            outputs=outputs,
        )
        assert analysis.output_nodes == 3
        assert analysis.output_success == 2
        assert analysis.output_failure == 1


class TestComparisonOutcome:
    def test_with_error(self):
        outcome = ComparisonOutcome.with_error("something failed")
        assert outcome.error == "something failed"

    def test_to_dict(self):
        outcome = ComparisonOutcome(match_rate=0.5)
        d = outcome.to_dict()
        assert d["match_rate"] == 0.5


class TestResultData:
    def test_available_with_df(self):
        rd = ResultData(task_id="t1", source="path", dataframe=pd.DataFrame({"a": [1]}))
        assert rd.available is True

    def test_not_available_with_error(self):
        rd = ResultData(task_id="t1", source="path", error="failed")
        assert rd.available is False

    def test_not_available_no_df(self):
        rd = ResultData(task_id="t1", source="path")
        assert rd.available is False


class TestSqlData:
    def test_available(self):
        sd = SqlData(task_id="t1", source="path", sql="SELECT 1")
        assert sd.available is True

    def test_not_available(self):
        sd = SqlData(task_id="t1", source="path", error="failed")
        assert sd.available is False


class TestComparisonRecord:
    def test_to_dict(self):
        actual = ResultData(task_id="t1", source="a.csv", dataframe=pd.DataFrame())
        expected = ResultData(task_id="t1", source="g.csv", dataframe=pd.DataFrame())
        outcome = ComparisonOutcome(match_rate=1.0)
        record = ComparisonRecord(task_id="t1", actual=actual, expected=expected, outcome=outcome)
        d = record.to_dict()
        assert d["task_id"] == "t1"


class TestTaskEvaluation:
    def test_to_dict(self):
        analysis = WorkflowAnalysis(
            task_id="t1",
            completion_time="2025-01-01",
            status="completed",
            total_nodes=1,
        )
        te = TaskEvaluation(task_id="t1", analysis=analysis)
        d = te.to_dict()
        assert d["total_node_count"] == 1


class TestEvaluationReport:
    def test_to_dict(self):
        er = EvaluationReport(
            status="success",
            generated_time="2025-01-01",
            summary={"total": 1},
            task_ids={"matched": "1"},
            details={"t1": {}},
        )
        d = er.to_dict()
        assert d["status"] == "success"


# ===========================================================================
# compare_pandas_tables / columns_match / preview_dataframe
# ===========================================================================


class TestColumnsMatch:
    def test_exact_match(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([1, 2, 3])
        assert columns_match(a, b) is True

    def test_float_tolerance(self):
        a = pd.Series([1.0000001, 2.0, 3.0])
        b = pd.Series([1.0, 2.0, 3.0])
        assert columns_match(a, b) is True

    def test_different_order(self):
        a = pd.Series([3, 1, 2])
        b = pd.Series([1, 2, 3])
        assert columns_match(a, b) is True

    def test_different_length(self):
        a = pd.Series([1, 2])
        b = pd.Series([1, 2, 3])
        assert columns_match(a, b) is False

    def test_nan_values(self):
        a = pd.Series([1, None, 3])
        b = pd.Series([1, None, 3])
        assert columns_match(a, b) is True

    def test_string_values(self):
        a = pd.Series(["a", "b", "c"])
        b = pd.Series(["a", "b", "c"])
        assert columns_match(a, b) is True

    def test_mismatch(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([4, 5, 6])
        assert columns_match(a, b) is False


class TestComparePandasTables:
    def test_exact_match(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = compare_pandas_tables(df1, df2)
        assert result["match_rate"] == 1.0

    def test_different_rows(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        result = compare_pandas_tables(df1, df2)
        assert result["match_rate"] == 0.0

    def test_extra_columns(self):
        df1 = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df2 = pd.DataFrame({"a": [1]})
        result = compare_pandas_tables(df1, df2)
        assert result["match_rate"] == 1.0
        assert len(result["extra_columns"]) == 2

    def test_missing_columns(self):
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [1], "b": [2]})
        result = compare_pandas_tables(df1, df2)
        assert result["match_rate"] == 0.5
        assert len(result["missing_columns"]) == 1


class TestPreviewDataframe:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = preview_dataframe(df)
        assert "a" in result

    def test_none(self):
        assert preview_dataframe(None) == "No data"

    def test_many_rows(self):
        df = pd.DataFrame({"a": range(10)})
        result = preview_dataframe(df, max_rows=3)
        assert "..." in result

    def test_many_columns(self):
        df = pd.DataFrame({f"c{i}": [i] for i in range(10)})
        result = preview_dataframe(df, max_cols=3)
        assert "..." in result


# ===========================================================================
# Providers
# ===========================================================================


class TestCsvPerTaskResultProvider:
    def test_fetch_existing(self, tmp_path):
        csv_file = tmp_path / "t1.csv"
        csv_file.write_text("a,b\n1,2\n")
        provider = CsvPerTaskResultProvider(str(tmp_path))
        result = provider.fetch("t1")
        assert result.available
        assert len(result.dataframe) == 1

    def test_fetch_missing(self, tmp_path):
        provider = CsvPerTaskResultProvider(str(tmp_path))
        result = provider.fetch("nonexistent")
        assert not result.available
        assert "not found" in result.error

    def test_with_namespace_and_run(self, tmp_path):
        run_dir = tmp_path / "ns1" / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "t1.csv").write_text("a\n1\n")
        provider = CsvPerTaskResultProvider(str(tmp_path), namespace="ns1", run_id="run1")
        result = provider.fetch("t1")
        assert result.available

    def test_with_namespace_auto_latest(self, tmp_path):
        run1 = tmp_path / "ns1" / "20250101"
        run1.mkdir(parents=True)
        run2 = tmp_path / "ns1" / "20250102"
        run2.mkdir(parents=True)
        (run2 / "t1.csv").write_text("a\n1\n")
        provider = CsvPerTaskResultProvider(str(tmp_path), namespace="ns1")
        result = provider.fetch("t1")
        assert result.available


class TestDirectorySqlProvider:
    def test_fetch_existing(self, tmp_path):
        sql_file = tmp_path / "t1.sql"
        sql_file.write_text("SELECT * FROM users")
        provider = DirectorySqlProvider(str(tmp_path))
        result = provider.fetch("t1")
        assert result.available
        assert "users" in result.sql.lower()

    def test_fetch_missing(self, tmp_path):
        provider = DirectorySqlProvider(str(tmp_path))
        result = provider.fetch("nonexistent")
        assert not result.available


class TestJsonMappingSqlProvider:
    def test_json_file(self, tmp_path):
        json_file = tmp_path / "sqls.json"
        json_file.write_text(json.dumps([{"task_id": "t1", "SQL": "SELECT 1"}]))
        provider = JsonMappingSqlProvider(str(json_file))
        result = provider.fetch("t1")
        assert result.available
        assert result.sql == "SELECT 1"

    def test_jsonl_file(self, tmp_path):
        # For jsonl, each line is parsed by _ingest as a Mapping, which iterates over items.
        # Use nested dict format to avoid string item assignment errors.
        jsonl_file = tmp_path / "sqls.jsonl"
        line1 = json.dumps({"t1": {"task_id": "t1", "SQL": "SELECT 1"}})
        line2 = json.dumps({"t2": {"task_id": "t2", "SQL": "SELECT 2"}})
        jsonl_file.write_text(f"{line1}\n{line2}\n")
        provider = JsonMappingSqlProvider(str(jsonl_file))
        result = provider.fetch("t1")
        assert result.available
        result2 = provider.fetch("t2")
        assert result2.available

    def test_missing_file(self, tmp_path):
        provider = JsonMappingSqlProvider(str(tmp_path / "missing.json"))
        result = provider.fetch("t1")
        assert not result.available

    def test_not_found_task(self, tmp_path):
        json_file = tmp_path / "sqls.json"
        json_file.write_text(json.dumps([{"task_id": "t1", "SQL": "SELECT 1"}]))
        provider = JsonMappingSqlProvider(str(json_file))
        result = provider.fetch("unknown")
        assert not result.available


class TestCsvColumnSqlProvider:
    def test_basic(self, tmp_path):
        csv_file = tmp_path / "sqls.csv"
        csv_file.write_text("task_id,SQL\nt1,SELECT 1\nt2,SELECT 2\n")
        provider = CsvColumnSqlProvider(str(csv_file), task_id_key="task_id", sql_key="SQL")
        result = provider.fetch("t1")
        assert result.available
        assert result.sql == "SELECT 1"

    def test_missing_file(self, tmp_path):
        provider = CsvColumnSqlProvider(str(tmp_path / "missing.csv"), task_id_key="id", sql_key="sql")
        result = provider.fetch("t1")
        assert not result.available

    def test_not_found_task(self, tmp_path):
        csv_file = tmp_path / "sqls.csv"
        csv_file.write_text("task_id,SQL\nt1,SELECT 1\n")
        provider = CsvColumnSqlProvider(str(csv_file), task_id_key="task_id", sql_key="SQL")
        result = provider.fetch("unknown")
        assert not result.available


class TestAgentResultSqlProvider:
    def test_sql_file(self, tmp_path):
        ns_dir = tmp_path / "ns1" / "run1"
        ns_dir.mkdir(parents=True)
        (ns_dir / "t1.sql").write_text("SELECT * FROM orders")
        provider = AgentResultSqlProvider(str(tmp_path), namespace="ns1", run_id="run1")
        result = provider.fetch("t1")
        assert result.available
        assert "orders" in result.sql.lower()

    def test_json_file(self, tmp_path):
        ns_dir = tmp_path / "ns1" / "run1"
        ns_dir.mkdir(parents=True)
        (ns_dir / "t1.json").write_text(json.dumps({"gen_sql": "SELECT 1"}))
        provider = AgentResultSqlProvider(str(tmp_path), namespace="ns1", run_id="run1")
        result = provider.fetch("t1")
        assert result.available
        assert result.sql == "SELECT 1"

    def test_no_artifacts(self, tmp_path):
        ns_dir = tmp_path / "ns1" / "run1"
        ns_dir.mkdir(parents=True)
        provider = AgentResultSqlProvider(str(tmp_path), namespace="ns1", run_id="run1")
        result = provider.fetch("t1")
        assert not result.available

    def test_nonexistent_dir(self, tmp_path):
        provider = AgentResultSqlProvider(str(tmp_path), namespace="missing", run_id="run1")
        result = provider.fetch("t1")
        assert not result.available


# ===========================================================================
# TrajectoryParser
# ===========================================================================


class TestTrajectoryParser:
    def test_parse_valid(self, tmp_path):
        trajectory_data = {
            "workflow": {
                "completion_time": "2025-01-01",
                "status": "completed",
                "nodes": [
                    {"id": "n1", "type": "schema_linking", "result": {}},
                    {
                        "id": "n2",
                        "type": "output",
                        "result": {"success": True, "status": "success"},
                    },
                ],
            }
        }
        filepath = tmp_path / "t1.yml"
        filepath.write_text(yaml.dump(trajectory_data))
        parser = TrajectoryParser()
        result = parser.parse(filepath, "t1")
        assert result.status == "completed"
        assert result.total_nodes == 2
        assert result.output_success == 1

    def test_parse_missing_file(self, tmp_path):
        parser = TrajectoryParser()
        result = parser.parse(tmp_path / "missing.yml", "t1")
        assert result.status == "missing"
        assert len(result.errors) > 0

    def test_parse_invalid_format(self, tmp_path):
        filepath = tmp_path / "bad.yml"
        filepath.write_text("not_workflow: true")
        parser = TrajectoryParser()
        result = parser.parse(filepath, "t1")
        assert result.status == "invalid"

    def test_parse_no_nodes(self, tmp_path):
        trajectory_data = {
            "workflow": {
                "status": "completed",
                "type": "output",
                "id": "w1",
                "result": {"success": True, "status": "success"},
            }
        }
        filepath = tmp_path / "t1.yml"
        filepath.write_text(yaml.dump(trajectory_data))
        parser = TrajectoryParser()
        result = parser.parse(filepath, "t1")
        assert result.total_nodes == 1

    def test_parse_failed_output(self, tmp_path):
        trajectory_data = {
            "workflow": {
                "status": "completed",
                "nodes": [
                    {
                        "id": "n1",
                        "type": "output",
                        "result": {"success": False, "status": "failed", "error": "SQL error"},
                    }
                ],
            }
        }
        filepath = tmp_path / "t1.yml"
        filepath.write_text(yaml.dump(trajectory_data))
        parser = TrajectoryParser()
        result = parser.parse(filepath, "t1")
        assert result.output_failure == 1
        assert len(result.errors) > 0


# ===========================================================================
# TableComparator
# ===========================================================================


class TestTableComparator:
    def test_exact_match(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        comparator = TableComparator()
        outcome = comparator.compare(df1, df2)
        assert outcome.match_rate == 1.0

    def test_mismatch(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        comparator = TableComparator()
        outcome = comparator.compare(df1, df2)
        assert outcome.match_rate == 0.0


# ===========================================================================
# EvaluationReportBuilder
# ===========================================================================


class TestEvaluationReportBuilder:
    def test_basic(self):
        analysis = WorkflowAnalysis(
            task_id="t1",
            completion_time="2025-01-01",
            status="completed",
            total_nodes=1,
            outputs=[WorkflowOutput(node_id="n1", success=True, status="success")],
        )
        actual = ResultData(task_id="t1", source="a.csv", dataframe=pd.DataFrame({"a": [1]}))
        expected = ResultData(task_id="t1", source="g.csv", dataframe=pd.DataFrame({"a": [1]}))
        outcome = ComparisonOutcome(match_rate=1.0)
        record = ComparisonRecord(task_id="t1", actual=actual, expected=expected, outcome=outcome)
        te = TaskEvaluation(task_id="t1", analysis=analysis, comparisons=[record])

        builder = EvaluationReportBuilder()
        report = builder.build({"t1": te})
        assert report.status == "success"
        assert report.summary["comparison_summary"]["match_count"] == 1

    def test_empty_evaluations(self):
        builder = EvaluationReportBuilder()
        report = builder.build({})
        assert report.summary["total_files"] == 0


# ===========================================================================
# SingleFileGoldProvider
# ===========================================================================


class TestSingleFileGoldProvider:
    def test_csv_with_query_result(self, tmp_path):
        csv_file = tmp_path / "gold.csv"
        csv_file.write_text("task_id,result\nt1,\"a,b\n1,2\"\n")
        provider = SingleFileGoldProvider(
            result_file=str(csv_file),
            connections={},
            task_id_key="task_id",
            query_result_key="result",
        )
        result = provider.fetch("t1")
        # Should try to parse the result
        assert result.task_id == "t1"

    def test_json_file(self, tmp_path):
        json_file = tmp_path / "gold.json"
        data = [{"task_id": "t1", "result": "a,b\n1,2\n"}]
        json_file.write_text(json.dumps(data))
        provider = SingleFileGoldProvider(
            result_file=str(json_file),
            connections={},
            task_id_key="task_id",
            query_result_key="result",
        )
        result = provider.fetch("t1")
        assert result.task_id == "t1"

    def test_missing_file(self, tmp_path):
        provider = SingleFileGoldProvider(
            result_file=str(tmp_path / "missing.json"),
            connections={},
        )
        result = provider.fetch("t1")
        assert not result.available

    def test_get_artifacts(self, tmp_path):
        json_file = tmp_path / "gold.json"
        data = [{"task_id": "t1", "result": "a\n1\n", "file": "data.txt", "expected_sql": "SELECT 1"}]
        json_file.write_text(json.dumps(data))
        provider = SingleFileGoldProvider(
            result_file=str(json_file),
            connections={},
            task_id_key="task_id",
            query_result_key="result",
        )
        artifacts = provider.get_artifacts("t1")
        assert artifacts.file_reference == "data.txt"
        assert artifacts.expected_sql == "SELECT 1"

    def test_unsupported_format(self, tmp_path):
        xml_file = tmp_path / "gold.xml"
        xml_file.write_text("<data></data>")
        provider = SingleFileGoldProvider(
            result_file=str(xml_file),
            connections={},
        )
        result = provider.fetch("t1")
        assert not result.available

    def test_jsonl_file(self, tmp_path):
        jsonl_file = tmp_path / "gold.jsonl"
        jsonl_file.write_text('{"task_id": "t1", "result": "a\\n1\\n"}\n')
        provider = SingleFileGoldProvider(
            result_file=str(jsonl_file),
            connections={},
            task_id_key="task_id",
            query_result_key="result",
        )
        result = provider.fetch("t1")
        assert result.task_id == "t1"
