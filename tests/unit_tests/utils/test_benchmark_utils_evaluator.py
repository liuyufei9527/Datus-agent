# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for BenchmarkEvaluator and remaining benchmark_utils functions."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from datus.utils.benchmark_utils import (
    BenchmarkEvaluator,
    ComparisonOutcome,
    CsvPerTaskResultProvider,
    GoldArtifacts,
    ResultData,
    SingleFileGoldProvider,
    SqlData,
    TableComparator,
    TrajectoryParser,
    WorkflowArtifacts,
    _clean_table_identifier_part,
    _resolve_existing_directory,
    collect_latest_trajectory_files,
    compute_table_matches,
    csv_str_to_pands,
    detect_benchmark_type,
    list_save_runs,
    list_trajectory_runs,
    parse_trajectory_filename,
)


# ===========================================================================
# parse_trajectory_filename
# ===========================================================================


class TestParseTrajectoryFilename:
    def test_valid_filename(self):
        task_id, ts = parse_trajectory_filename("task_123_1234567890.yaml")
        assert task_id == "task_123"
        assert ts == 1234567890.0

    def test_float_timestamp(self):
        task_id, ts = parse_trajectory_filename("my_task_1234567890.123.yaml")
        assert task_id == "my_task"
        assert ts == 1234567890.123

    def test_no_underscore(self):
        task_id, ts = parse_trajectory_filename("nounderscore.yaml")
        assert task_id is None
        assert ts is None

    def test_invalid_timestamp(self):
        task_id, ts = parse_trajectory_filename("task_notanumber.yaml")
        assert task_id is None
        assert ts is None


# ===========================================================================
# list_trajectory_runs / list_save_runs
# ===========================================================================


class TestListTrajectoryRuns:
    def test_nonexistent_dir(self, tmp_path):
        result = list_trajectory_runs(str(tmp_path / "nonexistent"))
        assert result == {}

    def test_with_namespace(self, tmp_path):
        ns_dir = tmp_path / "ns1"
        (ns_dir / "run1").mkdir(parents=True)
        (ns_dir / "run2").mkdir(parents=True)
        result = list_trajectory_runs(str(tmp_path), namespace="ns1")
        assert "ns1" in result
        assert len(result["ns1"]) == 2

    def test_all_namespaces(self, tmp_path):
        (tmp_path / "ns1" / "run1").mkdir(parents=True)
        (tmp_path / "ns2" / "run1").mkdir(parents=True)
        result = list_trajectory_runs(str(tmp_path))
        assert "ns1" in result
        assert "ns2" in result

    def test_empty_namespace(self, tmp_path):
        ns_dir = tmp_path / "ns1"
        ns_dir.mkdir()
        # No run dirs inside
        result = list_trajectory_runs(str(tmp_path), namespace="ns1")
        assert result == {}


class TestListSaveRuns:
    def test_nonexistent_dir(self, tmp_path):
        result = list_save_runs(str(tmp_path / "nonexistent"))
        assert result == {}

    def test_with_namespace(self, tmp_path):
        ns_dir = tmp_path / "ns1"
        (ns_dir / "run1").mkdir(parents=True)
        (ns_dir / "run2").mkdir(parents=True)
        result = list_save_runs(str(tmp_path), namespace="ns1")
        assert "ns1" in result
        assert len(result["ns1"]) == 2

    def test_all_namespaces(self, tmp_path):
        (tmp_path / "ns1" / "run1").mkdir(parents=True)
        (tmp_path / "ns2" / "run1").mkdir(parents=True)
        result = list_save_runs(str(tmp_path))
        assert "ns1" in result
        assert "ns2" in result


# ===========================================================================
# collect_latest_trajectory_files
# ===========================================================================


class TestCollectLatestTrajectoryFiles:
    def test_nonexistent_dir(self, tmp_path):
        result = collect_latest_trajectory_files(str(tmp_path / "nonexistent"), "ns1")
        assert result == {}

    def test_specific_run(self, tmp_path):
        run_dir = tmp_path / "ns1" / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "task1_1000.yaml").write_text(yaml.dump({"workflow": {"status": "done"}}))
        (run_dir / "task1_2000.yaml").write_text(yaml.dump({"workflow": {"status": "done"}}))

        result = collect_latest_trajectory_files(str(tmp_path), "ns1", "run1")
        assert "task1" in result
        # Should pick the latest one (timestamp 2000)
        assert "2000" in result["task1"].name

    def test_auto_latest_run(self, tmp_path):
        run1 = tmp_path / "ns1" / "20250101"
        run1.mkdir(parents=True)
        run2 = tmp_path / "ns1" / "20250102"
        run2.mkdir(parents=True)
        (run2 / "task1_1000.yaml").write_text(yaml.dump({"workflow": {"status": "done"}}))

        result = collect_latest_trajectory_files(str(tmp_path), "ns1")
        assert "task1" in result

    def test_nonexistent_namespace(self, tmp_path):
        result = collect_latest_trajectory_files(str(tmp_path), "nonexistent")
        assert result == {}

    def test_namespace_no_run_dirs(self, tmp_path):
        ns_dir = tmp_path / "ns1"
        ns_dir.mkdir(parents=True)
        # Trajectory file directly in namespace dir (no run subdirs)
        (ns_dir / "task1_1000.yaml").write_text(yaml.dump({"workflow": {"status": "done"}}))
        result = collect_latest_trajectory_files(str(tmp_path), "ns1")
        assert "task1" in result


# ===========================================================================
# detect_benchmark_type
# ===========================================================================


class TestDetectBenchmarkType:
    def test_sub_agent_file(self, tmp_path):
        f = tmp_path / "benchmark.json"
        f.write_text("{}")
        assert detect_benchmark_type(f) == "sub_agent"

    def test_bird_dev(self, tmp_path):
        (tmp_path / "dev.json").write_text("{}")
        assert detect_benchmark_type(tmp_path) == "bird_dev"

    def test_spider2(self, tmp_path):
        gold_dir = tmp_path / "evaluation_suite" / "gold" / "sql"
        gold_dir.mkdir(parents=True)
        assert detect_benchmark_type(tmp_path) == "spider2"

    def test_bird_by_name(self, tmp_path):
        bird_dir = tmp_path / "bird_benchmark"
        bird_dir.mkdir()
        assert detect_benchmark_type(bird_dir) == "bird_dev"

    def test_spider_by_name(self, tmp_path):
        spider_dir = tmp_path / "spider_dataset"
        spider_dir.mkdir()
        assert detect_benchmark_type(spider_dir) == "spider2"

    def test_unknown(self, tmp_path):
        unknown_dir = tmp_path / "data"
        unknown_dir.mkdir()
        assert detect_benchmark_type(unknown_dir) == "unknown"


# ===========================================================================
# _resolve_existing_directory
# ===========================================================================


class TestResolveExistingDirectory:
    def test_existing(self, tmp_path):
        sub = tmp_path / "sub1"
        sub.mkdir()
        result = _resolve_existing_directory(tmp_path, ["sub1", "sub2"])
        assert result == sub

    def test_none_found(self, tmp_path):
        result = _resolve_existing_directory(tmp_path, ["nonexistent1", "nonexistent2"])
        assert result is None


# ===========================================================================
# csv_str_to_pands
# ===========================================================================


class TestCsvStrToPands:
    def test_basic(self):
        df = csv_str_to_pands("a,b\n1,2\n")
        assert len(df) == 1
        assert list(df.columns) == ["a", "b"]

    def test_multiline(self):
        df = csv_str_to_pands("x\n10\n20\n30\n")
        assert len(df) == 3


# ===========================================================================
# BenchmarkEvaluator
# ===========================================================================


class _DummyResultProvider:
    """Minimal result provider for testing."""

    def __init__(self, data_map=None):
        self._data = data_map or {}

    def fetch(self, task_id):
        if task_id in self._data:
            return ResultData(
                task_id=task_id,
                source="test",
                dataframe=self._data[task_id],
            )
        return ResultData(task_id=task_id, source="test", error=f"Not found: {task_id}")


class _DummySqlProvider:
    """Minimal sql provider for testing."""

    def __init__(self, sql_map=None):
        self._sql = sql_map or {}

    def fetch(self, task_id):
        if task_id in self._sql:
            return SqlData(task_id=task_id, source="test", sql=self._sql[task_id], tables=["t1"])
        return SqlData(task_id=task_id, source="test", error="No SQL")


class TestBenchmarkEvaluator:
    def _make_trajectory(self, tmp_path, task_id, success=True):
        trajectory_data = {
            "workflow": {
                "completion_time": "2025-01-01",
                "status": "completed",
                "nodes": [
                    {
                        "id": "n1",
                        "type": "output",
                        "result": {
                            "success": success,
                            "status": "success" if success else "failed",
                            "error": None if success else "test error",
                        },
                    }
                ],
            }
        }
        filepath = tmp_path / f"{task_id}_1000.yaml"
        filepath.write_text(yaml.dump(trajectory_data))
        return filepath

    def test_evaluate_basic(self, tmp_path):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        result_provider = _DummyResultProvider({"t1": df1})
        gold_provider = _DummyResultProvider({"t1": df2})

        evaluator = BenchmarkEvaluator(
            trajectory_parser=TrajectoryParser(),
            result_provider=result_provider,
            gold_result_provider=gold_provider,
        )

        trajectory_path = self._make_trajectory(tmp_path, "t1")
        report = evaluator.evaluate({"t1": trajectory_path})
        assert report.status == "success"

    def test_evaluate_with_sql_providers(self, tmp_path):
        df = pd.DataFrame({"a": [1]})

        evaluator = BenchmarkEvaluator(
            trajectory_parser=TrajectoryParser(),
            result_provider=_DummyResultProvider({"t1": df}),
            gold_result_provider=_DummyResultProvider({"t1": df}),
            result_sql_provider=_DummySqlProvider({"t1": "SELECT 1"}),
            gold_sql_provider=_DummySqlProvider({"t1": "SELECT 1"}),
        )

        trajectory_path = self._make_trajectory(tmp_path, "t1")
        report = evaluator.evaluate({"t1": trajectory_path})
        assert report.status == "success"

    def test_evaluate_actual_unavailable(self, tmp_path):
        df = pd.DataFrame({"a": [1]})

        evaluator = BenchmarkEvaluator(
            trajectory_parser=TrajectoryParser(),
            result_provider=_DummyResultProvider({}),  # empty
            gold_result_provider=_DummyResultProvider({"t1": df}),
        )

        trajectory_path = self._make_trajectory(tmp_path, "t1")
        report = evaluator.evaluate({"t1": trajectory_path})
        assert report.status == "success"

    def test_evaluate_gold_unavailable(self, tmp_path):
        df = pd.DataFrame({"a": [1]})

        evaluator = BenchmarkEvaluator(
            trajectory_parser=TrajectoryParser(),
            result_provider=_DummyResultProvider({"t1": df}),
            gold_result_provider=_DummyResultProvider({}),  # empty
        )

        trajectory_path = self._make_trajectory(tmp_path, "t1")
        report = evaluator.evaluate({"t1": trajectory_path})
        assert report.status == "success"

    def test_evaluate_failed_output(self, tmp_path):
        evaluator = BenchmarkEvaluator(
            trajectory_parser=TrajectoryParser(),
            result_provider=_DummyResultProvider({}),
            gold_result_provider=_DummyResultProvider({}),
        )

        trajectory_path = self._make_trajectory(tmp_path, "t1", success=False)
        report = evaluator.evaluate({"t1": trajectory_path})
        assert report.status == "success"

    def test_evaluate_with_gold_artifacts(self, tmp_path):
        df = pd.DataFrame({"a": [1]})

        class _GoldWithArtifacts(_DummyResultProvider):
            def get_artifacts(self, task_id):
                return GoldArtifacts(
                    file_reference="data.csv",
                    expected_sql="SELECT 1",
                    semantic_model="model1",
                    expected_metrics="metric1;metric2",
                )

        evaluator = BenchmarkEvaluator(
            trajectory_parser=TrajectoryParser(),
            result_provider=_DummyResultProvider({"t1": df}),
            gold_result_provider=_GoldWithArtifacts({"t1": df}),
        )

        trajectory_path = self._make_trajectory(tmp_path, "t1")
        report = evaluator.evaluate({"t1": trajectory_path})
        assert report.status == "success"

    def test_evaluate_directory(self, tmp_path):
        trajectory_dir = tmp_path / "trajectories" / "ns1" / "run1"
        trajectory_dir.mkdir(parents=True)
        traj_file = trajectory_dir / "t1_1000.yaml"
        traj_file.write_text(
            yaml.dump(
                {
                    "workflow": {
                        "status": "completed",
                        "nodes": [{"id": "n1", "type": "output", "result": {"success": True, "status": "success"}}],
                    }
                }
            )
        )

        df = pd.DataFrame({"a": [1]})
        evaluator = BenchmarkEvaluator(
            trajectory_parser=TrajectoryParser(),
            result_provider=_DummyResultProvider({"t1": df}),
            gold_result_provider=_DummyResultProvider({"t1": df}),
        )

        report = evaluator.evaluate_directory(
            str(tmp_path / "trajectories"),
            target_task_ids=["t1"],
            namespace="ns1",
            run_id="run1",
        )
        assert report.status == "success"


# ===========================================================================
# compute_table_matches
# ===========================================================================


class TestComputeTableMatches:
    def test_exact_match(self):
        result = compute_table_matches(["users", "orders"], ["users", "orders"])
        assert "users" in result
        assert "orders" in result

    def test_partial_match(self):
        result = compute_table_matches(["schema.users"], ["users"])
        assert "users" in result

    def test_empty_actual(self):
        assert compute_table_matches([], ["users"]) == []

    def test_empty_expected(self):
        assert compute_table_matches(["users"], []) == []

    def test_none_values(self):
        assert compute_table_matches(None, None) == []

    def test_no_match(self):
        result = compute_table_matches(["users"], ["orders"])
        assert result == []


# ===========================================================================
# _clean_table_identifier_part
# ===========================================================================


class TestCleanTableIdentifierPart:
    def test_basic(self):
        assert _clean_table_identifier_part("users") == "users"

    def test_quoted(self):
        assert _clean_table_identifier_part('"users"') == "users"

    def test_backtick(self):
        assert _clean_table_identifier_part("`users`") == "users"

    def test_brackets(self):
        assert _clean_table_identifier_part("[users]") == "users"

    def test_none(self):
        # _clean_table_identifier_part converts to str first, so None becomes "None"
        result = _clean_table_identifier_part(None)
        assert isinstance(result, str)

    def test_non_string(self):
        # Non-string input is converted via str()
        result = _clean_table_identifier_part(123)
        assert result == "123"
