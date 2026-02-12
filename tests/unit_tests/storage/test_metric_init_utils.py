"""Tests for datus/storage/metric/init_utils.py."""

import pytest

from datus.storage.metric.init_utils import gen_metric_id, gen_semantic_model_id


class TestGenSemanticModelId:
    def test_basic(self):
        result = gen_semantic_model_id("cat", "db", "sch", "tbl")
        assert result == "cat_db_sch_tbl"

    def test_empty_parts(self):
        result = gen_semantic_model_id("", "db", "", "tbl")
        assert result == "_db__tbl"


class TestGenMetricId:
    def test_basic(self):
        result = gen_metric_id(["Finance", "Revenue"], "orders", "total_revenue")
        assert result == "Finance/Revenue/orders_total_revenue"

    def test_empty_subject_path(self):
        result = gen_metric_id([], "model", "metric")
        assert result == "/model_metric"

    def test_none_subject_path(self):
        result = gen_metric_id(None, "model", "metric")
        assert result == "/model_metric"
