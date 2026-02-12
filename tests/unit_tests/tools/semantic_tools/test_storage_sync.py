# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for semantic_tools/storage_sync.py – 12.8% coverage target."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.tools.semantic_tools.storage_sync import SemanticStorageManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.db_path = ":memory:"
    return config


@pytest.fixture
def manager(mock_agent_config):
    return SemanticStorageManager(mock_agent_config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_init(self, manager):
        assert manager.semantic_model_store is None
        assert manager.metric_store is None
        assert manager.subject_tree_store is None


# ---------------------------------------------------------------------------
# store_semantic_model
# ---------------------------------------------------------------------------


class TestStoreSemanticModel:
    def test_missing_name_raises(self, manager):
        with pytest.raises(ValueError, match="semantic_model_name"):
            manager.store_semantic_model({"table_name": "t"})

    def test_basic_store(self, manager):
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "my_model",
                    "table_name": "orders",
                    "database_name": "mydb",
                    "schema_name": "public",
                    "description": "Order model",
                    "dimensions": [{"name": "region", "description": "Region"}],
                    "measures": [{"name": "revenue", "description": "Revenue"}],
                    "identifiers": [{"name": "order_id", "description": "PK"}],
                }
            )
            # Should call batch_store for table + dimensions + measures + identifiers
            assert mock_store.batch_store.call_count == 4

    def test_dimensions_without_name_skipped(self, manager):
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "m",
                    "dimensions": [{"description": "no name"}, {"name": "valid_dim"}],
                    "measures": [],
                }
            )
            # Table object + 1 dimension (invalid skipped)
            # batch_store called for table and for dims
            calls = mock_store.batch_store.call_args_list
            table_call = calls[0]
            dim_call = calls[1]
            assert len(dim_call[0][0]) == 1  # Only 1 valid dim

    def test_measures_without_name_skipped(self, manager):
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "m",
                    "dimensions": [],
                    "measures": ["not a dict", {"name": "valid_measure"}],
                }
            )
            # Table call + measures call
            calls = mock_store.batch_store.call_args_list
            measure_call = calls[1]
            assert len(measure_call[0][0]) == 1

    def test_identifiers_without_name_skipped(self, manager):
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "m",
                    "identifiers": [{}, {"name": "valid_id"}],
                }
            )
            calls = mock_store.batch_store.call_args_list
            id_call = calls[1]
            assert len(id_call[0][0]) == 1

    def test_empty_dims_measures_identifiers(self, manager):
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "m",
                    "dimensions": [],
                    "measures": [],
                    "identifiers": [],
                }
            )
            # Only the table object should be stored
            assert mock_store.batch_store.call_count == 1

    def test_fully_qualified_table_name(self, manager):
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "m",
                    "table_name": "orders",
                    "catalog_name": "cat",
                    "database_name": "db",
                    "schema_name": "schema",
                }
            )
            table_obj = mock_store.batch_store.call_args_list[0][0][0][0]
            assert table_obj["fq_name"] == "cat.db.schema.orders"


# ---------------------------------------------------------------------------
# store_metric
# ---------------------------------------------------------------------------


class TestStoreMetric:
    def test_missing_name_raises(self, manager):
        with pytest.raises(ValueError, match="name"):
            manager.store_metric({"description": "no name"})

    def test_basic_store(self, manager):
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_metric_store", return_value=mock_store):
            manager.store_metric(
                {
                    "name": "revenue",
                    "description": "Total revenue",
                    "metric_type": "simple",
                    "dimensions": ["region"],
                    "measures": ["amount"],
                },
                subject_path=["Finance", "Revenue"],
            )
            mock_store.batch_store_metrics.assert_called_once()
            stored = mock_store.batch_store_metrics.call_args[0][0][0]
            assert stored["name"] == "revenue"
            assert stored["subject_path"] == ["Finance", "Revenue"]

    def test_default_subject_path(self, manager):
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_metric_store", return_value=mock_store):
            manager.store_metric({"name": "m1"})
            stored = mock_store.batch_store_metrics.call_args[0][0][0]
            assert stored["subject_path"] == ["Uncategorized"]


# ---------------------------------------------------------------------------
# sync_from_adapter
# ---------------------------------------------------------------------------


class TestSyncFromAdapter:
    @pytest.mark.asyncio
    async def test_sync_semantic_models(self, manager):
        adapter = MagicMock()
        adapter.service_type = "test"
        adapter.list_semantic_models.return_value = ["model1"]
        adapter.get_semantic_model.return_value = {
            "semantic_model_name": "model1",
            "table_name": "t1",
        }
        adapter.list_metrics = AsyncMock(return_value=[])

        with patch.object(manager, "store_semantic_model") as mock_store:
            stats = await manager.sync_from_adapter(adapter, sync_metrics=False)
            assert stats["semantic_models_synced"] == 1
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_metrics(self, manager):
        adapter = MagicMock()
        adapter.service_type = "test"
        adapter.list_semantic_models.return_value = []

        mock_metric = MagicMock()
        mock_metric.name = "revenue"
        mock_metric.description = "Total revenue"
        mock_metric.type = "simple"
        mock_metric.dimensions = ["region"]
        mock_metric.measures = ["amount"]
        mock_metric.unit = "USD"
        mock_metric.format = None
        mock_metric.path = ["Finance"]
        adapter.list_metrics = AsyncMock(return_value=[mock_metric])

        with patch.object(manager, "store_metric") as mock_store:
            stats = await manager.sync_from_adapter(adapter, sync_semantic_models=False)
            assert stats["metrics_synced"] == 1
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_error_in_model(self, manager):
        adapter = MagicMock()
        adapter.service_type = "test"
        adapter.list_semantic_models.return_value = ["bad_model"]
        adapter.get_semantic_model.side_effect = Exception("fail")
        adapter.list_metrics = AsyncMock(return_value=[])

        stats = await manager.sync_from_adapter(adapter, sync_metrics=False)
        assert stats["semantic_models_synced"] == 0

    @pytest.mark.asyncio
    async def test_sync_error_listing_models(self, manager):
        adapter = MagicMock()
        adapter.service_type = "test"
        adapter.list_semantic_models.side_effect = Exception("fail")
        adapter.list_metrics = AsyncMock(return_value=[])

        stats = await manager.sync_from_adapter(adapter, sync_metrics=False)
        assert stats["semantic_models_synced"] == 0

    @pytest.mark.asyncio
    async def test_sync_error_listing_metrics(self, manager):
        adapter = MagicMock()
        adapter.service_type = "test"
        adapter.list_semantic_models.return_value = []
        adapter.list_metrics = AsyncMock(side_effect=Exception("fail"))

        stats = await manager.sync_from_adapter(adapter, sync_semantic_models=False)
        assert stats["metrics_synced"] == 0

    @pytest.mark.asyncio
    async def test_sync_metric_with_no_path(self, manager):
        adapter = MagicMock()
        adapter.service_type = "test"
        adapter.list_semantic_models.return_value = []

        mock_metric = MagicMock()
        mock_metric.name = "count"
        mock_metric.description = ""
        mock_metric.type = None
        mock_metric.dimensions = []
        mock_metric.measures = []
        mock_metric.unit = None
        mock_metric.format = None
        mock_metric.path = None
        adapter.list_metrics = AsyncMock(return_value=[mock_metric])

        with patch.object(manager, "store_metric") as mock_store:
            stats = await manager.sync_from_adapter(
                adapter, sync_semantic_models=False, subject_path=["Default"]
            )
            assert stats["metrics_synced"] == 1
            # Should use provided subject_path since metric.path is None
            call_kwargs = mock_store.call_args
            assert call_kwargs[1]["subject_path"] == ["Default"]


# ---------------------------------------------------------------------------
# _ensure_* lazy init
# ---------------------------------------------------------------------------


class TestLazyInit:
    def test_ensure_semantic_model_store_cached(self, manager):
        mock_store = MagicMock()
        manager.semantic_model_store = mock_store
        store = manager._ensure_semantic_model_store()
        assert store is mock_store

    def test_ensure_metric_store_cached(self, manager):
        mock_store = MagicMock()
        manager.metric_store = mock_store
        store = manager._ensure_metric_store()
        assert store is mock_store

    def test_ensure_subject_tree_store_cached(self, manager):
        mock_store = MagicMock()
        manager.subject_tree_store = mock_store
        store = manager._ensure_subject_tree_store()
        assert store is mock_store
