"""Tests for datus/storage/semantic_model/store.py — SemanticModelStorage and SemanticModelRAG."""

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from datus.storage.semantic_model.store import SemanticModelRAG, SemanticModelStorage


@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=SemanticModelStorage)
    return storage


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.current_namespace = "default"
    return config


# ── SemanticModelRAG ──


class TestSemanticModelRAG:
    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_init(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        rag = SemanticModelRAG(mock_agent_config)
        assert rag.storage is mock_storage

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_get_semantic_model_no_table_name(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        rag = SemanticModelRAG(mock_agent_config)
        result = rag.get_semantic_model(table_name="")
        assert result is None

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_get_semantic_model_found(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        # Mock table found
        table_row = {
            "kind": "table",
            "name": "orders",
            "table_name": "orders",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
            "description": "Orders table",
            "yaml_path": "/path/to/yaml",
        }
        # Mock children
        dim_row = {
            "kind": "column",
            "name": "region",
            "is_dimension": True,
            "is_measure": False,
            "is_entity_key": False,
            "description": "Region",
            "expr": "region",
            "column_type": "CATEGORICAL",
            "is_partition": False,
            "time_granularity": "",
        }
        measure_row = {
            "kind": "column",
            "name": "amount",
            "is_dimension": False,
            "is_measure": True,
            "is_entity_key": False,
            "description": "Amount",
            "expr": "amount",
            "agg": "SUM",
            "create_metric": True,
            "agg_time_dimension": "order_date",
        }
        ident_row = {
            "kind": "column",
            "name": "order_id",
            "is_dimension": False,
            "is_measure": False,
            "is_entity_key": True,
            "description": "Primary key",
            "expr": "order_id",
            "column_type": "PRIMARY",
            "entity": "order",
        }

        # First call returns table, second returns children
        arrow_table = MagicMock()
        arrow_table.to_pylist.side_effect = [[table_row], [dim_row, measure_row, ident_row]]
        mock_storage._search_all.return_value = arrow_table

        rag = SemanticModelRAG(mock_agent_config)
        result = rag.get_semantic_model(table_name="orders")
        assert result is not None
        assert result["table_name"] == "orders"
        assert len(result["dimensions"]) == 1
        assert len(result["measures"]) == 1
        assert len(result["identifiers"]) == 1

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_get_semantic_model_not_found(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        arrow_table = MagicMock()
        arrow_table.to_pylist.return_value = []
        mock_storage._search_all.return_value = arrow_table

        rag = SemanticModelRAG(mock_agent_config)
        result = rag.get_semantic_model(table_name="nonexistent")
        assert result is None

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_get_semantic_model_with_select_fields(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        table_row = {
            "kind": "table",
            "name": "orders",
            "table_name": "orders",
            "catalog_name": "",
            "database_name": "",
            "schema_name": "",
            "description": "Orders",
            "yaml_path": "",
        }

        arrow_table = MagicMock()
        arrow_table.to_pylist.side_effect = [[table_row], []]
        mock_storage._search_all.return_value = arrow_table

        rag = SemanticModelRAG(mock_agent_config)
        result = rag.get_semantic_model(table_name="orders", select_fields=["table_name", "description"])
        assert result is not None
        assert "table_name" in result
        assert "description" in result
        # Fields not selected should be absent
        assert "yaml_path" not in result

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_search_all(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        arrow_table = MagicMock()
        arrow_table.to_pylist.return_value = [{"name": "orders", "kind": "table"}]
        mock_storage._search_all.return_value = arrow_table

        rag = SemanticModelRAG(mock_agent_config)
        result = rag.search_all()
        assert len(result) == 1

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_get_size(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst
        mock_storage.table = MagicMock()
        mock_storage.table.count_rows.return_value = 10

        rag = SemanticModelRAG(mock_agent_config)
        assert rag.get_size() == 10

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_get_size_exception(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst
        mock_storage._ensure_table_ready.side_effect = RuntimeError("fail")

        rag = SemanticModelRAG(mock_agent_config)
        assert rag.get_size() == 0

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_store_batch(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        rag = SemanticModelRAG(mock_agent_config)
        rag.store_batch([{"id": "1"}])
        mock_storage.store_batch.assert_called_once()

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_upsert_batch(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        rag = SemanticModelRAG(mock_agent_config)
        rag.upsert_batch([{"id": "1"}])
        mock_storage.upsert_batch.assert_called_once()

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_create_indices(self, mock_cache, mock_agent_config):
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        rag = SemanticModelRAG(mock_agent_config)
        rag.create_indices()
        mock_storage.create_indices.assert_called_once()

    @patch("datus.storage.semantic_model.store.get_storage_cache_instance")
    def test_fallback_broad_match(self, mock_cache, mock_agent_config):
        """Test that get_semantic_model falls back to broad match."""
        cache_inst = MagicMock()
        mock_storage = MagicMock()
        cache_inst.semantic_storage.return_value = mock_storage
        mock_cache.return_value = cache_inst

        # First call (full filter) returns empty, second call (broad) returns match
        table_row = {
            "kind": "table",
            "name": "orders",
            "table_name": "orders",
            "catalog_name": "",
            "database_name": "other_db",
            "schema_name": "",
            "description": "Orders",
            "yaml_path": "",
        }

        call_count = [0]
        def mock_search_all(*args, **kwargs):
            call_count[0] += 1
            arrow = MagicMock()
            if call_count[0] == 1:
                arrow.to_pylist.return_value = []  # First: not found with full filters
            elif call_count[0] == 2:
                arrow.to_pylist.return_value = [table_row]  # Second: broad match found
            else:
                arrow.to_pylist.return_value = []
            return arrow

        mock_storage._search_all.side_effect = mock_search_all

        rag = SemanticModelRAG(mock_agent_config)
        result = rag.get_semantic_model(table_name="orders", database_name="my_db")
        assert result is not None
        assert result["table_name"] == "orders"


# ── SemanticModelStorage methods ──


class TestSemanticModelStorageMethods:
    def test_search_objects(self, mock_storage):
        """Test search_objects builds conditions correctly."""
        mock_storage.search.return_value = MagicMock()
        mock_storage.search.return_value.to_pylist.return_value = [{"name": "test"}]

        # Calling on actual instance needs full init, so test via mock
        storage = MagicMock(spec=SemanticModelStorage)
        storage.search_objects = SemanticModelStorage.search_objects.__get__(storage, SemanticModelStorage)
        storage.search.return_value = MagicMock()
        storage.search.return_value.to_pylist.return_value = [{"name": "col1"}]

        result = storage.search_objects("test query", kinds=["column"], table_name="orders", top_n=5)
        assert len(result) == 1
        storage.search.assert_called_once()

    def test_search_all(self, mock_storage):
        storage = MagicMock(spec=SemanticModelStorage)
        storage.search_all = SemanticModelStorage.search_all.__get__(storage, SemanticModelStorage)
        arrow = MagicMock()
        arrow.to_pylist.return_value = [{"kind": "table"}]
        storage._search_all.return_value = arrow

        result = storage.search_all()
        assert len(result) == 1
