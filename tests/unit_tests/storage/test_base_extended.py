"""Extended tests for datus/storage/base.py — covers uncovered lines in StorageBase, BaseEmbeddingStore."""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pyarrow as pa
import pytest

from datus.utils.exceptions import DatusException


# ── StorageBase ──


class TestStorageBase:
    @patch("datus.storage.base.lancedb")
    def test_init(self, mock_lancedb, tmp_path):
        from datus.storage.base import StorageBase

        db_path = str(tmp_path / "test.lancedb")
        mock_lancedb.connect.return_value = MagicMock()
        base = StorageBase(db_path)
        assert base.db_path == db_path
        mock_lancedb.connect.assert_called_once_with(db_path)

    @patch("datus.storage.base.lancedb")
    def test_ensure_success_story_table_exists(self, mock_lancedb, tmp_path):
        from datus.storage.base import StorageBase

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["success_story"]
        mock_lancedb.connect.return_value = mock_db

        base = StorageBase(str(tmp_path))
        base._ensure_success_story_table()
        mock_db.create_table.assert_not_called()

    @patch("datus.storage.base.lancedb")
    def test_ensure_success_story_table_creates(self, mock_lancedb, tmp_path):
        from datus.storage.base import StorageBase

        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db

        base = StorageBase(str(tmp_path))
        base._ensure_success_story_table()
        mock_db.create_table.assert_called_once_with("success_story", schema=pytest.approx(pa.schema([]), abs=None))

    @patch("datus.storage.base.lancedb")
    def test_ensure_success_story_table_error(self, mock_lancedb, tmp_path):
        from datus.storage.base import StorageBase

        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_db.create_table.side_effect = RuntimeError("fail")
        mock_lancedb.connect.return_value = mock_db

        base = StorageBase(str(tmp_path))
        with pytest.raises(DatusException):
            base._ensure_success_story_table()

    @patch("datus.storage.base.lancedb")
    def test_get_current_timestamp(self, mock_lancedb, tmp_path):
        from datus.storage.base import StorageBase

        mock_lancedb.connect.return_value = MagicMock()
        base = StorageBase(str(tmp_path))
        ts = base._get_current_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts  # ISO format


# ── BaseEmbeddingStore._check_embedding_model_ready ──


class TestCheckEmbeddingModelReady:
    def _make_store(self):
        from datus.storage.base import BaseEmbeddingStore

        with patch("datus.storage.base.lancedb"):
            store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
            store.db_path = "/tmp/test"
            store.db = MagicMock()
            store.model = MagicMock()
            store.batch_size = 100
            store.table_name = "test"
            store.vector_source_name = "definition"
            store.vector_column_name = "vector"
            store.on_duplicate_columns = "vector"
            store._schema = None
            store.table = None
            store._table_initialized = False
            store._table_lock = MagicMock()
            store._write_lock = MagicMock()
            return store

    def test_model_already_failed(self):
        store = self._make_store()
        store.model.is_model_failed = True
        store.model.model_error_message = "download failed"
        with pytest.raises(DatusException):
            store._check_embedding_model_ready()

    def test_model_init_datus_exception(self):
        store = self._make_store()
        store.model.is_model_failed = False
        store.model.model = PropertyMock(side_effect=DatusException(message="init fail"))
        type(store.model).model = property(lambda self: (_ for _ in ()).throw(DatusException(message="init fail")))
        with pytest.raises(DatusException):
            store._check_embedding_model_ready()

    def test_model_init_other_exception(self):
        store = self._make_store()
        store.model.is_model_failed = False
        type(store.model).model = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(DatusException):
            store._check_embedding_model_ready()

    def test_model_ready(self):
        store = self._make_store()
        store.model.is_model_failed = False
        store.model.model = MagicMock()
        # Should not raise
        store._check_embedding_model_ready()


# ── BaseEmbeddingStore._ensure_table ──


class TestEnsureTable:
    def _make_store(self):
        from datus.storage.base import BaseEmbeddingStore

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db_path = "/tmp/test"
        store.db = MagicMock()
        store.model = MagicMock()
        store.model.model = MagicMock()
        store.batch_size = 100
        store.table_name = "test"
        store.vector_source_name = "definition"
        store.vector_column_name = "vector"
        store.on_duplicate_columns = "vector"
        store._schema = None
        store.table = None
        store._table_initialized = False
        store._table_lock = MagicMock()
        store._write_lock = MagicMock()
        return store

    def test_table_exists(self):
        store = self._make_store()
        mock_table = MagicMock()
        store.db.table_names.return_value = ["test"]
        store.db.open_table.return_value = mock_table

        store._ensure_table()
        assert store.table is mock_table
        store.db.open_table.assert_called_once_with("test")

    def test_table_not_exists_creates(self):
        store = self._make_store()
        mock_table = MagicMock()
        store.db.table_names.return_value = []
        store.db.create_table.return_value = mock_table

        store._ensure_table()
        assert store.table is mock_table
        store.db.create_table.assert_called_once()

    def test_create_table_fails(self):
        store = self._make_store()
        store.db.table_names.return_value = []
        store.db.create_table.side_effect = RuntimeError("create fail")

        with pytest.raises(DatusException):
            store._ensure_table()


# ── BaseEmbeddingStore.store / store_batch ──


class TestStoreOperations:
    def _make_ready_store(self):
        from datus.storage.base import BaseEmbeddingStore
        from threading import Lock

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db_path = "/tmp/test"
        store.db = MagicMock()
        store.model = MagicMock()
        store.model.is_model_failed = False
        store.model.model = MagicMock()
        store.batch_size = 2
        store.table_name = "test"
        store.vector_source_name = "definition"
        store.vector_column_name = "vector"
        store.on_duplicate_columns = "vector"
        store._schema = None
        store.table = MagicMock()
        store._table_initialized = True
        store._table_lock = Lock()
        store._write_lock = Lock()
        return store

    def test_store_batch_empty(self):
        store = self._make_ready_store()
        store.store_batch([])
        store.table.add.assert_not_called()

    def test_store_batch_single_batch(self):
        store = self._make_ready_store()
        data = [{"col": "a"}, {"col": "b"}]
        store.store_batch(data)
        store.table.add.assert_called_once()

    def test_store_batch_multi_batch(self):
        store = self._make_ready_store()
        data = [{"col": "a"}, {"col": "b"}, {"col": "c"}]
        store.store_batch(data)
        assert store.table.add.call_count == 2  # batch_size=2, so 2+1

    def test_store_exception(self):
        store = self._make_ready_store()
        store.table.add.side_effect = RuntimeError("write fail")
        with pytest.raises(DatusException):
            store.store([{"col": "a"}])


# ── BaseEmbeddingStore._add_with_retry ──


class TestAddWithRetry:
    def _make_ready_store(self):
        from datus.storage.base import BaseEmbeddingStore
        from threading import Lock

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db = MagicMock()
        store.table = MagicMock()
        store.table_name = "test"
        store._table_initialized = True
        store._write_lock = Lock()
        return store

    def test_no_table(self):
        store = self._make_ready_store()
        store.table = None
        with pytest.raises(DatusException):
            store._add_with_retry(pd.DataFrame([{"a": 1}]))

    def test_success_first_try(self):
        store = self._make_ready_store()
        store._add_with_retry(pd.DataFrame([{"a": 1}]))
        store.table.add.assert_called_once()

    def test_commit_conflict_retry(self):
        store = self._make_ready_store()
        store.table.add.side_effect = [Exception("Commit conflict"), None]
        store.db.open_table.return_value = store.table
        store._add_with_retry(pd.DataFrame([{"a": 1}]), max_attempts=2, initial_delay=0.001)
        assert store.table.add.call_count == 2

    def test_non_commit_conflict_raises_immediately(self):
        store = self._make_ready_store()
        store.table.add.side_effect = RuntimeError("disk full")
        with pytest.raises(RuntimeError, match="disk full"):
            store._add_with_retry(pd.DataFrame([{"a": 1}]))

    def test_all_retries_exhausted(self):
        store = self._make_ready_store()
        store.table.add.side_effect = Exception("Commit conflict")
        store.db.open_table.return_value = store.table
        with pytest.raises(Exception, match="Commit conflict"):
            store._add_with_retry(pd.DataFrame([{"a": 1}]), max_attempts=2, initial_delay=0.001)


# ── BaseEmbeddingStore._upsert_with_retry ──


class TestUpsertWithRetry:
    def _make_ready_store(self):
        from datus.storage.base import BaseEmbeddingStore
        from threading import Lock

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db = MagicMock()
        store.table = MagicMock()
        store.table_name = "test"
        store._table_initialized = True
        store._write_lock = Lock()
        return store

    def test_no_table(self):
        store = self._make_ready_store()
        store.table = None
        with pytest.raises(DatusException):
            store._upsert_with_retry(pd.DataFrame([{"a": 1}]), "id")

    def test_success_first_try(self):
        store = self._make_ready_store()
        mock_chain = MagicMock()
        store.table.merge_insert.return_value = mock_chain
        mock_chain.when_matched_update_all.return_value = mock_chain
        mock_chain.when_not_matched_insert_all.return_value = mock_chain

        store._upsert_with_retry(pd.DataFrame([{"id": "1", "val": "a"}]), "id")
        store.table.merge_insert.assert_called_once_with("id")

    def test_commit_conflict_retry(self):
        store = self._make_ready_store()
        mock_chain = MagicMock()
        store.table.merge_insert.return_value = mock_chain
        mock_chain.when_matched_update_all.return_value = mock_chain
        mock_chain.when_not_matched_insert_all.return_value = mock_chain
        mock_chain.execute.side_effect = [Exception("Commit conflict"), None]
        store.db.open_table.return_value = store.table

        store._upsert_with_retry(pd.DataFrame([{"id": "1"}]), "id", max_attempts=2, initial_delay=0.001)
        assert mock_chain.execute.call_count == 2

    def test_non_commit_conflict_raises(self):
        store = self._make_ready_store()
        mock_chain = MagicMock()
        store.table.merge_insert.return_value = mock_chain
        mock_chain.when_matched_update_all.return_value = mock_chain
        mock_chain.when_not_matched_insert_all.return_value = mock_chain
        mock_chain.execute.side_effect = RuntimeError("disk error")

        with pytest.raises(RuntimeError, match="disk error"):
            store._upsert_with_retry(pd.DataFrame([{"id": "1"}]), "id")


# ── BaseEmbeddingStore.upsert_batch ──


class TestUpsertBatch:
    def _make_ready_store(self):
        from datus.storage.base import BaseEmbeddingStore
        from threading import Lock

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db = MagicMock()
        store.model = MagicMock()
        store.model.is_model_failed = False
        store.model.model = MagicMock()
        store.batch_size = 2
        store.table = MagicMock()
        store.table_name = "test"
        store._table_initialized = True
        store._table_lock = Lock()
        store._write_lock = Lock()
        store.vector_source_name = "definition"
        store.vector_column_name = "vector"
        store.on_duplicate_columns = "vector"
        store._schema = None
        store.db_path = "/tmp/test"
        return store

    def test_empty_data(self):
        store = self._make_ready_store()
        store.upsert_batch([])
        store.table.merge_insert.assert_not_called()

    def test_deduplicates_input(self):
        store = self._make_ready_store()
        mock_chain = MagicMock()
        store.table.merge_insert.return_value = mock_chain
        mock_chain.when_matched_update_all.return_value = mock_chain
        mock_chain.when_not_matched_insert_all.return_value = mock_chain

        data = [
            {"id": "1", "val": "a"},
            {"id": "1", "val": "b"},  # duplicate, should keep last
            {"id": "2", "val": "c"},
        ]
        store.upsert_batch(data, on_column="id")
        # Should have been called (data after dedup = 2 items, batch_size=2, so 1 call)
        store.table.merge_insert.assert_called()


# ── BaseEmbeddingStore.search ──


class TestSearch:
    def _make_ready_store(self):
        from datus.storage.base import BaseEmbeddingStore
        from threading import Lock

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db = MagicMock()
        store.model = MagicMock()
        store.model.is_model_failed = False
        store.model.model = MagicMock()
        store.batch_size = 100
        store.table = MagicMock()
        store.table_name = "test"
        store._table_initialized = True
        store._table_lock = Lock()
        store._write_lock = Lock()
        store.vector_source_name = "definition"
        store.vector_column_name = "vector"
        store.on_duplicate_columns = "vector"
        store._schema = None
        store.db_path = "/tmp/test"
        return store

    def test_search_vector(self):
        store = self._make_ready_store()
        mock_builder = MagicMock()
        store.table.search.return_value = mock_builder
        mock_builder.where.return_value = mock_builder
        mock_builder.select.return_value = mock_builder
        mock_builder.limit.return_value = mock_builder

        mock_result = MagicMock()
        mock_result.column_names = ["vector", "name"]
        mock_result.drop.return_value = mock_result
        mock_builder.to_arrow.return_value = mock_result
        store.table.count_rows.return_value = 5

        result = store.search("test query", top_n=5)
        assert result is mock_result

    def test_search_vector_error(self):
        store = self._make_ready_store()
        store.table.search.side_effect = RuntimeError("search fail")

        with pytest.raises(DatusException):
            store.search("test query")


# ── BaseEmbeddingStore.update ──


class TestUpdate:
    def _make_ready_store(self):
        from datus.storage.base import BaseEmbeddingStore
        from threading import Lock

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db = MagicMock()
        store.model = MagicMock()
        store.model.is_model_failed = False
        store.model.model = MagicMock()
        store.batch_size = 100
        store.table = MagicMock()
        store.table_name = "test"
        store._table_initialized = True
        store._table_lock = Lock()
        store._write_lock = Lock()
        store.vector_source_name = "definition"
        store.vector_column_name = "vector"
        store.on_duplicate_columns = "vector"
        store._schema = None
        store.db_path = "/tmp/test"
        return store

    def test_empty_update_values(self):
        from datus.storage.lancedb_conditions import eq

        store = self._make_ready_store()
        store.update(eq("name", "test"), {})
        store.table.update.assert_not_called()

    def test_no_where_clause(self):
        store = self._make_ready_store()
        store.update(None, {"name": "new"})
        store.table.update.assert_not_called()

    def test_successful_update(self):
        from datus.storage.lancedb_conditions import eq

        store = self._make_ready_store()
        store.update(eq("name", "old"), {"name": "new"})
        store.table.update.assert_called_once()

    def test_unique_filter_conflict(self):
        from datus.storage.lancedb_conditions import eq

        store = self._make_ready_store()
        store.table.count_rows.return_value = 1  # conflict exists

        with pytest.raises(DatusException):
            store.update(eq("name", "old"), {"name": "new"}, unique_filter=eq("name", "new"))


# ── BaseEmbeddingStore.create_vector_index ──


class TestCreateVectorIndex:
    def _make_ready_store(self):
        from datus.storage.base import BaseEmbeddingStore
        from threading import Lock

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db = MagicMock()
        store.model = MagicMock()
        store.model.is_model_failed = False
        store.model.model = MagicMock()
        store.model.dim_size = 384
        store.model.device = None
        store.batch_size = 100
        store.table = MagicMock()
        store.table_name = "test"
        store._table_initialized = True
        store._table_lock = Lock()
        store._write_lock = Lock()
        store.vector_source_name = "definition"
        store.vector_column_name = "vector"
        store.on_duplicate_columns = "vector"
        store._schema = None
        store.db_path = "/tmp/test"
        return store

    def test_small_dataset_ivf_flat(self):
        store = self._make_ready_store()
        store.table.count_rows.return_value = 100
        store.create_vector_index()
        store.table.create_index.assert_called_once()
        call_kwargs = store.table.create_index.call_args[1]
        assert call_kwargs["index_type"] == "IVF_FLAT"

    def test_large_dataset_ivf_pq(self):
        store = self._make_ready_store()
        store.table.count_rows.return_value = 10000
        store.create_vector_index()
        call_kwargs = store.table.create_index.call_args[1]
        assert call_kwargs["index_type"] == "IVF_PQ"
        assert "num_sub_vectors" in call_kwargs

    def test_medium_dataset(self):
        store = self._make_ready_store()
        store.table.count_rows.return_value = 2000
        store.create_vector_index()
        call_kwargs = store.table.create_index.call_args[1]
        assert call_kwargs["index_type"] == "IVF_FLAT"

    def test_with_cuda_accelerator(self):
        store = self._make_ready_store()
        store.model.device = "cuda"
        store.table.count_rows.return_value = 100
        store.create_vector_index()
        call_kwargs = store.table.create_index.call_args[1]
        assert call_kwargs.get("accelerator") == "cuda"

    def test_index_creation_failure_logged(self):
        store = self._make_ready_store()
        store.table.count_rows.side_effect = RuntimeError("fail")
        # Should not raise
        store.create_vector_index()


# ── BaseEmbeddingStore.create_fts_index ──


class TestCreateFtsIndex:
    def _make_ready_store(self):
        from datus.storage.base import BaseEmbeddingStore
        from threading import Lock

        store = BaseEmbeddingStore.__new__(BaseEmbeddingStore)
        store.db = MagicMock()
        store.model = MagicMock()
        store.model.is_model_failed = False
        store.model.model = MagicMock()
        store.batch_size = 100
        store.table = MagicMock()
        store.table_name = "test"
        store._table_initialized = True
        store._table_lock = Lock()
        store._write_lock = Lock()
        store.vector_source_name = "definition"
        store.vector_column_name = "vector"
        store.on_duplicate_columns = "vector"
        store._schema = None
        store.db_path = "/tmp/test"
        return store

    def test_success(self):
        store = self._make_ready_store()
        store.create_fts_index(["name", "description"])
        store.table.create_fts_index.assert_called_once()

    def test_failure_logged(self):
        store = self._make_ready_store()
        store.table.create_fts_index.side_effect = RuntimeError("fts fail")
        # Should not raise
        store.create_fts_index(["name"])
