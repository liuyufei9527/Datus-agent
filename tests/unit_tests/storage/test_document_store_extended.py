"""Extended tests for datus/storage/document/store.py — DocumentStore additional coverage."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.document.store import DocumentStore, get_platform_doc_schema, _SAFE_IDENTIFIER_RE


# ── get_platform_doc_schema ──


class TestGetPlatformDocSchema:
    def test_default_dim(self):
        schema = get_platform_doc_schema()
        assert "chunk_id" in [f.name for f in schema]
        vec_field = [f for f in schema if f.name == "vector"][0]
        # Check list_size == 384
        assert vec_field.type.list_size == 384

    def test_custom_dim(self):
        schema = get_platform_doc_schema(embedding_dim=1024)
        vec_field = [f for f in schema if f.name == "vector"][0]
        assert vec_field.type.list_size == 1024


# ── _validate_identifier ──


class TestValidateIdentifier:
    def test_valid(self):
        DocumentStore._validate_identifier("v1.2.3", "version")

    def test_valid_with_hyphens(self):
        DocumentStore._validate_identifier("my-version_2.0", "version")

    def test_invalid_chars(self):
        with pytest.raises(ValueError, match="Invalid version"):
            DocumentStore._validate_identifier("v1; DROP TABLE", "version")

    def test_empty_fails(self):
        with pytest.raises(ValueError):
            DocumentStore._validate_identifier("", "version")


# ── DocumentStore methods ──


class TestDocumentStore:
    def _make_mock_store(self):
        store = MagicMock(spec=DocumentStore)
        store.table = MagicMock()
        store._table_initialized = True
        return store

    def test_store_chunks_empty(self):
        store = MagicMock(spec=DocumentStore)
        store.store_chunks = DocumentStore.store_chunks.__get__(store, DocumentStore)
        result = store.store_chunks([])
        assert result == 0

    def test_store_chunks_with_data(self):
        store = MagicMock(spec=DocumentStore)
        store.store_chunks = DocumentStore.store_chunks.__get__(store, DocumentStore)
        store._table_initialized = True
        store.table = MagicMock()
        store.table.count_rows.return_value = 0

        chunk = MagicMock()
        chunk.chunk_id = "c1"
        chunk.version = "v1"
        chunk.to_dict.return_value = {"chunk_id": "c1", "chunk_text": "hello"}

        result = store.store_chunks([chunk])
        assert result == 1
        store.store_batch.assert_called_once()

    def test_store_chunks_deduplication(self):
        store = MagicMock(spec=DocumentStore)
        store.store_chunks = DocumentStore.store_chunks.__get__(store, DocumentStore)
        store._table_initialized = True
        store.table = MagicMock()
        store.table.count_rows.return_value = 5

        chunk = MagicMock()
        chunk.chunk_id = "c1"
        chunk.version = "v1"
        chunk.to_dict.return_value = {"chunk_id": "c1", "chunk_text": "hello"}

        result = store.store_chunks([chunk])
        assert result == 1
        # Should have called delete for existing chunks
        store.table.delete.assert_called()

    def test_search_docs_no_version(self):
        store = MagicMock(spec=DocumentStore)
        store.search_docs = DocumentStore.search_docs.__get__(store, DocumentStore)
        mock_result = MagicMock()
        mock_result.to_pylist.return_value = [{"chunk_text": "data"}]
        store.search.return_value = mock_result

        results = store.search_docs("test query")
        assert len(results) == 1

    def test_search_docs_with_version(self):
        store = MagicMock(spec=DocumentStore)
        store.search_docs = DocumentStore.search_docs.__get__(store, DocumentStore)
        mock_result = MagicMock()
        mock_result.to_pylist.return_value = []
        store.search.return_value = mock_result

        results = store.search_docs("test query", version="v1.0")
        assert len(results) == 0
        store.search.assert_called_once()

    def test_list_versions(self):
        store = MagicMock(spec=DocumentStore)
        store.list_versions = DocumentStore.list_versions.__get__(store, DocumentStore)

        arrow_mock = MagicMock()
        arrow_mock.to_pylist.return_value = [
            {"version": "v1"},
            {"version": "v1"},
            {"version": "v2"},
        ]
        store._search_all.return_value = arrow_mock

        versions = store.list_versions()
        assert len(versions) == 2
        v_map = {v["version"]: v["chunk_count"] for v in versions}
        assert v_map["v1"] == 2
        assert v_map["v2"] == 1

    def test_get_stats_empty(self):
        store = MagicMock(spec=DocumentStore)
        store.get_stats = DocumentStore.get_stats.__get__(store, DocumentStore)

        arrow_mock = MagicMock()
        arrow_mock.to_pylist.return_value = []
        store._search_all.return_value = arrow_mock

        stats = store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["versions"] == []

    def test_get_stats_with_data(self):
        store = MagicMock(spec=DocumentStore)
        store.get_stats = DocumentStore.get_stats.__get__(store, DocumentStore)

        arrow_mock = MagicMock()
        arrow_mock.to_pylist.return_value = [
            {"version": "v1", "doc_path": "/a.md", "created_at": "2024-01-01"},
            {"version": "v1", "doc_path": "/b.md", "created_at": "2024-01-02"},
            {"version": "v2", "doc_path": "/a.md", "created_at": "2024-01-03"},
        ]
        store._search_all.return_value = arrow_mock

        stats = store.get_stats()
        assert stats["total_chunks"] == 3
        assert len(stats["versions"]) == 2
        assert stats["doc_count"] == 2
        assert stats["latest_update"] == "2024-01-03"

    def test_delete_docs_empty_table(self):
        store = MagicMock(spec=DocumentStore)
        store.delete_docs = DocumentStore.delete_docs.__get__(store, DocumentStore)
        store.table = MagicMock()
        store.table.count_rows.return_value = 0

        result = store.delete_docs(version="v1")
        assert result == 0

    def test_delete_docs_by_version(self):
        store = MagicMock(spec=DocumentStore)
        store.delete_docs = DocumentStore.delete_docs.__get__(store, DocumentStore)
        store._validate_identifier = DocumentStore._validate_identifier
        store.table = MagicMock()
        store.table.count_rows.side_effect = [10, 7]  # before, after

        result = store.delete_docs(version="v1.0")
        assert result == 3
        store.table.delete.assert_called_once()

    def test_delete_docs_all(self):
        store = MagicMock(spec=DocumentStore)
        store.delete_docs = DocumentStore.delete_docs.__get__(store, DocumentStore)
        store.table = MagicMock()
        store.table.count_rows.return_value = 5
        store._table_initialized = True

        with patch("datus.storage.document.store.shutil") as mock_shutil, patch(
            "datus.storage.document.store.lancedb"
        ) as mock_lancedb:
            store.db_path = "/tmp/test_db"
            mock_lancedb.connect.return_value = MagicMock()
            result = store.delete_docs()
            assert result == 5
            mock_shutil.rmtree.assert_called_once()

    def test_get_all_rows(self):
        store = MagicMock(spec=DocumentStore)
        store.get_all_rows = DocumentStore.get_all_rows.__get__(store, DocumentStore)

        arrow_mock = MagicMock()
        arrow_mock.to_pylist.return_value = [{"chunk_id": "c1"}]
        store._search_all.return_value = arrow_mock

        result = store.get_all_rows()
        assert len(result) == 1

    def test_create_indices(self):
        store = MagicMock(spec=DocumentStore)
        store.create_indices = DocumentStore.create_indices.__get__(store, DocumentStore)
        store.TABLE_NAME = "document"
        store.create_indices()
        store.create_vector_index.assert_called_once()
        store.create_fts_index.assert_called_once()
