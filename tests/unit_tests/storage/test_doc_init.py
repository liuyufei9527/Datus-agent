"""Tests for datus/storage/document/doc_init.py"""

from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from datus.storage.document.doc_init import (
    InitResult,
    _delete_existing_versions,
    _detect_versions_from_paths,
    _make_empty_result,
    import_documents,
)


class TestInitResult:
    def test_init_result_fields(self):
        result = InitResult(
            platform="test",
            version="1.0",
            source="https://example.com",
            total_docs=10,
            total_chunks=50,
            success=True,
            errors=[],
            duration_seconds=1.5,
        )
        assert result.platform == "test"
        assert result.version == "1.0"
        assert result.total_docs == 10
        assert result.total_chunks == 50
        assert result.success is True
        assert result.errors == []


class TestDetectVersionsFromPaths:
    def test_version_paths(self):
        paths = ["1.3.0/docs/intro.md", "1.3.0/docs/guide.md"]
        versions = _detect_versions_from_paths(paths)
        assert "1.3.0" in versions

    def test_multiple_versions(self):
        paths = ["1.3.0/docs/intro.md", "1.2.0/docs/intro.md"]
        versions = _detect_versions_from_paths(paths)
        assert "1.3.0" in versions
        assert "1.2.0" in versions

    def test_non_version_paths(self):
        paths = ["docs/intro.md", "README.md"]
        versions = _detect_versions_from_paths(paths)
        assert len(versions) == 0

    def test_mixed_paths(self):
        # If not ALL paths start with version, return empty
        paths = ["1.3.0/docs/intro.md", "README.md"]
        versions = _detect_versions_from_paths(paths)
        assert len(versions) == 0

    def test_empty_paths(self):
        versions = _detect_versions_from_paths([])
        assert len(versions) == 0

    def test_v_prefixed_versions(self):
        paths = ["v2.0.0/docs/intro.md", "v2.0.0/docs/guide.md"]
        versions = _detect_versions_from_paths(paths)
        assert "2.0.0" in versions

    def test_beta_version(self):
        paths = ["1.2.3-beta/docs/intro.md", "1.2.3-beta/docs/guide.md"]
        versions = _detect_versions_from_paths(paths)
        assert "1.2.3-beta" in versions


class TestMakeEmptyResult:
    def test_basic_empty_result(self):
        from datetime import datetime, timezone

        start_time = datetime.now(timezone.utc)
        result = _make_empty_result("test", "1.0", "https://example.com", start_time)
        assert result.platform == "test"
        assert result.version == "1.0"
        assert result.total_docs == 0
        assert result.total_chunks == 0
        assert result.success is True
        assert result.errors == []

    def test_empty_result_with_errors(self):
        from datetime import datetime, timezone

        start_time = datetime.now(timezone.utc)
        result = _make_empty_result("test", "", "src", start_time, errors=["No docs"])
        assert result.version == "unknown"
        assert result.errors == ["No docs"]


class TestDeleteExistingVersions:
    def test_single_version(self):
        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 5
        _delete_existing_versions(mock_store, "1.0", set())
        mock_store.delete_docs.assert_called_once_with(version="1.0")

    def test_multi_version(self):
        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 3
        _delete_existing_versions(mock_store, "1.0", {"1.0", "2.0"})
        assert mock_store.delete_docs.call_count == 2

    def test_no_delete_needed(self):
        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 0
        _delete_existing_versions(mock_store, "1.0", set())
        mock_store.delete_docs.assert_called_once()


class TestImportDocuments:
    def test_nonexistent_directory(self, tmp_path):
        mock_store = MagicMock()
        chunks, titles = import_documents(mock_store, str(tmp_path / "nonexistent"))
        assert chunks == 0
        assert titles == []

    def test_empty_directory(self, tmp_path):
        mock_store = MagicMock()
        chunks, titles = import_documents(mock_store, str(tmp_path))
        assert chunks == 0
        assert titles == []

    def test_import_markdown_files(self, tmp_path):
        (tmp_path / "doc1.md").write_text("# Doc 1\n\nContent 1.")
        (tmp_path / "doc2.md").write_text("# Doc 2\n\nContent 2.")

        mock_store = MagicMock()
        mock_store.store_chunks = MagicMock()
        mock_store.create_indices = MagicMock()

        chunks, titles = import_documents(mock_store, str(tmp_path), platform="test", version="1.0")
        assert chunks >= 1
        assert len(titles) == 2

    def test_import_recursive(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "top.md").write_text("# Top\n\nContent.")
        (subdir / "nested.md").write_text("# Nested\n\nContent.")

        mock_store = MagicMock()
        mock_store.store_chunks = MagicMock()
        mock_store.create_indices = MagicMock()

        chunks, titles = import_documents(mock_store, str(tmp_path), recursive=True)
        assert chunks >= 2
        assert len(titles) == 2

    def test_import_handles_exception(self, tmp_path):
        (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
        mock_store = MagicMock()
        mock_store.store_chunks = MagicMock(side_effect=Exception("Store error"))

        chunks, titles = import_documents(mock_store, str(tmp_path))
        # Even if storing fails, the function handles it
        assert isinstance(chunks, int)
