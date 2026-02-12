# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for search_tool.py – 22.2% coverage target."""

import os
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.doc_search_node_models import DocSearchInput
from datus.tools.search_tools.search_tool import SearchTool, search_by_tavily


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.document_storage_path.return_value = "/nonexistent/path"
    return config


@pytest.fixture
def search_tool(mock_agent_config):
    return SearchTool(agent_config=mock_agent_config)


# ---------------------------------------------------------------------------
# _version_sort_key
# ---------------------------------------------------------------------------


class TestVersionSortKey:
    def test_semantic_version(self):
        key = SearchTool._version_sort_key
        versions = ["v3.9", "v3.10", "v3.2"]
        sorted_versions = sorted(versions, key=key)
        assert sorted_versions == ["v3.2", "v3.9", "v3.10"]

    def test_numeric_prefix(self):
        key = SearchTool._version_sort_key
        assert key("1.0") < key("2.0")
        assert key("1.2") < key("1.10")


# ---------------------------------------------------------------------------
# _normalize_list_field
# ---------------------------------------------------------------------------


class TestNormalizeListField:
    def test_list(self):
        assert SearchTool._normalize_list_field(["a", "b"]) == ["a", "b"]

    def test_string(self):
        assert SearchTool._normalize_list_field("a > b > c") == ["a", "b", "c"]

    def test_empty(self):
        assert SearchTool._normalize_list_field("") == []
        assert SearchTool._normalize_list_field(None) == []


# ---------------------------------------------------------------------------
# _resolve_latest_version
# ---------------------------------------------------------------------------


class TestResolveLatestVersion:
    def test_with_versions(self):
        store = MagicMock()
        store.list_versions.return_value = [
            {"version": "v1.0"},
            {"version": "v2.0"},
            {"version": "v1.10"},
        ]
        result = SearchTool._resolve_latest_version(store)
        assert result == "v2.0"

    def test_empty(self):
        store = MagicMock()
        store.list_versions.return_value = []
        assert SearchTool._resolve_latest_version(store) is None


# ---------------------------------------------------------------------------
# _get_document_store
# ---------------------------------------------------------------------------


class TestGetDocumentStore:
    def test_path_not_exists(self, search_tool, mock_agent_config):
        mock_agent_config.document_storage_path.return_value = "/nonexistent"
        store = search_tool._get_document_store("snowflake")
        assert store is None

    def test_path_exists(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        with patch("datus.tools.search_tools.search_tool.document_store") as mock_ds:
            mock_ds.return_value = MagicMock()
            store = search_tool._get_document_store("snowflake")
            assert store is not None


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_execute_delegates(self, search_tool):
        inp = DocSearchInput(platform="snowflake", keywords=["CREATE TABLE"])
        with patch.object(search_tool, "search_document") as mock_search:
            mock_search.return_value = MagicMock(success=True)
            result = search_tool.execute(inp)
            mock_search.assert_called_once()


# ---------------------------------------------------------------------------
# list_document_nav
# ---------------------------------------------------------------------------


class TestListDocumentNav:
    def test_no_store(self, search_tool):
        result = search_tool.list_document_nav("snowflake")
        assert result.success is True
        assert result.total_docs == 0

    def test_no_rows(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        mock_store = MagicMock()
        mock_store.list_versions.return_value = [{"version": "v1.0"}]
        mock_store.get_all_rows.return_value = []
        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            result = search_tool.list_document_nav("snowflake")
            assert result.success is True
            assert result.total_docs == 0

    def test_with_rows(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        mock_store = MagicMock()
        mock_store.list_versions.return_value = [{"version": "v1.0"}]
        mock_store.get_all_rows.return_value = [
            {
                "doc_path": "/doc1",
                "version": "v1.0",
                "title": "CREATE TABLE",
                "nav_path": ["DDL"],
            },
            {
                "doc_path": "/doc2",
                "version": "v1.0",
                "title": "ALTER TABLE",
                "nav_path": ["DDL"],
            },
        ]
        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            result = search_tool.list_document_nav("snowflake", version="v1.0")
            assert result.success is True
            assert result.total_docs == 2

    def test_error_handling(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        with patch("datus.tools.search_tools.search_tool.document_store", side_effect=Exception("fail")):
            result = search_tool.list_document_nav("snowflake")
            assert result.success is False


# ---------------------------------------------------------------------------
# _build_nav_tree
# ---------------------------------------------------------------------------


class TestBuildNavTree:
    def test_basic(self, search_tool):
        doc_map = {
            "/a": {"nav_path": ["DDL"], "title": "CREATE TABLE"},
            "/b": {"nav_path": ["DDL"], "title": "ALTER TABLE"},
            "/c": {"nav_path": ["DML"], "title": "INSERT"},
        }
        tree = search_tool._build_nav_tree(doc_map)
        assert len(tree) == 2  # DDL and DML
        names = {n["name"] for n in tree}
        assert "DDL" in names
        assert "DML" in names

    def test_empty(self, search_tool):
        tree = search_tool._build_nav_tree({})
        assert tree == []

    def test_no_nav_path(self, search_tool):
        doc_map = {"/a": {"nav_path": [], "title": "Root Doc"}}
        tree = search_tool._build_nav_tree(doc_map)
        assert len(tree) == 1
        assert tree[0]["name"] == "Root Doc"

    def test_title_same_as_nav_path(self, search_tool):
        doc_map = {"/a": {"nav_path": ["DDL"], "title": "DDL"}}
        tree = search_tool._build_nav_tree(doc_map)
        # Title same as last nav segment => should not duplicate
        assert len(tree) == 1
        assert tree[0]["name"] == "DDL"
        assert len(tree[0]["children"]) == 0

    def test_string_nav_path(self, search_tool):
        doc_map = {"/a": {"nav_path": "SQL Reference > DDL", "title": "CREATE TABLE"}}
        tree = search_tool._build_nav_tree(doc_map)
        assert len(tree) == 1
        assert tree[0]["name"] == "SQL Reference"


# ---------------------------------------------------------------------------
# get_document
# ---------------------------------------------------------------------------


class TestGetDocument:
    def test_empty_platform(self, search_tool):
        result = search_tool.get_document(platform="", titles=["DDL"])
        assert result.success is False

    def test_empty_titles(self, search_tool):
        result = search_tool.get_document(platform="snowflake", titles=[])
        assert result.success is False

    def test_no_store(self, search_tool):
        result = search_tool.get_document(platform="snowflake", titles=["DDL", "CREATE TABLE"])
        assert result.success is True
        assert result.chunk_count == 0

    def test_with_results(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        mock_store = MagicMock()
        mock_store.list_versions.return_value = [{"version": "v1.0"}]
        mock_store.get_all_rows.return_value = [
            {
                "doc_path": "/doc1",
                "chunk_id": "c1",
                "chunk_index": 0,
                "chunk_text": "Hello",
                "title": "CREATE TABLE",
                "hierarchy": "DDL > CREATE TABLE",
                "version": "v1.0",
            },
            {
                "doc_path": "/doc1",
                "chunk_id": "c2",
                "chunk_index": 1,
                "chunk_text": "World",
                "title": "CREATE TABLE",
                "hierarchy": "DDL > CREATE TABLE",
                "version": "v1.0",
            },
        ]
        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            result = search_tool.get_document("snowflake", ["DDL", "CREATE TABLE"])
            assert result.success is True
            assert result.chunk_count == 2

    def test_error(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        with patch("datus.tools.search_tools.search_tool.document_store", side_effect=Exception("fail")):
            result = search_tool.get_document("snowflake", ["DDL"])
            assert result.success is False


# ---------------------------------------------------------------------------
# search_document
# ---------------------------------------------------------------------------


class TestSearchDocument:
    def test_empty_platform(self, search_tool):
        result = search_tool.search_document(platform="", keywords=["test"])
        assert result.success is False

    def test_empty_keywords(self, search_tool):
        result = search_tool.search_document(platform="snowflake", keywords=[])
        assert result.success is False

    def test_no_store(self, search_tool):
        result = search_tool.search_document(platform="snowflake", keywords=["CREATE TABLE"])
        assert result.success is True
        assert result.doc_count == 0

    def test_with_results(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        mock_store = MagicMock()
        mock_store.list_versions.return_value = [{"version": "v1.0"}]
        mock_store.search_docs.return_value = [
            {"chunk_id": "c1", "chunk_text": "Hello"},
        ]
        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            result = search_tool.search_document("snowflake", ["DDL"])
            assert result.success is True
            assert result.doc_count == 1

    def test_search_error_per_keyword(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        mock_store = MagicMock()
        mock_store.list_versions.return_value = [{"version": "v1.0"}]
        mock_store.search_docs.side_effect = Exception("search fail")
        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            result = search_tool.search_document("snowflake", ["DDL"])
            assert result.success is True
            assert result.doc_count == 0

    def test_error(self, search_tool, mock_agent_config, tmp_path):
        mock_agent_config.document_storage_path.return_value = str(tmp_path)
        with patch("datus.tools.search_tools.search_tool.document_store", side_effect=Exception("fail")):
            result = search_tool.search_document("snowflake", ["DDL"])
            assert result.success is False


# ---------------------------------------------------------------------------
# search_by_tavily
# ---------------------------------------------------------------------------


class TestSearchByTavily:
    def test_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = search_by_tavily(keywords=["test"], api_key=None)
            assert result.success is False
            assert "TAVILY_API_KEY" in result.error

    def test_empty_keywords(self):
        result = search_by_tavily(keywords=[], api_key="test-key")
        assert result.success is True
        assert result.doc_count == 0

    def test_success(self):
        import requests as real_requests

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"content": "Result text", "raw_content": "Raw result"}],
            "answer": "Summary answer",
        }
        mock_response.raise_for_status.return_value = None

        with patch.object(real_requests, "post", return_value=mock_response):
            result = search_by_tavily(keywords=["test"], api_key="key")
            assert result.success is True
            assert result.doc_count == 2  # 1 result + 1 answer

    def test_http_error(self):
        import requests as real_requests

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"
        mock_response.raise_for_status.side_effect = real_requests.HTTPError(response=mock_response)

        with patch.object(real_requests, "post", return_value=mock_response):
            result = search_by_tavily(keywords=["test"], api_key="key")
            assert result.success is False

    def test_generic_error(self):
        import requests as real_requests

        with patch.object(real_requests, "post", side_effect=Exception("network error")):
            result = search_by_tavily(keywords=["test"], api_key="key")
            assert result.success is False
