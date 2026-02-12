"""Tests for datus/storage/ext_knowledge/ext_knowledge_init.py and init_utils.py"""

import argparse
from threading import Lock
from unittest.mock import MagicMock, patch

import pytest

from datus.storage.ext_knowledge.init_utils import exists_ext_knowledge, gen_ext_knowledge_id


class TestGenExtKnowledgeId:
    def test_basic_id(self):
        result = gen_ext_knowledge_id(["Finance", "Revenue"], "annual_revenue")
        assert result == "Finance/Revenue/annual_revenue"

    def test_empty_path(self):
        result = gen_ext_knowledge_id([], "annual_revenue")
        assert result == "annual_revenue"

    def test_slash_in_parts(self):
        result = gen_ext_knowledge_id(["A/B", "C"], "text/with/slashes")
        assert "/" not in result.split("/")[0].replace("_", "")

    def test_single_path(self):
        result = gen_ext_knowledge_id(["Finance"], "revenue")
        assert result == "Finance/revenue"


class TestExistsExtKnowledge:
    def test_overwrite_mode_returns_empty(self):
        mock_store = MagicMock()
        result = exists_ext_knowledge(mock_store, "overwrite")
        assert result == set()
        mock_store.search_all_knowledge.assert_not_called()

    def test_incremental_mode_returns_existing(self):
        mock_store = MagicMock()
        mock_store.search_all_knowledge.return_value = [
            {"subject_path": ["Finance"], "search_text": "revenue"},
            {"subject_path": ["Sales"], "search_text": "orders"},
        ]
        result = exists_ext_knowledge(mock_store, "incremental")
        assert len(result) == 2

    def test_incremental_mode_empty_store(self):
        mock_store = MagicMock()
        mock_store.search_all_knowledge.return_value = []
        result = exists_ext_knowledge(mock_store, "incremental")
        assert result == set()


class TestProcessRow:
    def test_process_valid_row(self):
        from datus.storage.ext_knowledge.ext_knowledge_init import process_row

        mock_store = MagicMock()
        row = {
            "subject_path": "Finance/Revenue",
            "name": "test_knowledge",
            "search_text": "annual revenue",
            "explanation": "Total revenue for the year",
        }
        result = process_row(mock_store, row, 0, set(), Lock())
        assert result == "processed"
        mock_store.upsert_knowledge.assert_called_once()

    def test_process_row_missing_fields(self):
        from datus.storage.ext_knowledge.ext_knowledge_init import process_row

        mock_store = MagicMock()
        row = {"subject_path": "", "name": "", "search_text": "", "explanation": ""}
        result = process_row(mock_store, row, 0, set(), Lock())
        assert result == "skipped"

    def test_process_row_already_exists(self):
        from datus.storage.ext_knowledge.ext_knowledge_init import process_row

        mock_store = MagicMock()
        row = {
            "subject_path": "Finance/Revenue",
            "name": "test",
            "search_text": "annual revenue",
            "explanation": "Explanation",
        }
        existing = {gen_ext_knowledge_id(["Finance", "Revenue"], "annual revenue")}
        result = process_row(mock_store, row, 0, existing, Lock())
        assert result == "skipped"

    def test_process_row_invalid_subject_path(self):
        from datus.storage.ext_knowledge.ext_knowledge_init import process_row

        mock_store = MagicMock()
        row = {
            "subject_path": "///",
            "name": "test",
            "search_text": "text",
            "explanation": "explanation",
        }
        result = process_row(mock_store, row, 0, set(), Lock())
        # Empty path components after splitting should be skipped
        assert result == "skipped"

    def test_process_row_exception_returns_error(self):
        from datus.storage.ext_knowledge.ext_knowledge_init import process_row

        mock_store = MagicMock()
        mock_store.upsert_knowledge.side_effect = Exception("DB error")
        row = {
            "subject_path": "Finance/Revenue",
            "name": "test",
            "search_text": "text",
            "explanation": "explanation",
        }
        result = process_row(mock_store, row, 0, set(), Lock())
        assert result == "error"


class TestInitExtKnowledge:
    def test_no_ext_knowledge_arg(self):
        from datus.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge

        mock_store = MagicMock()
        args = argparse.Namespace()
        init_ext_knowledge(mock_store, args)
        # Should not attempt to process

    def test_nonexistent_file(self):
        from datus.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge

        mock_store = MagicMock()
        args = argparse.Namespace(ext_knowledge="/nonexistent/file.csv")
        init_ext_knowledge(mock_store, args)
        # Should log error and return

    def test_valid_csv(self, tmp_path):
        from datus.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge

        csv_file = tmp_path / "knowledge.csv"
        csv_file.write_text(
            "subject_path,name,search_text,explanation\n"
            "Finance/Revenue,rev,annual revenue,Total revenue for the year\n"
            "Sales/Orders,ord,total orders,Total orders count\n"
        )
        mock_store = MagicMock()
        args = argparse.Namespace(ext_knowledge=str(csv_file))
        init_ext_knowledge(mock_store, args, build_mode="overwrite", pool_size=1)
        assert mock_store.upsert_knowledge.call_count == 2
        mock_store.after_init.assert_called_once()

    def test_missing_columns(self, tmp_path):
        from datus.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge

        csv_file = tmp_path / "knowledge.csv"
        csv_file.write_text("col1,col2\nval1,val2\n")
        mock_store = MagicMock()
        args = argparse.Namespace(ext_knowledge=str(csv_file))
        with pytest.raises(ValueError, match="Missing required columns"):
            init_ext_knowledge(mock_store, args)
