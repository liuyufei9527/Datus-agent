"""Tests for datus/storage/reference_sql/init_utils.py."""

from unittest.mock import MagicMock

import pytest

from datus.storage.reference_sql.init_utils import exists_reference_sql, gen_reference_sql_id


class TestGenReferenceSqlId:
    def test_basic(self):
        result = gen_reference_sql_id("SELECT 1")
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex digest

    def test_deterministic(self):
        assert gen_reference_sql_id("SELECT 1") == gen_reference_sql_id("SELECT 1")

    def test_different_sql(self):
        assert gen_reference_sql_id("SELECT 1") != gen_reference_sql_id("SELECT 2")


class TestExistsReferenceSql:
    def test_overwrite_mode_returns_empty(self):
        mock_storage = MagicMock()
        result = exists_reference_sql(mock_storage, "overwrite")
        assert result == set()
        mock_storage.search_all_reference_sql.assert_not_called()

    def test_incremental_mode_returns_existing(self):
        mock_storage = MagicMock()
        mock_storage.search_all_reference_sql.return_value = [
            {"id": "abc123"},
            {"id": "def456"},
        ]
        result = exists_reference_sql(mock_storage, "incremental")
        assert result == {"abc123", "def456"}

    def test_unknown_mode_returns_empty(self):
        mock_storage = MagicMock()
        result = exists_reference_sql(mock_storage, "unknown")
        assert result == set()
