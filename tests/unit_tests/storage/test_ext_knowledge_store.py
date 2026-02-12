"""Tests for datus/storage/ext_knowledge/store.py utility functions"""

import pytest

from datus.storage.ext_knowledge.store import ExtKnowledgeRAG, gen_subject_item_id


class TestGenSubjectItemId:
    def test_basic_id(self):
        result = gen_subject_item_id(["Finance", "Revenue"], "annual_revenue")
        assert result == "Finance/Revenue/annual_revenue"

    def test_single_path(self):
        result = gen_subject_item_id(["Finance"], "revenue")
        assert result == "Finance/revenue"

    def test_empty_path(self):
        result = gen_subject_item_id([], "revenue")
        assert result == "revenue"

    def test_deep_path(self):
        result = gen_subject_item_id(["A", "B", "C", "D"], "item")
        assert result == "A/B/C/D/item"


class TestExtKnowledgeRAGParseSubjectPath:
    @pytest.fixture
    def rag(self):
        """Create an ExtKnowledgeRAG instance without calling __init__."""
        instance = object.__new__(ExtKnowledgeRAG)
        return instance

    def test_parse_string_path(self, rag):
        result = rag._parse_subject_path("Finance/Revenue/Q1")
        assert result == ["Finance", "Revenue", "Q1"]

    def test_parse_list_path(self, rag):
        result = rag._parse_subject_path(["Finance", "Revenue"])
        assert result == ["Finance", "Revenue"]

    def test_parse_invalid_type(self, rag):
        with pytest.raises(ValueError, match="must be string or list"):
            rag._parse_subject_path(123)

    def test_parse_string_with_spaces(self, rag):
        result = rag._parse_subject_path("Finance / Revenue / Q1")
        assert result == ["Finance", "Revenue", "Q1"]

    def test_parse_empty_string(self, rag):
        result = rag._parse_subject_path("")
        assert result == []

    def test_parse_single_segment(self, rag):
        result = rag._parse_subject_path("Finance")
        assert result == ["Finance"]
