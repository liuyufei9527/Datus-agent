"""Tests for datus/storage/document/parser/metadata_extractor.py"""

import pytest

from datus.storage.document.parser.metadata_extractor import MetadataExtractor
from datus.storage.document.schemas import FetchedDocument, ParsedDocument, ParsedSection


@pytest.fixture
def extractor():
    return MetadataExtractor()


def _make_parsed_doc(title="Test", content="", sections=None, metadata=None, version="1.0"):
    source_doc = FetchedDocument(
        platform="test",
        version=version,
        source_url="https://example.com",
        source_type="github",
        doc_path="docs/test.md",
        raw_content="raw",
        content_type="markdown",
    )
    if sections is None:
        sections = [ParsedSection(level=1, title=title, content=content, children=[])]
    return ParsedDocument(
        title=title,
        sections=sections,
        metadata=metadata or {},
        source_doc=source_doc,
    )


class TestExtract:
    def test_extract_basic_metadata(self, extractor):
        doc = _make_parsed_doc(title="SQL Guide", content="This is about SELECT and WHERE queries.")
        metadata = extractor.extract(doc, platform="snowflake")
        assert "keywords" in metadata
        assert "language" in metadata
        assert "word_count" in metadata

    def test_extract_version_from_source_doc(self, extractor):
        doc = _make_parsed_doc(version="2.0")
        metadata = extractor.extract(doc)
        assert metadata["version"] == "2.0"

    def test_extract_has_code_blocks(self, extractor):
        doc = _make_parsed_doc(content="Here is code:\n```sql\nSELECT 1;\n```")
        metadata = extractor.extract(doc)
        assert metadata["has_code_blocks"] is True

    def test_extract_no_code_blocks(self, extractor):
        doc = _make_parsed_doc(content="No code here.")
        metadata = extractor.extract(doc)
        assert metadata["has_code_blocks"] is False

    def test_extract_has_tables(self, extractor):
        doc = _make_parsed_doc(content="| Col1 | Col2 |\n|---|---|")
        metadata = extractor.extract(doc)
        assert metadata["has_tables"] is True


class TestExtractKeywords:
    def test_extract_sql_keywords(self, extractor):
        # Keywords require count >= 2, so repeat keywords
        text = (
            "Use SELECT to query data. The WHERE clause filters results. "
            "Then SELECT again with another WHERE clause. JOIN tables together. "
            "We can JOIN multiple tables."
        )
        keywords = extractor.extract_keywords(text)
        assert any(k in keywords for k in ["select", "where", "join"])

    def test_extract_platform_keywords(self, extractor):
        text = "Snowflake warehouse stage pipe stream task snowpipe. " * 3
        keywords = extractor.extract_keywords(text, platform="snowflake")
        assert any(k in keywords for k in ["warehouse", "stage", "pipe", "stream"])

    def test_max_keywords_limit(self, extractor):
        text = "select where join table index view function procedure column key " * 5
        keywords = extractor.extract_keywords(text, max_keywords=5)
        assert len(keywords) <= 5

    def test_keywords_require_minimum_frequency(self, extractor):
        text = "select"  # Only appears once
        keywords = extractor.extract_keywords(text)
        # Keywords need count >= 2
        assert "select" not in keywords


class TestDetectVersion:
    def test_detect_semantic_version(self, extractor):
        version = extractor._detect_version("This is version v1.2.3 of the documentation.")
        assert version == "1.2.3"

    def test_detect_date_version(self, extractor):
        version = extractor._detect_version("Released 2024-01-15.")
        assert version == "2024-01-15"

    def test_detect_major_version(self, extractor):
        version = extractor._detect_version("Documentation for version 15")
        assert version is not None

    def test_no_version_found(self, extractor):
        version = extractor._detect_version("No version information here.")
        assert version is None


class TestDetectLanguage:
    def test_english_text(self, extractor):
        lang = extractor._detect_language("This is English text about SQL databases.")
        assert lang == "en"

    def test_chinese_text(self, extractor):
        lang = extractor._detect_language("这是一个关于数据库查询的中文文档。数据库是非常重要的。")
        assert lang == "zh"

    def test_russian_text(self, extractor):
        lang = extractor._detect_language("Это документация на русском языке о базах данных. " * 10)
        assert lang == "ru"


class TestFindCompoundTerms:
    def test_find_create_table(self, extractor):
        terms = extractor._find_compound_terms("use create table to define schemas")
        assert any("create" in t and "table" in t for t in terms)

    def test_find_primary_key(self, extractor):
        terms = extractor._find_compound_terms("every table should have a primary key")
        assert any("primary" in t and "key" in t for t in terms)

    def test_find_group_by(self, extractor):
        terms = extractor._find_compound_terms("use group by to aggregate data")
        assert any("group" in t and "by" in t for t in terms)

    def test_no_compound_terms(self, extractor):
        terms = extractor._find_compound_terms("nothing special here")
        assert len(terms) == 0


class TestGetFullText:
    def test_collects_all_content(self, extractor):
        sections = [
            ParsedSection(
                level=1,
                title="Title",
                content="Title content.",
                children=[
                    ParsedSection(level=2, title="Child", content="Child content.", children=[]),
                ],
            )
        ]
        doc = _make_parsed_doc(title="Doc", sections=sections)
        text = extractor._get_full_text(doc)
        assert "Doc" in text
        assert "Title" in text
        assert "Title content" in text
        assert "Child" in text
        assert "Child content" in text
