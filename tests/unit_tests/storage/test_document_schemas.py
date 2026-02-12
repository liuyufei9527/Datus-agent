"""Tests for datus/storage/document/schemas.py"""

import pytest

from datus.storage.document.schemas import (
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_MARKDOWN,
    CONTENT_TYPE_RST,
    SOURCE_TYPE_GITHUB,
    SOURCE_TYPE_LOCAL,
    SOURCE_TYPE_WEBSITE,
    FetchedDocument,
    ParsedDocument,
    ParsedSection,
    PlatformDocChunk,
)


class TestFetchedDocument:
    def test_create_basic(self):
        doc = FetchedDocument(
            platform="test",
            version="1.0",
            source_url="https://example.com",
            source_type=SOURCE_TYPE_GITHUB,
            doc_path="docs/guide.md",
            raw_content="# Guide\n\nContent.",
            content_type=CONTENT_TYPE_MARKDOWN,
        )
        assert doc.platform == "test"
        assert doc.version == "1.0"
        assert doc.content_type == CONTENT_TYPE_MARKDOWN

    def test_metadata_default_empty(self):
        doc = FetchedDocument(
            platform="test",
            version="1.0",
            source_url="https://example.com",
            source_type=SOURCE_TYPE_LOCAL,
            doc_path="doc.md",
            raw_content="Content",
            content_type=CONTENT_TYPE_MARKDOWN,
        )
        assert doc.metadata == {} or doc.metadata is None

    def test_fetch_timestamp_auto(self):
        doc = FetchedDocument(
            platform="test",
            version="1.0",
            source_url="https://example.com",
            source_type=SOURCE_TYPE_LOCAL,
            doc_path="doc.md",
            raw_content="Content",
            content_type=CONTENT_TYPE_MARKDOWN,
        )
        assert doc.fetch_timestamp is not None


class TestParsedSection:
    def test_basic_section(self):
        section = ParsedSection(level=1, title="Title", content="Content.", children=[])
        assert section.level == 1
        assert section.title == "Title"
        assert section.content == "Content."
        assert section.children == []

    def test_nested_sections(self):
        child = ParsedSection(level=2, title="Child", content="Child content.", children=[])
        parent = ParsedSection(level=1, title="Parent", content="Parent content.", children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].title == "Child"

    def test_get_all_content(self):
        child = ParsedSection(level=2, title="Child", content="Child content.", children=[])
        parent = ParsedSection(level=1, title="Parent", content="Parent content.", children=[child])
        all_content = parent.get_all_content()
        assert "Parent content" in all_content
        assert "Child content" in all_content


class TestParsedDocument:
    def test_create_basic(self):
        source_doc = FetchedDocument(
            platform="test",
            version="1.0",
            source_url="https://example.com",
            source_type="github",
            doc_path="doc.md",
            raw_content="raw",
            content_type="markdown",
        )
        doc = ParsedDocument(
            title="Test",
            sections=[ParsedSection(level=1, title="Title", content="Content", children=[])],
            metadata={},
            source_doc=source_doc,
        )
        assert doc.title == "Test"
        assert len(doc.sections) == 1

    def test_get_section_titles(self):
        source_doc = FetchedDocument(
            platform="test",
            version="1.0",
            source_url="https://example.com",
            source_type="github",
            doc_path="doc.md",
            raw_content="raw",
            content_type="markdown",
        )
        child = ParsedSection(level=2, title="Sub", content="", children=[])
        parent = ParsedSection(level=1, title="Main", content="", children=[child])
        doc = ParsedDocument(
            title="Doc",
            sections=[parent],
            metadata={},
            source_doc=source_doc,
        )
        titles = doc.get_section_titles()
        assert "Main" in titles
        assert "Sub" in titles


class TestPlatformDocChunk:
    def test_create_chunk(self):
        chunk = PlatformDocChunk(
            chunk_id="id-1",
            chunk_text="Some text",
            chunk_index=0,
            title="Title",
            titles=["Doc", "Title"],
            nav_path=["Guides", "SQL"],
            group_name="Guides",
            hierarchy="Guides > SQL > Title",
            version="1.0",
            source_type="github",
            source_url="https://example.com",
            doc_path="docs/page.md",
        )
        assert chunk.chunk_id == "id-1"
        assert chunk.chunk_index == 0
        assert chunk.nav_path == ["Guides", "SQL"]

    def test_generate_chunk_id(self):
        id1 = PlatformDocChunk.generate_chunk_id("docs/page.md", 0, "1.0")
        id2 = PlatformDocChunk.generate_chunk_id("docs/page.md", 0, "1.0")
        assert id1 == id2

    def test_to_dict(self):
        chunk = PlatformDocChunk(
            chunk_id="id-1",
            chunk_text="Some text",
            chunk_index=0,
            title="Title",
            titles=["Title"],
            nav_path=[],
            group_name="",
            hierarchy="Title",
            version="1.0",
            source_type="github",
            source_url="https://example.com",
            doc_path="docs/page.md",
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "id-1"
        assert d["chunk_text"] == "Some text"
        assert d["version"] == "1.0"
        assert "content_hash" in d


class TestConstants:
    def test_content_types(self):
        assert CONTENT_TYPE_MARKDOWN == "markdown"
        assert CONTENT_TYPE_HTML == "html"
        assert CONTENT_TYPE_RST == "rst"

    def test_source_types(self):
        assert SOURCE_TYPE_GITHUB == "github"
        assert SOURCE_TYPE_LOCAL == "local"
        assert SOURCE_TYPE_WEBSITE == "website"
