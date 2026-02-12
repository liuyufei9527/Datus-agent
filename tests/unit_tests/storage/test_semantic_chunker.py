"""Tests for datus/storage/document/chunker/semantic_chunker.py"""

import pytest

from datus.storage.document.chunker.semantic_chunker import (
    ChunkingConfig,
    SemanticChunker,
)
from datus.storage.document.schemas import (
    FetchedDocument,
    ParsedDocument,
    ParsedSection,
    PlatformDocChunk,
)


@pytest.fixture
def chunker():
    config = ChunkingConfig(
        chunk_size=200,
        chunk_overlap=20,
        min_chunk_size=50,
        max_chunk_size=400,
    )
    return SemanticChunker(config=config)


@pytest.fixture
def base_metadata():
    return {
        "platform": "test",
        "version": "1.0",
        "source_type": "github",
        "source_url": "https://example.com/docs/page",
        "doc_path": "docs/page.md",
        "language": "en",
        "nav_path": [],
        "group_name": "",
    }


def _make_parsed_doc(sections, title="Test Doc", nav_path=None, group_name=""):
    return ParsedDocument(
        title=title,
        sections=sections,
        metadata={"nav_path": nav_path or [], "group_name": group_name},
        source_doc=FetchedDocument(
            platform="test",
            version="1.0",
            source_url="https://example.com",
            source_type="github",
            doc_path="docs/test.md",
            raw_content="raw",
            content_type="markdown",
        ),
    )


class TestChunkingConfig:
    def test_default_config(self):
        config = ChunkingConfig()
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 128
        assert config.min_chunk_size == 256
        assert config.max_chunk_size == 2048
        assert config.preserve_code_blocks is True
        assert config.preserve_paragraphs is True
        assert config.add_context_prefix is True

    def test_custom_config(self):
        config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50


class TestSemanticChunkerInit:
    def test_default_init(self):
        chunker = SemanticChunker()
        assert chunker.config.chunk_size == 1024

    def test_custom_init(self):
        config = ChunkingConfig(chunk_size=256)
        chunker = SemanticChunker(config=config)
        assert chunker.config.chunk_size == 256


class TestChunk:
    def test_single_small_section(self, chunker, base_metadata):
        sections = [ParsedSection(level=1, title="Title", content="Short content.", children=[])]
        doc = _make_parsed_doc(sections)
        chunks = chunker.chunk(doc, base_metadata)
        assert len(chunks) >= 1
        assert "Short content" in chunks[0].chunk_text

    def test_multiple_sections(self, chunker, base_metadata):
        sections = [
            ParsedSection(
                level=1,
                title="Title",
                content="Intro text.",
                children=[
                    ParsedSection(level=2, title="Section A", content="A " * 50, children=[]),
                    ParsedSection(level=2, title="Section B", content="B " * 50, children=[]),
                ],
            )
        ]
        doc = _make_parsed_doc(sections)
        chunks = chunker.chunk(doc, base_metadata)
        assert len(chunks) >= 1

    def test_chunk_indices_sequential(self, chunker, base_metadata):
        sections = [
            ParsedSection(
                level=1,
                title="Title",
                content="Content " * 100,
                children=[],
            )
        ]
        doc = _make_parsed_doc(sections)
        chunks = chunker.chunk(doc, base_metadata)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_ids_unique(self, chunker, base_metadata):
        sections = [
            ParsedSection(
                level=1,
                title="Title",
                content="Content " * 100,
                children=[],
            )
        ]
        doc = _make_parsed_doc(sections)
        chunks = chunker.chunk(doc, base_metadata)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_nav_path_in_metadata(self, chunker, base_metadata):
        sections = [ParsedSection(level=1, title="Title", content="Content.", children=[])]
        doc = _make_parsed_doc(sections, nav_path=["Guides", "SQL"])
        base_metadata["nav_path"] = ["Guides", "SQL"]
        base_metadata["group_name"] = "Guides"
        chunks = chunker.chunk(doc, base_metadata)
        assert len(chunks) >= 1
        assert chunks[0].nav_path == ["Guides", "SQL"]
        assert chunks[0].group_name == "Guides"

    def test_empty_sections(self, chunker, base_metadata):
        doc = _make_parsed_doc([])
        chunks = chunker.chunk(doc, base_metadata)
        assert len(chunks) == 0


class TestFlattenSectionContent:
    def test_flatten_simple(self):
        section = ParsedSection(
            level=1,
            title="Parent",
            content="Parent content.",
            children=[
                ParsedSection(level=2, title="Child", content="Child content.", children=[]),
            ],
        )
        result = SemanticChunker._flatten_section_content(section)
        assert "Parent content" in result
        assert "## Child" in result
        assert "Child content" in result

    def test_flatten_deep_nesting(self):
        section = ParsedSection(
            level=1,
            title="L1",
            content="L1 content.",
            children=[
                ParsedSection(
                    level=2,
                    title="L2",
                    content="L2 content.",
                    children=[
                        ParsedSection(level=3, title="L3", content="L3 content.", children=[]),
                    ],
                )
            ],
        )
        result = SemanticChunker._flatten_section_content(section)
        assert "L1 content" in result
        assert "L2 content" in result
        assert "L3 content" in result

    def test_flatten_empty_content(self):
        section = ParsedSection(level=1, title="Empty", content="", children=[])
        result = SemanticChunker._flatten_section_content(section)
        assert result == ""


class TestSplitContent:
    def test_short_content_single_chunk(self, chunker, base_metadata):
        chunks = chunker._split_content("Short text.", ["Title"], base_metadata, 0)
        assert len(chunks) == 1

    def test_long_content_multiple_chunks(self, chunker, base_metadata):
        long_text = "This is a paragraph. " * 50
        chunks = chunker._split_content(long_text, ["Title"], base_metadata, 0)
        assert len(chunks) >= 2

    def test_empty_content(self, chunker, base_metadata):
        chunks = chunker._split_content("", ["Title"], base_metadata, 0)
        assert len(chunks) == 0

    def test_whitespace_only_content(self, chunker, base_metadata):
        chunks = chunker._split_content("   \n\n  ", ["Title"], base_metadata, 0)
        assert len(chunks) == 0

    def test_code_block_preserved(self, chunker, base_metadata):
        content = "Intro.\n\n```sql\nSELECT * FROM users;\n```\n\nAfter."
        chunks = chunker._split_content(content, ["Title"], base_metadata, 0)
        all_text = " ".join(c.chunk_text for c in chunks)
        assert "```sql" in all_text


class TestSplitTextHierarchically:
    def test_short_text_unchanged(self, chunker):
        result = chunker._split_text_hierarchically("Short.", 100)
        assert result == ["Short."]

    def test_split_by_paragraph(self, chunker):
        text = "First paragraph.\n\nSecond paragraph."
        result = chunker._split_text_hierarchically(text, 20)
        assert len(result) >= 2

    def test_split_by_sentence(self, chunker):
        text = "First sentence. Second sentence. Third sentence."
        result = chunker._split_text_hierarchically(text, 25)
        assert len(result) >= 2

    def test_character_fallback(self, chunker):
        # A long string with no natural break points
        text = "a" * 100
        result = chunker._split_text_hierarchically(text, 30)
        assert all(len(piece) <= 30 for piece in result)


class TestSplitLargeParagraph:
    def test_splits_large_paragraph(self, chunker, base_metadata):
        large_para = "This is a sentence. " * 50
        chunks = chunker._split_large_paragraph(large_para, ["Title"], base_metadata, 0)
        assert len(chunks) >= 2

    def test_single_paragraph_fits(self, chunker, base_metadata):
        para = "Short paragraph."
        chunks = chunker._split_large_paragraph(para, ["Title"], base_metadata, 0)
        assert len(chunks) == 1


class TestCreateChunk:
    def test_basic_chunk_creation(self, chunker, base_metadata):
        chunk = chunker._create_chunk("Test content", ["Doc", "Section"], base_metadata, 0)
        assert isinstance(chunk, PlatformDocChunk)
        assert "Test content" in chunk.chunk_text
        assert chunk.title == "Section"
        assert chunk.titles == ["Doc", "Section"]
        assert chunk.chunk_index == 0

    def test_hierarchy_from_titles(self, chunker, base_metadata):
        chunk = chunker._create_chunk("Content", ["Doc", "Section", "Sub"], base_metadata, 0)
        assert "Doc > Section > Sub" in chunk.hierarchy

    def test_context_prefix_added(self, chunker, base_metadata):
        chunk = chunker._create_chunk("Content", ["Doc", "Section"], base_metadata, 0)
        assert chunk.chunk_text.startswith("[")

    def test_context_prefix_not_added_for_heading(self, chunker, base_metadata):
        chunk = chunker._create_chunk("# Heading\nContent", ["Doc"], base_metadata, 0)
        assert chunk.chunk_text.startswith("# Heading")

    def test_nav_path_in_hierarchy(self, chunker, base_metadata):
        base_metadata["nav_path"] = ["Guides", "SQL"]
        chunk = chunker._create_chunk("Content", ["Doc"], base_metadata, 0)
        assert "Guides" in chunk.hierarchy
        assert "SQL" in chunk.hierarchy

    def test_empty_titles(self, chunker, base_metadata):
        chunk = chunker._create_chunk("Content", [], base_metadata, 0)
        assert chunk.title == ""
        assert chunk.titles == []


class TestMergeSmallChunks:
    def test_merge_adjacent_small_chunks(self, chunker):
        chunks = [
            PlatformDocChunk(
                chunk_id="1",
                chunk_text="Short",
                chunk_index=0,
                title="T",
                titles=["T"],
                nav_path=[],
                group_name="",
                hierarchy="T",
                version="1.0",
                source_type="github",
                source_url="",
                doc_path="test.md",
            ),
            PlatformDocChunk(
                chunk_id="2",
                chunk_text="Also short",
                chunk_index=1,
                title="T",
                titles=["T"],
                nav_path=[],
                group_name="",
                hierarchy="T",
                version="1.0",
                source_type="github",
                source_url="",
                doc_path="test.md",
            ),
        ]
        result = chunker._merge_small_chunks(chunks)
        assert len(result) == 1
        assert "Short" in result[0].chunk_text
        assert "Also short" in result[0].chunk_text

    def test_no_merge_different_hierarchy(self, chunker):
        chunks = [
            PlatformDocChunk(
                chunk_id="1",
                chunk_text="Short",
                chunk_index=0,
                title="A",
                titles=["A"],
                nav_path=[],
                group_name="",
                hierarchy="A",
                version="1.0",
                source_type="github",
                source_url="",
                doc_path="test.md",
            ),
            PlatformDocChunk(
                chunk_id="2",
                chunk_text="Also short",
                chunk_index=1,
                title="B",
                titles=["B"],
                nav_path=[],
                group_name="",
                hierarchy="B",
                version="1.0",
                source_type="github",
                source_url="",
                doc_path="test.md",
            ),
        ]
        result = chunker._merge_small_chunks(chunks)
        assert len(result) == 2

    def test_empty_chunks_list(self, chunker):
        result = chunker._merge_small_chunks([])
        assert result == []


class TestPlatformDocChunkGenerateId:
    def test_deterministic_ids(self):
        id1 = PlatformDocChunk.generate_chunk_id("docs/page.md", 0, "1.0")
        id2 = PlatformDocChunk.generate_chunk_id("docs/page.md", 0, "1.0")
        assert id1 == id2

    def test_different_index_different_id(self):
        id1 = PlatformDocChunk.generate_chunk_id("docs/page.md", 0, "1.0")
        id2 = PlatformDocChunk.generate_chunk_id("docs/page.md", 1, "1.0")
        assert id1 != id2

    def test_different_path_different_id(self):
        id1 = PlatformDocChunk.generate_chunk_id("docs/a.md", 0, "1.0")
        id2 = PlatformDocChunk.generate_chunk_id("docs/b.md", 0, "1.0")
        assert id1 != id2
