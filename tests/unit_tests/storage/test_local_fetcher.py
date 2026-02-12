"""Tests for datus/storage/document/fetcher/local_fetcher.py"""

import pytest

from datus.storage.document.fetcher.local_fetcher import LocalFetcher
from datus.storage.document.schemas import CONTENT_TYPE_HTML, CONTENT_TYPE_MARKDOWN, SOURCE_TYPE_LOCAL


@pytest.fixture
def fetcher():
    return LocalFetcher(platform="test", version="1.0")


class TestLocalFetcherInit:
    def test_basic_init(self):
        fetcher = LocalFetcher(platform="test")
        assert fetcher.platform == "test"
        assert fetcher.version is None

    def test_init_with_version(self):
        fetcher = LocalFetcher(platform="test", version="2.0")
        assert fetcher.version == "2.0"


class TestFetch:
    def test_fetch_markdown_files(self, fetcher, tmp_path):
        (tmp_path / "doc1.md").write_text("# Doc 1\n\nContent 1.")
        (tmp_path / "doc2.md").write_text("# Doc 2\n\nContent 2.")
        docs = fetcher.fetch(str(tmp_path))
        assert len(docs) == 2

    def test_fetch_html_files(self, fetcher, tmp_path):
        (tmp_path / "page.html").write_text("<html><body><p>Content</p></body></html>")
        docs = fetcher.fetch(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].content_type == CONTENT_TYPE_HTML

    def test_fetch_recursive(self, fetcher, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "top.md").write_text("# Top\n\nTop content.")
        (subdir / "nested.md").write_text("# Nested\n\nNested content.")
        docs = fetcher.fetch(str(tmp_path), recursive=True)
        assert len(docs) == 2

    def test_fetch_non_recursive(self, fetcher, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "top.md").write_text("# Top\n\nTop content.")
        (subdir / "nested.md").write_text("# Nested\n\nNested content.")
        docs = fetcher.fetch(str(tmp_path), recursive=False)
        assert len(docs) == 1

    def test_fetch_skips_non_doc_files(self, fetcher, tmp_path):
        (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.json").write_text('{"key": "value"}')
        docs = fetcher.fetch(str(tmp_path))
        assert len(docs) == 1

    def test_fetch_skips_empty_files(self, fetcher, tmp_path):
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "nonempty.md").write_text("# Content\n\nText.")
        docs = fetcher.fetch(str(tmp_path))
        assert len(docs) == 1

    def test_fetch_nonexistent_directory(self, fetcher):
        docs = fetcher.fetch("/nonexistent/path")
        assert docs == []

    def test_fetch_not_a_directory(self, fetcher, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        docs = fetcher.fetch(str(file_path))
        assert docs == []

    def test_fetch_include_patterns(self, fetcher, tmp_path):
        (tmp_path / "guide.md").write_text("# Guide\n\nGuide content.")
        (tmp_path / "readme.md").write_text("# README\n\nReadme content.")
        docs = fetcher.fetch(str(tmp_path), include_patterns=["*guide*"])
        assert len(docs) == 1
        assert "guide" in docs[0].doc_path.lower()

    def test_fetch_exclude_patterns(self, fetcher, tmp_path):
        (tmp_path / "guide.md").write_text("# Guide\n\nGuide content.")
        (tmp_path / "readme.md").write_text("# README\n\nReadme content.")
        docs = fetcher.fetch(str(tmp_path), exclude_patterns=["*readme*"])
        assert len(docs) == 1

    def test_fetch_sets_source_type(self, fetcher, tmp_path):
        (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
        docs = fetcher.fetch(str(tmp_path))
        assert docs[0].source_type == SOURCE_TYPE_LOCAL

    def test_fetch_sets_version(self, fetcher, tmp_path):
        (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
        docs = fetcher.fetch(str(tmp_path))
        assert docs[0].version == "1.0"

    def test_fetch_auto_version(self, tmp_path):
        fetcher = LocalFetcher(platform="test")  # No version specified
        (tmp_path / "doc.md").write_text("# Doc\n\nContent.")
        docs = fetcher.fetch(str(tmp_path))
        # Version should be set to current date
        assert len(docs[0].version) >= 8


class TestFetchSingle:
    def test_fetch_single_file(self, fetcher, tmp_path):
        file_path = tmp_path / "doc.md"
        file_path.write_text("# Single Doc\n\nContent.")
        doc = fetcher.fetch_single(str(file_path))
        assert doc is not None
        assert doc.content_type == CONTENT_TYPE_MARKDOWN
        assert "# Single Doc" in doc.raw_content

    def test_fetch_single_nonexistent(self, fetcher, tmp_path):
        doc = fetcher.fetch_single(str(tmp_path / "nonexistent.md"))
        assert doc is None

    def test_fetch_single_not_a_file(self, fetcher, tmp_path):
        doc = fetcher.fetch_single(str(tmp_path))
        assert doc is None

    def test_fetch_single_with_base_url(self, fetcher, tmp_path):
        file_path = tmp_path / "docs" / "guide.md"
        file_path.parent.mkdir()
        file_path.write_text("# Guide\n\nContent.")
        doc = fetcher.fetch_single(str(file_path), base_url=str(tmp_path))
        assert doc is not None
        assert doc.doc_path == "docs/guide.md"


class TestFetchFile:
    def test_metadata_populated(self, fetcher, tmp_path):
        file_path = tmp_path / "doc.md"
        file_path.write_text("# Doc\n\nContent.")
        doc = fetcher._fetch_file(file_path, tmp_path, "1.0")
        assert doc is not None
        assert "file_name" in doc.metadata
        assert "file_size" in doc.metadata
        assert "last_modified" in doc.metadata
        assert "absolute_path" in doc.metadata

    def test_source_url_is_file_uri(self, fetcher, tmp_path):
        file_path = tmp_path / "doc.md"
        file_path.write_text("# Doc\n\nContent.")
        doc = fetcher._fetch_file(file_path, tmp_path, "1.0")
        assert doc.source_url.startswith("file://")


class TestExtractTitle:
    def test_markdown_heading(self, fetcher):
        title = fetcher._extract_title("# My Title\n\nContent.", "doc.md", "markdown")
        assert title == "My Title"

    def test_markdown_setext_heading(self, fetcher):
        title = fetcher._extract_title("My Title\n=======\n\nContent.", "doc.md", "markdown")
        assert title == "My Title"

    def test_rst_heading(self, fetcher):
        title = fetcher._extract_title("My Title\n--------\n\nContent.", "doc.rst", "rst")
        assert title == "My Title"

    def test_html_title_tag(self, fetcher):
        title = fetcher._extract_title("<html><head><title>My Title</title></head></html>", "page.html", "html")
        assert title == "My Title"

    def test_html_h1_tag(self, fetcher):
        title = fetcher._extract_title("<html><body><h1>My Title</h1></body></html>", "page.html", "html")
        assert title == "My Title"

    def test_fallback_to_filename(self, fetcher):
        title = fetcher._extract_title("Just plain text.", "my-document.md", "markdown")
        assert "My Document" in title


class TestDetectContentType:
    def test_markdown_extension(self, fetcher):
        assert fetcher._detect_content_type("file.md", "") == "markdown"
        assert fetcher._detect_content_type("file.markdown", "") == "markdown"

    def test_html_extension(self, fetcher):
        assert fetcher._detect_content_type("file.html", "") == "html"
        assert fetcher._detect_content_type("file.htm", "") == "html"

    def test_rst_extension(self, fetcher):
        assert fetcher._detect_content_type("file.rst", "") == "rst"

    def test_html_from_content(self, fetcher):
        assert fetcher._detect_content_type("file.txt", "<!DOCTYPE html>") == "html"
        assert fetcher._detect_content_type("file.txt", "<html>") == "html"

    def test_default_to_markdown(self, fetcher):
        assert fetcher._detect_content_type("file.txt", "Just text.") == "markdown"


class TestIsDocFile:
    def test_doc_extensions(self, fetcher):
        assert fetcher._is_doc_file("file.md") is True
        assert fetcher._is_doc_file("file.rst") is True
        assert fetcher._is_doc_file("file.html") is True
        assert fetcher._is_doc_file("file.htm") is True
        assert fetcher._is_doc_file("file.txt") is True

    def test_non_doc_extensions(self, fetcher):
        assert fetcher._is_doc_file("file.py") is False
        assert fetcher._is_doc_file("file.json") is False
        assert fetcher._is_doc_file("file.png") is False

    def test_special_files(self, fetcher):
        assert fetcher._is_doc_file("README") is True
        assert fetcher._is_doc_file("CHANGELOG") is True
        assert fetcher._is_doc_file("CONTRIBUTING") is True
