"""Tests for datus/storage/document/cleaner/doc_cleaner.py"""

import pytest

from datus.storage.document.cleaner.doc_cleaner import DocumentCleaner, clean_document, clean_text
from datus.storage.document.schemas import FetchedDocument


def _make_doc(content: str, content_type: str = "markdown"):
    return FetchedDocument(
        platform="test",
        version="1.0",
        source_url="https://example.com/doc",
        source_type="github",
        doc_path="docs/test.md",
        raw_content=content,
        content_type=content_type,
    )


class TestDocumentCleanerInit:
    def test_default_init(self):
        cleaner = DocumentCleaner()
        assert cleaner.normalize_unicode is True
        assert cleaner.remove_control_chars is True
        assert cleaner.normalize_whitespace is True
        assert cleaner.preserve_code_blocks is True

    def test_custom_init(self):
        cleaner = DocumentCleaner(
            normalize_unicode=False,
            remove_control_chars=False,
            normalize_whitespace=False,
            preserve_code_blocks=False,
        )
        assert cleaner.normalize_unicode is False
        assert cleaner.remove_control_chars is False


class TestClean:
    def test_clean_markdown(self):
        cleaner = DocumentCleaner()
        doc = _make_doc("# Title\n\nContent.")
        result = cleaner.clean(doc)
        assert result.raw_content == "# Title\n\nContent."
        assert result.content_type == "markdown"
        assert result.platform == "test"

    def test_clean_html(self):
        cleaner = DocumentCleaner()
        doc = _make_doc("<html><script>alert('hi')</script><body>Content</body></html>", content_type="html")
        result = cleaner.clean(doc)
        assert "<script>" not in result.raw_content
        assert "Content" in result.raw_content

    def test_clean_preserves_metadata(self):
        cleaner = DocumentCleaner()
        doc = _make_doc("Content")
        doc.metadata = {"key": "value"}
        result = cleaner.clean(doc)
        assert result.metadata == {"key": "value"}


class TestCleanText:
    def test_unicode_normalization(self):
        cleaner = DocumentCleaner()
        # Composed vs decomposed forms
        result = cleaner.clean_text("caf\u0065\u0301")  # e + combining accent
        assert len(result) > 0

    def test_remove_control_chars(self):
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("Hello\x00World\x01Test")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "HelloWorldTest" in result

    def test_preserve_newlines_and_tabs(self):
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("Line1\nLine2\tTabbed")
        assert "\n" in result
        assert "\t" in result

    def test_normalize_whitespace_multiple_blank_lines(self):
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("Line1\n\n\n\n\nLine2")
        assert result == "Line1\n\nLine2"

    def test_normalize_whitespace_windows_line_endings(self):
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("Line1\r\nLine2\rLine3")
        assert "\r" not in result
        assert "Line1\nLine2\nLine3" == result

    def test_preserve_code_blocks(self):
        cleaner = DocumentCleaner()
        content = "Text\n\n```python\ncode   with   spaces\n```\n\nMore text"
        result = cleaner.clean_text(content)
        assert "code   with   spaces" in result

    def test_no_code_block_preservation(self):
        cleaner = DocumentCleaner(preserve_code_blocks=False)
        content = "Text\n\n```python\ncode\n```\n\nMore text"
        result = cleaner.clean_text(content)
        assert "code" in result

    def test_trailing_whitespace_removed(self):
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("Line1   \nLine2  ")
        assert "Line1\nLine2" == result

    def test_empty_string(self):
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("")
        assert result == ""


class TestCleanHtml:
    def test_removes_script_tags(self):
        cleaner = DocumentCleaner()
        result = cleaner._clean_html("<script>var x=1;</script>Content")
        assert "var x=1" not in result
        assert "Content" in result

    def test_removes_style_tags(self):
        cleaner = DocumentCleaner()
        result = cleaner._clean_html("<style>.foo{color:red}</style>Content")
        assert "color:red" not in result
        assert "Content" in result

    def test_removes_html_comments(self):
        cleaner = DocumentCleaner()
        result = cleaner._clean_html("<!-- comment -->Content")
        assert "comment" not in result
        assert "Content" in result

    def test_removes_noscript(self):
        cleaner = DocumentCleaner()
        result = cleaner._clean_html("<noscript>Enable JS</noscript>Content")
        assert "Enable JS" not in result


class TestRemoveControlChars:
    def test_removes_null(self):
        cleaner = DocumentCleaner()
        result = cleaner._remove_control_chars("Hello\x00World")
        assert result == "HelloWorld"

    def test_removes_bel(self):
        cleaner = DocumentCleaner()
        result = cleaner._remove_control_chars("Hello\x07World")
        assert result == "HelloWorld"

    def test_removes_delete(self):
        cleaner = DocumentCleaner()
        result = cleaner._remove_control_chars("Hello\x7fWorld")
        assert result == "HelloWorld"

    def test_preserves_tab_newline(self):
        cleaner = DocumentCleaner()
        result = cleaner._remove_control_chars("Hello\t\nWorld")
        assert result == "Hello\t\nWorld"


class TestNormalizeWhitespace:
    def test_windows_line_endings(self):
        cleaner = DocumentCleaner()
        result = cleaner._normalize_whitespace("a\r\nb\rc")
        assert result == "a\nb\nc"

    def test_collapse_blank_lines(self):
        cleaner = DocumentCleaner()
        result = cleaner._normalize_whitespace("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_trailing_whitespace(self):
        cleaner = DocumentCleaner()
        result = cleaner._normalize_whitespace("a   \nb  \nc")
        assert result == "a\nb\nc"


class TestConvenienceFunctions:
    def test_clean_document(self):
        doc = _make_doc("# Title\n\nContent.")
        result = clean_document(doc)
        assert result.raw_content == "# Title\n\nContent."

    def test_clean_text(self):
        result = clean_text("Hello\x00World")
        assert "HelloWorld" in result
