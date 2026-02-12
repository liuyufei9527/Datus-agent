"""Tests for datus/storage/document/parser/markdown_parser.py"""

import pytest

from datus.storage.document.parser.markdown_parser import MarkdownParser
from datus.storage.document.schemas import FetchedDocument


@pytest.fixture
def parser():
    return MarkdownParser()


def _make_doc(content: str, doc_path: str = "docs/page.md"):
    return FetchedDocument(
        platform="test",
        version="1.0",
        source_url="https://example.com/" + doc_path,
        source_type="github",
        doc_path=doc_path,
        raw_content=content,
        content_type="markdown",
    )


class TestMarkdownParserInit:
    def test_init(self):
        parser = MarkdownParser()
        assert parser._md is not None


class TestParse:
    def test_parse_basic_markdown(self, parser):
        content = "# Hello World\n\nSome content here."
        result = parser.parse(_make_doc(content))
        assert result.title == "Hello World"
        assert len(result.sections) >= 1

    def test_parse_with_frontmatter(self, parser):
        content = """---
title: My Page
description: A test page
---

# Content

Some text.
"""
        result = parser.parse(_make_doc(content))
        assert result.title == "My Page"
        assert result.metadata.get("description") == "A test page"

    def test_parse_title_from_h1(self, parser):
        content = "# First Heading\n\nText\n\n## Second\n\nMore text"
        result = parser.parse(_make_doc(content))
        assert result.title == "First Heading"

    def test_parse_title_fallback_to_doc_path(self, parser):
        content = "Just some text without headings."
        result = parser.parse(_make_doc(content, doc_path="docs/my-page.md"))
        assert "My Page" in result.title

    def test_parse_hierarchical_sections(self, parser):
        content = """# Title

Intro text.

## Section 1

Section 1 content.

### Sub Section

Sub section content.

## Section 2

Section 2 content.
"""
        result = parser.parse(_make_doc(content))
        assert len(result.sections) >= 1
        # h1 section should have children
        top = result.sections[0]
        assert top.title == "Title"
        assert len(top.children) >= 1


class TestExtractFrontmatter:
    def test_yaml_frontmatter(self, parser):
        content = """---
title: Test
author: Tester
---

Content here."""
        metadata, remaining = parser._extract_frontmatter(content)
        assert metadata["title"] == "Test"
        assert metadata["author"] == "Tester"
        assert "Content here." in remaining

    def test_no_frontmatter(self, parser):
        content = "# Just Markdown\n\nNo frontmatter."
        metadata, remaining = parser._extract_frontmatter(content)
        assert metadata == {}
        assert remaining == content

    def test_frontmatter_with_nav_hints(self, parser):
        content = """---
title: Test
sidebar_position: 3
sidebar_label: Custom Label
---

Content."""
        metadata, remaining = parser._extract_frontmatter(content)
        assert metadata["title"] == "Test"
        assert "_nav_hints" in metadata
        assert metadata["_nav_hints"]["sidebar_position"] == "3"
        assert metadata["_nav_hints"]["sidebar_label"] == "Custom Label"

    def test_frontmatter_with_quoted_values(self, parser):
        content = """---
title: "My Title"
author: 'Author Name'
---

Content."""
        metadata, remaining = parser._extract_frontmatter(content)
        assert metadata["title"] == "My Title"
        assert metadata["author"] == "Author Name"

    def test_incomplete_frontmatter(self, parser):
        content = "---\ntitle: Test\nNo closing delimiter"
        metadata, remaining = parser._extract_frontmatter(content)
        # Should not parse if there's no closing ---
        assert metadata == {}


class TestParseWithMarkdownIt:
    def test_code_fence(self, parser):
        content = """# Example

```sql
SELECT * FROM users;
```
"""
        result = parser.parse(_make_doc(content))
        assert len(result.sections) >= 1
        section = result.sections[0]
        assert "```sql" in section.content
        assert "SELECT * FROM users;" in section.content

    def test_indented_code_block(self, parser):
        content = """# Example

    indented code block
    line 2
"""
        result = parser.parse(_make_doc(content))
        section = result.sections[0]
        assert "```" in section.content

    def test_unordered_list(self, parser):
        content = """# List

- Item 1
- Item 2
- Item 3
"""
        result = parser.parse(_make_doc(content))
        section = result.sections[0]
        assert "- Item 1" in section.content

    def test_ordered_list(self, parser):
        content = """# Steps

1. First
2. Second
3. Third
"""
        result = parser.parse(_make_doc(content))
        section = result.sections[0]
        assert "1." in section.content
        assert "First" in section.content

    def test_blockquote(self, parser):
        content = """# Quote

> This is a quote
"""
        result = parser.parse(_make_doc(content))
        section = result.sections[0]
        assert "> This is a quote" in section.content

    def test_table(self, parser):
        content = """# Data

| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |
"""
        result = parser.parse(_make_doc(content))
        section = result.sections[0]
        assert "Name" in section.content
        assert "Value" in section.content

    def test_content_before_heading(self, parser):
        content = """Some intro text.

# First Heading

Content.
"""
        result = parser.parse(_make_doc(content))
        # Should have an intro section (level 0) plus the heading section
        assert len(result.sections) >= 1

    def test_nested_headings(self, parser):
        content = """# H1

## H2

### H3

Content.
"""
        result = parser.parse(_make_doc(content))
        assert result.sections[0].title == "H1"
        assert result.sections[0].children[0].title == "H2"
        assert result.sections[0].children[0].children[0].title == "H3"


class TestParseWithRegex:
    def test_regex_fallback_parsing(self, parser):
        """Test the regex fallback parser directly."""
        content = """# Title

Intro text.

## Section 1

Section 1 content.

## Section 2

Section 2 content.
"""
        sections = parser._parse_with_regex(content)
        assert len(sections) >= 1
        # First section should be h1
        assert sections[0].title == "Title"

    def test_regex_content_before_heading(self, parser):
        content = """Some text before any heading.

# First Heading

Content.
"""
        sections = parser._parse_with_regex(content)
        assert len(sections) >= 1
        # Should have a level-0 section for pre-heading content
        assert any(s.level == 0 for s in sections)

    def test_regex_nested_sections(self, parser):
        content = """# Parent

## Child 1

### Grandchild

## Child 2
"""
        sections = parser._parse_with_regex(content)
        assert sections[0].title == "Parent"
        assert len(sections[0].children) >= 1


class TestExtractListContent:
    def test_basic_list(self, parser):
        content = """# Test

- Item A
- Item B
"""
        result = parser.parse(_make_doc(content))
        assert "Item A" in result.sections[0].content
        assert "Item B" in result.sections[0].content


class TestExtractTableContent:
    def test_basic_table(self, parser):
        content = """# Test

| Col1 | Col2 |
|------|------|
| A    | B    |
"""
        result = parser.parse(_make_doc(content))
        assert "Col1" in result.sections[0].content


class TestParsedDocumentMethods:
    def test_get_section_titles(self, parser):
        content = """# Title

## Section A

### Sub A

## Section B
"""
        result = parser.parse(_make_doc(content))
        titles = result.get_section_titles()
        assert "Title" in titles
        assert "Section A" in titles
        assert "Sub A" in titles
        assert "Section B" in titles


class TestParsedSectionMethods:
    def test_get_all_content(self, parser):
        content = """# Title

Title content.

## Child

Child content.
"""
        result = parser.parse(_make_doc(content))
        top = result.sections[0]
        all_content = top.get_all_content()
        assert "Title content" in all_content
        assert "Child content" in all_content
