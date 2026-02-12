"""Tests for datus/storage/document/parser/html_parser.py"""

import pytest

from datus.storage.document.parser.html_parser import HTMLParser
from datus.storage.document.schemas import FetchedDocument


@pytest.fixture
def parser():
    return HTMLParser(parser="html.parser")


def _make_doc(html_content: str, source_url: str = "https://example.com/docs/page", doc_path: str = "docs/page.html"):
    return FetchedDocument(
        platform="test",
        version="1.0",
        source_url=source_url,
        source_type="website",
        doc_path=doc_path,
        raw_content=html_content,
        content_type="html",
    )


class TestHTMLParserInit:
    def test_init_default_parser(self):
        parser = HTMLParser()
        assert parser._parser == "lxml"

    def test_init_custom_parser(self):
        parser = HTMLParser(parser="html.parser")
        assert parser._parser == "html.parser"


class TestParse:
    def test_parse_basic_html(self, parser):
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Title</h1>
            <p>Some paragraph text.</p>
        </body>
        </html>
        """
        result = parser.parse(_make_doc(html))
        assert result.title == "Test Page"
        assert len(result.sections) > 0

    def test_parse_title_with_pipe(self, parser):
        html = "<html><head><title>Docs | SiteName</title></head><body><p>Text</p></body></html>"
        result = parser.parse(_make_doc(html))
        assert result.title == "Docs"

    def test_parse_title_with_dash(self, parser):
        html = "<html><head><title>Docs - SiteName</title></head><body><p>Text</p></body></html>"
        result = parser.parse(_make_doc(html))
        assert result.title == "Docs"

    def test_parse_title_fallback_to_doc_path(self, parser):
        html = "<html><body><p>No title</p></body></html>"
        result = parser.parse(_make_doc(html, doc_path="docs/my-page.html"))
        assert "My Page" in result.title

    def test_parse_extracts_metadata(self, parser):
        html = """
        <html>
        <head>
            <title>Test</title>
            <meta name="description" content="A test page">
            <meta name="author" content="TestAuthor">
            <meta name="keywords" content="sql, database, test">
        </head>
        <body><p>Content</p></body>
        </html>
        """
        result = parser.parse(_make_doc(html))
        assert result.metadata.get("description") == "A test page"
        assert result.metadata.get("author") == "TestAuthor"
        assert result.metadata.get("keywords") == ["sql", "database", "test"]

    def test_parse_og_title_fallback(self, parser):
        html = """
        <html>
        <head>
            <meta property="og:title" content="OG Title">
        </head>
        <body><p>Content</p></body>
        </html>
        """
        result = parser.parse(_make_doc(html))
        assert result.metadata.get("title") == "OG Title"


class TestExtractNavigationPath:
    def test_breadcrumb_list_items(self, parser):
        html = """
        <html><body>
        <nav aria-label="breadcrumb">
            <ul>
                <li>Guides</li>
                <li>SQL</li>
                <li>SELECT</li>
            </ul>
        </nav>
        <p>Content</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        nav_path = result.metadata.get("nav_path", [])
        assert len(nav_path) >= 2

    def test_breadcrumb_links(self, parser):
        html = """
        <html><body>
        <nav class="breadcrumb">
            <a href="/guides">Guides</a>
            <a href="/guides/sql">SQL</a>
            <span class="current">SELECT</span>
        </nav>
        <p>Content</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        nav_path = result.metadata.get("nav_path", [])
        assert len(nav_path) >= 2

    def test_url_path_fallback(self, parser):
        html = "<html><body><p>Content</p></body></html>"
        result = parser.parse(_make_doc(html, source_url="https://example.com/en/user-guide/data-load"))
        nav_path = result.metadata.get("nav_path", [])
        # URL path segments: user-guide, data-load (en is filtered)
        assert len(nav_path) >= 1

    def test_sidebar_path(self, parser):
        html = """
        <html><body>
        <nav aria-label="Docs pages">
            <ul>
                <li>
                    <a href="/guides">Guides</a>
                    <ul>
                        <li>
                            <a href="/guides/sql">SQL</a>
                            <ul>
                                <li><a href="/guides/sql/select">SELECT</a></li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </nav>
        <main><p>Content</p></main>
        </body></html>
        """
        result = parser.parse(_make_doc(html, source_url="https://example.com/guides/sql/select"))
        nav_path = result.metadata.get("nav_path", [])
        assert len(nav_path) >= 1


class TestExtractPathFromUrl:
    def test_basic_url_path(self, parser):
        result = parser._extract_path_from_url("https://example.com/en/user-guide/data-load-intro")
        # 'en' is filtered, remaining: user-guide, data-load-intro
        assert len(result) >= 1
        assert "User Guide" in result or "Data Load Intro" in result

    def test_empty_url_path(self, parser):
        result = parser._extract_path_from_url("https://example.com/")
        assert result == []

    def test_url_with_docs_prefix(self, parser):
        result = parser._extract_path_from_url("https://example.com/docs/guide/intro")
        # 'docs' is filtered
        assert len(result) >= 1


class TestExtractGroupName:
    def test_known_group_first_element(self, parser):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<html></html>", "html.parser")
        result = parser._extract_group_name(["Guides", "SQL", "SELECT"], soup)
        assert result == "Guides"

    def test_known_group_contains(self, parser):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<html></html>", "html.parser")
        result = parser._extract_group_name(["API Reference", "Functions"], soup)
        assert result == "API Reference"

    def test_short_first_element_fallback(self, parser):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<html></html>", "html.parser")
        result = parser._extract_group_name(["MyGroup", "SubPage"], soup)
        assert result == "MyGroup"

    def test_empty_nav_path(self, parser):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<html></html>", "html.parser")
        result = parser._extract_group_name([], soup)
        assert result == ""


class TestCleanSoup:
    def test_removes_script_and_style(self, parser):
        html = """
        <html><body>
        <script>alert('hi')</script>
        <style>.foo { color: red; }</style>
        <p>Visible content</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        for section in result.sections:
            assert "alert" not in section.content
            assert "color: red" not in section.content

    def test_removes_footer_header(self, parser):
        html = """
        <html><body>
        <header>Site Header</header>
        <main><p>Main content</p></main>
        <footer>Site Footer</footer>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        found_content = " ".join(s.content for s in result.sections)
        assert "Main content" in found_content


class TestExtractSections:
    def test_hierarchical_sections(self, parser):
        html = """
        <html><body>
        <h1>Title</h1>
        <p>Intro text</p>
        <h2>Section One</h2>
        <p>Section one text</p>
        <h3>Sub Section</h3>
        <p>Sub section text</p>
        <h2>Section Two</h2>
        <p>Section two text</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        # Should have a top-level h1 section
        assert len(result.sections) >= 1
        top = result.sections[0]
        assert top.title == "Title"
        assert len(top.children) >= 1

    def test_code_blocks_preserved(self, parser):
        html = """
        <html><body>
        <h1>Code Example</h1>
        <pre><code class="language-sql">SELECT * FROM table;</code></pre>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "```sql" in content
        assert "SELECT * FROM table;" in content

    def test_code_block_without_language(self, parser):
        html = """
        <html><body>
        <h1>Code</h1>
        <pre>Some preformatted text</pre>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "```" in content

    def test_unordered_list(self, parser):
        html = """
        <html><body>
        <h1>List</h1>
        <ul>
            <li>Item A</li>
            <li>Item B</li>
            <li>Item C</li>
        </ul>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "- Item A" in content
        assert "- Item B" in content

    def test_ordered_list(self, parser):
        html = """
        <html><body>
        <h1>Steps</h1>
        <ol>
            <li>First</li>
            <li>Second</li>
            <li>Third</li>
        </ol>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "1. First" in content
        assert "2. Second" in content

    def test_table_extraction(self, parser):
        html = """
        <html><body>
        <h1>Data</h1>
        <table>
            <thead><tr><th>Name</th><th>Age</th></tr></thead>
            <tbody><tr><td>Alice</td><td>30</td></tr></tbody>
        </table>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "Name" in content
        assert "Alice" in content

    def test_table_without_thead(self, parser):
        html = """
        <html><body>
        <h1>Data</h1>
        <table>
            <tr><td>Name</td><td>Age</td></tr>
            <tr><td>Bob</td><td>25</td></tr>
        </table>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "Name" in content

    def test_blockquote(self, parser):
        html = """
        <html><body>
        <h1>Quote</h1>
        <blockquote>This is a quote</blockquote>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "> This is a quote" in content

    def test_no_sections_creates_default(self, parser):
        html = """
        <html><body>
        <p>Just some text without headings.</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        assert len(result.sections) >= 1
        assert "Just some text" in result.sections[0].content


class TestExtractTextWithFormatting:
    def test_bold_text(self, parser):
        html = """
        <html><body>
        <h1>Title</h1>
        <p>This is <strong>bold</strong> text.</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "**bold**" in content

    def test_italic_text(self, parser):
        html = """
        <html><body>
        <h1>Title</h1>
        <p>This is <em>italic</em> text.</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "*italic*" in content

    def test_inline_code(self, parser):
        html = """
        <html><body>
        <h1>Title</h1>
        <p>Use the <code>SELECT</code> statement.</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "`SELECT`" in content

    def test_link_formatting(self, parser):
        html = """
        <html><body>
        <h1>Title</h1>
        <p>See <a href="/docs/guide">the guide</a>.</p>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        content = result.sections[0].content
        assert "[the guide](/docs/guide)" in content


class TestFindMainContent:
    def test_finds_main_tag(self, parser):
        html = """
        <html><body>
        <div>Other stuff</div>
        <main><p>Main content</p></main>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        found = " ".join(s.content for s in result.sections)
        assert "Main content" in found

    def test_finds_article_tag(self, parser):
        html = """
        <html><body>
        <article><p>Article content</p></article>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        found = " ".join(s.content for s in result.sections)
        assert "Article content" in found

    def test_finds_role_main(self, parser):
        html = """
        <html><body>
        <div role="main"><p>Role main content</p></div>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        found = " ".join(s.content for s in result.sections)
        assert "Role main content" in found

    def test_fallback_to_body(self, parser):
        html = """
        <html><body>
        <div><p>Body content</p></div>
        </body></html>
        """
        result = parser.parse(_make_doc(html))
        found = " ".join(s.content for s in result.sections)
        assert "Body content" in found
