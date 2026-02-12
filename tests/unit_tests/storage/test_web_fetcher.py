"""Tests for datus/storage/document/fetcher/web_fetcher.py"""

import re
from unittest.mock import MagicMock, patch

import httpx
import pytest

from datus.storage.document.fetcher.web_fetcher import WebFetcher


@pytest.fixture
def mock_rate_limiter():
    limiter = MagicMock()
    limiter.wait = MagicMock()
    return limiter


@pytest.fixture
def fetcher(mock_rate_limiter):
    with patch.object(WebFetcher, "__init__", lambda self, **kwargs: None):
        f = WebFetcher.__new__(WebFetcher)
        f.platform = "test"
        f.version = "1.0"
        f.rate_limiter = mock_rate_limiter
        f.pool_size = 2
        f.timeout = 10.0
        f._client = MagicMock()
    return f


class TestShouldInclude:
    def test_no_patterns_includes_all(self, fetcher):
        assert fetcher._should_include("https://example.com/docs", [], []) is True

    def test_exclude_pattern_matches(self, fetcher):
        exclude = [re.compile(r"/blog/")]
        assert fetcher._should_include("https://example.com/blog/post", [], exclude) is False

    def test_exclude_pattern_no_match(self, fetcher):
        exclude = [re.compile(r"/blog/")]
        assert fetcher._should_include("https://example.com/docs/guide", [], exclude) is True

    def test_include_pattern_matches(self, fetcher):
        include = [re.compile(r"/docs/")]
        assert fetcher._should_include("https://example.com/docs/guide", include, []) is True

    def test_include_pattern_no_match(self, fetcher):
        include = [re.compile(r"/docs/")]
        assert fetcher._should_include("https://example.com/blog/post", include, []) is False

    def test_exclude_takes_priority(self, fetcher):
        include = [re.compile(r"/docs/")]
        exclude = [re.compile(r"/docs/private")]
        assert fetcher._should_include("https://example.com/docs/private/page", include, exclude) is False


class TestExtractPathPrefix:
    def test_basic_prefix(self, fetcher):
        assert fetcher._extract_path_prefix("/en/docs/overview") == "/en/"

    def test_docs_prefix(self, fetcher):
        assert fetcher._extract_path_prefix("/docs/v1.2/guide") == "/docs/"

    def test_root_path(self, fetcher):
        assert fetcher._extract_path_prefix("/") is None

    def test_empty_path(self, fetcher):
        assert fetcher._extract_path_prefix("") is None

    def test_single_segment(self, fetcher):
        assert fetcher._extract_path_prefix("/developer/guide") == "/developer/"


class TestDetectVersionFromUrl:
    def test_detect_semantic_version(self, fetcher):
        version = fetcher._detect_version_from_url("https://example.com/docs/v1.2.3/guide")
        assert version == "1.2.3"

    def test_detect_major_minor_version(self, fetcher):
        version = fetcher._detect_version_from_url("https://example.com/docs/1.2/guide")
        assert version == "1.2"

    def test_detect_version_prefix(self, fetcher):
        version = fetcher._detect_version_from_url("https://example.com/version/3.5/docs")
        assert version == "3.5"

    def test_detect_date_version(self, fetcher):
        version = fetcher._detect_version_from_url("https://example.com/docs/2024-01/guide")
        assert version == "2024-01"

    def test_no_version_returns_date(self, fetcher):
        version = fetcher._detect_version_from_url("https://example.com/docs/guide")
        # Returns current date
        assert len(version) == 10  # YYYY-MM-DD format


class TestExtractLinks:
    def test_extract_same_domain_links(self, fetcher):
        from bs4 import BeautifulSoup

        html = """
        <html><body>
        <a href="/docs/guide">Guide</a>
        <a href="/docs/tutorial">Tutorial</a>
        <a href="https://other.com/page">Other</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = fetcher._extract_links(soup, "https://example.com/docs", "example.com")
        assert any("/docs/guide" in link for link in links)
        assert any("/docs/tutorial" in link for link in links)
        assert not any("other.com" in link for link in links)

    def test_skip_anchor_links(self, fetcher):
        from bs4 import BeautifulSoup

        html = '<html><body><a href="#section">Section</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = fetcher._extract_links(soup, "https://example.com/docs", "example.com")
        assert len(links) == 0

    def test_skip_javascript_links(self, fetcher):
        from bs4 import BeautifulSoup

        html = '<html><body><a href="javascript:void(0)">Click</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = fetcher._extract_links(soup, "https://example.com/docs", "example.com")
        assert len(links) == 0

    def test_skip_non_doc_files(self, fetcher):
        from bs4 import BeautifulSoup

        html = """
        <html><body>
        <a href="/image.png">Image</a>
        <a href="/style.css">Style</a>
        <a href="/data.json">Data</a>
        <a href="/docs/page">Page</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = fetcher._extract_links(soup, "https://example.com", "example.com")
        assert any("/docs/page" in link for link in links)
        assert not any(".png" in link for link in links)
        assert not any(".css" in link for link in links)

    def test_relative_links_resolved(self, fetcher):
        from bs4 import BeautifulSoup

        html = '<html><body><a href="guide">Guide</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = fetcher._extract_links(soup, "https://example.com/docs/", "example.com")
        assert any("example.com/docs/guide" in link for link in links)


class TestFetchPage:
    def test_fetch_html_page(self, fetcher, mock_rate_limiter):
        mock_response = MagicMock()
        mock_response.text = "<html><head><title>Test Page</title></head><body><p>Content</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()
        fetcher._client.get = MagicMock(return_value=mock_response)

        result = fetcher._fetch_page("https://example.com/docs/page", 0, "example.com", "1.0")
        assert result is not None
        doc, links = result
        assert doc.content_type == "html"
        assert "Test Page" in doc.metadata.get("title", "")

    def test_fetch_non_html_skipped(self, fetcher, mock_rate_limiter):
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = MagicMock()
        fetcher._client.get = MagicMock(return_value=mock_response)

        result = fetcher._fetch_page("https://example.com/api/data", 0, "example.com", "1.0")
        assert result is None

    def test_fetch_404_returns_none(self, fetcher, mock_rate_limiter):
        response_mock = MagicMock()
        response_mock.status_code = 404
        fetcher._client.get = MagicMock(
            side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=response_mock)
        )

        result = fetcher._fetch_page("https://example.com/missing", 0, "example.com", "1.0")
        assert result is None

    def test_fetch_request_error_returns_none(self, fetcher, mock_rate_limiter):
        fetcher._client.get = MagicMock(
            side_effect=httpx.RequestError("Connection failed", request=MagicMock())
        )

        result = fetcher._fetch_page("https://example.com/down", 0, "example.com", "1.0")
        assert result is None


class TestFetchSingle:
    def test_fetch_single_basic(self, fetcher, mock_rate_limiter):
        mock_response = MagicMock()
        mock_response.text = "<html><head><title>Single</title></head><body><p>Content</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()
        fetcher._client.get = MagicMock(return_value=mock_response)

        doc = fetcher.fetch_single("https://example.com/docs/page")
        assert doc is not None

    def test_fetch_single_with_base_url(self, fetcher, mock_rate_limiter):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Content</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()
        fetcher._client.get = MagicMock(return_value=mock_response)

        doc = fetcher.fetch_single("/docs/page", base_url="https://example.com")
        assert doc is not None

    def test_fetch_single_returns_none_on_failure(self, fetcher, mock_rate_limiter):
        fetcher._client.get = MagicMock(
            side_effect=httpx.RequestError("Failed", request=MagicMock())
        )
        doc = fetcher.fetch_single("https://example.com/missing")
        assert doc is None


class TestContextManager:
    def test_context_manager(self, mock_rate_limiter):
        with patch.object(WebFetcher, "__init__", lambda self, **kwargs: None):
            f = WebFetcher.__new__(WebFetcher)
            f.platform = "test"
            f.version = "1.0"
            f.rate_limiter = mock_rate_limiter
            f.pool_size = 2
            f.timeout = 10.0
            mock_client = MagicMock()
            f._client = mock_client

            result = f.__enter__()
            assert result is f

            f.__exit__(None, None, None)
            # After close(), _client is set to None
            mock_client.close.assert_called_once()
            assert f._client is None
