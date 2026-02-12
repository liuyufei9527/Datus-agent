"""Tests for datus/storage/document/streaming_processor.py"""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.document.schemas import (
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_MARKDOWN,
    FetchedDocument,
)
from datus.storage.document.streaming_processor import ProcessingStats, StreamingDocProcessor


def _make_doc(content="# Test\n\nContent.", content_type=CONTENT_TYPE_MARKDOWN, doc_path="docs/test.md"):
    return FetchedDocument(
        platform="test",
        version="1.0",
        source_url="https://example.com/" + doc_path,
        source_type="github",
        doc_path=doc_path,
        raw_content=content,
        content_type=content_type,
        metadata={},
    )


class TestProcessingStats:
    def test_init(self):
        stats = ProcessingStats()
        assert stats.total_docs == 0
        assert stats.total_chunks == 0
        assert stats.errors == []

    def test_increment(self):
        stats = ProcessingStats()
        stats.increment(docs=1, chunks=5)
        assert stats.total_docs == 1
        assert stats.total_chunks == 5

    def test_increment_multiple(self):
        stats = ProcessingStats()
        stats.increment(docs=1, chunks=3)
        stats.increment(docs=1, chunks=2)
        assert stats.total_docs == 2
        assert stats.total_chunks == 5

    def test_add_error(self):
        stats = ProcessingStats()
        stats.add_error("Test error")
        assert len(stats.errors) == 1
        assert stats.errors[0] == "Test error"

    def test_duration_seconds(self):
        stats = ProcessingStats()
        assert stats.duration_seconds >= 0

    def test_thread_safe_increment(self):
        import threading

        stats = ProcessingStats()
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: stats.increment(docs=1, chunks=1))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert stats.total_docs == 10
        assert stats.total_chunks == 10


class TestStreamingDocProcessor:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.store_chunks = MagicMock()
        return store

    @pytest.fixture
    def processor(self, mock_store):
        return StreamingDocProcessor(store=mock_store, chunk_size=512, pool_size=2)

    def test_init(self, mock_store):
        processor = StreamingDocProcessor(store=mock_store, chunk_size=1024, pool_size=4)
        assert processor.pool_size == 4
        assert processor.store == mock_store

    def test_process_single_document_markdown(self, processor, mock_store):
        doc = _make_doc("# Title\n\nSome content paragraph.")
        stats = ProcessingStats()
        base_metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com/docs/test.md",
            "doc_path": "docs/test.md",
        }
        chunks = processor._process_single_document(doc, base_metadata, stats)
        assert len(chunks) >= 1
        assert stats.total_docs == 1
        assert stats.total_chunks >= 1
        mock_store.store_chunks.assert_called_once()

    def test_process_single_document_html(self, processor, mock_store):
        html_content = "<html><body><h1>Title</h1><p>Some content.</p></body></html>"
        doc = _make_doc(html_content, content_type=CONTENT_TYPE_HTML, doc_path="docs/page.html")
        stats = ProcessingStats()
        base_metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "website",
            "source_url": "https://example.com/docs/page.html",
            "doc_path": "docs/page.html",
        }
        chunks = processor._process_single_document(doc, base_metadata, stats)
        assert len(chunks) >= 1

    def test_process_single_document_error(self, processor, mock_store):
        mock_store.store_chunks.side_effect = Exception("Store failed")
        doc = _make_doc()
        stats = ProcessingStats()
        base_metadata = {"platform": "test", "version": "1.0", "source_type": "github"}
        chunks = processor._process_single_document(doc, base_metadata, stats)
        assert len(chunks) == 0
        assert len(stats.errors) == 1

    def test_mark_visited(self, processor):
        assert processor._mark_visited("https://example.com/page1") is True
        assert processor._mark_visited("https://example.com/page1") is False
        assert processor._mark_visited("https://example.com/page2") is True

    def test_is_visited(self, processor):
        processor._mark_visited("https://example.com/page1")
        assert processor._is_visited("https://example.com/page1") is True
        assert processor._is_visited("https://example.com/page2") is False


class TestProcessLocal:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.store_chunks = MagicMock()
        return store

    @pytest.fixture
    def processor(self, mock_store):
        return StreamingDocProcessor(store=mock_store, chunk_size=512, pool_size=2)

    def test_process_local_basic(self, processor, mock_store):
        docs = [
            _make_doc("# Doc 1\n\nContent 1.", doc_path="doc1.md"),
            _make_doc("# Doc 2\n\nContent 2.", doc_path="doc2.md"),
        ]
        fetcher = MagicMock()
        stats = processor.process_local(
            fetcher=fetcher,
            documents=docs,
            version="1.0",
            platform="test",
        )
        assert stats.total_docs == 2
        assert stats.total_chunks >= 2

    def test_process_local_empty(self, processor, mock_store):
        fetcher = MagicMock()
        stats = processor.process_local(
            fetcher=fetcher,
            documents=[],
            version="1.0",
            platform="test",
        )
        assert stats.total_docs == 0
        assert stats.total_chunks == 0


class TestProcessGithub:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.store_chunks = MagicMock()
        return store

    @pytest.fixture
    def processor(self, mock_store):
        return StreamingDocProcessor(store=mock_store, chunk_size=512, pool_size=2)

    def test_process_github_empty_metadata(self, processor, mock_store):
        fetcher = MagicMock()
        metadata = MagicMock()
        metadata.file_paths = []
        stats = processor.process_github(
            fetcher=fetcher,
            metadata=metadata,
            version="1.0",
            platform="test",
        )
        assert stats.total_docs == 0

    def test_process_github_with_files(self, processor, mock_store):
        doc = _make_doc("# Guide\n\nContent.", doc_path="docs/guide.md")
        fetcher = MagicMock()
        fetcher.fetch_batch.return_value = [doc]

        metadata = MagicMock()
        metadata.file_paths = ["docs/guide.md"]

        stats = processor.process_github(
            fetcher=fetcher,
            metadata=metadata,
            version="1.0",
            platform="test",
        )
        assert stats.total_docs >= 1
