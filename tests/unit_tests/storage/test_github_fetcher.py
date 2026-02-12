"""Tests for datus/storage/document/fetcher/github_fetcher.py"""

import base64
from unittest.mock import MagicMock, patch

import pytest

from datus.storage.document.fetcher.github_fetcher import GitHubFetcher, GitHubFetchMetadata


@pytest.fixture
def mock_rate_limiter():
    limiter = MagicMock()
    limiter.wait = MagicMock()
    limiter.configure_github_authenticated = MagicMock()
    return limiter


@pytest.fixture
def mock_github():
    with patch("datus.storage.document.fetcher.github_fetcher.Github") as MockGithub:
        mock_instance = MagicMock()
        MockGithub.return_value = mock_instance
        yield MockGithub, mock_instance


@pytest.fixture
def fetcher(mock_rate_limiter, mock_github):
    _, mock_instance = mock_github
    f = GitHubFetcher(
        platform="test",
        version="1.0",
        token="fake-token",
        rate_limiter=mock_rate_limiter,
    )
    return f


class TestGitHubFetchMetadata:
    def test_metadata_dataclass(self):
        metadata = GitHubFetchMetadata(
            repo=MagicMock(),
            branch="main",
            version="1.0",
            source="owner/repo",
            file_paths=["docs/guide.md"],
            nav_map={"docs/guide.md": ["Guide"]},
        )
        assert metadata.branch == "main"
        assert metadata.version == "1.0"
        assert len(metadata.file_paths) == 1


class TestDetectVersionFromPath:
    def test_release_path(self):
        version = GitHubFetcher._detect_version_from_path("releases/0.2.0/foo.md")
        assert version == "0.2.0"

    def test_versioned_path(self):
        version = GitHubFetcher._detect_version_from_path("content/releases/v3.4.0/bar.md")
        assert version == "3.4.0"

    def test_beta_version(self):
        version = GitHubFetcher._detect_version_from_path("versioned_docs/1.2.3-beta/baz.md")
        assert version == "1.2.3-beta"

    def test_no_version(self):
        version = GitHubFetcher._detect_version_from_path("docs/guide.md")
        assert version is None


class TestDetectDefaultBranch:
    def test_returns_default_branch(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()
        mock_repo.default_branch = "develop"
        result = fetcher._detect_default_branch(mock_repo)
        assert result == "develop"

    def test_fallback_to_main(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()
        type(mock_repo).default_branch = property(lambda self: (_ for _ in ()).throw(Exception("not found")))
        mock_repo.get_branch = MagicMock(return_value=MagicMock())
        result = fetcher._detect_default_branch(mock_repo)
        assert result == "main"


class TestDetectVersion:
    def test_detect_from_release(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()
        mock_release = MagicMock()
        mock_release.tag_name = "v2.0.0"
        mock_repo.get_releases.return_value = [mock_release]
        result = fetcher._detect_version(mock_repo, "main")
        assert result == "v2.0.0"

    def test_non_default_branch_used(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()
        mock_repo.get_releases.return_value = []
        mock_repo.get_tags.return_value = []
        result = fetcher._detect_version(mock_repo, "feature-branch")
        assert result == "feature-branch"


class TestFetchFile:
    def test_fetch_markdown_file(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()
        mock_content = MagicMock()
        mock_content.content = base64.b64encode(b"# Hello\n\nContent.").decode()
        mock_content.encoding = "base64"
        mock_content.name = "guide.md"
        mock_content.sha = "abc123"
        mock_content.size = 100
        mock_repo.get_contents.return_value = mock_content

        doc = fetcher._fetch_file(mock_repo, "docs/guide.md", "main", "1.0", "owner/repo")
        assert doc is not None
        assert "# Hello" in doc.raw_content
        assert doc.content_type == "markdown"
        assert doc.doc_path == "docs/guide.md"

    def test_fetch_returns_none_for_directory(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = [MagicMock(), MagicMock()]  # list = directory

        doc = fetcher._fetch_file(mock_repo, "docs/", "main", "1.0", "owner/repo")
        assert doc is None

    def test_fetch_handles_404(self, fetcher, mock_rate_limiter):
        from unittest.mock import PropertyMock

        mock_repo = MagicMock()
        exc = type("GithubException", (Exception,), {"status": 404})()
        exc.status = 404
        # Need to create a proper GithubException-like
        from github import GithubException

        mock_repo.get_contents.side_effect = GithubException(404, "Not Found", None)

        doc = fetcher._fetch_file(mock_repo, "missing.md", "main", "1.0", "owner/repo")
        assert doc is None


class TestCollectFilePaths:
    def test_collect_files_recursively(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()

        mock_file1 = MagicMock()
        mock_file1.type = "file"
        mock_file1.name = "guide.md"
        mock_file1.path = "docs/guide.md"

        mock_file2 = MagicMock()
        mock_file2.type = "file"
        mock_file2.name = "intro.md"
        mock_file2.path = "docs/intro.md"

        mock_repo.get_contents.return_value = [mock_file1, mock_file2]

        paths = fetcher._collect_file_paths(mock_repo, "docs", "main")
        assert len(paths) == 2
        assert "docs/guide.md" in paths
        assert "docs/intro.md" in paths

    def test_collect_recurses_into_dirs(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()

        mock_dir = MagicMock()
        mock_dir.type = "dir"
        mock_dir.path = "docs/sub"

        mock_file = MagicMock()
        mock_file.type = "file"
        mock_file.name = "page.md"
        mock_file.path = "docs/sub/page.md"

        mock_repo.get_contents.side_effect = lambda path, ref: {
            "docs": [mock_dir],
            "docs/sub": [mock_file],
        }.get(path, [])

        paths = fetcher._collect_file_paths(mock_repo, "docs", "main")
        assert "docs/sub/page.md" in paths

    def test_skips_non_doc_files(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()

        mock_file = MagicMock()
        mock_file.type = "file"
        mock_file.name = "image.png"
        mock_file.path = "docs/image.png"

        mock_repo.get_contents.return_value = [mock_file]

        paths = fetcher._collect_file_paths(mock_repo, "docs", "main")
        assert len(paths) == 0


class TestFetchBatch:
    def test_fetch_batch_basic(self, fetcher, mock_rate_limiter):
        mock_repo = MagicMock()
        mock_content = MagicMock()
        mock_content.content = base64.b64encode(b"# Hello").decode()
        mock_content.encoding = "base64"
        mock_content.name = "guide.md"
        mock_content.sha = "abc123"
        mock_content.size = 100
        mock_repo.get_contents.return_value = mock_content

        metadata = GitHubFetchMetadata(
            repo=mock_repo,
            branch="main",
            version="1.0",
            source="owner/repo",
            file_paths=["docs/guide.md"],
            nav_map={"docs/guide.md": ["Guides", "Guide"]},
        )

        docs = fetcher.fetch_batch(metadata, ["docs/guide.md"])
        assert len(docs) == 1
        assert docs[0].metadata.get("nav_path") == ["Guides", "Guide"]

    def test_fetch_batch_empty(self, fetcher, mock_rate_limiter):
        metadata = GitHubFetchMetadata(
            repo=MagicMock(),
            branch="main",
            version="1.0",
            source="owner/repo",
            file_paths=[],
            nav_map={},
        )
        docs = fetcher.fetch_batch(metadata, [])
        assert docs == []


class TestFetchSingle:
    def test_fetch_single_requires_repo_name(self, fetcher):
        with pytest.raises(ValueError, match="repo_name"):
            fetcher.fetch_single("docs/guide.md")


class TestIsDocFile:
    def test_doc_extensions(self, fetcher):
        assert fetcher._is_doc_file("file.md") is True
        assert fetcher._is_doc_file("file.rst") is True
        assert fetcher._is_doc_file("file.html") is True

    def test_non_doc_extensions(self, fetcher):
        assert fetcher._is_doc_file("file.py") is False
        assert fetcher._is_doc_file("file.png") is False
