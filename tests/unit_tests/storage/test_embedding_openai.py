"""Tests for datus/storage/embedding_openai.py — OpenAIEmbeddings."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.embedding_openai import OpenAIEmbeddings


@pytest.fixture
def embeddings():
    return OpenAIEmbeddings(name="text-embedding-ada-002")


# ── ndims / _ndims ──


class TestNdims:
    def test_ada_002(self):
        emb = OpenAIEmbeddings(name="text-embedding-ada-002")
        assert emb.ndims() == 1536

    def test_3_large_default(self):
        emb = OpenAIEmbeddings(name="text-embedding-3-large")
        assert emb.ndims() == 3072

    def test_3_large_with_dim(self):
        emb = OpenAIEmbeddings(name="text-embedding-3-large", dim=256)
        assert emb.ndims() == 256

    def test_3_small_default(self):
        emb = OpenAIEmbeddings(name="text-embedding-3-small")
        assert emb.ndims() == 1536

    def test_3_small_with_dim(self):
        emb = OpenAIEmbeddings(name="text-embedding-3-small", dim=512)
        assert emb.ndims() == 512

    def test_unknown_model_raises(self):
        emb = OpenAIEmbeddings(name="unknown-model")
        with pytest.raises(ValueError, match="Unknown model name"):
            emb.ndims()


# ── static methods ──


class TestStaticMethods:
    def test_sensitive_keys(self):
        assert OpenAIEmbeddings.sensitive_keys() == []

    def test_model_names(self):
        names = OpenAIEmbeddings.model_names()
        assert "text-embedding-ada-002" in names
        assert "text-embedding-3-large" in names
        assert "text-embedding-3-small" in names


# ── __setattr__ ──


class TestSetattr:
    def test_set_name_triggers_debug_log(self, embeddings):
        # Should not raise
        embeddings.name = "text-embedding-3-small"
        assert embeddings.name == "text-embedding-3-small"

    def test_set_dim(self, embeddings):
        embeddings.dim = 256
        assert embeddings.dim == 256

    def test_set_arbitrary_attr(self, embeddings):
        embeddings.custom_attr = "value"
        assert embeddings.custom_attr == "value"


# ── generate_embeddings ──


class TestGenerateEmbeddings:
    def test_empty_text_filtered(self, embeddings):
        """Empty strings should be filtered out, result should have None for them."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        # Inject mock client
        embeddings.__dict__["_openai_client"] = mock_client

        results = embeddings.generate_embeddings(["", "hello", ""])
        assert results[0] is None
        assert results[1] is not None
        assert results[2] is None

    def test_bad_request_returns_none_list(self, embeddings):
        """BadRequestError should return [None, ...] for all texts."""
        from openai import BadRequestError

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = BadRequestError(
            message="bad", response=MagicMock(status_code=400), body={}
        )
        embeddings.__dict__["_openai_client"] = mock_client

        results = embeddings.generate_embeddings(["hello", "world"])
        assert results == [None, None]

    def test_other_exception_propagates(self, embeddings):
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = RuntimeError("API down")
        embeddings.__dict__["_openai_client"] = mock_client

        with pytest.raises(RuntimeError, match="API down"):
            embeddings.generate_embeddings(["hello"])

    def test_dimensions_passed_for_non_ada(self):
        emb = OpenAIEmbeddings(name="text-embedding-3-small", dim=256)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 256
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response
        emb.__dict__["_openai_client"] = mock_client

        emb.generate_embeddings(["hello"])
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["dimensions"] == 256


# ── _openai_client ──


class TestOpenAIClient:
    @patch("datus.storage.embedding_openai.OpenAI")
    def test_default_client(self, mock_openai_cls):
        emb = OpenAIEmbeddings(name="text-embedding-ada-002", api_key="test-key")
        _ = emb._openai_client
        mock_openai_cls.assert_called_once_with(api_key="test-key")

    @patch("datus.storage.embedding_openai.OpenAI")
    def test_client_with_base_url(self, mock_openai_cls):
        emb = OpenAIEmbeddings(name="text-embedding-ada-002", base_url="http://localhost:8080", api_key="k")
        _ = emb._openai_client
        mock_openai_cls.assert_called_once()
        call_kwargs = mock_openai_cls.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:8080"

    @patch("datus.storage.embedding_openai.AzureOpenAI")
    def test_azure_client(self, mock_azure_cls):
        emb = OpenAIEmbeddings(name="text-embedding-ada-002", use_azure=True, api_key="az-key")
        _ = emb._openai_client
        mock_azure_cls.assert_called_once()

    @patch("datus.storage.embedding_openai.OpenAI")
    def test_client_with_all_options(self, mock_openai_cls):
        emb = OpenAIEmbeddings(
            name="text-embedding-ada-002",
            api_key="k",
            base_url="http://test",
            default_headers={"X-Custom": "val"},
            organization="org-123",
        )
        _ = emb._openai_client
        call_kwargs = mock_openai_cls.call_args[1]
        assert call_kwargs["api_key"] == "k"
        assert call_kwargs["base_url"] == "http://test"
        assert call_kwargs["default_headers"] == {"X-Custom": "val"}
        assert call_kwargs["organization"] == "org-123"
