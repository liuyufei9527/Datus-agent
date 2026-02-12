"""Tests for datus/storage/document/store.py"""

import pytest

from datus.storage.document.store import _SAFE_IDENTIFIER_RE, DocumentStore, get_platform_doc_schema


class TestGetPlatformDocSchema:
    def test_default_schema(self):
        schema = get_platform_doc_schema()
        assert "chunk_id" in schema.names
        assert "chunk_text" in schema.names
        assert "vector" in schema.names
        assert len(schema.names) == 18

    def test_custom_embedding_dim(self):
        schema = get_platform_doc_schema(embedding_dim=768)
        vector_type = schema.field("vector").type
        assert vector_type.list_size == 768

    def test_field_types(self):
        import pyarrow as pa

        schema = get_platform_doc_schema()
        assert schema.field("chunk_id").type == pa.string()
        assert schema.field("chunk_index").type == pa.int32()
        assert isinstance(schema.field("titles").type, pa.ListType)
        assert isinstance(schema.field("nav_path").type, pa.ListType)


class TestSafeIdentifierRegex:
    def test_valid_identifiers(self):
        assert _SAFE_IDENTIFIER_RE.match("v1.2.3")
        assert _SAFE_IDENTIFIER_RE.match("1.0.0")
        assert _SAFE_IDENTIFIER_RE.match("my-version_1.0")
        assert _SAFE_IDENTIFIER_RE.match("latest")
        assert _SAFE_IDENTIFIER_RE.match("2024-01-15")

    def test_invalid_identifiers(self):
        assert not _SAFE_IDENTIFIER_RE.match("'; DROP TABLE --")
        assert not _SAFE_IDENTIFIER_RE.match("version\nOR 1=1")
        assert not _SAFE_IDENTIFIER_RE.match("")


class TestDocumentStoreValidateIdentifier:
    def test_valid_version(self):
        # Should not raise
        DocumentStore._validate_identifier("v1.2.3", "version")
        DocumentStore._validate_identifier("latest", "version")

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Invalid version"):
            DocumentStore._validate_identifier("'; DROP TABLE", "version")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            DocumentStore._validate_identifier("", "test")
