# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for semantic_tools/registry.py"""

from unittest.mock import MagicMock, patch

import pytest

from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.registry import (
    AdapterMetadata,
    SemanticAdapterRegistry,
)
from datus.utils.exceptions import DatusException


class MockAdapter(BaseSemanticAdapter):
    """Mock adapter for testing."""

    def __init__(self, config=None):
        self.config = config

    async def list_metrics(self, **kwargs):
        return []

    async def get_dimensions(self, **kwargs):
        return []

    async def query_metrics(self, **kwargs):
        return MagicMock()

    async def validate_semantic(self):
        return MagicMock()


class MockConfig:
    """Mock config class for testing, NOT a Pydantic BaseModel."""
    pass


# ---------------------------------------------------------------------------
# AdapterMetadata
# ---------------------------------------------------------------------------


class TestAdapterMetadata:
    def test_init(self):
        meta = AdapterMetadata(
            service_type="test",
            adapter_class=MockAdapter,
        )
        assert meta.service_type == "test"
        assert meta.display_name == "Test"

    def test_custom_display_name(self):
        meta = AdapterMetadata(
            service_type="test",
            adapter_class=MockAdapter,
            display_name="My Adapter",
        )
        assert meta.display_name == "My Adapter"

    def test_get_config_fields_no_config(self):
        meta = AdapterMetadata(
            service_type="test",
            adapter_class=MockAdapter,
        )
        assert meta.get_config_fields() == {}

    def test_get_config_fields_non_pydantic(self):
        meta = AdapterMetadata(
            service_type="test",
            adapter_class=MockAdapter,
            config_class=MockConfig,
        )
        # MockConfig is not a Pydantic BaseModel
        assert meta.get_config_fields() == {}


# ---------------------------------------------------------------------------
# SemanticAdapterRegistry
# ---------------------------------------------------------------------------


class TestSemanticAdapterRegistry:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry state between tests."""
        original_adapters = SemanticAdapterRegistry._adapters.copy()
        original_factories = SemanticAdapterRegistry._factories.copy()
        original_metadata = SemanticAdapterRegistry._metadata.copy()
        original_initialized = SemanticAdapterRegistry._initialized
        yield
        SemanticAdapterRegistry._adapters = original_adapters
        SemanticAdapterRegistry._factories = original_factories
        SemanticAdapterRegistry._metadata = original_metadata
        SemanticAdapterRegistry._initialized = original_initialized

    def test_register_adapter(self):
        SemanticAdapterRegistry.register("test_type", MockAdapter)
        assert SemanticAdapterRegistry.is_registered("test_type")

    def test_register_with_factory(self):
        factory = MagicMock(return_value=MockAdapter())
        SemanticAdapterRegistry.register("test_factory", MockAdapter, factory=factory)
        assert SemanticAdapterRegistry.is_registered("test_factory")

    def test_create_adapter_default(self):
        SemanticAdapterRegistry.register("test_default", MockAdapter)
        adapter = SemanticAdapterRegistry.create_adapter("test_default", None)
        assert isinstance(adapter, MockAdapter)

    def test_create_adapter_with_factory(self):
        mock_adapter = MockAdapter()
        factory = MagicMock(return_value=mock_adapter)
        SemanticAdapterRegistry.register("test_fac", MockAdapter, factory=factory)
        result = SemanticAdapterRegistry.create_adapter("test_fac", {"key": "val"})
        assert result is mock_adapter
        factory.assert_called_once_with({"key": "val"})

    def test_create_adapter_not_found(self):
        with pytest.raises(DatusException):
            SemanticAdapterRegistry.create_adapter("nonexistent_adapter", None)

    def test_list_adapters(self):
        SemanticAdapterRegistry.register("test_list", MockAdapter)
        adapters = SemanticAdapterRegistry.list_adapters()
        assert "test_list" in adapters

    def test_is_registered_false(self):
        assert SemanticAdapterRegistry.is_registered("nonexistent") is False

    def test_get_metadata(self):
        SemanticAdapterRegistry.register(
            "test_meta",
            MockAdapter,
            display_name="Test Meta",
        )
        meta = SemanticAdapterRegistry.get_metadata("test_meta")
        assert meta is not None
        assert meta.display_name == "Test Meta"

    def test_get_metadata_not_found(self):
        meta = SemanticAdapterRegistry.get_metadata("nonexistent")
        assert meta is None

    def test_case_insensitive(self):
        SemanticAdapterRegistry.register("TestCase", MockAdapter)
        assert SemanticAdapterRegistry.is_registered("testcase")
        assert SemanticAdapterRegistry.is_registered("TESTCASE")

    def test_discover_adapters_idempotent(self):
        SemanticAdapterRegistry._initialized = False
        SemanticAdapterRegistry.discover_adapters()
        assert SemanticAdapterRegistry._initialized is True
        # Second call should be a no-op
        SemanticAdapterRegistry.discover_adapters()
        assert SemanticAdapterRegistry._initialized is True

    def test_list_available_adapters(self):
        SemanticAdapterRegistry._initialized = False
        SemanticAdapterRegistry.register("test_avail", MockAdapter)
        result = SemanticAdapterRegistry.list_available_adapters()
        assert "test_avail" in result

    def test_try_load_adapter_import_error(self):
        """_try_load_adapter should handle ImportError gracefully."""
        # This should not raise
        SemanticAdapterRegistry._try_load_adapter("definitely_nonexistent_adapter_xyz")
        assert not SemanticAdapterRegistry.is_registered("definitely_nonexistent_adapter_xyz")
