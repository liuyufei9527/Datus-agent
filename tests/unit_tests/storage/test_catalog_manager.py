"""Tests for datus/storage/catalog_manager.py — CatalogUpdater."""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.storage.catalog_manager import CatalogUpdater


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.agentic_nodes = {}
    config.current_namespace = "default"
    return config


@pytest.fixture
def mock_semantic_storage():
    return MagicMock()


@pytest.fixture
def updater(mock_agent_config, mock_semantic_storage):
    with patch("datus.storage.catalog_manager.get_storage_cache_instance") as mock_cache:
        cache_inst = MagicMock()
        cache_inst.semantic_storage.return_value = mock_semantic_storage
        mock_cache.return_value = cache_inst
        return CatalogUpdater(mock_agent_config)


# ── _parse_json_field ──


class TestParseJsonField:
    def test_none_returns_none(self, updater):
        assert updater._parse_json_field(None) is None

    def test_list_returns_as_is(self, updater):
        data = [{"name": "col1"}]
        assert updater._parse_json_field(data) is data

    def test_json_string_list(self, updater):
        data = json.dumps([{"name": "col1"}])
        result = updater._parse_json_field(data)
        assert result == [{"name": "col1"}]

    def test_json_string_non_list(self, updater):
        # If parsed JSON is not a list, return None
        data = json.dumps({"name": "col1"})
        assert updater._parse_json_field(data) is None

    def test_invalid_json_string(self, updater):
        assert updater._parse_json_field("not-json{") is None

    def test_other_type_returns_none(self, updater):
        assert updater._parse_json_field(12345) is None


# ── _get_all_storages ──


class TestGetAllStorages:
    def test_no_agentic_nodes(self, updater, mock_semantic_storage):
        storages = updater._get_all_storages()
        assert storages == [mock_semantic_storage]

    def test_with_sub_agent_in_namespace(self, mock_agent_config):
        sub_agent_dict = {
            "system_prompt": "sub1",
            "scoped_context": {"tables": ["t1"]},
        }
        mock_agent_config.agentic_nodes = {"agent1": sub_agent_dict}

        with patch("datus.storage.catalog_manager.get_storage_cache_instance") as mock_cache:
            cache_inst = MagicMock()
            main_storage = MagicMock()
            sub_storage = MagicMock()
            cache_inst.semantic_storage.side_effect = lambda name=None: sub_storage if name else main_storage
            mock_cache.return_value = cache_inst

            with patch("datus.storage.catalog_manager.SubAgentConfig") as MockSubAgentConfig:
                mock_sub = MagicMock()
                mock_sub.is_in_namespace.return_value = True
                mock_sub.has_scoped_context.return_value = True
                mock_sub.system_prompt = "sub1"
                MockSubAgentConfig.model_validate.return_value = mock_sub

                updater = CatalogUpdater(mock_agent_config)
                storages = updater._get_all_storages()
                assert len(storages) == 2

    def test_sub_agent_not_in_namespace(self, mock_agent_config):
        sub_agent_dict = {"system_prompt": "sub1"}
        mock_agent_config.agentic_nodes = {"agent1": sub_agent_dict}

        with patch("datus.storage.catalog_manager.get_storage_cache_instance") as mock_cache:
            cache_inst = MagicMock()
            main_storage = MagicMock()
            cache_inst.semantic_storage.return_value = main_storage
            mock_cache.return_value = cache_inst

            with patch("datus.storage.catalog_manager.SubAgentConfig") as MockSubAgentConfig:
                mock_sub = MagicMock()
                mock_sub.is_in_namespace.return_value = False
                mock_sub.has_scoped_context.return_value = True
                MockSubAgentConfig.model_validate.return_value = mock_sub

                updater = CatalogUpdater(mock_agent_config)
                storages = updater._get_all_storages()
                assert storages == [main_storage]


# ── update_semantic_model ──


class TestUpdateSemanticModel:
    def test_update_description(self, updater, mock_semantic_storage):
        old = {
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "tbl",
            "semantic_model_name": "model",
        }
        new = {"description": "new desc"}
        updater.update_semantic_model(old, new)
        mock_semantic_storage.update.assert_called_once()

    def test_update_columns_dimensions(self, updater, mock_semantic_storage):
        old = {
            "catalog_name": "",
            "database_name": "",
            "schema_name": "",
            "table_name": "t",
            "semantic_model_name": "m",
            "dimensions": [{"name": "d1", "description": "old desc"}],
        }
        new_values = {
            "dimensions": [{"name": "d1", "description": "new desc"}],
        }
        updater.update_semantic_model(old, new_values)
        # description changed -> update should be called
        mock_semantic_storage.update.assert_called()

    def test_update_no_changes(self, updater, mock_semantic_storage):
        old = {
            "catalog_name": "",
            "database_name": "",
            "schema_name": "",
            "table_name": "t",
            "semantic_model_name": "m",
            "dimensions": [{"name": "d1", "description": "same"}],
        }
        new_values = {
            "dimensions": [{"name": "d1", "description": "same"}],
        }
        updater.update_semantic_model(old, new_values)
        # No description key in new_values top level, and dim desc is same
        mock_semantic_storage.update.assert_not_called()


# ── _update_columns ──


class TestUpdateColumns:
    def test_empty_columns(self, updater, mock_semantic_storage):
        updater._update_columns(
            storages=[mock_semantic_storage],
            catalog_name="",
            database_name="",
            schema_name="",
            table_name="t",
            old_columns=None,
            new_columns=None,
            kind_field="is_dimension",
            allowed_fields={"description"},
        )
        mock_semantic_storage.update.assert_not_called()

    def test_new_column_without_name_skipped(self, updater, mock_semantic_storage):
        updater._update_columns(
            storages=[mock_semantic_storage],
            catalog_name="",
            database_name="",
            schema_name="",
            table_name="t",
            old_columns=[],
            new_columns=[{"description": "no name"}],
            kind_field="is_dimension",
            allowed_fields={"description"},
        )
        mock_semantic_storage.update.assert_not_called()

    def test_column_with_changes(self, updater, mock_semantic_storage):
        updater._update_columns(
            storages=[mock_semantic_storage],
            catalog_name="c",
            database_name="d",
            schema_name="s",
            table_name="t",
            old_columns=[{"name": "col1", "description": "old"}],
            new_columns=[{"name": "col1", "description": "new"}],
            kind_field="is_measure",
            allowed_fields={"description"},
        )
        mock_semantic_storage.update.assert_called_once()

    def test_column_type_mapping(self, updater, mock_semantic_storage):
        """column_type in storage maps to 'type' in the column dict."""
        updater._update_columns(
            storages=[mock_semantic_storage],
            catalog_name="",
            database_name="",
            schema_name="",
            table_name="t",
            old_columns=[{"name": "col1", "type": "CATEGORICAL"}],
            new_columns=[{"name": "col1", "type": "TIME"}],
            kind_field="is_dimension",
            allowed_fields={"column_type"},
        )
        mock_semantic_storage.update.assert_called_once()
        call_args = mock_semantic_storage.update.call_args
        update_values = call_args[0][1]
        assert "column_type" in update_values
        assert update_values["column_type"] == "TIME"
