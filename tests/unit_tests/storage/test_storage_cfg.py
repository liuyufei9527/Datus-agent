"""Tests for datus/storage/storage_cfg.py"""

import pytest

from datus.storage.storage_cfg import (
    _find_config_differences,
    check_storage_config,
    load_storage_config,
    save_storage_config,
    save_storage_configs,
)
from datus.utils.constants import EmbeddingProvider
from datus.utils.exceptions import DatusException


class TestSaveStorageConfigs:
    def test_save_and_load(self, tmp_path):
        configs = {
            "test_store": {
                "model_name": "test-model",
                "dim_size": 384,
                "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS,
            }
        }
        save_storage_configs(configs, str(tmp_path))
        loaded = load_storage_config(str(tmp_path))
        assert "test_store" in loaded
        assert loaded["test_store"]["model_name"] == "test-model"
        assert loaded["test_store"]["dim_size"] == "384"

    def test_save_multiple_stores(self, tmp_path):
        configs = {
            "store_a": {"model_name": "model-a", "dim_size": 384},
            "store_b": {"model_name": "model-b", "dim_size": 768},
        }
        save_storage_configs(configs, str(tmp_path))
        loaded = load_storage_config(str(tmp_path))
        assert "store_a" in loaded
        assert "store_b" in loaded


class TestLoadStorageConfig:
    def test_load_nonexistent(self, tmp_path):
        result = load_storage_config(str(tmp_path))
        assert result == {}

    def test_load_after_save(self, tmp_path):
        configs = {"test": {"key": "value"}}
        save_storage_configs(configs, str(tmp_path))
        loaded = load_storage_config(str(tmp_path))
        assert loaded["test"]["key"] == "value"


class TestSaveStorageConfig:
    def test_save_new_config(self, tmp_path):
        from unittest.mock import MagicMock

        model = MagicMock()
        model.to_dict.return_value = {"model_name": "test", "dim_size": 384}
        save_storage_config("test_store", str(tmp_path), model)
        loaded = load_storage_config(str(tmp_path))
        assert "test_store" in loaded

    def test_save_none_config_removes(self, tmp_path):
        # First save a config
        configs = {"test_store": {"model_name": "test"}}
        save_storage_configs(configs, str(tmp_path))
        # Then save None to remove it
        save_storage_config("test_store", str(tmp_path), None)
        loaded = load_storage_config(str(tmp_path))
        assert "test_store" not in loaded


class TestFindConfigDifferences:
    def test_no_differences(self):
        config = {"model_name": "test", "dim_size": "384", "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS}
        diffs = _find_config_differences("test", config, config.copy())
        assert diffs == []

    def test_value_mismatch(self):
        existing = {"model_name": "model-a", "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS}
        new = {"model_name": "model-b", "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS}
        diffs = _find_config_differences("test", existing, new)
        assert len(diffs) >= 1
        assert "model_name" in diffs[0]

    def test_missing_key(self):
        existing = {"model_name": "test", "extra_key": "value", "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS}
        new = {"model_name": "test", "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS}
        diffs = _find_config_differences("test", existing, new)
        assert len(diffs) >= 1
        assert "Missing key" in diffs[0]

    def test_none_configs_use_defaults(self):
        diffs = _find_config_differences("test", None, None)
        assert diffs == []


class TestCheckStorageConfig:
    def test_no_existing_config(self, tmp_path):
        check_storage_config("test", {"model_name": "test"}, str(tmp_path))
        loaded = load_storage_config(str(tmp_path))
        assert "test" in loaded

    def test_matching_config(self, tmp_path):
        config = {"model_name": "test", "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS}
        save_storage_configs({"test": config}, str(tmp_path))
        # Should not raise
        check_storage_config("test", config, str(tmp_path))

    def test_mismatched_config_raises(self, tmp_path):
        existing = {"model_name": "model-a", "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS}
        save_storage_configs({"test": existing}, str(tmp_path))
        new = {"model_name": "model-b", "registry_name": EmbeddingProvider.SENTENCE_TRANSFORMERS}
        with pytest.raises(DatusException):
            check_storage_config("test", new, str(tmp_path))

    def test_save_config_false(self, tmp_path):
        check_storage_config("test", {"model_name": "test"}, str(tmp_path), save_config=False)
        loaded = load_storage_config(str(tmp_path))
        assert "test" not in loaded
