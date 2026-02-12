"""Tests for datus/storage/semantic_model/semantic_model_init.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

from datus.storage.semantic_model.semantic_model_init import (
    init_semantic_yaml_semantic_model,
    init_success_story_semantic_model,
    process_semantic_yaml_file,
)


# ── init_semantic_yaml_semantic_model ──


class TestInitSemanticYamlSemanticModel:
    def test_file_not_found(self, tmp_path):
        config = MagicMock()
        success, error = init_semantic_yaml_semantic_model(str(tmp_path / "missing.yaml"), config)
        assert success is False
        assert "not found" in error

    @patch("datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file")
    def test_delegates_to_process(self, mock_process, tmp_path):
        yaml_path = tmp_path / "model.yaml"
        yaml_path.write_text("model: test")
        config = MagicMock()
        mock_process.return_value = (True, "")

        success, error = init_semantic_yaml_semantic_model(str(yaml_path), config)
        assert success is True
        mock_process.assert_called_once_with(str(yaml_path), config, include_metrics=False)


# ── process_semantic_yaml_file ──


class TestProcessSemanticYamlFile:
    def test_file_not_found(self, tmp_path):
        config = MagicMock()
        success, error = process_semantic_yaml_file(str(tmp_path / "missing.yaml"), config)
        assert success is False
        assert "not found" in error

    @patch("datus.storage.semantic_model.semantic_model_init.GenerationHooks")
    def test_successful_sync(self, mock_hooks_cls, tmp_path):
        yaml_path = tmp_path / "model.yaml"
        yaml_path.write_text("model: test")
        config = MagicMock()
        mock_hooks_cls._sync_semantic_to_db.return_value = {"success": True, "message": "ok"}

        success, error = process_semantic_yaml_file(str(yaml_path), config)
        assert success is True
        assert error == ""

    @patch("datus.storage.semantic_model.semantic_model_init.GenerationHooks")
    def test_sync_failure(self, mock_hooks_cls, tmp_path):
        yaml_path = tmp_path / "model.yaml"
        yaml_path.write_text("model: test")
        config = MagicMock()
        mock_hooks_cls._sync_semantic_to_db.return_value = {"success": False, "error": "sync failed"}

        success, error = process_semantic_yaml_file(str(yaml_path), config)
        assert success is False
        assert "sync failed" in error

    @patch("datus.storage.semantic_model.semantic_model_init.GenerationHooks")
    def test_sync_exception(self, mock_hooks_cls, tmp_path):
        yaml_path = tmp_path / "model.yaml"
        yaml_path.write_text("model: test")
        config = MagicMock()
        mock_hooks_cls._sync_semantic_to_db.side_effect = RuntimeError("boom")

        success, error = process_semantic_yaml_file(str(yaml_path), config)
        assert success is False
        assert "boom" in error


# ── init_success_story_semantic_model ──


class TestInitSuccessStorySemanticModel:
    def _make_args(self, csv_path):
        args = MagicMock()
        args.success_story = csv_path
        return args

    def test_file_not_found(self, tmp_path):
        args = self._make_args(str(tmp_path / "missing.csv"))
        config = MagicMock()
        success, error = init_success_story_semantic_model(args, config)
        assert success is False
        assert "not found" in error

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        args = self._make_args(str(csv_path))
        config = MagicMock()
        success, error = init_success_story_semantic_model(args, config)
        assert success is False

    def test_missing_columns(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("col1,col2\na,b\n")
        args = self._make_args(str(csv_path))
        config = MagicMock()
        success, error = init_success_story_semantic_model(args, config)
        assert success is False
        assert "missing" in error.lower()

    def test_empty_data_rows(self, tmp_path):
        csv_path = tmp_path / "no_data.csv"
        csv_path.write_text("sql,question\n")
        args = self._make_args(str(csv_path))
        config = MagicMock()
        success, error = init_success_story_semantic_model(args, config)
        assert success is False
        assert "no data" in error.lower()
