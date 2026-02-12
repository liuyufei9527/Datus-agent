# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/configuration/agent_config_loader.py"""

import os
from pathlib import Path

import pytest
import yaml

from datus.configuration.agent_config_loader import (
    ConfigurationManager,
    configuration_manager,
    load_node_config,
    parse_config_path,
)
from datus.configuration.node_type import NodeType
from datus.utils.exceptions import DatusException


# =====================================================================
# parse_config_path
# =====================================================================


class TestParseConfigPath:
    def test_explicit_path_exists(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")
        result = parse_config_path(str(config_file))
        assert result == config_file

    def test_explicit_path_not_found(self):
        with pytest.raises(DatusException, match="not found"):
            parse_config_path("/nonexistent/path/agent.yml")

    def test_default_config_file_not_found(self, tmp_path, monkeypatch):
        # Change to tmp_path so no conf/agent.yml exists
        monkeypatch.chdir(tmp_path)
        # Also ensure ~/.datus/conf/agent.yml doesn't exist for this test
        home_config = Path.home() / ".datus" / "conf" / "agent.yml"
        if not home_config.exists():
            with pytest.raises(DatusException, match="not found"):
                parse_config_path("")

    def test_local_config_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        config_file = conf_dir / "agent.yml"
        config_file.write_text("agent: {}")
        result = parse_config_path("")
        assert result == Path("conf/agent.yml")


# =====================================================================
# load_node_config
# =====================================================================


class TestLoadNodeConfig:
    def test_with_model(self):
        data = {"model": "gpt-4o"}
        result = load_node_config(NodeType.TYPE_SCHEMA_LINKING, data)
        assert result.model == "gpt-4o"

    def test_without_model(self):
        data = {}
        result = load_node_config(NodeType.TYPE_OUTPUT, data)
        assert result.model == ""

    def test_none_data(self):
        result = load_node_config(NodeType.TYPE_OUTPUT, None)
        assert result.model == ""


# =====================================================================
# ConfigurationManager
# =====================================================================


class TestConfigurationManager:
    def test_load_config(self, tmp_path):
        config_data = {
            "agent": {
                "target": "test",
                "models": {
                    "test": {
                        "type": "openai",
                        "api_key": "key",
                        "model": "gpt-4o",
                    }
                },
            }
        }
        config_file = tmp_path / "agent.yml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        cm = ConfigurationManager(str(config_file))
        assert cm.get("target") == "test"

    def test_get_default(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  target: test")
        cm = ConfigurationManager(str(config_file))
        assert cm.get("nonexistent", "default") == "default"

    def test_update_item(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  target: test")
        cm = ConfigurationManager(str(config_file))
        cm.update_item("target", "new_target")
        assert cm.get("target") == "new_target"

        # Verify persisted
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["agent"]["target"] == "new_target"

    def test_update_dict_merges(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  models:\n    test:\n      type: openai")
        cm = ConfigurationManager(str(config_file))
        cm.update_item("models", {"test2": {"type": "claude"}})
        models = cm.get("models")
        assert "test" in models
        assert "test2" in models

    def test_update_item_delete_old_key(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  models:\n    old_key: old_value")
        cm = ConfigurationManager(str(config_file))
        cm.update_item("models", {"new_key": "new_value"}, delete_old_key=True)
        models = cm.get("models")
        assert "old_key" not in models
        assert "new_key" in models

    def test_update_batch(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  target: old\n  schema_linking_rate: fast")
        cm = ConfigurationManager(str(config_file))
        result = cm.update({"target": "new", "schema_linking_rate": "slow"})
        assert result is True
        assert cm.get("target") == "new"
        assert cm.get("schema_linking_rate") == "slow"

    def test_remove_item_recursively(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  models:\n    test:\n      type: openai")
        cm = ConfigurationManager(str(config_file))
        result = cm.remove_item_recursively("models", "test")
        assert result is True
        assert "test" not in cm.get("models")

    def test_remove_item_recursively_invalid_path(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  target: test")
        cm = ConfigurationManager(str(config_file))
        with pytest.raises(DatusException, match="does not exist"):
            cm.remove_item_recursively("nonexistent", "key")

    def test_remove_item_recursively_empty_keys(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  target: test")
        cm = ConfigurationManager(str(config_file))
        result = cm.remove_item_recursively()
        assert result is False

    def test_getitem_setitem(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  target: test")
        cm = ConfigurationManager(str(config_file))
        assert cm["target"] == "test"
        cm["target"] = "new"
        assert cm["target"] == "new"

    def test_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("invalid: yaml: content: [")
        cm = ConfigurationManager(str(config_file))
        # Should handle gracefully
        assert cm.data == {}


# =====================================================================
# configuration_manager singleton
# =====================================================================


class TestConfigurationManagerSingleton:
    def test_reload(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  target: test")

        cm1 = configuration_manager(str(config_file), reload=True)
        cm2 = configuration_manager(str(config_file))
        assert cm1 is cm2

    def test_reload_flag(self, tmp_path):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent:\n  target: test")
        cm1 = configuration_manager(str(config_file), reload=True)

        config_file.write_text("agent:\n  target: new_test")
        cm2 = configuration_manager(str(config_file), reload=True)
        assert cm2.get("target") == "new_test"
