# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/models/base.py"""

from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import AgentConfig, ModelConfig, NodeConfig
from datus.models.base import LLMBaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(tmp_path) -> AgentConfig:
    config_kwargs = {
        "home": str(tmp_path),
        "target": "test_model",
        "models": {
            "test_model": {
                "type": "openai",
                "api_key": "test-key",
                "model": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
            },
            "claude_model": {
                "type": "claude",
                "api_key": "claude-key",
                "model": "claude-sonnet-4",
                "base_url": "https://api.anthropic.com",
            },
        },
        "namespace": {
            "test_ns": {
                "type": "sqlite",
                "dbs": [
                    {
                        "uri": f"sqlite:///{tmp_path}/test.db",
                        "name": "test_db",
                    }
                ],
            },
        },
    }
    nodes: dict[str, NodeConfig] = {}
    return AgentConfig(nodes=nodes, **config_kwargs)


# =====================================================================
# LLMBaseModel class-level
# =====================================================================


class TestLLMBaseModel:
    def test_model_type_map(self):
        assert "openai" in LLMBaseModel.MODEL_TYPE_MAP
        assert "claude" in LLMBaseModel.MODEL_TYPE_MAP
        assert "deepseek" in LLMBaseModel.MODEL_TYPE_MAP
        assert "qwen" in LLMBaseModel.MODEL_TYPE_MAP
        assert "gemini" in LLMBaseModel.MODEL_TYPE_MAP
        assert "kimi" in LLMBaseModel.MODEL_TYPE_MAP

    def test_create_model_default(self, tmp_path):
        agent_config = _make_agent_config(tmp_path)
        model = LLMBaseModel.create_model(agent_config)
        assert model.model_config.model == "gpt-4o"

    def test_create_model_by_name(self, tmp_path):
        agent_config = _make_agent_config(tmp_path)
        with patch("datus.models.claude_model.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            model = LLMBaseModel.create_model(agent_config, model_name="claude_model")
            assert model.model_config.model == "claude-sonnet-4"

    def test_create_model_not_found(self, tmp_path):
        agent_config = _make_agent_config(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            LLMBaseModel.create_model(agent_config, model_name="nonexistent")

    def test_create_model_unsupported_type(self, tmp_path):
        agent_config = _make_agent_config(tmp_path)
        # Temporarily add an unsupported model type
        agent_config.models["bad_model"] = ModelConfig(
            type="unsupported_provider",
            api_key="key",
            model="model",
        )
        with pytest.raises(KeyError, match="Unsupported"):
            LLMBaseModel.create_model(agent_config, model_name="bad_model")

    def test_create_model_with_default_keyword(self, tmp_path):
        agent_config = _make_agent_config(tmp_path)
        model = LLMBaseModel.create_model(agent_config, model_name="default")
        assert model.model_config.model == "gpt-4o"

    def test_to_dict(self, tmp_path):
        agent_config = _make_agent_config(tmp_path)
        model = LLMBaseModel.create_model(agent_config)
        d = model.to_dict()
        assert d["model_name"] == "gpt-4o"

    def test_set_context(self, tmp_path):
        agent_config = _make_agent_config(tmp_path)
        model = LLMBaseModel.create_model(agent_config)
        mock_workflow = MagicMock()
        mock_node = MagicMock()
        model.set_context(workflow=mock_workflow, current_node=mock_node)
        assert model.workflow is mock_workflow
        assert model.current_node is mock_node

    def test_session_manager_lazy_init(self, tmp_path):
        agent_config = _make_agent_config(tmp_path)
        model = LLMBaseModel.create_model(agent_config)
        assert model._session_manager is None
        # Access triggers lazy init; SessionManager is imported lazily in the property
        with patch("datus.models.session_manager.SessionManager") as mock_sm:
            mock_sm.return_value = MagicMock()
            sm = model.session_manager
            assert sm is not None
