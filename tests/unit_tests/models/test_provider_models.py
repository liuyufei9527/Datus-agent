# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for provider-specific model implementations:
- datus/models/openai_model.py
- datus/models/qwen_model.py
- datus/models/gemini_model.py
- datus/models/kimi_model.py
- datus/models/deepseek_model.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import ModelConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_config(provider: str, model: str, api_key: str = "test-key", **kwargs) -> ModelConfig:
    return ModelConfig(
        type=provider,
        api_key=api_key,
        model=model,
        base_url=kwargs.pop("base_url", "https://test.api.com"),
        **kwargs,
    )


# =====================================================================
# OpenAIModel
# =====================================================================


class TestOpenAIModel:
    def test_init(self):
        from datus.models.openai_model import OpenAIModel

        cfg = _make_config("openai", "gpt-4o", api_key="openai-key")
        model = OpenAIModel(cfg)
        assert model.model_name == "gpt-4o"
        assert model.api_key == "openai-key"

    def test_get_api_key_from_env(self):
        from datus.models.openai_model import OpenAIModel

        cfg = _make_config("openai", "gpt-4o", api_key="")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}):
            model = OpenAIModel(cfg)
            assert model.api_key == "env-openai-key"

    def test_get_api_key_missing(self):
        from datus.models.openai_model import OpenAIModel

        cfg = _make_config("openai", "gpt-4o", api_key="")
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="OpenAI API key"):
                    OpenAIModel(cfg)
            finally:
                if env_backup:
                    os.environ["OPENAI_API_KEY"] = env_backup

    def test_uses_completion_tokens_for_o_series(self):
        from datus.models.openai_model import OpenAIModel

        for model_name in ["o1", "o1-mini", "o3", "o3-mini", "o4-mini"]:
            cfg = _make_config("openai", model_name, api_key="key")
            model = OpenAIModel(cfg)
            assert model._uses_completion_tokens_parameter() is True

    def test_does_not_use_completion_tokens_for_gpt(self):
        from datus.models.openai_model import OpenAIModel

        cfg = _make_config("openai", "gpt-4o", api_key="key")
        model = OpenAIModel(cfg)
        assert model._uses_completion_tokens_parameter() is False

    def test_gpt_o4_pattern(self):
        from datus.models.openai_model import OpenAIModel

        cfg = _make_config("openai", "gpt-o4-mini", api_key="key")
        model = OpenAIModel(cfg)
        assert model._uses_completion_tokens_parameter() is True

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_reasoning_model_transforms_params(self, mock_litellm):
        from datus.models.openai_model import OpenAIModel

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "o3"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        cfg = _make_config("openai", "o3", api_key="key")
        model = OpenAIModel(cfg)
        result = model.generate("test", max_tokens=1000, temperature=0.7)
        assert result == "result"

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_gpt5_sets_temperature(self, mock_litellm):
        from datus.models.openai_model import OpenAIModel

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-5"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        cfg = _make_config("openai", "gpt-5", api_key="key")
        model = OpenAIModel(cfg)
        model.generate("test")

        call_kwargs = mock_litellm.completion.call_args
        # gpt-5 should have temperature=1
        assert call_kwargs[1].get("temperature") == 1 or call_kwargs.kwargs.get("temperature") == 1


# =====================================================================
# DeepSeekModel
# =====================================================================


class TestDeepSeekModel:
    def test_init(self):
        from datus.models.deepseek_model import DeepSeekModel

        cfg = _make_config("deepseek", "deepseek-chat", api_key="ds-key")
        model = DeepSeekModel(cfg)
        assert model.model_name == "deepseek-chat"

    def test_get_api_key_from_env(self):
        from datus.models.deepseek_model import DeepSeekModel

        cfg = _make_config("deepseek", "deepseek-chat", api_key="")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-ds-key"}):
            model = DeepSeekModel(cfg)
            assert model.api_key == "env-ds-key"

    def test_get_api_key_missing(self):
        from datus.models.deepseek_model import DeepSeekModel

        cfg = _make_config("deepseek", "deepseek-chat", api_key="")
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("DEEPSEEK_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="DeepSeek API key"):
                    DeepSeekModel(cfg)
            finally:
                if env_backup:
                    os.environ["DEEPSEEK_API_KEY"] = env_backup

    def test_get_base_url_default(self):
        from datus.models.deepseek_model import DeepSeekModel

        cfg = _make_config("deepseek", "deepseek-chat", api_key="key", base_url=None)
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("DEEPSEEK_API_BASE", None)
            try:
                model = DeepSeekModel(cfg)
                assert model.base_url == "https://api.deepseek.com"
            finally:
                if env_backup:
                    os.environ["DEEPSEEK_API_BASE"] = env_backup


# =====================================================================
# QwenModel
# =====================================================================


class TestQwenModel:
    def test_init(self):
        from datus.models.qwen_model import QwenModel

        cfg = _make_config("qwen", "qwen3-coder", api_key="qwen-key")
        model = QwenModel(cfg)
        assert model.model_name == "qwen3-coder"

    def test_get_api_key_from_env(self):
        from datus.models.qwen_model import QwenModel

        cfg = _make_config("qwen", "qwen3-coder", api_key="")
        with patch.dict(os.environ, {"QWEN_API_KEY": "env-qwen-key"}):
            model = QwenModel(cfg)
            assert model.api_key == "env-qwen-key"

    def test_get_api_key_missing(self):
        from datus.models.qwen_model import QwenModel

        cfg = _make_config("qwen", "qwen3-coder", api_key="")
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("QWEN_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="Qwen API key"):
                    QwenModel(cfg)
            finally:
                if env_backup:
                    os.environ["QWEN_API_KEY"] = env_backup

    def test_get_base_url_default(self):
        from datus.models.qwen_model import QwenModel

        cfg = _make_config("qwen", "qwen3-coder", api_key="key", base_url=None)
        model = QwenModel(cfg)
        assert "dashscope" in model.base_url


# =====================================================================
# GeminiModel
# =====================================================================


class TestGeminiModel:
    def test_init(self):
        from datus.models.gemini_model import GeminiModel

        cfg = _make_config("gemini", "gemini-2.5-pro", api_key="gemini-key")
        model = GeminiModel(cfg)
        assert model.model_name == "gemini-2.5-pro"

    def test_get_api_key_from_env(self):
        from datus.models.gemini_model import GeminiModel

        cfg = _make_config("gemini", "gemini-2.5-pro", api_key="")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-gemini-key"}):
            model = GeminiModel(cfg)
            assert model.api_key == "env-gemini-key"

    def test_get_api_key_missing(self):
        from datus.models.gemini_model import GeminiModel

        cfg = _make_config("gemini", "gemini-2.5-pro", api_key="")
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("GEMINI_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="Gemini API key"):
                    GeminiModel(cfg)
            finally:
                if env_backup:
                    os.environ["GEMINI_API_KEY"] = env_backup

    def test_get_base_url_none_default(self):
        from datus.models.gemini_model import GeminiModel

        cfg = _make_config("gemini", "gemini-2.5-pro", api_key="key", base_url=None)
        model = GeminiModel(cfg)
        assert model.base_url is None

    def test_model_specs(self):
        from datus.models.gemini_model import GeminiModel

        cfg = _make_config("gemini", "gemini-2.5-pro", api_key="key")
        model = GeminiModel(cfg)
        specs = model.model_specs
        assert "gemini-2.5-pro" in specs
        assert "gemini-2.0-flash" in specs
        assert "gemini-1.5-pro" in specs


# =====================================================================
# KimiModel
# =====================================================================


class TestKimiModel:
    def test_init(self):
        from datus.models.kimi_model import KimiModel

        cfg = _make_config("kimi", "kimi-k2.5", api_key="kimi-key")
        model = KimiModel(cfg)
        assert model.model_name == "kimi-k2.5"

    def test_get_api_key_from_env(self):
        from datus.models.kimi_model import KimiModel

        cfg = _make_config("kimi", "kimi-k2", api_key="")
        with patch.dict(os.environ, {"KIMI_API_KEY": "env-kimi-key"}):
            model = KimiModel(cfg)
            assert model.api_key == "env-kimi-key"

    def test_get_api_key_missing(self):
        from datus.models.kimi_model import KimiModel

        cfg = _make_config("kimi", "kimi-k2", api_key="")
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("KIMI_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="Kimi API key"):
                    KimiModel(cfg)
            finally:
                if env_backup:
                    os.environ["KIMI_API_KEY"] = env_backup

    def test_get_base_url_default(self):
        from datus.models.kimi_model import KimiModel

        cfg = _make_config("kimi", "kimi-k2", api_key="key", base_url=None)
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("KIMI_API_BASE", None)
            try:
                model = KimiModel(cfg)
                assert "moonshot" in model.base_url
            finally:
                if env_backup:
                    os.environ["KIMI_API_BASE"] = env_backup
