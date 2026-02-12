# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/models/litellm_adapter.py"""

from unittest.mock import MagicMock, patch

import pytest

from datus.models.litellm_adapter import LiteLLMAdapter, create_litellm_adapter


# =====================================================================
# LiteLLMAdapter initialization
# =====================================================================


class TestLiteLLMAdapterInit:
    def test_basic_init(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
        )
        assert adapter.provider == "openai"
        assert adapter.model == "gpt-4o"
        assert adapter.api_key == "test-key"

    def test_auto_detect_kimi_from_model(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="kimi-k2.5",
            api_key="test-key",
        )
        assert adapter.provider == "kimi"

    def test_auto_detect_claude_from_model(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="claude-sonnet-4",
            api_key="test-key",
        )
        assert adapter.provider == "claude"

    def test_auto_detect_deepseek_from_model(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="deepseek-chat",
            api_key="test-key",
        )
        assert adapter.provider == "deepseek"

    def test_auto_detect_qwen_from_model(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="qwen3-coder",
            api_key="test-key",
        )
        assert adapter.provider == "qwen"

    def test_auto_detect_gemini_from_model(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gemini-2.5-pro",
            api_key="test-key",
        )
        assert adapter.provider == "gemini"

    def test_auto_detect_moonshot_from_model(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="moonshot-v1-8k",
            api_key="test-key",
        )
        assert adapter.provider == "kimi"

    def test_no_auto_detect_for_unknown(self):
        adapter = LiteLLMAdapter(
            provider="custom",
            model="my-custom-model",
            api_key="test-key",
        )
        assert adapter.provider == "custom"

    def test_custom_base_url(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            base_url="https://custom.api.com",
        )
        assert adapter.base_url == "https://custom.api.com"

    def test_default_base_url_for_deepseek(self):
        adapter = LiteLLMAdapter(
            provider="deepseek",
            model="deepseek-chat",
            api_key="test-key",
        )
        assert adapter.base_url == "https://api.deepseek.com"

    def test_default_base_url_for_openai(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
        )
        assert adapter.base_url is None

    def test_enable_thinking(self):
        adapter = LiteLLMAdapter(
            provider="deepseek",
            model="deepseek-reasoner",
            api_key="test-key",
            enable_thinking=True,
        )
        assert adapter.is_thinking_model is True

    def test_thinking_disabled_by_default(self):
        adapter = LiteLLMAdapter(
            provider="deepseek",
            model="deepseek-chat",
            api_key="test-key",
        )
        assert adapter.is_thinking_model is False


# =====================================================================
# litellm_model_name
# =====================================================================


class TestLiteLLMModelName:
    def test_openai_no_prefix(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        assert adapter.litellm_model_name == "gpt-4o"

    def test_claude_prefix(self):
        adapter = LiteLLMAdapter(provider="claude", model="claude-sonnet-4", api_key="key")
        assert adapter.litellm_model_name == "anthropic/claude-sonnet-4"

    def test_deepseek_prefix(self):
        adapter = LiteLLMAdapter(provider="deepseek", model="deepseek-chat", api_key="key")
        assert adapter.litellm_model_name == "deepseek/deepseek-chat"

    def test_qwen_prefix(self):
        adapter = LiteLLMAdapter(provider="qwen", model="qwen3-coder", api_key="key")
        assert adapter.litellm_model_name == "dashscope/qwen3-coder"

    def test_gemini_prefix(self):
        adapter = LiteLLMAdapter(provider="gemini", model="gemini-2.5-pro", api_key="key")
        assert adapter.litellm_model_name == "gemini/gemini-2.5-pro"

    def test_kimi_prefix(self):
        adapter = LiteLLMAdapter(provider="kimi", model="kimi-k2.5", api_key="key")
        assert adapter.litellm_model_name == "moonshot/kimi-k2.5"

    def test_model_with_existing_prefix(self):
        adapter = LiteLLMAdapter(provider="openai", model="custom/my-model", api_key="key")
        assert adapter.litellm_model_name == "custom/my-model"

    def test_unknown_provider_no_prefix(self):
        adapter = LiteLLMAdapter(provider="my_provider", model="my-model", api_key="key")
        assert adapter.litellm_model_name == "my-model"

    def test_caching_model_name(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        name1 = adapter.litellm_model_name
        name2 = adapter.litellm_model_name
        assert name1 is name2  # Cached


# =====================================================================
# get_agents_sdk_model
# =====================================================================


class TestGetAgentsSdkModel:
    @patch("datus.models.litellm_adapter.LiteLLMAdapter.get_agents_sdk_model")
    def test_returns_litellm_model(self, mock_get):
        mock_model = MagicMock()
        mock_get.return_value = mock_model
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        result = adapter.get_agents_sdk_model()
        assert result is mock_model


# =====================================================================
# get_completion_kwargs
# =====================================================================


class TestGetCompletionKwargs:
    def test_basic_kwargs(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        kwargs = adapter.get_completion_kwargs()
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["api_key"] == "key"
        assert "api_base" not in kwargs

    def test_kwargs_with_base_url(self):
        adapter = LiteLLMAdapter(
            provider="deepseek", model="deepseek-chat", api_key="key", base_url="https://api.deepseek.com"
        )
        kwargs = adapter.get_completion_kwargs()
        assert kwargs["api_base"] == "https://api.deepseek.com"


# =====================================================================
# create_litellm_adapter factory
# =====================================================================


class TestCreateLiteLLMAdapter:
    def test_factory(self):
        adapter = create_litellm_adapter(
            provider="openai",
            model="gpt-4o",
            api_key="key",
            base_url="https://custom.api.com",
            enable_thinking=True,
        )
        assert isinstance(adapter, LiteLLMAdapter)
        assert adapter.provider == "openai"
        assert adapter.is_thinking_model is True
