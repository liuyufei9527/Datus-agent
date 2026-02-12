# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/models/sdk_patches.py"""

import copy
from unittest.mock import MagicMock, patch

import pytest

from datus.models.sdk_patches import (
    _is_kimi_model,
    _normalize_provider_data,
    _postprocess_messages_for_reasoning,
    _preprocess_items_for_reasoning,
    apply_sdk_patches,
    remove_sdk_patches,
)


# =====================================================================
# _is_kimi_model
# =====================================================================


class TestIsKimiModel:
    def test_kimi_k2(self):
        assert _is_kimi_model("kimi-k2") is True

    def test_kimi_k25(self):
        assert _is_kimi_model("kimi-k2.5") is True

    def test_kimi_k2_thinking(self):
        assert _is_kimi_model("kimi-k2-thinking") is True

    def test_moonshot(self):
        assert _is_kimi_model("moonshot-v1-8k") is True

    def test_k25_standalone(self):
        assert _is_kimi_model("k2.5") is True

    def test_k2_dash(self):
        assert _is_kimi_model("k2-something") is True

    def test_gpt_not_kimi(self):
        assert _is_kimi_model("gpt-4o") is False

    def test_deepseek_not_kimi(self):
        assert _is_kimi_model("deepseek-chat") is False

    def test_claude_not_kimi(self):
        assert _is_kimi_model("claude-sonnet-4") is False


# =====================================================================
# _normalize_provider_data
# =====================================================================


class TestNormalizeProviderData:
    def test_non_dict_passthrough(self):
        assert _normalize_provider_data("string") == "string"
        assert _normalize_provider_data(123) == 123

    def test_no_provider_data(self):
        item = {"role": "assistant", "content": "hello"}
        assert _normalize_provider_data(item) is item

    def test_provider_data_non_dict(self):
        item = {"provider_data": "not a dict"}
        assert _normalize_provider_data(item) is item

    def test_provider_data_no_model(self):
        item = {"provider_data": {"some_key": "some_value"}}
        assert _normalize_provider_data(item) is item

    def test_non_kimi_model_passthrough(self):
        item = {"provider_data": {"model": "gpt-4o"}}
        assert _normalize_provider_data(item) is item

    def test_kimi_model_normalized(self):
        item = {"provider_data": {"model": "kimi-k2.5"}}
        result = _normalize_provider_data(item)
        assert result["provider_data"]["model"] == "deepseek-kimi-k2.5"
        # Original should not be modified
        assert item["provider_data"]["model"] == "kimi-k2.5"

    def test_moonshot_model_normalized(self):
        item = {"provider_data": {"model": "moonshot-v1"}}
        result = _normalize_provider_data(item)
        assert result["provider_data"]["model"] == "deepseek-moonshot-v1"


# =====================================================================
# _preprocess_items_for_reasoning
# =====================================================================


class TestPreprocessItemsForReasoning:
    def test_string_items(self):
        items, model = _preprocess_items_for_reasoning("hello", "kimi-k2.5")
        assert items == "hello"
        assert model == "deepseek-kimi-k2.5"

    def test_string_items_non_kimi(self):
        items, model = _preprocess_items_for_reasoning("hello", "gpt-4o")
        assert items == "hello"
        assert model == "gpt-4o"

    def test_list_items_kimi(self):
        items_input = [
            {"provider_data": {"model": "kimi-k2.5"}},
            {"provider_data": {"model": "gpt-4o"}},
        ]
        items, model = _preprocess_items_for_reasoning(items_input, "kimi-k2.5")
        assert model == "deepseek-kimi-k2.5"
        # First item should be normalized
        assert items[0]["provider_data"]["model"] == "deepseek-kimi-k2.5"
        # Second item should not be modified
        assert items[1]["provider_data"]["model"] == "gpt-4o"

    def test_none_model(self):
        items, model = _preprocess_items_for_reasoning("hello", None)
        assert model is None


# =====================================================================
# _postprocess_messages_for_reasoning
# =====================================================================


class TestPostprocessMessagesForReasoning:
    def test_non_kimi_model_passthrough(self):
        messages = [{"role": "assistant", "content": "hello"}]
        result = _postprocess_messages_for_reasoning(messages, "gpt-4o")
        assert result is messages

    def test_none_model_passthrough(self):
        messages = [{"role": "assistant", "content": "hello"}]
        result = _postprocess_messages_for_reasoning(messages, None)
        assert result is messages

    def test_kimi_model_copies_reasoning_content(self):
        messages = [
            {"role": "assistant", "reasoning_content": "I think because...", "content": "answer"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "tool_calls": [{"id": "call_1"}], "content": None, "reasoning_content": ""},
        ]
        result = _postprocess_messages_for_reasoning(messages, "kimi-k2.5")
        # Tool call message should get the previous reasoning content
        assert result[2]["reasoning_content"] == "I think because..."
        # Content should be empty string, not None
        assert result[2]["content"] == ""

    def test_kimi_model_no_reasoning_to_copy(self):
        messages = [
            {"role": "assistant", "content": "hello"},
            {"role": "assistant", "tool_calls": [{"id": "call_1"}], "content": None},
        ]
        result = _postprocess_messages_for_reasoning(messages, "kimi-k2.5")
        # No reasoning to copy, content still becomes ""
        assert result[1]["content"] == ""

    def test_non_dict_messages_skipped(self):
        messages = [
            "string message",
            {"role": "assistant", "reasoning_content": "thinking...", "content": "answer"},
        ]
        result = _postprocess_messages_for_reasoning(messages, "kimi-k2.5")
        assert result[0] == "string message"


# =====================================================================
# apply_sdk_patches / remove_sdk_patches
# =====================================================================


class TestPatchLifecycle:
    def test_apply_and_remove_patches(self):
        apply_sdk_patches()
        # After applying, the patched methods should be in place
        from agents.models.chatcmpl_converter import Converter

        # The method should be patched (a classmethod wrapping our function)
        assert Converter.items_to_messages is not None

        remove_sdk_patches()
        # After removal, originals should be restored

    def test_double_apply_idempotent(self):
        apply_sdk_patches()
        apply_sdk_patches()  # Should not error
        remove_sdk_patches()

    def test_remove_without_apply_safe(self):
        # Reset globals to ensure clean state
        import datus.models.sdk_patches as sp

        sp._original_items_to_messages = None
        sp._original_acompletion = None
        # Should not error
        remove_sdk_patches()
