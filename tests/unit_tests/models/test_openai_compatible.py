# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/models/openai_compatible.py"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import (
    OpenAICompatibleModel,
    classify_openai_compatible_error,
)
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.utils.exceptions import DatusException, ErrorCode


# ---------------------------------------------------------------------------
# Helper: Concrete subclass for testing (OpenAICompatibleModel is abstract-ish
# because _get_api_key raises NotImplementedError)
# ---------------------------------------------------------------------------

class ConcreteModel(OpenAICompatibleModel):
    """Minimal concrete subclass for testing."""

    def _get_api_key(self) -> str:
        return self.model_config.api_key or "test-key"


def _make_model_config(**overrides) -> ModelConfig:
    defaults = {
        "type": "openai",
        "api_key": "test-api-key",
        "model": "gpt-4o",
        "base_url": "https://api.test.com/v1",
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_model(**overrides) -> ConcreteModel:
    cfg = _make_model_config(**overrides)
    return ConcreteModel(cfg)


# =====================================================================
# classify_openai_compatible_error
# =====================================================================


class TestClassifyError:
    def test_auth_error(self):
        err = APIError(message="401 unauthorized", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_AUTHENTICATION_ERROR
        assert retryable is False

    def test_permission_error(self):
        err = APIError(message="403 forbidden", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_PERMISSION_ERROR
        assert retryable is False

    def test_not_found_error(self):
        err = APIError(message="404 not found", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_NOT_FOUND
        assert retryable is False

    def test_request_too_large(self):
        err = APIError(message="413 too large", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_REQUEST_TOO_LARGE
        assert retryable is False

    def test_rate_limit_429(self):
        err = APIError(message="429 rate limit exceeded", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_RATE_LIMIT
        assert retryable is True

    def test_quota_exceeded(self):
        err = APIError(message="429 quota exceeded", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_QUOTA_EXCEEDED
        assert retryable is False

    def test_billing_error(self):
        err = APIError(message="billing limit exceeded", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_QUOTA_EXCEEDED
        assert retryable is False

    def test_server_error_500(self):
        err = APIError(message="500 internal server error", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_API_ERROR
        assert retryable is True

    def test_overloaded_503(self):
        err = APIError(message="503 overloaded", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_OVERLOADED
        assert retryable is True

    def test_bad_request_400(self):
        err = APIError(message="400 bad request", request=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_INVALID_RESPONSE
        assert retryable is False

    def test_rate_limit_error_class(self):
        err = RateLimitError(message="rate limit", response=MagicMock(), body=None)
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_RATE_LIMIT
        assert retryable is True

    def test_timeout_error_class(self):
        err = APITimeoutError(request=MagicMock())
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_TIMEOUT_ERROR
        assert retryable is True

    def test_connection_error_class(self):
        err = APIConnectionError(request=MagicMock())
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_CONNECTION_ERROR
        assert retryable is True

    def test_generic_exception(self):
        err = Exception("something unexpected")
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_REQUEST_FAILED
        assert retryable is False


# =====================================================================
# OpenAICompatibleModel.__init__ and properties
# =====================================================================


class TestModelInit:
    def test_basic_init(self):
        model = _make_model()
        assert model.model_name == "gpt-4o"
        assert model.api_key == "test-api-key"
        assert model.base_url == "https://api.test.com/v1"

    def test_default_headers(self):
        model = _make_model(default_headers={"X-Custom": "val"})
        assert model.default_headers == {"X-Custom": "val"}

    def test_no_base_url(self):
        model = _make_model(base_url=None)
        assert model.base_url is None


# =====================================================================
# model_specs / max_tokens / context_length
# =====================================================================


class TestModelSpecs:
    def test_exact_match(self):
        model = _make_model(model="gpt-4o")
        assert model.context_length() == 128000
        assert model.max_tokens() == 16384

    def test_prefix_match(self):
        model = _make_model(model="gpt-4o-mini-2024")
        assert model.context_length() == 128000
        assert model.max_tokens() == 16384

    def test_unknown_model(self):
        model = _make_model(model="unknown-model")
        assert model.context_length() is None
        assert model.max_tokens() is None

    def test_deepseek_chat(self):
        model = _make_model(model="deepseek-chat")
        assert model.context_length() == 65535
        assert model.max_tokens() == 8192

    def test_kimi_k2(self):
        model = _make_model(model="kimi-k2")
        assert model.context_length() == 256000
        assert model.max_tokens() == 8192

    def test_gemini_model(self):
        model = _make_model(model="gemini-2.5-pro")
        assert model.context_length() == 1048576
        assert model.max_tokens() == 65535


# =====================================================================
# _with_retry (sync)
# =====================================================================


class TestWithRetry:
    def test_success_first_try(self):
        model = _make_model()
        result = model._with_retry(lambda: "ok", "test_op", max_retries=3, base_delay=0.001)
        assert result == "ok"

    def test_retries_on_retryable_error(self):
        model = _make_model()
        call_count = 0

        def op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APITimeoutError(request=MagicMock())
            return "success"

        result = model._with_retry(op, "test_op", max_retries=3, base_delay=0.001)
        assert result == "success"
        assert call_count == 3

    def test_raises_datus_exception_on_non_retryable(self):
        model = _make_model()

        def op():
            raise APIError(message="401 unauthorized", request=MagicMock(), body=None)

        with pytest.raises(DatusException):
            model._with_retry(op, "test_op", max_retries=3, base_delay=0.001)

    def test_raises_datus_exception_after_max_retries(self):
        model = _make_model()

        def op():
            raise APITimeoutError(request=MagicMock())

        with pytest.raises(DatusException):
            model._with_retry(op, "test_op", max_retries=1, base_delay=0.001)

    def test_unexpected_error_propagates(self):
        model = _make_model()

        def op():
            raise ValueError("unexpected")

        with pytest.raises(ValueError, match="unexpected"):
            model._with_retry(op, "test_op", max_retries=3, base_delay=0.001)


# =====================================================================
# _with_retry_async
# =====================================================================


class TestWithRetryAsync:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        model = _make_model()

        async def op():
            return "ok"

        result = await model._with_retry_async(op, "test_op", max_retries=3, base_delay=0.001)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self):
        model = _make_model()
        call_count = 0

        async def op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIConnectionError(request=MagicMock())
            return "success"

        result = await model._with_retry_async(op, "test_op", max_retries=3, base_delay=0.001)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_datus_exception_after_max_retries(self):
        model = _make_model()

        async def op():
            raise APITimeoutError(request=MagicMock())

        with pytest.raises(DatusException):
            await model._with_retry_async(op, "test_op", max_retries=1, base_delay=0.001)

    @pytest.mark.asyncio
    async def test_unexpected_error_propagates(self):
        model = _make_model()

        async def op():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await model._with_retry_async(op, "test_op", max_retries=3, base_delay=0.001)


# =====================================================================
# generate (mocking litellm.completion)
# =====================================================================


class TestGenerate:
    @patch("datus.models.openai_compatible.litellm")
    def test_generate_string_prompt(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response
        mock_litellm.token_counter.return_value = 5

        model = _make_model()
        result = model.generate("test prompt")
        assert result == "Hello world"

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_list_prompt(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate([{"role": "user", "content": "hello"}])
        assert result == "result"

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_with_temperature_from_config(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "test"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model(model="test-model", temperature=0.5)
        model.generate("hello")

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.5 or call_kwargs[1].get("temperature") == 0.5

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_with_thinking(self, mock_litellm):
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "final answer"
        mock_msg.reasoning_content = "I think..."
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate("hello", enable_thinking=True)
        assert result == "final answer"

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_null_content(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate("hello")
        assert result == ""


# =====================================================================
# generate_with_json_output
# =====================================================================


class TestGenerateWithJsonOutput:
    @patch("datus.models.openai_compatible.litellm")
    def test_valid_json(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate_with_json_output("give me json")
        assert result == {"key": "value"}

    @patch("datus.models.openai_compatible.litellm")
    def test_json_in_text(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Here is your json: {"key": "value"} and some extra text'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate_with_json_output("give me json")
        assert result == {"key": "value"}

    @patch("datus.models.openai_compatible.litellm")
    def test_invalid_json(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not json at all"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate_with_json_output("give me json")
        assert "error" in result
        assert "raw_response" in result


# =====================================================================
# _format_tool_result / _format_tool_result_from_dict
# =====================================================================


class TestFormatToolResult:
    def test_empty_content(self):
        model = _make_model()
        assert model._format_tool_result("") == "Empty result"

    def test_json_dict_with_result_list(self):
        model = _make_model()
        content = json.dumps({"result": [1, 2, 3]})
        assert model._format_tool_result(content) == "3 items"

    def test_json_dict_with_result_int(self):
        model = _make_model()
        content = json.dumps({"result": 42})
        assert model._format_tool_result(content) == "42 rows"

    def test_json_dict_with_rows(self):
        model = _make_model()
        content = json.dumps({"rows": 10})
        assert model._format_tool_result(content) == "10 rows"

    def test_json_dict_with_items(self):
        model = _make_model()
        content = json.dumps({"items": [1, 2]})
        assert model._format_tool_result(content) == "2 items"

    def test_json_list(self):
        model = _make_model()
        content = json.dumps([1, 2, 3, 4])
        assert model._format_tool_result(content) == "4 items"

    def test_non_json_string(self):
        model = _make_model()
        result = model._format_tool_result("plain text result")
        assert "plain text result" in result

    def test_long_non_json_string(self):
        model = _make_model()
        long_text = "a" * 200
        result = model._format_tool_result(long_text)
        assert result.endswith("...")

    def test_from_dict_success_only(self):
        model = _make_model()
        assert model._format_tool_result_from_dict({"success": True}) == "Success"
        assert model._format_tool_result_from_dict({"success": False}) == "Failed"

    def test_from_dict_count(self):
        model = _make_model()
        assert model._format_tool_result_from_dict({"count": 5}) == "5 items"

    def test_from_dict_result_dict_with_count(self):
        model = _make_model()
        assert model._format_tool_result_from_dict({"result": {"count": 3}}) == "3 items"

    def test_from_dict_result_dict_without_count(self):
        model = _make_model()
        assert model._format_tool_result_from_dict({"result": {"foo": "bar"}}) == "Success"

    def test_from_dict_result_str(self):
        model = _make_model()
        assert model._format_tool_result_from_dict({"result": "some string"}) == "Success"

    def test_from_dict_generic(self):
        model = _make_model()
        assert model._format_tool_result_from_dict({"foo": "bar"}) == "Success"

    def test_from_dict_rows_non_int(self):
        model = _make_model()
        assert model._format_tool_result_from_dict({"rows": "many"}) == "Success"


# =====================================================================
# _distribute_token_usage_to_actions / _add_usage_to_action
# =====================================================================


class TestDistributeTokenUsage:
    def test_add_usage_to_none_output(self):
        model = _make_model()
        action = ActionHistory(
            action_id="test",
            role=ActionRole.ASSISTANT,
            messages="test",
            action_type="response",
            input={},
            output=None,
            status=ActionStatus.SUCCESS,
        )
        model._add_usage_to_action(action, {"total_tokens": 100})
        assert action.output == {"usage": {"total_tokens": 100}}

    def test_add_usage_to_non_dict_output(self):
        model = _make_model()
        action = ActionHistory(
            action_id="test",
            role=ActionRole.ASSISTANT,
            messages="test",
            action_type="response",
            input={},
            output="some string",
            status=ActionStatus.SUCCESS,
        )
        model._add_usage_to_action(action, {"total_tokens": 100})
        assert action.output["raw_output"] == "some string"
        assert action.output["usage"] == {"total_tokens": 100}

    def test_distribute_to_final_assistant(self):
        model = _make_model()
        manager = ActionHistoryManager()

        tool_action = ActionHistory(
            action_id="tool_1",
            role=ActionRole.TOOL,
            messages="tool call",
            action_type="read_query",
            input={},
            output={},
            status=ActionStatus.SUCCESS,
        )
        assistant_action = ActionHistory(
            action_id="asst_1",
            role=ActionRole.ASSISTANT,
            messages="response",
            action_type="response",
            input={},
            output={},
            status=ActionStatus.SUCCESS,
        )
        manager.add_action(tool_action)
        manager.add_action(assistant_action)

        usage = {"total_tokens": 500, "input_tokens": 300, "output_tokens": 200}
        model._distribute_token_usage_to_actions(manager, usage)

        assert assistant_action.output["usage"]["total_tokens"] == 500
        assert "usage" not in (tool_action.output or {})


# =====================================================================
# token_count
# =====================================================================


class TestTokenCount:
    @patch("datus.models.openai_compatible.litellm")
    def test_token_count(self, mock_litellm):
        mock_litellm.token_counter.return_value = 42
        model = _make_model()
        assert model.token_count("hello world") == 42

    @patch("datus.models.openai_compatible.litellm")
    def test_token_count_fallback(self, mock_litellm):
        mock_litellm.token_counter.side_effect = Exception("fail")
        model = _make_model()
        result = model.token_count("hello world")
        # Fallback is len // 4
        assert result == len("hello world") // 4


# =====================================================================
# _setup_custom_json_encoder
# =====================================================================


class TestCustomJsonEncoder:
    def test_anyurl_serialization(self):
        from pydantic import AnyUrl

        model = _make_model()
        model._setup_custom_json_encoder()
        url = AnyUrl("https://example.com")
        result = json.dumps({"url": url})
        assert "example.com" in result

    def test_date_serialization(self):
        from datetime import date

        model = _make_model()
        model._setup_custom_json_encoder()
        d = date(2024, 1, 15)
        result = json.dumps({"date": d})
        assert "2024-01-15" in result

    def test_datetime_serialization(self):
        from datetime import datetime

        model = _make_model()
        model._setup_custom_json_encoder()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = json.dumps({"datetime": dt})
        assert "2024-01-15" in result


# =====================================================================
# _save_llm_trace
# =====================================================================


class TestSaveLlmTrace:
    def test_trace_disabled(self):
        model = _make_model(save_llm_trace=False)
        # Should return early without error
        model._save_llm_trace("prompt", "response")

    def test_trace_enabled_no_workflow(self):
        model = _make_model(save_llm_trace=True)
        # No workflow set, should return early
        model._save_llm_trace("prompt", "response")

    def test_trace_enabled_with_workflow(self, tmp_path):
        model = _make_model(save_llm_trace=True)

        # Mock workflow and node
        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task.id = "test_task"
        model.workflow = mock_workflow

        mock_node = MagicMock()
        mock_node.id = "test_node"
        model.current_node = mock_node

        # Test with string prompt
        model._save_llm_trace("test prompt", "test response", "reasoning")
        trace_file = tmp_path / "test_task" / "test_node.yml"
        assert trace_file.exists()

    def test_trace_with_message_list(self, tmp_path):
        model = _make_model(save_llm_trace=True)

        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task.id = "test_task"
        model.workflow = mock_workflow

        mock_node = MagicMock()
        mock_node.id = "test_node"
        model.current_node = mock_node

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        model._save_llm_trace(messages, "final response")
        trace_file = tmp_path / "test_task" / "test_node.yml"
        assert trace_file.exists()


# =====================================================================
# generate - additional parameter paths
# =====================================================================


class TestGenerateAdvanced:
    @patch("datus.models.openai_compatible.litellm")
    def test_generate_with_top_p_from_config(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "test"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model(model="test-model", top_p=0.9)
        model.generate("hello")

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("top_p") == 0.9 or call_kwargs[1].get("top_p") == 0.9

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_with_max_tokens_kwarg(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        model.generate("hello", max_tokens=2000)
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 2000 or call_kwargs[1].get("max_tokens") == 2000

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_with_base_url(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model(base_url="https://custom.api.com/v1")
        model.generate("hello")
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("api_base") == "https://custom.api.com/v1" or call_kwargs[1].get("api_base") == "https://custom.api.com/v1"

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_with_usage_logging(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate("hello")
        assert result == "result"

    @patch("datus.models.openai_compatible.litellm")
    def test_generate_with_thinking_reasoning_content_only(self, mock_litellm):
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = ""  # Empty main content
        mock_msg.reasoning_content = "I think the answer is..."
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "deepseek-r1"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model(model="deepseek-r1")
        result = model.generate("hello", enable_thinking=True)
        assert "I think the answer is..." in result


# =====================================================================
# generate_with_json_output - additional paths
# =====================================================================


class TestGenerateWithJsonOutputAdvanced:
    @patch("datus.models.openai_compatible.litellm")
    def test_json_in_markdown_code_block(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate_with_json_output("give me json")
        assert result == {"key": "value"}

    @patch("datus.models.openai_compatible.litellm")
    def test_nested_json(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"outer": {"inner": [1, 2, 3]}}'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate_with_json_output("give me json")
        assert result == {"outer": {"inner": [1, 2, 3]}}

    @patch("datus.models.openai_compatible.litellm")
    def test_with_enable_thinking(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "thinking"}'
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_litellm.completion.return_value = mock_response

        model = _make_model()
        result = model.generate_with_json_output("give me json", enable_thinking=True)
        assert result == {"result": "thinking"}


# =====================================================================
# _save_llm_trace - additional paths
# =====================================================================


class TestSaveLlmTraceAdvanced:
    def test_trace_with_multiple_system_and_user_messages(self, tmp_path):
        model = _make_model(save_llm_trace=True)

        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task.id = "test_task"
        model.workflow = mock_workflow

        mock_node = MagicMock()
        mock_node.id = "multi_msg_node"
        model.current_node = mock_node

        messages = [
            {"role": "system", "content": "System msg 1"},
            {"role": "system", "content": "System msg 2"},
            {"role": "user", "content": "User msg 1"},
            {"role": "user", "content": "User msg 2"},
        ]
        model._save_llm_trace(messages, "response")
        trace_file = tmp_path / "test_task" / "multi_msg_node.yml"
        assert trace_file.exists()

        import yaml

        with open(trace_file) as f:
            data = yaml.safe_load(f)
        assert "System msg 1" in data["system_prompt"]
        assert "System msg 2" in data["system_prompt"]
        assert "User msg 1" in data["user_prompt"]
        assert "User msg 2" in data["user_prompt"]

    def test_trace_with_non_string_prompt(self, tmp_path):
        model = _make_model(save_llm_trace=True)

        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task.id = "test_task"
        model.workflow = mock_workflow

        mock_node = MagicMock()
        mock_node.id = "nonstr_node"
        model.current_node = mock_node

        model._save_llm_trace(12345, "response")
        trace_file = tmp_path / "test_task" / "nonstr_node.yml"
        assert trace_file.exists()


# =====================================================================
# model_specs - additional entries
# =====================================================================


class TestModelSpecsAdditional:
    def test_o3_model(self):
        model = _make_model(model="o3")
        assert model.context_length() == 200000
        assert model.max_tokens() == 200000

    def test_o4_model(self):
        model = _make_model(model="o4")
        assert model.context_length() == 200000
        assert model.max_tokens() == 200000

    def test_gpt5_model(self):
        model = _make_model(model="gpt-5")
        assert model.context_length() == 400000
        assert model.max_tokens() == 128000

    def test_gpt41_model(self):
        model = _make_model(model="gpt-4.1")
        assert model.context_length() == 400000
        assert model.max_tokens() == 128000

    def test_deepseek_reasoner(self):
        model = _make_model(model="deepseek-reasoner")
        assert model.context_length() == 65535
        assert model.max_tokens() == 65535

    def test_kimi_k2_turbo(self):
        model = _make_model(model="kimi-k2-turbo")
        assert model.context_length() == 256000
        assert model.max_tokens() == 8192

    def test_gemini_2_5_flash(self):
        model = _make_model(model="gemini-2.5-flash")
        assert model.context_length() == 1048576
        assert model.max_tokens() == 8192

    def test_qwen3_coder(self):
        model = _make_model(model="qwen3-coder")
        assert model.context_length() == 128000
        assert model.max_tokens() == 8192


# =====================================================================
# _format_tool_result - additional paths
# =====================================================================


class TestFormatToolResultAdvanced:
    def test_json_non_dict_non_list(self):
        model = _make_model()
        content = json.dumps(42)
        result = model._format_tool_result(content)
        assert "42" in result

    def test_medium_non_json_string(self):
        model = _make_model()
        text = "a" * 50
        result = model._format_tool_result(text)
        assert not result.endswith("...")

    def test_from_dict_items_non_list(self):
        model = _make_model()
        result = model._format_tool_result_from_dict({"items": "not a list"})
        # len("not a list") == 10
        assert result == "10 items"


# =====================================================================
# generate_with_tools (async, mocking Agent/Runner)
# =====================================================================


class TestGenerateWithTools:
    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_basic_generate_with_tools(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        # Setup MCP context manager
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        # Setup Runner.run result
        mock_result = MagicMock()
        mock_result.final_output = "Generated SQL"
        mock_result.to_input_list.return_value = []
        mock_result.turn_count = 1

        # Mock context_wrapper.usage
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.requests = 1
        mock_usage.input_tokens_details = None
        mock_usage.output_tokens_details = None
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = mock_usage

        mock_runner.run = AsyncMock(return_value=mock_result)

        model = _make_model()
        result = await model.generate_with_tools(
            prompt="Generate a SQL query",
            instruction="You are a SQL expert",
            max_turns=5,
        )

        assert result["content"] == "Generated SQL"
        assert result["model"] == "gpt-4o"
        assert result["usage"]["total_tokens"] == 150
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_generate_with_tools_with_tools_list(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.turn_count = 1
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15, requests=1,
            input_tokens_details=None, output_tokens_details=None,
        )
        mock_runner.run = AsyncMock(return_value=mock_result)

        model = _make_model()
        mock_tool = MagicMock()
        result = await model.generate_with_tools(
            prompt="test",
            tools=[mock_tool],
            instruction="instructions",
        )
        assert result["content"] == "result"
        # Verify tools were passed to Agent
        agent_call_kwargs = mock_agent_cls.call_args
        assert "tools" in agent_call_kwargs.kwargs

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_generate_with_tools_max_turns_exceeded(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        from agents import MaxTurnsExceeded

        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_runner.run = AsyncMock(side_effect=MaxTurnsExceeded("max turns exceeded"))

        model = _make_model()
        with pytest.raises(DatusException):
            await model.generate_with_tools(
                prompt="test",
                instruction="instructions",
                max_turns=3,
            )

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_generate_with_tools_no_usage(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.turn_count = 1
        # No context_wrapper
        del mock_result.context_wrapper
        mock_runner.run = AsyncMock(return_value=mock_result)

        model = _make_model()
        result = await model.generate_with_tools(prompt="test", instruction="inst")
        assert result["content"] == "result"
        assert result["usage"] == {}

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_generate_with_tools_with_cache_and_reasoning(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.turn_count = 2

        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 100
        mock_usage.total_tokens = 300
        mock_usage.requests = 2
        mock_usage.input_tokens_details = MagicMock(cached_tokens=50)
        mock_usage.output_tokens_details = MagicMock(reasoning_tokens=30)
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = mock_usage
        mock_runner.run = AsyncMock(return_value=mock_result)

        model = _make_model()
        result = await model.generate_with_tools(prompt="test", instruction="inst")
        assert result["usage"]["cached_tokens"] == 50
        assert result["usage"]["reasoning_tokens"] == 30
        assert result["usage"]["cache_hit_rate"] == round(50 / 200, 3)
        assert result["usage"]["context_usage_ratio"] > 0

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_generate_with_tools_mcp_connected(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_server = MagicMock()
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={"db_server": mock_server})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.turn_count = 1
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15, requests=1,
            input_tokens_details=None, output_tokens_details=None,
        )
        mock_runner.run = AsyncMock(return_value=mock_result)

        model = _make_model()
        result = await model.generate_with_tools(
            prompt="test",
            mcp_servers={"db": MagicMock()},
            instruction="inst",
        )
        assert result["content"] == "result"
        # MCP servers should be passed
        agent_kwargs = mock_agent_cls.call_args.kwargs
        assert "mcp_servers" in agent_kwargs

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_generate_with_tools_with_hooks(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.turn_count = 1
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15, requests=1,
            input_tokens_details=None, output_tokens_details=None,
        )
        mock_runner.run = AsyncMock(return_value=mock_result)

        model = _make_model()
        mock_hooks = MagicMock()
        result = await model.generate_with_tools(
            prompt="test",
            instruction="inst",
            hooks=mock_hooks,
        )
        assert result["content"] == "result"
        agent_kwargs = mock_agent_cls.call_args.kwargs
        assert "hooks" in agent_kwargs

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_generate_with_tools_thinking_model(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.turn_count = 1
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15, requests=1,
            input_tokens_details=None, output_tokens_details=None,
        )
        mock_runner.run = AsyncMock(return_value=mock_result)

        model = _make_model(model="deepseek-r1", enable_thinking=True)
        result = await model.generate_with_tools(prompt="test", instruction="inst")
        assert result["content"] == "result"


# =====================================================================
# _extract_and_distribute_token_usage
# =====================================================================


class TestExtractAndDistributeTokenUsage:
    @pytest.mark.asyncio
    async def test_no_context_wrapper(self):
        model = _make_model()
        mock_result = MagicMock(spec=[])  # No context_wrapper attr
        manager = ActionHistoryManager()
        # Should not raise
        await model._extract_and_distribute_token_usage(mock_result, manager)

    @pytest.mark.asyncio
    async def test_with_usage(self):
        model = _make_model()
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.requests = 1
        mock_usage.input_tokens_details = None
        mock_usage.output_tokens_details = None
        mock_result.context_wrapper.usage = mock_usage

        manager = ActionHistoryManager()
        action = ActionHistory(
            action_id="test",
            role=ActionRole.ASSISTANT,
            messages="response",
            action_type="response",
            input={},
            output={},
            status=ActionStatus.SUCCESS,
        )
        manager.add_action(action)

        await model._extract_and_distribute_token_usage(mock_result, manager)
        assert action.output["usage"]["total_tokens"] == 150


# =====================================================================
# generate_with_tools_stream (async streaming)
# =====================================================================


class TestGenerateWithToolsStream:
    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_stream_basic(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        # Create a mock streamed result
        mock_result = MagicMock()
        mock_result.is_complete = False
        mock_result.final_output = "final result"
        mock_result.to_input_list.return_value = []
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15, requests=1,
            input_tokens_details=None, output_tokens_details=None,
        )

        # Create tool_call event
        tool_call_event = MagicMock()
        tool_call_event.type = "run_item_stream_event"
        tool_call_item = MagicMock()
        tool_call_item.type = "tool_call_item"
        raw_tool = MagicMock()
        raw_tool.name = "read_query"
        raw_tool.arguments = '{"query": "SELECT 1"}'
        raw_tool.call_id = "call_abc123"
        tool_call_item.raw_item = raw_tool
        tool_call_event.item = tool_call_item

        # Create tool output event
        tool_output_event = MagicMock()
        tool_output_event.type = "run_item_stream_event"
        tool_output_item = MagicMock()
        tool_output_item.type = "tool_call_output_item"
        tool_output_item.output = '[{"id": 1}]'
        tool_output_item.raw_item = MagicMock(call_id="call_abc123")
        tool_output_event.item = tool_output_item

        # Create message event
        message_event = MagicMock()
        message_event.type = "run_item_stream_event"
        message_item = MagicMock()
        message_item.type = "message_output_item"
        message_content = MagicMock()
        message_content.text = "Based on the results, the answer is 1."
        message_item.raw_item = MagicMock(content=[message_content])
        message_event.item = message_item

        # Set up stream_events to yield events, then mark complete
        call_count = 0

        async def fake_stream_events():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield tool_call_event
                yield tool_output_event
                yield message_event

        mock_result.stream_events = fake_stream_events

        # Make is_complete return False first, then True
        is_complete_calls = [False, True]
        type(mock_result).is_complete = property(lambda self: is_complete_calls.pop(0))

        mock_runner.run_streamed.return_value = mock_result

        model = _make_model()
        actions = []
        async for action in model.generate_with_tools_stream(
            prompt="test prompt",
            instruction="system instruction",
        ):
            actions.append(action)

        # Should have tool_call, tool_result, and message actions
        assert len(actions) >= 2
        # First action should be tool call
        assert actions[0].action_type == "read_query"
        assert actions[0].status == ActionStatus.PROCESSING

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_stream_with_orphan_tool_result(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15, requests=1,
            input_tokens_details=None, output_tokens_details=None,
        )

        # Create orphan tool output (no matching tool call)
        orphan_event = MagicMock()
        orphan_event.type = "run_item_stream_event"
        orphan_item = MagicMock()
        orphan_item.type = "tool_call_output_item"
        orphan_item.output = "orphan result"
        orphan_item.raw_item = MagicMock(call_id="orphan_id")
        orphan_event.item = orphan_item

        async def fake_stream_events():
            yield orphan_event

        mock_result.stream_events = fake_stream_events
        is_complete_calls = [False, True]
        type(mock_result).is_complete = property(lambda self: is_complete_calls.pop(0))

        mock_runner.run_streamed.return_value = mock_result

        model = _make_model()
        actions = []
        async for action in model.generate_with_tools_stream(
            prompt="test",
            instruction="inst",
        ):
            actions.append(action)

        assert len(actions) >= 1
        # The orphan should still be yielded
        orphan_actions = [a for a in actions if "orphan" in str(a.action_id)]
        assert len(orphan_actions) >= 1

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_stream_with_dict_output(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15, requests=1,
            input_tokens_details=None, output_tokens_details=None,
        )

        # Create tool call event
        tool_call_event = MagicMock()
        tool_call_event.type = "run_item_stream_event"
        tool_call_item = MagicMock()
        tool_call_item.type = "tool_call_item"
        raw_tool = MagicMock()
        raw_tool.name = "list_tables"
        raw_tool.arguments = "{}"
        raw_tool.call_id = "call_dict"
        tool_call_item.raw_item = raw_tool
        tool_call_event.item = tool_call_item

        # Create tool output with dict content
        output_event = MagicMock()
        output_event.type = "run_item_stream_event"
        output_item = MagicMock()
        output_item.type = "tool_call_output_item"
        output_item.output = {"result": [1, 2, 3]}  # Dict output
        output_item.raw_item = MagicMock(call_id="call_dict")
        output_event.item = output_item

        async def fake_stream_events():
            yield tool_call_event
            yield output_event

        mock_result.stream_events = fake_stream_events
        is_complete_calls = [False, True]
        type(mock_result).is_complete = property(lambda self: is_complete_calls.pop(0))

        mock_runner.run_streamed.return_value = mock_result

        model = _make_model()
        actions = []
        async for action in model.generate_with_tools_stream(
            prompt="test",
            instruction="inst",
        ):
            actions.append(action)

        # Should have tool_call and tool_result
        assert len(actions) >= 2
        # The result action should have summary
        result_actions = [a for a in actions if a.status == ActionStatus.SUCCESS]
        assert len(result_actions) >= 1

    @pytest.mark.asyncio
    @patch("datus.models.openai_compatible.multiple_mcp_servers")
    @patch("datus.models.openai_compatible.Runner")
    @patch("datus.models.openai_compatible.Agent")
    async def test_stream_no_events(self, mock_agent_cls, mock_runner, mock_mcp_ctx):
        mock_mcp_ctx.return_value.__aenter__ = AsyncMock(return_value={})
        mock_mcp_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.final_output = "result"
        mock_result.to_input_list.return_value = []
        mock_result.context_wrapper = MagicMock()
        mock_result.context_wrapper.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15, requests=1,
            input_tokens_details=None, output_tokens_details=None,
        )

        async def fake_stream_events():
            return
            yield  # Make it an empty async generator

        mock_result.stream_events = fake_stream_events
        type(mock_result).is_complete = property(lambda self: True)

        mock_runner.run_streamed.return_value = mock_result

        model = _make_model()
        actions = []
        async for action in model.generate_with_tools_stream(
            prompt="test",
            instruction="inst",
        ):
            actions.append(action)

        assert len(actions) == 0
