# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OpenAI-compatible model implementation.

Drives the OpenAI Chat Completions endpoint via the openai SDK directly
(no LiteLLM, no openai-agents-sdk).  Used as the runtime for any
provider whose ``ModelConfig.type`` is ``openai`` and — once their
catalog entries flip back from the stub registry — also for DeepSeek,
Qwen, Kimi, GLM, MiniMax, and OpenRouter, which all speak the same
wire format.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError

from datus.configuration.agent_config import ModelConfig
from datus.models.agent_loop import AgentLoop
from datus.models.base import LLMBaseModel
from datus.models.mcp_client import MCPServerStdio
from datus.models.session import SQLiteSession
from datus.models.token_counter import context_length as ctx_lookup
from datus.models.token_counter import count_tokens
from datus.models.tool import FunctionTool
from datus.models.transports.openai_chat import OpenAIChatTransport
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def classify_openai_compatible_error(error: Exception) -> Tuple[ErrorCode, bool]:
    """Map an openai SDK exception to ``(ErrorCode, is_retryable)``.

    Mirrors the classifier from the deleted module so existing
    error-code consumers keep working.
    """
    if isinstance(error, RateLimitError):
        return ErrorCode.MODEL_RATE_LIMIT, True
    if isinstance(error, APITimeoutError):
        return ErrorCode.MODEL_TIMEOUT_ERROR, True
    if isinstance(error, APIConnectionError):
        return ErrorCode.MODEL_CONNECTION_ERROR, True
    if isinstance(error, APIError):
        msg = str(error).lower()
        if any(s in msg for s in ("401", "unauthorized", "authentication")):
            return ErrorCode.MODEL_AUTHENTICATION_ERROR, False
        if any(s in msg for s in ("403", "forbidden", "permission")):
            return ErrorCode.MODEL_PERMISSION_ERROR, False
        if any(s in msg for s in ("404", "not found")):
            return ErrorCode.MODEL_NOT_FOUND, False
        if any(s in msg for s in ("413", "too large", "request size")):
            return ErrorCode.MODEL_REQUEST_TOO_LARGE, False
        if any(s in msg for s in ("429", "rate limit", "quota", "billing")):
            if any(s in msg for s in ("quota", "billing")):
                return ErrorCode.MODEL_QUOTA_EXCEEDED, False
            return ErrorCode.MODEL_RATE_LIMIT, True
        if any(s in msg for s in ("502", "503", "overloaded")):
            return ErrorCode.MODEL_OVERLOADED, True
        if any(s in msg for s in ("500", "internal", "server error")):
            return ErrorCode.MODEL_API_ERROR, True
        if any(s in msg for s in ("400", "bad request", "invalid")):
            return ErrorCode.MODEL_INVALID_RESPONSE, False
    return ErrorCode.MODEL_REQUEST_FAILED, False


class OpenAICompatibleModel(LLMBaseModel):
    """LLM driver for any provider speaking OpenAI Chat Completions."""

    def __init__(self, model_config: ModelConfig, **kwargs: Any) -> None:
        super().__init__(model_config, **kwargs)
        self.model_name = model_config.model
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        self.default_headers = dict(model_config.default_headers or {})
        self.transport = OpenAIChatTransport()
        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _get_api_key(self) -> str:
        return self.model_config.api_key or ""

    def _get_base_url(self) -> Optional[str]:
        return self.model_config.base_url or None

    def _client_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"api_key": self.api_key or "missing"}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = dict(self.default_headers)
        return kwargs

    @property
    def sync_client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(**self._client_kwargs())
        return self._sync_client

    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._client_kwargs())
        return self._async_client

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_call_params(self, **overrides: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if "temperature" in overrides:
            params["temperature"] = overrides["temperature"]
        elif self.model_config.temperature is not None:
            params["temperature"] = self.model_config.temperature
        else:
            params["temperature"] = 0.7
        if "top_p" in overrides:
            params["top_p"] = overrides["top_p"]
        elif self.model_config.top_p is not None:
            params["top_p"] = self.model_config.top_p
        else:
            params["top_p"] = 1.0
        for passthrough in ("max_tokens", "max_completion_tokens", "response_format", "stop", "seed"):
            if passthrough in overrides:
                params[passthrough] = overrides[passthrough]
        if self.default_headers:
            params["extra_headers"] = dict(self.default_headers)
        return params

    def _retry_loop_sync(self, callable_, label: str = "operation"):
        delays = [self.model_config.retry_interval * (2**i) for i in range(self.model_config.max_retry + 1)]
        for attempt, delay in enumerate(delays):
            try:
                return callable_()
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
                code, retryable = classify_openai_compatible_error(exc)
                if retryable and attempt < self.model_config.max_retry:
                    logger.warning(
                        "%s attempt %d/%d failed (%s); retrying in %.1fs",
                        label,
                        attempt + 1,
                        self.model_config.max_retry + 1,
                        code,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise DatusException(code, message_args={"error_message": str(exc)}) from exc

    async def _retry_loop_async(self, callable_, label: str = "operation"):
        delays = [self.model_config.retry_interval * (2**i) for i in range(self.model_config.max_retry + 1)]
        for attempt, delay in enumerate(delays):
            try:
                return await callable_()
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
                code, retryable = classify_openai_compatible_error(exc)
                if retryable and attempt < self.model_config.max_retry:
                    logger.warning(
                        "%s attempt %d/%d failed (%s); retrying in %.1fs",
                        label,
                        attempt + 1,
                        self.model_config.max_retry + 1,
                        code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise DatusException(code, message_args={"error_message": str(exc)}) from exc

    # ------------------------------------------------------------------
    # LLMBaseModel: required methods
    # ------------------------------------------------------------------

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs: Any) -> str:
        if isinstance(prompt, list):
            messages = list(prompt)
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        call_params = self._build_call_params(**kwargs)
        request = self.transport.build_kwargs(model=self.model_name, messages=messages, tools=None, **call_params)

        def _do_call():
            response = self.sync_client.chat.completions.create(**request)
            normalised = self.transport.normalize_response(response)
            text = normalised.content or ""
            if enable_thinking and normalised.reasoning and not text.strip():
                text = normalised.reasoning
            return text

        return self._retry_loop_sync(_do_call, "generate")

    def generate_with_json_output(self, prompt: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("response_format", {"type": "json_object"})
        text = self.generate(prompt, **kwargs)
        result = llm_result2json(text)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"items": result}
        return {"content": text}

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[FunctionTool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        loop = AgentLoop(
            model=self,
            transport=self.transport,
            local_tools=tools or [],
            mcp_servers=mcp_servers or {},
            instruction=instruction,
            output_type=output_type,
            max_turns=max_turns,
            session=session,
            hooks=kwargs.get("hooks"),
            interrupt_controller=kwargs.get("interrupt_controller"),
            db_type=kwargs.get("db_type", ""),
        )
        return await loop.run(prompt)

    async def generate_with_tools_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[FunctionTool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks: Any = None,
        interrupt_controller: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ActionHistory, None]:
        loop = AgentLoop(
            model=self,
            transport=self.transport,
            local_tools=tools or [],
            mcp_servers=mcp_servers or {},
            instruction=instruction,
            output_type=output_type,
            max_turns=max_turns,
            session=session,
            hooks=hooks,
            interrupt_controller=interrupt_controller,
            action_history_manager=action_history_manager,
            db_type=kwargs.get("db_type", ""),
        )
        async for action in loop.run_streamed(prompt):
            yield action

    def token_count(self, prompt: str) -> int:
        return count_tokens(self.model_name, prompt or "", model_type=self.model_config.type)

    def context_length(self) -> Optional[int]:
        return ctx_lookup(self.model_name)

    # ------------------------------------------------------------------
    # Single-turn API helpers used by AgentLoop
    # ------------------------------------------------------------------

    async def complete_once(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        **call_overrides: Any,
    ):
        """Single async chat-completion call returning a NormalizedResponse."""
        params = self._build_call_params(**call_overrides)
        request = self.transport.build_kwargs(
            model=self.model_name,
            messages=messages,
            tools=tools_schema,
            **params,
        )

        async def _call():
            response = await self.async_client.chat.completions.create(**request)
            return self.transport.normalize_response(response)

        return await self._retry_loop_async(_call, "complete_once")

    async def stream_once(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        **call_overrides: Any,
    ):
        """Async generator yielding raw streaming chunks from the SDK."""
        params = self._build_call_params(**call_overrides)
        params["stream_options"] = {"include_usage": True}
        request = self.transport.build_kwargs(
            model=self.model_name,
            messages=messages,
            tools=tools_schema,
            **params,
        )
        request["stream"] = True

        async def _call():
            return await self.async_client.chat.completions.create(**request)

        stream = await self._retry_loop_async(_call, "stream_once")
        async for chunk in stream:
            yield chunk


__all__ = ["OpenAICompatibleModel", "classify_openai_compatible_error"]


# Used by AgentLoop's stream collector — a quick parse for tool argument JSON.
def parse_tool_arguments(arguments: str) -> Any:
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return arguments
