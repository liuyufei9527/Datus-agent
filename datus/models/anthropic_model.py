# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Anthropic Claude model implementation.

Drives the Anthropic Messages API directly via the ``anthropic`` SDK
(no LiteLLM, no openai-agents-sdk). Reuses :class:`AgentLoop` for
multi-turn tool-calling so the orchestration logic is shared with
:class:`OpenAICompatibleModel`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import anthropic
import httpx
from anthropic import APIConnectionError, APIError, APITimeoutError, AsyncAnthropic, RateLimitError

from datus.configuration.agent_config import ModelConfig
from datus.models.agent_loop import AgentLoop
from datus.models.base import LLMBaseModel
from datus.models.mcp_client import MCPServerStdio
from datus.models.session import SQLiteSession
from datus.models.token_counter import context_length as ctx_lookup
from datus.models.token_counter import count_tokens
from datus.models.tool import FunctionTool
from datus.models.transports.anthropic import AnthropicTransport
from datus.models.transports.types import NormalizedResponse, ToolCall, Usage
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def classify_anthropic_error(error: Exception) -> Tuple[ErrorCode, bool]:
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
        if "overload" in msg or "529" in msg or "503" in msg:
            return ErrorCode.MODEL_OVERLOADED, True
        if "rate" in msg or "429" in msg:
            return ErrorCode.MODEL_RATE_LIMIT, True
    return ErrorCode.MODEL_REQUEST_FAILED, False


class AnthropicModel(LLMBaseModel):
    """LLM driver for Anthropic Claude models."""

    def __init__(self, model_config: ModelConfig, **kwargs: Any) -> None:
        super().__init__(model_config, **kwargs)
        self.model_name = model_config.model
        self.api_key = model_config.api_key or ""
        self.base_url = model_config.base_url or None
        self.default_headers = dict(model_config.default_headers or {})
        self.transport = AnthropicTransport()
        self._sync_client: Optional[anthropic.Anthropic] = None
        self._async_client: Optional[AsyncAnthropic] = None

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _client_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"api_key": self.api_key or "missing"}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = dict(self.default_headers)
        return kwargs

    @property
    def sync_client(self) -> anthropic.Anthropic:
        if self._sync_client is None:
            kwargs = self._client_kwargs()
            # Always supply our own httpx.Client. The vendored ``SyncHttpxClientWrapper``
            # in anthropic-python 0.51 trips Cloudflare's "Request not allowed" 403
            # when reused across SDK versions / proxy combinations; constructing a
            # plain ``httpx.Client`` (with ``trust_env=True`` so users keep their
            # proxy config) avoids the issue.
            kwargs["http_client"] = httpx.Client(timeout=httpx.Timeout(60.0))
            self._sync_client = anthropic.Anthropic(**kwargs)
        return self._sync_client

    @property
    def async_client(self) -> AsyncAnthropic:
        if self._async_client is None:
            kwargs = self._client_kwargs()
            kwargs["http_client"] = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            self._async_client = AsyncAnthropic(**kwargs)
        return self._async_client

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_call_params(self, **overrides: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if "max_tokens" in overrides:
            params["max_tokens"] = overrides["max_tokens"]
        else:
            params["max_tokens"] = 4096
        if "temperature" in overrides:
            params["temperature"] = overrides["temperature"]
        elif self.model_config.temperature is not None:
            params["temperature"] = self.model_config.temperature
        if "top_p" in overrides:
            params["top_p"] = overrides["top_p"]
        elif self.model_config.top_p is not None:
            params["top_p"] = self.model_config.top_p
        if self.default_headers:
            params["extra_headers"] = dict(self.default_headers)
        if self.model_config.enable_thinking:
            # Conservative budget; provider clamps at 2x max_tokens anyway.
            params["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        return params

    def _retry_loop_sync(self, callable_, label: str = "operation"):
        delays = [self.model_config.retry_interval * (2**i) for i in range(self.model_config.max_retry + 1)]
        for attempt, delay in enumerate(delays):
            try:
                return callable_()
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
                code, retryable = classify_anthropic_error(exc)
                if retryable and attempt < self.model_config.max_retry:
                    logger.warning(
                        "%s attempt %d/%d (%s); retrying in %.1fs",
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
                code, retryable = classify_anthropic_error(exc)
                if retryable and attempt < self.model_config.max_retry:
                    logger.warning(
                        "%s attempt %d/%d (%s); retrying in %.1fs",
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
        params = self._build_call_params(**kwargs)
        if enable_thinking and "thinking" not in params:
            params["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        request = self.transport.build_kwargs(model=self.model_name, messages=messages, tools=None, **params)

        def _do_call():
            response = self.sync_client.messages.create(**request)
            normalised = self.transport.normalize_response(response)
            text = normalised.content or ""
            if enable_thinking and normalised.reasoning and not text.strip():
                text = normalised.reasoning
            return text

        return self._retry_loop_sync(_do_call, "generate")

    def generate_with_json_output(self, prompt: Any, **kwargs: Any) -> Dict[str, Any]:
        # Anthropic does not support response_format; rely on the model
        # following the JSON instructions in the prompt and re-parse here.
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
        # AgentLoop expects model to expose ``stream_once`` returning a chat-style
        # async iterator. Anthropic's stream is event-based with a different
        # shape, so we adapt it to look the same here.
        async for action in loop.run_streamed(prompt):
            yield action

    def token_count(self, prompt: str) -> int:
        return count_tokens(self.model_name, prompt or "", model_type="claude")

    def context_length(self) -> Optional[int]:
        return ctx_lookup(self.model_name)

    # ------------------------------------------------------------------
    # AgentLoop callbacks
    # ------------------------------------------------------------------

    async def complete_once(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        **call_overrides: Any,
    ) -> NormalizedResponse:
        params = self._build_call_params(**call_overrides)
        request = self.transport.build_kwargs(
            model=self.model_name,
            messages=messages,
            tools=tools_schema,
            **params,
        )

        async def _call():
            response = await self.async_client.messages.create(**request)
            return self.transport.normalize_response(response)

        return await self._retry_loop_async(_call, "complete_once")

    async def stream_once(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        **call_overrides: Any,
    ):
        """Adapt Anthropic streaming events into chat-completion-shaped chunks.

        :class:`AgentLoop` expects each yielded chunk to look like an
        OpenAI ``ChatCompletionChunk`` (``choices[0].delta`` plus an
        optional ``usage``).  We translate Anthropic's event types
        (``content_block_*``, ``message_delta``, ``message_stop``) on
        the fly so the loop's :class:`_ChatStreamCollector` can consume
        them unchanged.
        """
        params = self._build_call_params(**call_overrides)
        request = self.transport.build_kwargs(
            model=self.model_name,
            messages=messages,
            tools=tools_schema,
            **params,
        )

        async def _stream_iter():
            async with self.async_client.messages.stream(**request) as stream:
                tool_use_state: Dict[int, Dict[str, Any]] = {}
                async for event in stream:
                    chunk = _anthropic_event_to_chat_chunk(event, tool_use_state)
                    if chunk is not None:
                        yield chunk

        async for chunk in _stream_iter():
            yield chunk


# ---------------------------------------------------------------------------
# Anthropic event → OpenAI chunk shim
# ---------------------------------------------------------------------------


class _Choice:
    def __init__(self, delta: "_Delta") -> None:
        self.delta = delta
        self.finish_reason = None


class _Delta:
    def __init__(
        self,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        tool_calls: Optional[List["_ToolCallChunk"]] = None,
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _ToolCallChunk:
    def __init__(self, index: int, id_: Optional[str], name: Optional[str], arguments: Optional[str]) -> None:
        self.index = index
        self.id = id_
        self.function = _ToolCallFunction(name=name, arguments=arguments)


class _ToolCallFunction:
    def __init__(self, name: Optional[str], arguments: Optional[str]) -> None:
        self.name = name
        self.arguments = arguments


class _ChatChunk:
    def __init__(self, choices: List[_Choice], usage: Optional[Usage] = None) -> None:
        self.choices = choices
        self.usage = usage


def _anthropic_event_to_chat_chunk(event: Any, tool_state: Dict[int, Dict[str, Any]]) -> Optional[_ChatChunk]:
    event_type = getattr(event, "type", None)

    if event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        index = int(getattr(event, "index", 0) or 0)
        if block is None:
            return None
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_state[index] = {
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", None),
            }
            chunk_tool = _ToolCallChunk(
                index=index,
                id_=tool_state[index]["id"],
                name=tool_state[index]["name"],
                arguments="",
            )
            return _ChatChunk(choices=[_Choice(_Delta(tool_calls=[chunk_tool]))])
        return None

    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        index = int(getattr(event, "index", 0) or 0)
        if delta is None:
            return None
        delta_type = getattr(delta, "type", None)
        if delta_type == "text_delta":
            return _ChatChunk(choices=[_Choice(_Delta(content=getattr(delta, "text", "") or ""))])
        if delta_type == "thinking_delta":
            return _ChatChunk(choices=[_Choice(_Delta(reasoning_content=getattr(delta, "thinking", "") or ""))])
        if delta_type == "input_json_delta":
            partial = getattr(delta, "partial_json", "") or ""
            stored = tool_state.get(index, {})
            chunk_tool = _ToolCallChunk(
                index=index,
                id_=stored.get("id"),
                name=stored.get("name"),
                arguments=partial,
            )
            return _ChatChunk(choices=[_Choice(_Delta(tool_calls=[chunk_tool]))])
        return None

    if event_type == "message_delta":
        usage_obj = getattr(event, "usage", None)
        if usage_obj is None:
            return None
        return _ChatChunk(
            choices=[],
            usage=Usage(
                prompt_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
                completion_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
                total_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0)
                + int(getattr(usage_obj, "output_tokens", 0) or 0),
                cached_tokens=int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0),
            ),
        )

    return None


# Used by AgentLoop; a no-op here just to silence ruff F401 if needed.
_ = ToolCall  # noqa: F841
_ = json  # noqa: F841
