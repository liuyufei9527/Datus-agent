# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Placeholder model implementation for providers awaiting a follow-up PR.

The first wave of the rewrite (PR #N) only ships native ``openai`` and
``claude`` adapters. The other 8 providers (deepseek/qwen/kimi/glm/
minimax/openrouter/gemini/codex) keep their entries in
:data:`MODEL_TYPE_MAP` so config loading and ``/model`` selection still
work; calling any generation method just raises a clear error pointing
at the deferred PR.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.mcp_client import MCPServerStdio
from datus.models.session import SQLiteSession
from datus.models.tool import FunctionTool
from datus.schemas.action_history import ActionHistory, ActionHistoryManager


class StubModel(LLMBaseModel):
    """Always raises ``NotImplementedError`` — wire this provider up later."""

    def __init__(self, model_config: ModelConfig, **kwargs: Any) -> None:
        super().__init__(model_config, **kwargs)
        self._provider = model_config.type

    def _unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            f"Provider {self._provider!r} is not yet supported in this build. "
            "Native adapter will land in a follow-up PR."
        )

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs: Any) -> str:
        raise self._unsupported()

    def generate_with_json_output(self, prompt: Any, **kwargs: Any) -> Dict[str, Any]:
        raise self._unsupported()

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
        raise self._unsupported()

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
        raise self._unsupported()
        yield  # type: ignore[unreachable]  # makes the function a generator

    def token_count(self, prompt: str) -> int:
        return max(1, len(str(prompt or "")) // 4)

    def context_length(self) -> Optional[int]:
        return None
