# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""MCP server client wrappers (stdio / SSE / streamable-HTTP).

Replaces ``agents.mcp.{MCPServerStdio,MCPServerSse,MCPServerStreamableHttp}``.
Each class wraps an :class:`mcp.ClientSession` plus the appropriate
transport context manager and exposes the methods that the openai-agents
SDK exposed (``connect`` / ``list_tools`` / ``call_tool`` /
``invalidate_tools_cache`` / ``cleanup`` / ``name``), so the rest of the
project (``datus/tools/mcp_tools/mcp_manager.py``) can switch its imports
without changing call shapes.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Dict, List, NotRequired, Optional, TypedDict, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TypedDict parameter shapes (mirror agents.mcp.* TypedDicts)
# ---------------------------------------------------------------------------


class MCPServerStdioParams(TypedDict):
    """Stdio MCP server parameters (mirror of ``mcp.StdioServerParameters``)."""

    command: str
    args: NotRequired[List[str]]
    env: NotRequired[Optional[Dict[str, str]]]
    cwd: NotRequired[Optional[str]]
    encoding: NotRequired[str]
    encoding_error_handler: NotRequired[str]


class MCPServerSseParams(TypedDict):
    url: str
    headers: NotRequired[Optional[Dict[str, str]]]
    timeout: NotRequired[float]
    sse_read_timeout: NotRequired[float]


class MCPServerStreamableHttpParams(TypedDict):
    url: str
    headers: NotRequired[Optional[Dict[str, str]]]
    timeout: NotRequired[float]
    sse_read_timeout: NotRequired[float]
    terminate_on_close: NotRequired[bool]


# ---------------------------------------------------------------------------
# Base — owns the ClientSession lifecycle + tool cache
# ---------------------------------------------------------------------------


class _MCPServerBase:
    """Shared connect / list_tools / call_tool plumbing.

    Subclasses provide :meth:`_open_streams`, an async context manager
    that yields ``(read_stream, write_stream)`` (and optionally a
    ``get_session_id`` callable that we ignore).
    """

    def __init__(
        self,
        *,
        cache_tools_list: bool = False,
        name: Optional[str] = None,
        client_session_timeout_seconds: Optional[float] = 5,
        tool_filter: Any = None,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.cache_tools_list = bool(cache_tools_list)
        self.client_session_timeout_seconds = client_session_timeout_seconds
        self.tool_filter = tool_filter
        self.max_retry_attempts = int(max_retry_attempts)
        self.retry_backoff_seconds_base = float(retry_backoff_seconds_base)

        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[contextlib.AsyncExitStack] = None
        self._tool_cache: Optional[List[MCPTool]] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _open_streams(self):  # pragma: no cover — overridden
        raise NotImplementedError
        yield  # type: ignore[unreachable]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Spin up the transport and initialise the MCP session."""
        if self._session is not None:
            return
        async with self._lock:
            if self._session is not None:
                return
            stack = contextlib.AsyncExitStack()
            try:
                streams = await stack.enter_async_context(self._open_streams())
                # ``streamablehttp_client`` returns a 3-tuple; the rest are 2.
                read_stream, write_stream = streams[0], streams[1]
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                if self.client_session_timeout_seconds is not None:
                    await asyncio.wait_for(session.initialize(), timeout=self.client_session_timeout_seconds)
                else:
                    await session.initialize()
            except BaseException:
                await stack.aclose()
                raise
            self._exit_stack = stack
            self._session = session

    async def cleanup(self) -> None:
        """Tear down the MCP session and the underlying transport."""
        if self._exit_stack is None:
            self._session = None
            return
        async with self._lock:
            if self._exit_stack is None:
                return
            stack, self._exit_stack = self._exit_stack, None
            self._session = None
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.debug("MCP server %s cleanup error: %s", self.name, exc)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def invalidate_tools_cache(self) -> None:
        self._tool_cache = None

    async def list_tools(self) -> List[MCPTool]:
        if self.cache_tools_list and self._tool_cache is not None:
            return self._tool_cache
        if self._session is None:
            await self.connect()
        assert self._session is not None
        result = await self._session.list_tools()
        tools = list(result.tools)
        if callable(self.tool_filter):
            tools = [t for t in tools if self.tool_filter(t)]
        if self.cache_tools_list:
            self._tool_cache = tools
        return tools

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> CallToolResult:
        if self._session is None:
            await self.connect()
        assert self._session is not None
        return await self._session.call_tool(name, arguments or {})

    async def list_prompts(self) -> Any:
        if self._session is None:
            await self.connect()
        assert self._session is not None
        return await self._session.list_prompts()

    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        if self._session is None:
            await self.connect()
        assert self._session is not None
        return await self._session.get_prompt(name, arguments or {})


# ---------------------------------------------------------------------------
# Concrete servers
# ---------------------------------------------------------------------------


class MCPServerStdio(_MCPServerBase):
    """MCP server launched as a subprocess speaking JSON-RPC over stdio."""

    def __init__(
        self,
        params: Union[MCPServerStdioParams, StdioServerParameters],
        cache_tools_list: bool = False,
        name: Optional[str] = None,
        client_session_timeout_seconds: Optional[float] = 5,
        tool_filter: Any = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: Any = None,
    ) -> None:
        super().__init__(
            cache_tools_list=cache_tools_list,
            name=name,
            client_session_timeout_seconds=client_session_timeout_seconds,
            tool_filter=tool_filter,
            max_retry_attempts=max_retry_attempts,
            retry_backoff_seconds_base=retry_backoff_seconds_base,
        )
        self.params = self._coerce_stdio_params(params)
        self.use_structured_content = bool(use_structured_content)
        self.message_handler = message_handler

    @staticmethod
    def _coerce_stdio_params(
        raw: Union[MCPServerStdioParams, StdioServerParameters, Dict[str, Any]],
    ) -> StdioServerParameters:
        if isinstance(raw, StdioServerParameters):
            return raw
        data = dict(raw)
        return StdioServerParameters(
            command=data["command"],
            args=list(data.get("args") or []),
            env=data.get("env"),
            cwd=data.get("cwd"),
            encoding=data.get("encoding") or "utf-8",
            encoding_error_handler=data.get("encoding_error_handler") or "strict",
        )

    @contextlib.asynccontextmanager
    async def _open_streams(self):
        async with stdio_client(self.params) as (read_stream, write_stream):
            yield (read_stream, write_stream)


class MCPServerSse(_MCPServerBase):
    """MCP server reachable over Server-Sent Events."""

    def __init__(
        self,
        params: MCPServerSseParams,
        cache_tools_list: bool = False,
        name: Optional[str] = None,
        client_session_timeout_seconds: Optional[float] = 5,
        tool_filter: Any = None,
    ) -> None:
        super().__init__(
            cache_tools_list=cache_tools_list,
            name=name,
            client_session_timeout_seconds=client_session_timeout_seconds,
            tool_filter=tool_filter,
        )
        self.params = dict(params)

    @contextlib.asynccontextmanager
    async def _open_streams(self):
        async with sse_client(
            url=self.params["url"],
            headers=self.params.get("headers"),
            timeout=self.params.get("timeout", 5.0),
            sse_read_timeout=self.params.get("sse_read_timeout", 300.0),
        ) as (read_stream, write_stream):
            yield (read_stream, write_stream)


class MCPServerStreamableHttp(_MCPServerBase):
    """MCP server reachable over streamable HTTP (sse-encoded responses)."""

    def __init__(
        self,
        params: MCPServerStreamableHttpParams,
        cache_tools_list: bool = False,
        name: Optional[str] = None,
        client_session_timeout_seconds: Optional[float] = 5,
        tool_filter: Any = None,
    ) -> None:
        super().__init__(
            cache_tools_list=cache_tools_list,
            name=name,
            client_session_timeout_seconds=client_session_timeout_seconds,
            tool_filter=tool_filter,
        )
        self.params = dict(params)

    @contextlib.asynccontextmanager
    async def _open_streams(self):
        async with streamablehttp_client(
            url=self.params["url"],
            headers=self.params.get("headers"),
            timeout=self.params.get("timeout", 30.0),
            sse_read_timeout=self.params.get("sse_read_timeout", 300.0),
            terminate_on_close=self.params.get("terminate_on_close", True),
        ) as streams:
            # streams is (read, write, get_session_id) — _MCPServerBase only
            # uses the first two, so just forward the tuple unchanged.
            yield streams
