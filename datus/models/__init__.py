# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Public surface of ``datus.models``.

Importing from ``datus.models`` is the supported path for caller code;
sub-modules may be reorganised. The names re-exported here mirror the
old ``agents``-package types they replace, so call sites can switch
``from agents import Tool`` → ``from datus.models import Tool`` (and
similarly for Session, MCPServer, RunHooks).
"""

from datus.models.base import LLMBaseModel
from datus.models.hooks import (
    Agent,
    AgentHookContext,
    AgentHooks,
    CompositeHooks,
    RunContextWrapper,
    RunHooks,
    Usage,
)
from datus.models.mcp_client import (
    MCPServerSse,
    MCPServerSseParams,
    MCPServerStdio,
    MCPServerStdioParams,
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
)
from datus.models.registry import MODEL_TYPE_MAP, resolve_model_class
from datus.models.result import NewItem, RunResult, RunResultBase
from datus.models.session import SQLiteSession
from datus.models.tool import FunctionTool, Tool, ToolContext, function_tool, tool_to_openai_schema

__all__ = [
    "LLMBaseModel",
    "MODEL_TYPE_MAP",
    "resolve_model_class",
    # Tool types
    "Tool",
    "FunctionTool",
    "ToolContext",
    "function_tool",
    "tool_to_openai_schema",
    # Session
    "SQLiteSession",
    # MCP
    "MCPServerStdio",
    "MCPServerStdioParams",
    "MCPServerSse",
    "MCPServerSseParams",
    "MCPServerStreamableHttp",
    "MCPServerStreamableHttpParams",
    # Hooks / context
    "RunHooks",
    "AgentHooks",
    "CompositeHooks",
    "RunContextWrapper",
    "AgentHookContext",
    "Agent",
    "Usage",
    # Result
    "RunResult",
    "RunResultBase",
    "NewItem",
]
