# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Multi-turn tool-calling orchestrator.

Replaces the SDK's ``agents.Runner.run`` / ``agents.Runner.run_streamed``.
Provider-agnostic — the model object provides ``complete_once`` /
``stream_once`` and a transport that knows the wire format; this loop
just shuttles tool calls between the LLM and either a local
:class:`FunctionTool` or an MCP server.

Streaming model: :meth:`run_streamed` is an ``async def`` returning an
``AsyncGenerator[ActionHistory, None]``; one item per assistant text
chunk, one item per tool call (``status=PROCESSING``), one item per tool
result (``status=SUCCESS``/``FAILED``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from mcp.types import Tool as MCPTool

from datus.models.hooks import RunContextWrapper, RunHooks
from datus.models.hooks import Usage as RunUsage
from datus.models.mcp_client import _MCPServerBase
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.result import NewItem, RunResult
from datus.models.session import SQLiteSession
from datus.models.tool import FunctionTool, ToolContext, tool_to_openai_schema
from datus.models.transports.base import ProviderTransport
from datus.models.transports.types import NormalizedResponse, ToolCall
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.tool_summary import TOOL_SUMMARY_REGISTRY, looks_like_failure
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import to_str
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _new_call_id(prefix: str = "call") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _short(text: str, limit: int = 80) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "..."


class AgentLoop:
    """Run an agent end-to-end against a single LLM model + tool set."""

    def __init__(
        self,
        *,
        model: Any,
        transport: ProviderTransport,
        local_tools: List[FunctionTool],
        mcp_servers: Dict[str, _MCPServerBase],
        instruction: str,
        output_type: type,
        max_turns: int,
        session: Optional[SQLiteSession] = None,
        hooks: Optional[RunHooks] = None,
        interrupt_controller: Any = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        db_type: str = "",
    ) -> None:
        self.model = model
        self.transport = transport
        self.local_tools = list(local_tools)
        self.mcp_servers = dict(mcp_servers or {})
        self.instruction = instruction or ""
        self.output_type = output_type
        self.max_turns = max(1, int(max_turns))
        self.session = session
        self.hooks = hooks
        self.interrupt_controller = interrupt_controller
        self.action_history_manager = action_history_manager or ActionHistoryManager()
        self.db_type = db_type
        self._context = RunContextWrapper(usage=RunUsage())

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run(self, prompt: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        async with self._mcp_lifecycle() as mcp_tool_lookup:
            tools_schema, tool_lookup = self._collect_tools_schema(mcp_tool_lookup)
            messages, user_input_items = await self._initial_messages(prompt)
            await self._persist_user_items(user_input_items)

            new_items: List[NewItem] = []
            final_text: str = ""
            turns = 0

            for _ in range(self.max_turns):
                turns += 1
                self._raise_if_interrupted()
                if self.hooks:
                    await self.hooks.on_llm_start(self._context, self.model, self.instruction, messages)
                response = await self.model.complete_once(messages, tools_schema)
                if self.hooks:
                    await self.hooks.on_llm_end(self._context, self.model, response)
                self._track_usage(response)

                assistant_text = response.content or ""
                if assistant_text:
                    new_items.append(NewItem(type="message", data={"content": assistant_text}))
                    final_text = assistant_text

                if not response.tool_calls:
                    await self._append_assistant_to_session(messages, response)
                    break

                tool_call_payloads = []
                for call in response.tool_calls:
                    call_id = call.id or _new_call_id()
                    tool_call_payloads.append((call_id, call))
                    new_items.append(
                        NewItem(
                            type="tool_call",
                            data={"call_id": call_id, "name": call.name, "arguments": call.arguments},
                        )
                    )

                # Append the assistant message with tool_calls into the conversation.
                messages.append(self._assistant_with_tool_calls(assistant_text, tool_call_payloads))
                if self.session:
                    items_to_save: List[Dict[str, Any]] = []
                    if assistant_text:
                        items_to_save.append({"role": "assistant", "content": assistant_text})
                    for cid, call in tool_call_payloads:
                        items_to_save.append(
                            {
                                "type": "function_call",
                                "call_id": cid,
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                        )
                    await self.session.add_items(items_to_save)

                # Execute every tool call sequentially (parallel exec is left to a future PR).
                for call_id, call in tool_call_payloads:
                    self._raise_if_interrupted()
                    output_text = await self._dispatch_tool_call(call, call_id, tool_lookup)
                    new_items.append(NewItem(type="tool_result", data={"call_id": call_id, "output": output_text}))
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": output_text})
                    if self.session:
                        await self.session.add_items(
                            [{"type": "function_call_output", "call_id": call_id, "output": output_text}]
                        )

            result = RunResult(
                final_output=self._coerce_output(final_text),
                new_items=new_items,
                context_wrapper=self._context,
                turn_count=turns,
                finish_reason=response.finish_reason if turns else "stop",
            )
            sql_contexts = extract_sql_contexts(result, self.db_type)
            usage_dict = self._usage_to_dict()
            await self._persist_turn_usage()
            return {
                "content": result.final_output,
                "sql_contexts": sql_contexts,
                "usage": usage_dict,
                "model": getattr(self.model, "model_name", ""),
                "turns_used": result.turn_count,
                "final_output_length": len(final_text or ""),
            }

    async def run_streamed(self, prompt: Union[str, List[Dict[str, Any]]]) -> AsyncGenerator[ActionHistory, None]:
        async with self._mcp_lifecycle() as mcp_tool_lookup:
            tools_schema, tool_lookup = self._collect_tools_schema(mcp_tool_lookup)
            messages, user_input_items = await self._initial_messages(prompt)
            await self._persist_user_items(user_input_items)

            for _ in range(self.max_turns):
                self._raise_if_interrupted()
                if self.hooks:
                    await self.hooks.on_llm_start(self._context, self.model, self.instruction, messages)

                stream_collector = _ChatStreamCollector()
                async for chunk in self.model.stream_once(messages, tools_schema):
                    self._raise_if_interrupted()
                    for action in stream_collector.feed(chunk):
                        self._record_action(action)
                        yield action
                final = stream_collector.finalize()
                if final is not None:
                    self._record_action(final)
                    yield final

                if self.hooks:
                    await self.hooks.on_llm_end(self._context, self.model, stream_collector.snapshot)

                self._track_usage_chunked(stream_collector.usage)
                assistant_text = stream_collector.text
                tool_calls = stream_collector.tool_calls
                if assistant_text:
                    if self.session:
                        await self.session.add_items([{"role": "assistant", "content": assistant_text}])

                if not tool_calls:
                    break

                tool_call_payloads: List[Tuple[str, ToolCall]] = []
                for call in tool_calls:
                    call_id = call.id or _new_call_id()
                    tool_call_payloads.append((call_id, call))

                messages.append(self._assistant_with_tool_calls(assistant_text, tool_call_payloads))
                if self.session:
                    await self.session.add_items(
                        [
                            {
                                "type": "function_call",
                                "call_id": cid,
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                            for cid, call in tool_call_payloads
                        ]
                    )

                for call_id, call in tool_call_payloads:
                    self._raise_if_interrupted()
                    start_action = self._tool_start_action(call_id, call)
                    self._record_action(start_action)
                    yield start_action

                    output_text = await self._dispatch_tool_call(call, call_id, tool_lookup)
                    success_action = self._tool_result_action(call_id, call, output_text)
                    self._record_action(success_action)
                    yield success_action

                    messages.append({"role": "tool", "tool_call_id": call_id, "content": output_text})
                    if self.session:
                        await self.session.add_items(
                            [{"type": "function_call_output", "call_id": call_id, "output": output_text}]
                        )

            await self._persist_turn_usage()

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _mcp_lifecycle(self) -> AsyncGenerator[Dict[str, Tuple[_MCPServerBase, MCPTool]], None]:
        """Connect every MCP server, yield ``{tool_name: (server, mcp_tool)}``."""
        if not self.mcp_servers:
            yield {}
            return
        connected: List[_MCPServerBase] = []
        try:
            for server in self.mcp_servers.values():
                await server.connect()
                connected.append(server)
            lookup: Dict[str, Tuple[_MCPServerBase, MCPTool]] = {}
            for server in connected:
                tools = await server.list_tools()
                for tool in tools:
                    lookup[tool.name] = (server, tool)
            yield lookup
        finally:
            for server in connected:
                with contextlib.suppress(Exception):
                    await server.cleanup()

    def _collect_tools_schema(
        self, mcp_lookup: Dict[str, Tuple[_MCPServerBase, MCPTool]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Combine local FunctionTool list + MCP tools into one OpenAI tool list."""
        schema: List[Dict[str, Any]] = []
        lookup: Dict[str, Any] = {}
        for tool in self.local_tools:
            schema.append(tool_to_openai_schema(tool))
            lookup[tool.name] = ("local", tool)
        for name, (server, mcp_tool) in mcp_lookup.items():
            schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description or mcp_tool.name,
                        "parameters": mcp_tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )
            lookup[name] = ("mcp", server, mcp_tool)
        return schema, lookup

    async def _initial_messages(
        self, prompt: Union[str, List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Compose system + history + new prompt into the LLM message list."""
        messages: List[Dict[str, Any]] = []
        if self.instruction:
            messages.append({"role": "system", "content": self.instruction})

        if self.session is not None:
            history = await self.session.get_items()
            if history:
                messages.extend(history)

        if isinstance(prompt, list):
            user_items = list(prompt)
        else:
            user_items = [{"role": "user", "content": str(prompt)}]
        messages.extend(user_items)
        return messages, user_items

    async def _persist_user_items(self, items: List[Dict[str, Any]]) -> None:
        if self.session is not None and items:
            await self.session.add_items(items)

    async def _append_assistant_to_session(self, messages: List[Dict[str, Any]], response: NormalizedResponse) -> None:
        if self.session is None:
            return
        if not response.content:
            return
        await self.session.add_items([{"role": "assistant", "content": response.content}])

    def _assistant_with_tool_calls(self, assistant_text: str, payloads: List[Tuple[str, ToolCall]]) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": assistant_text or None,
            "tool_calls": [
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for cid, call in payloads
            ],
        }

    async def _dispatch_tool_call(
        self,
        call: ToolCall,
        call_id: str,
        tool_lookup: Dict[str, Any],
    ) -> str:
        entry = tool_lookup.get(call.name)
        if entry is None:
            err = {"success": 0, "error": f"Unknown tool: {call.name}"}
            return json.dumps(err)

        if entry[0] == "local":
            tool: FunctionTool = entry[1]
            ctx = ToolContext(tool_call_id=call_id, tool_name=call.name)
            if self.hooks:
                await self.hooks.on_tool_start(self._context, self.model, tool)
            try:
                result = await tool.on_invoke_tool(ctx, call.arguments)
                output_text = self._stringify_tool_output(result)
            except Exception as exc:  # noqa: BLE001 — surface to LLM as tool error
                logger.exception("Local tool %s raised: %s", tool.name, exc)
                output_text = json.dumps({"success": 0, "error": str(exc)})
            if self.hooks:
                await self.hooks.on_tool_end(self._context, self.model, tool, output_text)
            return output_text

        if entry[0] == "mcp":
            _, server, mcp_tool = entry
            try:
                args = json.loads(call.arguments) if call.arguments else {}
            except (TypeError, json.JSONDecodeError):
                args = {}
            try:
                response = await server.call_tool(mcp_tool.name, args)
            except Exception as exc:  # noqa: BLE001 — propagate to LLM
                logger.exception("MCP tool %s failed: %s", mcp_tool.name, exc)
                return json.dumps({"success": 0, "error": str(exc)})
            return self._mcp_result_to_text(response)

        return json.dumps({"success": 0, "error": "Tool dispatcher misconfigured"})

    @staticmethod
    def _stringify_tool_output(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, default=str)
            except (TypeError, ValueError):
                return str(result)
        return str(result)

    @staticmethod
    def _mcp_result_to_text(result: Any) -> str:
        content = getattr(result, "content", None) or []
        parts: List[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
                continue
            data = getattr(block, "data", None)
            if data:
                parts.append(str(data))
        if parts:
            return "\n".join(parts)
        return json.dumps({"isError": getattr(result, "isError", False), "content": []})

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def _tool_start_action(self, call_id: str, call: ToolCall) -> ActionHistory:
        try:
            args_dict = json.loads(call.arguments) if call.arguments else {}
            args_repr = to_str(args_dict)[:80]
        except (TypeError, json.JSONDecodeError):
            args_repr = str(call.arguments)[:80]
        return ActionHistory(
            action_id=call_id,
            role=ActionRole.TOOL,
            messages=f"Tool call: {call.name}",
            action_type=call.name,
            input={"function_name": call.name, "arguments": call.arguments, "args_display": args_repr},
            output=None,
            status=ActionStatus.PROCESSING,
            start_time=datetime.now(),
        )

    def _tool_result_action(self, call_id: str, call: ToolCall, output_text: str) -> ActionHistory:
        try:
            output_data = json.loads(output_text) if output_text else {}
        except (TypeError, json.JSONDecodeError):
            output_data = {"result": output_text}

        status = ActionStatus.SUCCESS
        if isinstance(output_data, dict) and looks_like_failure(output_data):
            status = ActionStatus.FAILED

        if isinstance(output_data, dict):
            try:
                summary = TOOL_SUMMARY_REGISTRY.summarize_dict(output_data, call.name)
            except Exception:  # noqa: BLE001 — never let the formatter abort the run
                summary = ""
            output_data.setdefault("summary", summary or "")
        return ActionHistory(
            action_id="complete_" + call_id,
            role=ActionRole.TOOL,
            messages=f"Tool result: {call.name}",
            action_type=call.name,
            input={"function_name": call.name, "arguments": call.arguments},
            output=output_data,
            status=status,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    def _record_action(self, action: ActionHistory) -> None:
        if action.action_type == "thinking_delta":
            return  # transient — UI consumes the event but we don't persist it
        self.action_history_manager.add_action(action)

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _track_usage(self, response: NormalizedResponse) -> None:
        if response.usage is None:
            return
        usage = self._context.usage
        usage.requests += 1
        usage.input_tokens += response.usage.prompt_tokens
        usage.output_tokens += response.usage.completion_tokens
        usage.total_tokens += response.usage.total_tokens
        usage.cached_tokens += response.usage.cached_tokens

    def _track_usage_chunked(self, usage: Optional[Any]) -> None:
        if usage is None:
            return
        self._context.usage.requests += 1
        self._context.usage.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self._context.usage.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self._context.usage.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            self._context.usage.cached_tokens += int(getattr(details, "cached_tokens", 0) or 0)

    def _usage_to_dict(self) -> Dict[str, Any]:
        usage = self._context.usage
        return {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cached_tokens": usage.cached_tokens,
        }

    async def _persist_turn_usage(self) -> None:
        if self.session is None:
            return
        usage = self._context.usage
        try:
            current_turn = await self._next_turn_number()
            await self.session.store_run_usage(
                turn_number=current_turn,
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cached_tokens=usage.cached_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — accounting must not break the run
            logger.debug("Failed to store turn usage: %s", exc)

    async def _next_turn_number(self) -> int:
        if self.session is None:
            return 1
        items = await self.session.get_items()
        return sum(1 for item in items if item.get("role") == "user") or 1

    def _coerce_output(self, text: str) -> Any:
        if self.output_type is str or not text:
            return text or ""
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return text
        try:
            return self.output_type(**payload) if isinstance(payload, dict) else payload
        except Exception:  # noqa: BLE001 — fall back to raw payload
            return payload

    def _raise_if_interrupted(self) -> None:
        if self.interrupt_controller is None:
            return
        flag = getattr(self.interrupt_controller, "is_interrupted", False)
        if isinstance(flag, bool) and flag:
            from datus.cli.execution_state import ExecutionInterrupted

            raise ExecutionInterrupted("Interrupted by user")
        if callable(flag) and flag():
            from datus.cli.execution_state import ExecutionInterrupted

            raise ExecutionInterrupted("Interrupted by user")


class _ChatStreamCollector:
    """Aggregates streaming Chat Completions chunks into ActionHistory items.

    The Chat Completions stream emits ``choices[0].delta`` with at most
    one of: ``content``, ``tool_calls[i].function`` (whose
    ``arguments`` are themselves chunked), or ``finish_reason``. We
    accumulate the text and the per-tool-call argument string, then
    flush a single ``response`` ActionHistory at the end and return
    ``ToolCall`` records for the agent loop to dispatch.
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.reasoning: str = ""
        self.snapshot: Any = None
        self.usage: Any = None
        self._tool_call_buffers: Dict[int, Dict[str, Any]] = {}
        self._stream_id = f"thinking_stream_{uuid.uuid4().hex[:8]}"

    @property
    def tool_calls(self) -> List[ToolCall]:
        result: List[ToolCall] = []
        for idx in sorted(self._tool_call_buffers):
            buf = self._tool_call_buffers[idx]
            result.append(
                ToolCall(
                    id=buf.get("id"),
                    name=buf.get("name", ""),
                    arguments=buf.get("arguments", "") or "",
                )
            )
        return result

    def feed(self, chunk: Any) -> List[ActionHistory]:
        actions: List[ActionHistory] = []
        self.snapshot = chunk
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.usage = usage
            return actions
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return actions

        text_delta = getattr(delta, "content", None) or ""
        if text_delta:
            self.text += text_delta
            actions.append(
                ActionHistory(
                    action_id=self._stream_id,
                    role=ActionRole.ASSISTANT,
                    messages="",
                    action_type="thinking_delta",
                    input={},
                    output={"delta": text_delta, "accumulated": self.text},
                    status=ActionStatus.PROCESSING,
                )
            )

        reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning_delta:
            self.reasoning += reasoning_delta

        tool_calls = getattr(delta, "tool_calls", None) or []
        for tc in tool_calls:
            idx = getattr(tc, "index", 0) or 0
            buf = self._tool_call_buffers.setdefault(idx, {"id": None, "name": "", "arguments": ""})
            tc_id = getattr(tc, "id", None)
            if tc_id and not buf["id"]:
                buf["id"] = tc_id
            fn = getattr(tc, "function", None)
            if fn is not None:
                fn_name = getattr(fn, "name", None) or ""
                if fn_name and not buf["name"]:
                    buf["name"] = fn_name
                args_chunk = getattr(fn, "arguments", None) or ""
                if args_chunk:
                    buf["arguments"] += args_chunk

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.usage = usage
        return actions

    def finalize(self) -> Optional[ActionHistory]:
        if not self.text:
            return None
        is_thinking = bool(self._tool_call_buffers)
        return ActionHistory(
            action_id=self._stream_id,
            role=ActionRole.ASSISTANT,
            messages=("Thinking: " + _short(self.text, 200)) if is_thinking else _short(self.text, 200),
            action_type="response",
            input={},
            output={"raw_output": self.text, "is_thinking": is_thinking},
            status=ActionStatus.SUCCESS,
        )


def _ensure_datus_exception(error: Exception, code: ErrorCode = ErrorCode.MODEL_REQUEST_FAILED) -> DatusException:
    if isinstance(error, DatusException):
        return error
    return DatusException(code, message_args={"error_message": str(error)})


# Re-export for callers that imported the helper from the agent loop.
__all__ = ["AgentLoop"]


# Tame the unused-loop helper warning when asyncio sleeps are short-circuited.
asyncio.iscoroutinefunction  # noqa: B018
