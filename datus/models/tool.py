# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""In-house Tool / FunctionTool replacement for ``agents.Tool``.

The previous implementation pulled :class:`agents.tool.FunctionTool`
from the openai-agents-sdk; this module re-implements the tiny subset
that Datus actually uses (function-based tools with a JSON-schema
parameter spec), so we can drop the SDK dependency entirely.

What we keep verbatim from the SDK contract:
* :class:`FunctionTool` is a ``@dataclass`` with ``name``,
  ``description``, ``params_json_schema``, ``on_invoke_tool``,
  ``strict_json_schema``, ``is_enabled``, ``tool_input_guardrails``,
  ``tool_output_guardrails``.
* :data:`Tool` is the ``Union`` exposed to type hints — only
  :class:`FunctionTool` is supported in our runtime.
* :func:`function_tool` is the decorator/factory that derives a
  :class:`FunctionTool` from a regular Python callable using its
  signature + docstring (mirroring the SDK helper).
* :class:`ToolContext` carries the per-invocation context the SDK used
  to pass into ``on_invoke_tool``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union, get_type_hints

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class ToolContext:
    """Per-invocation context passed to ``on_invoke_tool`` callbacks."""

    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    agent_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agent_context(
        cls,
        *,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        **extra: Any,
    ) -> "ToolContext":
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            agent_name=agent_name,
            extra=dict(extra),
        )


OnInvokeTool = Callable[[ToolContext, str], Awaitable[Any]]


@dataclass
class FunctionTool:
    """A tool backed by a Python callable.

    Mirrors the surface of ``agents.tool.FunctionTool`` so that callers
    in ``datus/tools/`` keep their existing construction code.
    """

    name: str
    description: str
    params_json_schema: Dict[str, Any]
    on_invoke_tool: OnInvokeTool
    strict_json_schema: bool = True
    is_enabled: Union[bool, Callable[..., Any]] = True
    tool_input_guardrails: Optional[List[Any]] = None
    tool_output_guardrails: Optional[List[Any]] = None


# ``Tool`` was a Union over many SDK tool kinds; in Datus we only ship
# function tools, so the alias collapses to FunctionTool. Keeping the
# name lets ``isinstance(t, Tool)``-style code keep working when written
# as ``isinstance(t, FunctionTool)``.
Tool = FunctionTool


# ---------------------------------------------------------------------------
# JSON schema helpers
# ---------------------------------------------------------------------------


_SIMPLE_TYPE_MAP: Dict[type, Dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array", "items": {}},
    dict: {"type": "object", "additionalProperties": True},
    type(None): {"type": "null"},
}


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    """Best-effort Python-annotation → JSON Schema fragment.

    Falls back to permissive ``{"type": "string"}`` when the annotation
    is too complex for a quick mapping; pydantic-based callers should
    pass an ``input_model`` to :func:`function_tool` instead.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    if annotation in _SIMPLE_TYPE_MAP:
        return dict(_SIMPLE_TYPE_MAP[annotation])
    # ``Optional[X]`` and other generics: try Pydantic for a proper schema.
    try:
        adapter = TypeAdapter(annotation)
        schema = adapter.json_schema(mode="validation")
        # Strip top-level ``$defs`` / ``title`` noise; keep the inline shape.
        schema.pop("title", None)
        return schema
    except Exception:  # noqa: BLE001 — fall back gracefully for exotic annotations
        logger.debug("Failed to derive JSON schema for annotation %r; using string fallback.", annotation)
        return {"type": "string"}


def _strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Apply OpenAI-style strict mode tweaks (``additionalProperties: False``).

    Only the top-level object schema is touched; nested object schemas
    are left as-is to avoid over-constraining tool inputs.
    """
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        if "properties" in schema and "required" not in schema:
            schema["required"] = list(schema["properties"].keys())
    return schema


def _build_params_schema(func: Callable[..., Any], strict: bool) -> Dict[str, Any]:
    """Derive a JSON schema describing *func*'s positional/keyword parameters."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 — unevaluable annotations should not abort tool registration
        hints = {}
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name in {"self", "cls"} or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, param.annotation)
        prop_schema = _annotation_to_schema(annotation)
        # Capture the parameter description from a Pydantic FieldInfo default
        # if the user wrote ``arg: str = Field(default=..., description="...")``.
        default = param.default
        if isinstance(default, FieldInfo):
            if default.description:
                prop_schema = {**prop_schema, "description": default.description}
            if default.default is not inspect.Parameter.empty and default.is_required() is False:
                pass  # default value handled below
            else:
                required.append(name)
        elif default is inspect.Parameter.empty:
            required.append(name)
        properties[name] = prop_schema

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if strict:
        schema = _strict_schema(schema)
    return schema


def _docstring_summary(func: Callable[..., Any]) -> str:
    """Pull a one-paragraph summary out of *func*'s docstring."""
    doc = inspect.getdoc(func) or ""
    if not doc:
        return ""
    return doc.split("\n\n", 1)[0].strip()


def function_tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    strict_mode: bool = True,
    is_enabled: Union[bool, Callable[..., Any]] = True,
    input_model: Optional[Type[BaseModel]] = None,
) -> Any:
    """Decorator/factory turning a callable into a :class:`FunctionTool`.

    Mirrors ``agents.function_tool`` for the subset Datus uses. Pass
    ``input_model`` (a Pydantic model class) for richer JSON schemas
    when the function takes a single complex argument.
    """

    def _wrap(target: Callable[..., Any]) -> FunctionTool:
        tool_name = name_override or target.__name__
        tool_desc = description_override or _docstring_summary(target) or tool_name

        if input_model is not None:
            schema = input_model.model_json_schema()
            schema.pop("title", None)
            if strict_mode:
                schema = _strict_schema(schema)
        else:
            schema = _build_params_schema(target, strict=strict_mode)

        async def on_invoke(ctx: ToolContext, args_str: str) -> Any:
            try:
                args = json.loads(args_str) if args_str else {}
            except (TypeError, json.JSONDecodeError) as exc:
                return {"success": 0, "error": f"Invalid JSON arguments ({exc})", "result": None}
            if not isinstance(args, dict):
                args = {"value": args}
            if input_model is not None:
                args = {"args": input_model(**args)}
            try:
                if asyncio.iscoroutinefunction(target):
                    result = await target(**args)
                else:
                    result = target(**args)
            except Exception as exc:  # noqa: BLE001 — surface tool errors to the LLM
                logger.exception("Tool %s raised: %s", tool_name, exc)
                return {"success": 0, "error": str(exc), "result": None}
            return result

        return FunctionTool(
            name=tool_name,
            description=tool_desc,
            params_json_schema=schema,
            on_invoke_tool=on_invoke,
            strict_json_schema=strict_mode,
            is_enabled=is_enabled,
        )

    # Support both ``@function_tool`` (no args) and ``function_tool(func)``.
    if func is not None and callable(func):
        return _wrap(func)
    return _wrap


def tool_to_openai_schema(tool: FunctionTool) -> Dict[str, Any]:
    """Render *tool* in the OpenAI ``tools=[{...}]`` request format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.params_json_schema,
        },
    }
