# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Literal

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

HAS_LANGSMITH = False
try:
    from langsmith.client import RUN_TYPE_T

    HAS_LANGSMITH = True
except ImportError:
    RUN_TYPE_T = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def optional_traceable(name: str = "", run_type: RUN_TYPE_T = "chain"):
    """
    Optional traceable decorator that wraps functions with LangSmith tracing.

    Args:
        name: The name of the trace. Defaults to the function name.
        run_type: The type of run (e.g., "chain", "llm", "tool").
    """

    def decorator(func):
        if not HAS_LANGSMITH:
            return func
        try:
            from langsmith import traceable

            trace_name = name or getattr(func, "__name__", "agent_operation")
            return traceable(name=trace_name, run_type=run_type)(func)
        except ImportError:
            return func

    return decorator


_tracing_initialized = False
_tracing_processor = None


def _is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is explicitly enabled via environment variables."""
    import os

    tracing_enabled = (
        os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )
    has_api_key = bool(os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY"))
    return tracing_enabled and has_api_key


def setup_tracing():
    """Set up LangSmith tracing with DatusTracingProcessor.

    Creates a DatusTracingProcessor (subclass of OpenAIAgentsTracingProcessor)
    that captures trace URLs on trace end, and registers it via set_trace_processors.

    Requires both a tracing env var (LANGSMITH_TRACING=true or LANGCHAIN_TRACING_V2=true)
    and a valid API key (LANGCHAIN_API_KEY or LANGSMITH_API_KEY) to be set.

    Safe to call multiple times; initialization only happens once.
    """
    global _tracing_initialized, _tracing_processor
    if _tracing_initialized:
        return
    _tracing_initialized = True

    if not HAS_LANGSMITH:
        return

    if not _is_tracing_enabled():
        logger.debug("LangSmith tracing not enabled (set LANGSMITH_TRACING=true and LANGCHAIN_API_KEY to enable)")
        return

    # The previous integration relied on ``agents.set_trace_processors`` from
    # openai-agents-sdk. Since that dependency was dropped, in-process LangSmith
    # tracing is currently a no-op; users who need traces should set up LangSmith
    # via the standard ``langsmith.trace`` decorators on the call sites that
    # matter to them. The follow-up native-tracing PR will restore parity here.
    logger.debug("LangSmith SDK tracing is currently disabled (post-litellm refactor).")


def get_trace_url() -> str | None:
    """Return the last captured LangSmith trace URL, or None."""
    if _tracing_processor is not None:
        return getattr(_tracing_processor, "_last_trace_url", None)
    return None
