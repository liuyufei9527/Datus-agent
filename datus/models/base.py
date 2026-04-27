# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Abstract base for every Datus LLM provider implementation.

The public surface (signatures of ``generate`` /
``generate_with_json_output`` / ``generate_with_tools`` /
``generate_with_tools_stream`` / ``token_count`` / ``context_length`` /
session helpers) is preserved byte-for-byte from the previous SDK-backed
implementation so the 50+ caller sites need no changes beyond their
import path.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import multiprocessing
import os
import platform
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, AsyncGenerator, ClassVar, Dict, List, Optional, Tuple, Union

from datus.configuration.agent_config import AgentConfig, ModelConfig
from datus.models.mcp_client import MCPServerStdio
from datus.models.registry import MODEL_TYPE_MAP, resolve_model_class
from datus.models.session import SQLiteSession
from datus.models.tool import FunctionTool
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def configure_multiprocessing_start_method() -> None:
    """Set a safe multiprocessing start method for the current platform.

    Kept on this module for backwards compatibility with the test suite
    in ``tests/unit_tests/utils/test_windows_compatibility.py`` (the
    previous implementation lived in ``datus.models.base`` and was
    invoked at import time).
    """
    try:
        if platform.system() == "Windows":
            multiprocessing.set_start_method("spawn", force=True)
        else:
            multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        # ``set_start_method`` may only be called once per process.
        pass


configure_multiprocessing_start_method()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class LLMBaseModel(ABC):
    """Abstract base class for all language model implementations.

    Concrete subclasses live in ``datus/models/<type>_model.py`` and are
    looked up via :data:`MODEL_TYPE_MAP`.
    """

    # Re-exported for backwards compatibility with callers that still
    # read ``LLMBaseModel.MODEL_TYPE_MAP`` directly.
    MODEL_TYPE_MAP: ClassVar[Dict[str, Tuple[str, str]]] = MODEL_TYPE_MAP

    # Process-wide LRU keyed on the resolved configuration fingerprint.
    # ``/model`` switching produces a different key and a fresh instance;
    # subsequent calls under the same selection reuse the cached client.
    _MODEL_CACHE_MAXSIZE: ClassVar[int] = 4
    _MODEL_CACHE: ClassVar["OrderedDict[Tuple, LLMBaseModel]"] = OrderedDict()
    _MODEL_CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_config: ModelConfig, **kwargs: Any) -> None:
        self.model_config = model_config
        self._session_manager = None
        self.session_dir = kwargs.get("session_dir")
        self.session_scope = kwargs.get("session_scope")
        self.workflow = None
        self.current_node = None

    # ------------------------------------------------------------------
    # Factory + cache
    # ------------------------------------------------------------------

    @classmethod
    def create_model(
        cls,
        agent_config: AgentConfig,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "LLMBaseModel":
        """Resolve the active LLM and return a (cached) instance.

        Resolution order:

        1. ``model_name == "default"`` or ``None`` → ``agent_config.active_model()``.
        2. ``model_name in agent_config.models`` → use that model entry.
        3. Otherwise raise :class:`KeyError`.
        """
        if not model_name or model_name == "default":
            target = agent_config.active_model()
        elif model_name in agent_config.models:
            target = agent_config.model_config(model_name)
        else:
            raise KeyError(f"Model {model_name} not found in agent_config")

        scope = kwargs.get("scope")
        api_key_digest = hashlib.sha1((target.api_key or "").encode("utf-8")).hexdigest()[:12] if target.api_key else ""
        cache_key: Tuple = (
            target.type,
            target.model,
            target.base_url or "",
            api_key_digest,
            target.auth_type,
            scope or "",
            bool(target.enable_thinking),
            target.reasoning_effort or "",
        )

        with cls._MODEL_CACHE_LOCK:
            cached = cls._MODEL_CACHE.get(cache_key)
            if cached is not None:
                cls._MODEL_CACHE.move_to_end(cache_key)
                return cached

        module_path, class_name = resolve_model_class(target.type)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        instance = model_class(
            model_config=target,
            session_dir=agent_config.session_dir,
            session_scope=scope,
        )

        with cls._MODEL_CACHE_LOCK:
            cls._MODEL_CACHE[cache_key] = instance
            cls._MODEL_CACHE.move_to_end(cache_key)
            while len(cls._MODEL_CACHE) > cls._MODEL_CACHE_MAXSIZE:
                cls._MODEL_CACHE.popitem(last=False)
        return instance

    # ------------------------------------------------------------------
    # Required generation interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs: Any) -> str:
        """Single-turn text generation."""

    @abstractmethod
    def generate_with_json_output(self, prompt: Any, **kwargs: Any) -> Dict[str, Any]:
        """Single-turn structured-JSON generation."""

    @abstractmethod
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
        """Multi-turn tool-calling generation; returns ``{content, sql_contexts, usage, ...}``."""

    @abstractmethod
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
        """Streaming version of :meth:`generate_with_tools`."""

    @abstractmethod
    def token_count(self, prompt: str) -> int:
        """Best-effort prompt-token count (used by compaction triggers)."""

    @abstractmethod
    def context_length(self) -> Optional[int]:
        """Maximum input context length for the active model."""

    # ------------------------------------------------------------------
    # Helpers shared by every subclass
    # ------------------------------------------------------------------

    def set_context(self, workflow: Any = None, current_node: Any = None) -> None:
        self.workflow = workflow
        self.current_node = current_node

    def to_dict(self) -> Dict[str, str]:
        return {"model_name": self.model_config.model}

    async def test_connection(self, timeout: float = 10.0) -> Tuple[bool, str]:
        """Probe the model with a 1-token request; return ``(ok, error)``."""
        try:
            probe = asyncio.to_thread(self.generate, "hi", max_tokens=1)
            response = await asyncio.wait_for(probe, timeout=timeout)
            if response is None or not str(response).strip():
                return False, "Empty response from model"
            return True, ""
        except asyncio.TimeoutError:
            return False, f"Timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001 — surface error string to caller
            return False, str(exc)

    # ------------------------------------------------------------------
    # Session glue
    # ------------------------------------------------------------------

    @property
    def session_manager(self):
        if self._session_manager is None:
            from datus.models.session_manager import SessionManager

            self._session_manager = SessionManager(session_dir=self.session_dir, scope=self.session_scope)
        return self._session_manager

    def create_session(self, session_id: str) -> SQLiteSession:
        return self.session_manager.create_session(session_id)

    def clear_session(self, session_id: str) -> None:
        self.session_manager.clear_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self.session_manager.delete_session(session_id)
