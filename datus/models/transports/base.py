# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Abstract base for provider transports.

A transport owns the *data path* for one wire-format / api_mode:
``convert_messages`` → ``convert_tools`` → ``build_kwargs`` →
``normalize_response``.  It does NOT own client construction, streaming
control, credential refresh, prompt caching, retry logic or interrupt
handling — those stay on the model class and the agent loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from datus.models.transports.types import NormalizedResponse


class ProviderTransport(ABC):
    """Base class for provider-specific format conversion + normalisation."""

    @property
    @abstractmethod
    def api_mode(self) -> str:
        """Identifier for the wire format (e.g. ``"chat_completions"``)."""
        raise NotImplementedError

    @abstractmethod
    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        """Convert OpenAI-format messages to the provider's native format.

        Returns whatever shape the provider needs (e.g. ``(system, msgs)``
        for Anthropic, the messages list unchanged for Chat Completions).
        """
        raise NotImplementedError

    @abstractmethod
    def convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Any:
        """Convert OpenAI-format tool schemas to the provider's tool list."""
        raise NotImplementedError

    @abstractmethod
    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """Assemble the complete request kwargs dict.

        Typically calls :meth:`convert_messages` + :meth:`convert_tools`
        and appends model-specific config; the result is ready for the
        provider's SDK call.
        """
        raise NotImplementedError

    @abstractmethod
    def normalize_response(self, response: Any, **kwargs: Any) -> NormalizedResponse:
        """Normalise a raw provider response to :class:`NormalizedResponse`."""
        raise NotImplementedError

    def validate_response(self, response: Any) -> bool:
        """Override to detect structurally-invalid responses early."""
        return True

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Override to surface provider-specific prompt cache statistics."""
        return None

    def map_finish_reason(self, raw_reason: str) -> str:
        """Override to map provider stop reasons to the OpenAI vocabulary."""
        return raw_reason
