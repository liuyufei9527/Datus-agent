# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Token counting and context-length lookup.

Replaces ``litellm.token_counter`` / ``litellm.get_max_tokens``.

* OpenAI-family models use ``tiktoken`` for byte-perfect counts.
* Claude models use ``anthropic.Anthropic().messages.count_tokens`` when
  the SDK is available, otherwise fall back to character/4.
* Everything else falls back to a deliberately rough character/4
  heuristic — good enough for compaction triggers, not for billing.

``context_length`` and ``max_tokens`` are loaded from
``conf/providers.yml`` (``model_specs`` block) with prefix-matching, so
versioned IDs like ``claude-sonnet-4-6-20250514`` resolve to
``claude-sonnet-4-6``.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import yaml

from datus.utils.loggings import get_logger
from datus.utils.resource_utils import read_data_file_text

logger = get_logger(__name__)


_SPECS_LOCK = threading.Lock()
_SPECS_CACHE: Optional[Dict[str, Dict[str, int]]] = None


def _load_model_specs() -> Dict[str, Dict[str, int]]:
    """Load and cache ``model_specs`` from the bundled providers catalog.

    Mirrors what the deleted openai_compatible.py did. The OpenRouter
    cache is intentionally not merged in here — it lives behind the
    ``cli.provider_model_catalog`` import which itself depends on this
    module during model construction.
    """
    global _SPECS_CACHE
    if _SPECS_CACHE is not None:
        return _SPECS_CACHE
    with _SPECS_LOCK:
        if _SPECS_CACHE is not None:
            return _SPECS_CACHE
        specs: Dict[str, Dict[str, int]] = {}
        try:
            text = read_data_file_text("conf/providers.yml")
            catalog = yaml.safe_load(text) or {}
            raw = catalog.get("model_specs") or {}
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if isinstance(key, str) and isinstance(value, dict):
                        specs[key] = dict(value)
        except Exception as exc:  # noqa: BLE001 — fall back to empty when YAML is missing
            logger.warning("Failed to load model_specs: %s", exc)
        _SPECS_CACHE = specs
    return _SPECS_CACHE


def _lookup_spec(model_name: str) -> Dict[str, int]:
    specs = _load_model_specs()
    if model_name in specs:
        return specs[model_name]
    # Prefix match for versioned IDs like "kimi-k2-turbo-0905-preview".
    best: Optional[tuple[int, str]] = None
    for key in specs:
        if model_name.startswith(key) and (best is None or len(key) > best[0]):
            best = (len(key), key)
    if best is not None:
        return specs[best[1]]
    return {}


def context_length(model_name: str) -> Optional[int]:
    """Return the maximum *input* context length for *model_name* (or ``None``)."""
    spec = _lookup_spec(model_name)
    value = spec.get("context_length")
    return int(value) if value else None


def max_output_tokens(model_name: str) -> Optional[int]:
    """Return the maximum *output* token budget for *model_name* (or ``None``)."""
    spec = _lookup_spec(model_name)
    value = spec.get("max_tokens")
    return int(value) if value else None


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


_TIKTOKEN_LOCK = threading.Lock()
_TIKTOKEN_CACHE: Dict[str, Any] = {}


def _get_tiktoken_encoder(model_name: str) -> Optional[Any]:
    """Return a tiktoken encoder for OpenAI-family models, or ``None``."""
    try:
        import tiktoken
    except ImportError:
        return None
    with _TIKTOKEN_LOCK:
        if model_name in _TIKTOKEN_CACHE:
            return _TIKTOKEN_CACHE[model_name]
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fall back to o200k_base (the GPT-4o / GPT-4.1 family default).
            enc = tiktoken.get_encoding("o200k_base")
        except Exception as exc:  # noqa: BLE001 — exotic build issues should not crash callers
            logger.debug("tiktoken init failed for %s: %s", model_name, exc)
            enc = None
        _TIKTOKEN_CACHE[model_name] = enc
        return enc


def _approximate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def count_tokens(model_name: str, prompt: str, model_type: str = "openai") -> int:
    """Best-effort token count for *prompt* under *model_name*.

    ``model_type`` controls fallback selection: ``"openai"`` and other
    OpenAI-compatible types use tiktoken; ``"claude"`` defers to the
    Anthropic SDK; everything else uses a chars/4 approximation.
    """
    if not prompt:
        return 0
    text = str(prompt)

    if model_type in {"openai", "deepseek", "qwen", "kimi", "glm", "minimax", "openrouter", "codex"}:
        encoder = _get_tiktoken_encoder(model_name)
        if encoder is not None:
            try:
                return len(encoder.encode(text))
            except Exception as exc:  # noqa: BLE001 — tokenizer corruption shouldn't crash callers
                logger.debug("tiktoken encode failed for %s: %s", model_name, exc)

    if model_type == "claude":
        try:
            import anthropic

            client = anthropic.Anthropic()
            result = client.messages.count_tokens(
                model=model_name,
                messages=[{"role": "user", "content": text}],
            )
            return int(getattr(result, "input_tokens", 0)) or _approximate_tokens(text)
        except Exception as exc:  # noqa: BLE001 — offline / quota / unsupported model
            logger.debug("anthropic token count failed for %s: %s", model_name, exc)

    return _approximate_tokens(text)
