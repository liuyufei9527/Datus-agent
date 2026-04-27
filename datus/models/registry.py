# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Provider registry — maps a ``ModelConfig.type`` value to the
implementation class that handles it.

Adding a new provider:

1. Implement the model class (subclass :class:`LLMBaseModel` or
   :class:`OpenAICompatibleModel`).
2. Add an entry below mapping the catalog ``type`` string to the
   ``(module_path, class_name)`` pair.

The registry is intentionally a dict literal rather than a decorator
chain so the supported providers are visible at a glance.
"""

from __future__ import annotations

from typing import Dict, Tuple

from datus.utils.constants import LLMProvider

ModuleClassRef = Tuple[str, str]

# ``ModelConfig.type`` value → ``(module_path, class_name)``.
MODEL_TYPE_MAP: Dict[str, ModuleClassRef] = {
    LLMProvider.OPENAI.value: ("datus.models.openai_compatible", "OpenAICompatibleModel"),
    LLMProvider.CLAUDE.value: ("datus.models.anthropic_model", "AnthropicModel"),
    # Stub placeholders — wire up in follow-up PRs.
    LLMProvider.DEEPSEEK.value: ("datus.models.stub_model", "StubModel"),
    LLMProvider.QWEN.value: ("datus.models.stub_model", "StubModel"),
    LLMProvider.KIMI.value: ("datus.models.stub_model", "StubModel"),
    LLMProvider.GLM.value: ("datus.models.stub_model", "StubModel"),
    LLMProvider.MINIMAX.value: ("datus.models.stub_model", "StubModel"),
    LLMProvider.OPENROUTER.value: ("datus.models.stub_model", "StubModel"),
    LLMProvider.GEMINI.value: ("datus.models.stub_model", "StubModel"),
    LLMProvider.CODEX.value: ("datus.models.stub_model", "StubModel"),
}


def resolve_model_class(model_type: str) -> Tuple[str, str]:
    """Return ``(module_path, class_name)`` for *model_type* or raise ``KeyError``."""
    try:
        return MODEL_TYPE_MAP[model_type]
    except KeyError as exc:
        raise KeyError(f"Unsupported model type: {model_type!r}. Known types: {sorted(MODEL_TYPE_MAP)}.") from exc
