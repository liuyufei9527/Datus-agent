# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the per-session system-prompt snapshot cache.

Covers:
- ``_get_session_system_prompt``: build-once / replay-verbatim, and rebuild on a
  meta change (model switch).
- ``_system_prompt_snapshot_meta``: identity keys exclude per-turn live values.
- ``_build_environment_block``: per-turn live datasource + non-default profile.
- ``_inject_runtime_context``: gated on ``db_func_tool``; renders the shared
  ``runtime_context`` partial (date + datasource catalog + workspace root).

A lightweight fake node bypasses the heavy ``AgenticNode.__init__`` and wires a
real :class:`SessionManager` over ``tmp_path`` — no LLM, no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.models.session_manager import SessionManager


def _agent_config(*, current_datasource=None, services=None, profile="normal", model="gpt-4.1"):
    return SimpleNamespace(
        prompt_version="1.2",
        current_datasource=current_datasource,
        services=services,
        active_profile_name=profile,
        active_model=lambda: SimpleNamespace(type="openai", model=model),
    )


class _SnapshotNode(AgenticNode):
    """Minimal node exposing the real snapshot/runtime-context methods."""

    def __init__(self, session_manager: SessionManager, agent_config, *, db_func_tool=None):
        self.session_id = "chat_session_x"
        self._session_manager = session_manager
        self.agent_config = agent_config
        self.db_func_tool = db_func_tool
        self.build_count = 0

    def get_node_name(self) -> str:
        return "chat"

    def _resolve_workspace_root(self) -> str:
        return "/tmp/ws"

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        # Each rebuild is observable and uniquely tagged so replay vs rebuild is
        # unambiguous in assertions.
        self.build_count += 1
        return f"SYS#{self.build_count}"


@pytest.fixture
def session_manager(tmp_path):
    return SessionManager(session_dir=str(tmp_path))


class TestGetSessionSystemPrompt:
    def test_first_turn_builds_and_persists(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config())
        prompt = node._get_session_system_prompt(prompt_version="1.2")
        assert prompt == "SYS#1"
        assert node.build_count == 1
        # Snapshot file now persists the exact prompt for replay.
        snapshot = session_manager.load_system_prompt_snapshot(node.session_id)
        assert snapshot["prompt"] == "SYS#1"
        assert snapshot["model_name"] == "openai:gpt-4.1"

    def test_second_turn_replays_verbatim(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config())
        first = node._get_session_system_prompt(prompt_version="1.2")
        second = node._get_session_system_prompt(prompt_version="1.2")
        assert first == second == "SYS#1"
        # The expensive builder ran exactly once across both turns.
        assert node.build_count == 1

    def test_model_switch_rebuilds(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config(model="gpt-4.1"))
        first = node._get_session_system_prompt(prompt_version="1.2")
        assert first == "SYS#1"
        # Switch model → meta mismatch → rebuild and overwrite.
        node.agent_config = _agent_config(model="gpt-5")
        second = node._get_session_system_prompt(prompt_version="1.2")
        assert second == "SYS#2"
        assert node.build_count == 2

    def test_no_session_id_skips_cache(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config())
        node.session_id = ""
        node._get_session_system_prompt(prompt_version="1.2")
        node._get_session_system_prompt(prompt_version="1.2")
        # Without a session id there is no snapshot to replay → always rebuilds.
        assert node.build_count == 2


class TestSnapshotMeta:
    def test_meta_excludes_per_turn_live_values(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config(current_datasource="main", profile="auto"))
        meta = node._system_prompt_snapshot_meta("1.2")
        assert meta == {"node_name": "chat", "prompt_version": "1.2", "model_name": "openai:gpt-4.1"}
        # Live values must NOT leak into the cache key.
        assert "datasource" not in meta
        assert "active_profile" not in meta


class TestEnvironmentBlock:
    def test_datasource_line_requires_db_tool(self, session_manager):
        cfg = _agent_config(current_datasource="main")
        without = _SnapshotNode(session_manager, cfg, db_func_tool=None)
        assert without._build_environment_block() == ""

        with_db = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        block = with_db._build_environment_block()
        assert "<environment>" in block
        assert "Current datasource: main" in block

    def test_dialect_included_when_available(self, session_manager):
        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="snowflake")})
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        block = node._build_environment_block()
        assert "Current datasource: main (dialect: snowflake)" in block

    def test_default_profile_omitted_nondefault_included(self, session_manager):
        normal = _SnapshotNode(
            session_manager, _agent_config(current_datasource="main", profile="normal"), db_func_tool=object()
        )
        assert "permission profile" not in normal._build_environment_block()

        auto = _SnapshotNode(
            session_manager, _agent_config(current_datasource="main", profile="auto"), db_func_tool=object()
        )
        assert "Current permission profile: auto" in auto._build_environment_block()

    def test_empty_when_no_datasource_and_default_profile(self, session_manager):
        node = _SnapshotNode(
            session_manager, _agent_config(current_datasource=None, profile="normal"), db_func_tool=object()
        )
        assert node._build_environment_block() == ""


class TestInjectRuntimeContext:
    def test_skipped_without_db_tool(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config(), db_func_tool=None)
        assert node._inject_runtime_context("BASE") == "BASE"

    def test_appends_catalog_and_workspace_for_db_node(self, session_manager):
        services = SimpleNamespace(
            datasources={"main": SimpleNamespace(type="snowflake"), "dev": SimpleNamespace(type="duckdb")}
        )
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        out = node._inject_runtime_context("BASE")
        assert out.startswith("BASE")
        assert "Current context:" in out
        assert "Available datasources:" in out
        assert "main (snowflake)" in out
        assert "/tmp/ws" in out
