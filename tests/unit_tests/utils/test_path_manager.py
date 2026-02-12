# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/path_manager.py."""

from pathlib import Path

import pytest

from datus.utils.path_manager import DatusPathManager, get_path_manager, reset_path_manager


class TestDatusPathManager:
    def test_default_home(self):
        pm = DatusPathManager()
        assert pm.datus_home == Path.home() / ".datus"

    def test_custom_home(self, tmp_path):
        pm = DatusPathManager(str(tmp_path / "custom"))
        assert pm.datus_home == tmp_path / "custom"

    def test_update_home(self, tmp_path):
        pm = DatusPathManager()
        pm.update_home(str(tmp_path / "new_home"))
        assert pm.datus_home == tmp_path / "new_home"

    def test_conf_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.conf_dir == tmp_path / "conf"

    def test_data_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.data_dir == tmp_path / "data"

    def test_logs_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.logs_dir == tmp_path / "logs"

    def test_sessions_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.sessions_dir == tmp_path / "sessions"

    def test_template_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.template_dir == tmp_path / "template"

    def test_sample_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.sample_dir == tmp_path / "sample"

    def test_run_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.run_dir == tmp_path / "run"

    def test_benchmark_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.benchmark_dir == tmp_path / "benchmark"

    def test_save_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.save_dir == tmp_path / "save"

    def test_workspace_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.workspace_dir == tmp_path / "workspace"

    def test_trajectory_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.trajectory_dir == tmp_path / "trajectory"

    def test_semantic_models_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.semantic_models_dir == tmp_path / "semantic_models"

    def test_sql_summaries_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.sql_summaries_dir == tmp_path / "sql_summaries"

    def test_ext_knowledge_dir(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.ext_knowledge_dir == tmp_path / "ext_knowledge"

    def test_agent_config_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.agent_config_path() == tmp_path / "conf" / "agent.yml"

    def test_mcp_config_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.mcp_config_path() == tmp_path / "conf" / ".mcp.json"

    def test_auth_config_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.auth_config_path() == tmp_path / "conf" / "auth_clients.yml"

    def test_history_file_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.history_file_path() == tmp_path / "history"

    def test_dashboard_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.dashboard_path() == tmp_path / "dashboard"

    def test_pid_file_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        assert pm.pid_file_path() == tmp_path / "run" / "datus-agent-api.pid"
        assert pm.pid_file_path("custom") == tmp_path / "run" / "custom.pid"

    def test_rag_storage_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        path = pm.rag_storage_path("test_ns")
        assert path == tmp_path / "data" / "datus_db_test_ns"
        assert path.exists()

    def test_sub_agent_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        path = pm.sub_agent_path("agent1")
        assert path == tmp_path / "data" / "sub_agents" / "agent1"
        assert path.exists()

    def test_session_db_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        path = pm.session_db_path("session_123")
        assert path == tmp_path / "sessions" / "session_123.db"

    def test_semantic_model_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        path = pm.semantic_model_path("ns1")
        assert path == tmp_path / "semantic_models" / "ns1"
        assert path.exists()

    def test_sql_summary_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        path = pm.sql_summary_path("ns1")
        assert path == tmp_path / "sql_summaries" / "ns1"
        assert path.exists()

    def test_ext_knowledge_path(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        path = pm.ext_knowledge_path("ns1")
        assert path == tmp_path / "ext_knowledge" / "ns1"
        assert path.exists()

    def test_resolve_config_path_explicit(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        # Create explicit path
        explicit = tmp_path / "explicit.yml"
        explicit.write_text("config")
        result = pm.resolve_config_path("agent.yml", local_path=str(explicit))
        assert result == explicit

    def test_resolve_config_path_default(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        result = pm.resolve_config_path("agent.yml")
        assert result == tmp_path / "conf" / "agent.yml"

    def test_ensure_dirs_all(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        pm.ensure_dirs()
        assert pm.conf_dir.exists()
        assert pm.data_dir.exists()
        assert pm.logs_dir.exists()

    def test_ensure_dirs_specific(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        pm.ensure_dirs("conf", "data")
        assert pm.conf_dir.exists()
        assert pm.data_dir.exists()

    def test_ensure_dirs_invalid(self, tmp_path):
        pm = DatusPathManager(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid directory name"):
            pm.ensure_dirs("invalid_name")

    def test_resolve_run_dir(self, tmp_path):
        result = DatusPathManager.resolve_run_dir(tmp_path, "ns1", "run_001")
        assert result == tmp_path / "ns1" / "run_001"
        assert result.exists()

    def test_resolve_run_dir_without_run_id(self, tmp_path):
        result = DatusPathManager.resolve_run_dir(tmp_path, "ns1")
        assert result == tmp_path / "ns1"
        assert result.exists()


class TestGetPathManager:
    def setup_method(self):
        reset_path_manager()

    def teardown_method(self):
        reset_path_manager()

    def test_returns_singleton(self):
        pm1 = get_path_manager()
        pm2 = get_path_manager()
        assert pm1 is pm2

    def test_reset_and_recreate(self):
        pm1 = get_path_manager()
        reset_path_manager()
        pm2 = get_path_manager()
        assert pm1 is not pm2
