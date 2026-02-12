# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/configuration/agent_config.py - additional coverage."""

import os
from dataclasses import asdict

import pytest

from datus.configuration.agent_config import (
    AgentConfig,
    BenchmarkConfig,
    DbConfig,
    DashboardConfig,
    DocumentConfig,
    ModelConfig,
    NodeConfig,
    file_stem_from_uri,
    load_model_config,
    rag_storage_path,
    resolve_env,
)
from datus.utils.exceptions import DatusException


# =====================================================================
# resolve_env
# =====================================================================


class TestResolveEnv:
    def test_simple_env_var(self):
        os.environ["TEST_VAR_ABC"] = "hello"
        result = resolve_env("${TEST_VAR_ABC}")
        assert result == "hello"
        del os.environ["TEST_VAR_ABC"]

    def test_missing_env_var(self):
        result = resolve_env("${NONEXISTENT_VAR_XYZ}")
        assert "<MISSING:NONEXISTENT_VAR_XYZ>" in result

    def test_mixed_text_and_env(self):
        os.environ["TEST_MIXED_VAR"] = "world"
        result = resolve_env("hello ${TEST_MIXED_VAR}")
        assert result == "hello world"
        del os.environ["TEST_MIXED_VAR"]

    def test_no_env_var(self):
        result = resolve_env("plain text")
        assert result == "plain text"

    def test_empty_string(self):
        assert resolve_env("") == ""

    def test_none_passthrough(self):
        assert resolve_env(None) is None

    def test_non_string_passthrough(self):
        assert resolve_env(123) == 123


# =====================================================================
# file_stem_from_uri
# =====================================================================


class TestFileStemFromUri:
    def test_duckdb_uri(self):
        assert file_stem_from_uri("duckdb:///path/to/demo.duckdb") == "demo"

    def test_sqlite_uri(self):
        assert file_stem_from_uri("sqlite:////tmp/foo.db") == "foo"

    def test_plain_path(self):
        assert file_stem_from_uri("/abs/path/bar.duckdb") == "bar"

    def test_filename_only(self):
        assert file_stem_from_uri("foo.db") == "foo"

    def test_empty_string(self):
        assert file_stem_from_uri("") == ""


# =====================================================================
# load_model_config
# =====================================================================


class TestLoadModelConfig:
    def test_basic_load(self):
        data = {
            "type": "openai",
            "api_key": "key",
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
        }
        cfg = load_model_config(data)
        assert cfg.type == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.api_key == "key"
        assert cfg.save_llm_trace is False
        assert cfg.enable_thinking is False

    def test_with_optional_fields(self):
        data = {
            "type": "deepseek",
            "api_key": "key",
            "model": "deepseek-chat",
            "save_llm_trace": True,
            "enable_thinking": True,
            "strict_json_schema": False,
            "default_headers": {"X-Custom": "val"},
            "max_retry": 5,
            "retry_interval": 3.0,
            "temperature": 0.5,
            "top_p": 0.9,
        }
        cfg = load_model_config(data)
        assert cfg.save_llm_trace is True
        assert cfg.enable_thinking is True
        assert cfg.strict_json_schema is False
        assert cfg.default_headers == {"X-Custom": "val"}
        assert cfg.max_retry == 5
        assert cfg.retry_interval == 3.0
        assert cfg.temperature == 0.5
        assert cfg.top_p == 0.9

    def test_defaults_for_retry(self):
        data = {
            "type": "openai",
            "api_key": "key",
            "model": "gpt-4o",
        }
        cfg = load_model_config(data)
        assert cfg.max_retry == 3
        assert cfg.retry_interval == 2.0
        assert cfg.temperature is None
        assert cfg.top_p is None


# =====================================================================
# ModelConfig
# =====================================================================


class TestModelConfig:
    def test_to_dict(self):
        cfg = ModelConfig(type="openai", api_key="key", model="gpt-4o")
        d = cfg.to_dict()
        assert d["type"] == "openai"
        assert d["model"] == "gpt-4o"


# =====================================================================
# DbConfig
# =====================================================================


class TestDbConfig:
    def test_to_dict(self):
        cfg = DbConfig(type="sqlite", uri="sqlite:///test.db", database="test")
        d = cfg.to_dict()
        assert d["type"] == "sqlite"
        assert d["database"] == "test"

    def test_filter_kwargs_basic(self):
        kwargs = {
            "type": "postgresql",
            "host": "localhost",
            "port": "5432",
            "username": "user",
            "password": "pass",
            "database": "mydb",
        }
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.type == "postgresql"
        assert cfg.host == "localhost"
        assert cfg.database == "mydb"

    def test_filter_kwargs_extra_fields(self):
        kwargs = {
            "type": "snowflake",
            "host": "account.snowflake.com",
            "custom_field": "value",
        }
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.extra is not None
        assert cfg.extra["custom_field"] == "value"

    def test_filter_kwargs_with_name(self):
        kwargs = {
            "type": "sqlite",
            "uri": "sqlite:///test.db",
            "name": "my_db",
        }
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.logic_name == "my_db"

    def test_filter_kwargs_env_resolution(self):
        os.environ["TEST_DB_HOST"] = "resolved_host"
        kwargs = {
            "type": "postgresql",
            "host": "${TEST_DB_HOST}",
        }
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.host == "resolved_host"
        del os.environ["TEST_DB_HOST"]


# =====================================================================
# BenchmarkConfig
# =====================================================================


class TestBenchmarkConfig:
    def test_validate_missing_question_key(self):
        cfg = BenchmarkConfig(question_file="test.json", question_id_key="id")
        with pytest.raises(DatusException, match="question_key"):
            cfg.validate()

    def test_validate_missing_question_file(self):
        cfg = BenchmarkConfig(question_key="question", question_id_key="id")
        with pytest.raises(DatusException, match="question_file"):
            cfg.validate()

    def test_validate_missing_question_id_key(self):
        cfg = BenchmarkConfig(question_file="test.json", question_key="question")
        with pytest.raises(DatusException, match="question_id_key"):
            cfg.validate()

    def test_validate_success(self):
        cfg = BenchmarkConfig(question_file="test.json", question_key="question", question_id_key="id")
        cfg.validate()  # Should not raise

    def test_filter_kwargs(self):
        kwargs = {
            "question_file": "test.json",
            "question_key": "q",
            "unknown_field": "ignored",
        }
        cfg = BenchmarkConfig.filter_kwargs(BenchmarkConfig, kwargs)
        assert cfg.question_file == "test.json"
        assert cfg.question_key == "q"


# =====================================================================
# DocumentConfig
# =====================================================================


class TestDocumentConfig:
    def test_from_dict(self):
        data = {
            "type": "github",
            "source": "owner/repo",
            "version": "1.0",
            "paths": ["docs"],
        }
        cfg = DocumentConfig.from_dict(data)
        assert cfg.type == "github"
        assert cfg.source == "owner/repo"

    def test_from_dict_ignores_unknown(self):
        data = {
            "type": "local",
            "unknown_field": "ignored",
        }
        cfg = DocumentConfig.from_dict(data)
        assert cfg.type == "local"

    def test_merge_cli_args(self):
        cfg = DocumentConfig(type="local", source="/path")

        class Args:
            source_type = "github"
            source = "owner/repo"
            version = None
            github_ref = None
            github_token = None
            paths = None
            chunk_size = None
            max_depth = None
            include_patterns = None
            exclude_patterns = None

        merged = cfg.merge_cli_args(Args())
        assert merged.type == "github"
        assert merged.source == "owner/repo"

    def test_merge_cli_args_no_override(self):
        cfg = DocumentConfig(type="local", source="/path")

        class Args:
            pass

        merged = cfg.merge_cli_args(Args())
        assert merged.type == "local"
        assert merged.source == "/path"


# =====================================================================
# DashboardConfig
# =====================================================================


class TestDashboardConfig:
    def test_basic_creation(self):
        cfg = DashboardConfig(platform="superset", username="admin", password="pass")
        assert cfg.platform == "superset"
        assert cfg.username == "admin"


# =====================================================================
# rag_storage_path
# =====================================================================


class TestRagStoragePath:
    def test_basic(self):
        result = rag_storage_path("my_ns", "data")
        assert result == os.path.join("data", "datus_db_my_ns")

    def test_default_base(self):
        result = rag_storage_path("ns")
        assert "datus_db_ns" in result


# =====================================================================
# AgentConfig additional tests
# =====================================================================


class TestAgentConfigExtended:
    def _make_config(self, tmp_path, **extra_kwargs):
        config_kwargs = {
            "home": str(tmp_path),
            "target": "test_model",
            "models": {
                "test_model": {
                    "type": "openai",
                    "api_key": "test-key",
                    "model": "gpt-4o",
                    "base_url": "https://api.openai.com/v1",
                },
            },
            "namespace": {
                "test_ns": {
                    "type": "sqlite",
                    "dbs": [
                        {
                            "uri": f"sqlite:///{tmp_path}/test.db",
                            "name": "test_db",
                        }
                    ],
                },
            },
        }
        config_kwargs.update(extra_kwargs)
        nodes: dict[str, NodeConfig] = {}
        return AgentConfig(nodes=nodes, **config_kwargs)

    def test_getitem(self, tmp_path):
        cfg = self._make_config(tmp_path)
        model = cfg["test_model"]
        assert model.model == "gpt-4o"

    def test_getitem_not_found(self, tmp_path):
        cfg = self._make_config(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            cfg["nonexistent"]

    def test_active_model(self, tmp_path):
        cfg = self._make_config(tmp_path)
        model = cfg.active_model()
        assert model.model == "gpt-4o"

    def test_model_config_by_name(self, tmp_path):
        cfg = self._make_config(tmp_path)
        model = cfg.model_config("test_model")
        assert model.model == "gpt-4o"

    def test_model_config_default(self, tmp_path):
        cfg = self._make_config(tmp_path)
        model = cfg.model_config()
        assert model.model == "gpt-4o"

    def test_model_config_not_found(self, tmp_path):
        cfg = self._make_config(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            cfg.model_config("nonexistent")

    def test_reflection_nodes(self, tmp_path):
        cfg = self._make_config(tmp_path)
        nodes = cfg.reflection_nodes("schema_linking")
        assert len(nodes) > 0

    def test_reflection_nodes_unsupported(self, tmp_path):
        cfg = self._make_config(tmp_path)
        with pytest.raises(DatusException):
            cfg.reflection_nodes("nonexistent_strategy")

    def test_max_export_lines_default(self, tmp_path):
        cfg = self._make_config(tmp_path)
        assert cfg.max_export_lines == 1000

    def test_max_export_lines_custom(self, tmp_path):
        cfg = self._make_config(tmp_path, export={"max_lines": 500})
        assert cfg.max_export_lines == 500

    def test_current_namespace_setter(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.current_namespace = "test_ns"
        assert cfg.current_namespace == "test_ns"

    def test_current_namespace_empty_raises(self, tmp_path):
        cfg = self._make_config(tmp_path)
        with pytest.raises(DatusException, match="namespace"):
            cfg.current_namespace = ""

    def test_current_namespace_same_value_noop(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.current_namespace = "test_ns"
        cfg.current_namespace = "test_ns"  # Same value, should be no-op
        assert cfg.current_namespace == "test_ns"

    def test_current_db_config_single(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.current_namespace = "test_ns"
        db_cfg = cfg.current_db_config()
        assert db_cfg is not None

    def test_current_db_config_by_name(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.current_namespace = "test_ns"
        db_cfg = cfg.current_db_config("test_db")
        assert db_cfg.logic_name == "test_db"

    def test_current_db_config_not_found(self, tmp_path):
        # Need 2+ dbs to trigger the not-found path (single db always returns first)
        cfg = self._make_config(
            tmp_path,
            namespace={
                "multi_ns": {
                    "type": "sqlite",
                    "dbs": [
                        {"uri": f"sqlite:///{tmp_path}/a.db", "name": "db_a"},
                        {"uri": f"sqlite:///{tmp_path}/b.db", "name": "db_b"},
                    ],
                },
            },
        )
        cfg.current_namespace = "multi_ns"
        with pytest.raises(DatusException, match="not found"):
            cfg.current_db_config("nonexistent_db")

    def test_benchmark_path_empty(self, tmp_path):
        cfg = self._make_config(tmp_path)
        with pytest.raises(DatusException, match="required"):
            cfg.benchmark_path("")

    def test_benchmark_config_unknown(self, tmp_path):
        cfg = self._make_config(tmp_path)
        with pytest.raises(DatusException):
            cfg.benchmark_config("nonexistent_benchmark")

    def test_init_dashboard(self, tmp_path):
        cfg = self._make_config(
            tmp_path,
            dashboard={
                "superset": {
                    "username": "admin",
                    "password": "pass123",
                    "api_key": "api-key",
                    "extra": {"custom": "value"},
                }
            },
        )
        assert "superset" in cfg.dashboard_config
        assert cfg.dashboard_config["superset"].username == "admin"

    def test_init_dashboard_non_dict(self, tmp_path):
        # Should handle gracefully
        cfg = self._make_config(tmp_path, dashboard="not_a_dict")
        assert len(cfg.dashboard_config) == 0

    def test_override_save_llm_trace(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.override_by_args(save_llm_trace=True, namespace="test_ns")
        for model in cfg.models.values():
            assert model.save_llm_trace is True

    def test_agentic_nodes_validation(self, tmp_path):
        with pytest.raises(DatusException, match="Invalid agentic_node name"):
            self._make_config(
                tmp_path,
                agentic_nodes={"invalid name!": {"system_prompt": "test"}},
            )

    def test_document_config_invalid_name(self, tmp_path):
        with pytest.raises(DatusException, match="Invalid document platform name"):
            self._make_config(
                tmp_path,
                document={"invalid name!": {"type": "local"}},
            )

    def test_sub_agent_config(self, tmp_path):
        cfg = self._make_config(
            tmp_path,
            agentic_nodes={"my_agent": {"system_prompt": "test", "tools": "db_tools.*"}},
        )
        sub_cfg = cfg.sub_agent_config("my_agent")
        assert sub_cfg["system_prompt"] == "test"

    def test_sub_agent_config_missing(self, tmp_path):
        cfg = self._make_config(tmp_path)
        sub_cfg = cfg.sub_agent_config("nonexistent")
        assert sub_cfg == {}

    def test_document_storage_path(self, tmp_path):
        cfg = self._make_config(tmp_path)
        path = cfg.document_storage_path("github")
        assert "document" in path
        assert "github" in path

    def test_document_storage_base_path(self, tmp_path):
        cfg = self._make_config(tmp_path)
        path = cfg.document_storage_base_path()
        assert "document" in path

    def test_output_dir(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.current_namespace = "test_ns"
        assert "test_ns" in cfg.output_dir

    def test_current_db_name_type(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.current_namespace = "test_ns"
        name, db_type = cfg.current_db_name_type("test_db")
        assert name == "test_db"
        assert db_type == "sqlite"
