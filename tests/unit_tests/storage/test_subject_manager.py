"""Tests for datus/storage/subject_manager.py — SubjectUpdater."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.subject_manager import SubjectUpdater


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.agentic_nodes = {}
    config.current_namespace = "default"
    return config


@pytest.fixture
def updater(mock_agent_config):
    with patch("datus.storage.subject_manager.get_storage_cache_instance") as mock_cache:
        cache_inst = MagicMock()
        cache_inst.metric_storage.return_value = MagicMock()
        cache_inst.reference_sql_storage.return_value = MagicMock()
        cache_inst.ext_knowledge_storage.return_value = MagicMock()
        mock_cache.return_value = cache_inst
        return SubjectUpdater(mock_agent_config)


# ── update_metrics_detail ──


class TestUpdateMetricsDetail:
    def test_empty_update_values(self, updater):
        updater.update_metrics_detail(["Finance"], "metric1", {})
        updater.metrics_storage.update_entry.assert_not_called()

    def test_update_main_and_sub_agents(self, mock_agent_config):
        mock_agent_config.agentic_nodes = {"sub1": {"system_prompt": "s1"}}

        with patch("datus.storage.subject_manager.get_storage_cache_instance") as mock_cache:
            cache_inst = MagicMock()
            main_metrics = MagicMock()
            sub_metrics = MagicMock()
            cache_inst.metric_storage.side_effect = lambda name=None: sub_metrics if name else main_metrics
            cache_inst.reference_sql_storage.return_value = MagicMock()
            cache_inst.ext_knowledge_storage.return_value = MagicMock()
            mock_cache.return_value = cache_inst

            with patch("datus.storage.subject_manager.SubAgentConfig") as MockSub:
                mock_sub = MagicMock()
                mock_sub.is_in_namespace.return_value = True
                mock_sub.system_prompt = "s1"
                MockSub.model_validate.return_value = mock_sub

                upd = SubjectUpdater(mock_agent_config)
                upd.update_metrics_detail(["Finance"], "m1", {"description": "new"})
                main_metrics.update_entry.assert_called_once()
                sub_metrics.update_entry.assert_called_once()

    def test_sub_agent_update_failure_logged(self, mock_agent_config):
        mock_agent_config.agentic_nodes = {"sub1": {"system_prompt": "s1"}}

        with patch("datus.storage.subject_manager.get_storage_cache_instance") as mock_cache:
            cache_inst = MagicMock()
            main_metrics = MagicMock()
            sub_metrics = MagicMock()
            sub_metrics.update_entry.side_effect = RuntimeError("fail")
            cache_inst.metric_storage.side_effect = lambda name=None: sub_metrics if name else main_metrics
            cache_inst.reference_sql_storage.return_value = MagicMock()
            cache_inst.ext_knowledge_storage.return_value = MagicMock()
            mock_cache.return_value = cache_inst

            with patch("datus.storage.subject_manager.SubAgentConfig") as MockSub:
                mock_sub = MagicMock()
                mock_sub.is_in_namespace.return_value = True
                mock_sub.system_prompt = "s1"
                MockSub.model_validate.return_value = mock_sub

                upd = SubjectUpdater(mock_agent_config)
                # Should not raise
                upd.update_metrics_detail(["Finance"], "m1", {"description": "new"})
                main_metrics.update_entry.assert_called_once()


# ── update_historical_sql ──


class TestUpdateHistoricalSql:
    def test_empty_update_values(self, updater):
        updater.update_historical_sql(["Analytics"], "sql1", {})
        updater.reference_sql_storage.update_entry.assert_not_called()

    def test_update_main_storage(self, updater):
        updater.update_historical_sql(["Analytics"], "sql1", {"summary": "updated"})
        updater.reference_sql_storage.update_entry.assert_called_once_with(["Analytics"], "sql1", {"summary": "updated"})

    def test_sub_agent_update_failure_logged(self, mock_agent_config):
        mock_agent_config.agentic_nodes = {"sub1": {"system_prompt": "s1"}}

        with patch("datus.storage.subject_manager.get_storage_cache_instance") as mock_cache:
            cache_inst = MagicMock()
            main_sql = MagicMock()
            sub_sql = MagicMock()
            sub_sql.update_entry.side_effect = RuntimeError("fail")
            cache_inst.reference_sql_storage.side_effect = lambda name=None: sub_sql if name else main_sql
            cache_inst.metric_storage.return_value = MagicMock()
            cache_inst.ext_knowledge_storage.return_value = MagicMock()
            mock_cache.return_value = cache_inst

            with patch("datus.storage.subject_manager.SubAgentConfig") as MockSub:
                mock_sub = MagicMock()
                mock_sub.is_in_namespace.return_value = True
                mock_sub.system_prompt = "s1"
                MockSub.model_validate.return_value = mock_sub

                upd = SubjectUpdater(mock_agent_config)
                upd.update_historical_sql(["A"], "s1", {"summary": "x"})
                main_sql.update_entry.assert_called_once()


# ── update_ext_knowledge ──


class TestUpdateExtKnowledge:
    def test_empty_update_values(self, updater):
        updater.update_ext_knowledge(["Topic"], "k1", {})
        updater.ext_knowledge_storage.update_entry.assert_not_called()

    def test_update_main_storage(self, updater):
        updater.update_ext_knowledge(["Topic"], "k1", {"content": "new"})
        updater.ext_knowledge_storage.update_entry.assert_called_once_with(["Topic"], "k1", {"content": "new"})
