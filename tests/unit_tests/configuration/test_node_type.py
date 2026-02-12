# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/configuration/node_type.py"""

import pytest

from datus.configuration.node_type import NodeType


# =====================================================================
# NodeType constants
# =====================================================================


class TestNodeTypeConstants:
    def test_control_types(self):
        assert NodeType.TYPE_BEGIN in NodeType.CONTROL_TYPES
        assert NodeType.TYPE_HITL in NodeType.CONTROL_TYPES
        assert NodeType.TYPE_REFLECT in NodeType.CONTROL_TYPES
        assert NodeType.TYPE_PARALLEL in NodeType.CONTROL_TYPES

    def test_action_types(self):
        assert NodeType.TYPE_SCHEMA_LINKING in NodeType.ACTION_TYPES
        assert NodeType.TYPE_GENERATE_SQL in NodeType.ACTION_TYPES
        assert NodeType.TYPE_EXECUTE_SQL in NodeType.ACTION_TYPES
        assert NodeType.TYPE_OUTPUT in NodeType.ACTION_TYPES
        assert NodeType.TYPE_CHAT in NodeType.ACTION_TYPES
        assert NodeType.TYPE_GENSQL in NodeType.ACTION_TYPES


# =====================================================================
# get_description
# =====================================================================


class TestGetDescription:
    def test_known_type(self):
        desc = NodeType.get_description(NodeType.TYPE_SCHEMA_LINKING)
        assert "schema" in desc.lower() or "Understand" in desc

    def test_unknown_type(self):
        desc = NodeType.get_description("nonexistent_type")
        assert "Unknown" in desc


# =====================================================================
# type_input
# =====================================================================


class TestTypeInput:
    def test_schema_linking(self):
        result = NodeType.type_input(NodeType.TYPE_SCHEMA_LINKING, {}, ignore_require_check=True)
        assert result is not None

    def test_generate_sql(self):
        result = NodeType.type_input(NodeType.TYPE_GENERATE_SQL, {}, ignore_require_check=True)
        assert result is not None

    def test_execute_sql(self):
        result = NodeType.type_input(NodeType.TYPE_EXECUTE_SQL, {}, ignore_require_check=True)
        assert result is not None

    def test_reflect(self):
        result = NodeType.type_input(NodeType.TYPE_REFLECT, {}, ignore_require_check=True)
        assert result is not None

    def test_output(self):
        result = NodeType.type_input(NodeType.TYPE_OUTPUT, {}, ignore_require_check=True)
        assert result is not None

    def test_reasoning(self):
        result = NodeType.type_input(NodeType.TYPE_REASONING, {}, ignore_require_check=True)
        assert result is not None

    def test_doc_search(self):
        result = NodeType.type_input(NodeType.TYPE_DOC_SEARCH, {}, ignore_require_check=True)
        assert result is not None

    def test_fix(self):
        result = NodeType.type_input(NodeType.TYPE_FIX, {}, ignore_require_check=True)
        assert result is not None

    def test_search_metrics(self):
        result = NodeType.type_input(NodeType.TYPE_SEARCH_METRICS, {}, ignore_require_check=True)
        assert result is not None

    def test_parallel(self):
        result = NodeType.type_input(NodeType.TYPE_PARALLEL, {}, ignore_require_check=True)
        assert result is not None

    def test_selection(self):
        result = NodeType.type_input(NodeType.TYPE_SELECTION, {}, ignore_require_check=True)
        assert result is not None

    def test_subworkflow(self):
        result = NodeType.type_input(NodeType.TYPE_SUBWORKFLOW, {}, ignore_require_check=True)
        assert result is not None

    def test_compare(self):
        result = NodeType.type_input(NodeType.TYPE_COMPARE, {}, ignore_require_check=True)
        assert result is not None

    def test_date_parser(self):
        result = NodeType.type_input(NodeType.TYPE_DATE_PARSER, {}, ignore_require_check=True)
        assert result is not None

    def test_chat(self):
        result = NodeType.type_input(NodeType.TYPE_CHAT, {}, ignore_require_check=True)
        assert result is not None

    def test_gensql(self):
        result = NodeType.type_input(NodeType.TYPE_GENSQL, {}, ignore_require_check=True)
        assert result is not None

    def test_semantic(self):
        result = NodeType.type_input(NodeType.TYPE_SEMANTIC, {}, ignore_require_check=True)
        assert result is not None

    def test_sql_summary(self):
        result = NodeType.type_input(NodeType.TYPE_SQL_SUMMARY, {}, ignore_require_check=True)
        assert result is not None

    def test_gen_report(self):
        result = NodeType.type_input(NodeType.TYPE_GEN_REPORT, {}, ignore_require_check=True)
        assert result is not None

    def test_ext_knowledge(self):
        result = NodeType.type_input(NodeType.TYPE_EXT_KNOWLEDGE, {}, ignore_require_check=True)
        assert result is not None

    def test_unknown_type_raises(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            NodeType.type_input("nonexistent_type", {})

    def test_with_input_data(self):
        result = NodeType.type_input(NodeType.TYPE_CHAT, {}, ignore_require_check=True)
        assert result is not None

    def test_without_ignore_require_check_raises_on_missing_fields(self):
        # Without ignore_require_check, required fields must be provided
        with pytest.raises(Exception):  # Pydantic ValidationError
            NodeType.type_input(NodeType.TYPE_OUTPUT, {}, ignore_require_check=False)

    def test_without_ignore_require_check_with_data(self):
        # If all required data is provided, it should work
        data = {
            "task_id": "test",
            "task": "test task",
            "database_name": "test_db",
            "output_dir": "/tmp",
            "gen_sql": "SELECT 1",
        }
        result = NodeType.type_input(NodeType.TYPE_OUTPUT, data, ignore_require_check=False)
        assert result is not None
        assert result.task_id == "test"


# =====================================================================
# make_optional_model
# =====================================================================


class TestMakeOptionalModel:
    def test_relaxes_required_fields(self):
        from datus.schemas.node_models import ExecuteSQLInput

        relaxed = NodeType.make_optional_model(ExecuteSQLInput)
        # Should be able to create with no arguments
        instance = relaxed()
        assert instance is not None

    def test_name_suffix(self):
        from datus.schemas.node_models import ExecuteSQLInput

        relaxed = NodeType.make_optional_model(ExecuteSQLInput, "_Custom")
        assert "Custom" in relaxed.__name__
