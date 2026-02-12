"""Tests for datus/storage/schema_metadata/init_utils.py"""

from unittest.mock import MagicMock, PropertyMock

import pyarrow as pa
import pytest

from datus.storage.schema_metadata.init_utils import exists_table_value
from datus.utils.exceptions import DatusException


class TestExistsTableValue:
    def test_overwrite_mode_returns_empty(self):
        mock_storage = MagicMock()
        tables, values = exists_table_value(mock_storage, build_mode="overwrite")
        assert tables == {}
        assert values == set()

    def test_incremental_mode_with_data(self):
        mock_storage = MagicMock()

        # Mock schemas result
        schemas_data = pa.table(
            {"identifier": ["table1", "table2"], "definition": ["def1", "def2"]}
        )
        mock_storage.search_all_schemas.return_value = schemas_data

        # Mock values result
        values_data = pa.table({"identifier": ["table1"]})
        mock_storage.search_all_value.return_value = values_data

        tables, values = exists_table_value(mock_storage, build_mode="incremental")
        assert len(tables) == 2
        assert "table1" in tables
        assert "table2" in tables
        assert "table1" in values

    def test_incremental_mode_empty_store(self):
        mock_storage = MagicMock()
        empty = pa.table({"identifier": pa.array([], type=pa.string()), "definition": pa.array([], type=pa.string())})
        mock_storage.search_all_schemas.return_value = empty
        mock_storage.search_all_value.return_value = pa.table(
            {"identifier": pa.array([], type=pa.string())}
        )
        tables, values = exists_table_value(mock_storage, build_mode="incremental")
        assert tables == {}
        assert values == set()

    def test_incremental_mode_exception_raises(self):
        mock_storage = MagicMock()
        mock_storage.search_all_schemas.side_effect = Exception("DB error")
        with pytest.raises(DatusException):
            exists_table_value(mock_storage, build_mode="incremental")

    def test_with_filters(self):
        mock_storage = MagicMock()
        empty_schemas = pa.table(
            {"identifier": pa.array([], type=pa.string()), "definition": pa.array([], type=pa.string())}
        )
        empty_values = pa.table({"identifier": pa.array([], type=pa.string())})
        mock_storage.search_all_schemas.return_value = empty_schemas
        mock_storage.search_all_value.return_value = empty_values

        exists_table_value(
            mock_storage,
            database_name="mydb",
            catalog_name="mycat",
            schema_name="myschema",
            table_type="view",
            build_mode="incremental",
        )
        mock_storage.search_all_schemas.assert_called_once_with(
            catalog_name="mycat",
            database_name="mydb",
            schema_name="myschema",
            table_type="view",
        )
