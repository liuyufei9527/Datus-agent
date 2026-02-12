# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for datus/utils/compress_utils.py to improve coverage."""

import pandas as pd
import pyarrow as pa
import pytest

from datus.utils.compress_utils import (
    DataCompressor,
    _compress_pyarrow_table,
    _format_as_csv,
    _format_as_table,
    _get_data_dimensions,
    _get_row_count_fast,
    _identify_id_time_columns,
    _is_empty_data,
    _to_dataframe_efficient,
)


class TestIdentifyIdTimeColumns:
    def test_basic(self):
        columns = ["user_id", "name", "created_time", "amount", "order_key"]
        id_cols, time_cols = _identify_id_time_columns(columns)
        assert "user_id" in id_cols
        assert "order_key" in id_cols
        assert "created_time" in time_cols

    def test_no_matches(self):
        columns = ["name", "amount", "score"]
        id_cols, time_cols = _identify_id_time_columns(columns)
        assert id_cols == []
        assert time_cols == []

    def test_date_and_timestamp(self):
        columns = ["date", "updated_at", "timestamp"]
        id_cols, time_cols = _identify_id_time_columns(columns)
        assert "date" in time_cols
        assert "updated_at" in time_cols
        assert "timestamp" in time_cols

    def test_empty_columns(self):
        id_cols, time_cols = _identify_id_time_columns([])
        assert id_cols == []
        assert time_cols == []


class TestCompressPyarrowTable:
    def test_small_table_no_compression(self):
        table = pa.table({"a": list(range(10)), "b": list(range(10))})
        result, head_idx, tail_idx = _compress_pyarrow_table(table)
        assert len(result) == 10
        assert head_idx == list(range(10))
        assert tail_idx == []

    def test_large_table_compression_string_columns(self):
        # Use string columns since _compress_pyarrow_table creates "..." ellipsis rows
        # which are string type and must match column types
        table = pa.table({"a": [str(i) for i in range(100)], "b": [str(i) for i in range(100)]})
        result, head_idx, tail_idx = _compress_pyarrow_table(table)
        assert len(result) == 21  # 10 head + 1 ellipsis + 10 tail
        assert head_idx == list(range(10))
        assert tail_idx == list(range(90, 100))

    def test_exactly_20_rows(self):
        table = pa.table({"a": list(range(20))})
        result, head_idx, tail_idx = _compress_pyarrow_table(table)
        assert len(result) == 20
        assert tail_idx == []


class TestGetRowCountFast:
    def test_list(self):
        assert _get_row_count_fast([{}, {}, {}]) == 3

    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert _get_row_count_fast(df) == 3

    def test_pyarrow_table(self):
        table = pa.table({"a": [1, 2, 3, 4]})
        assert _get_row_count_fast(table) == 4

    def test_unsupported_type(self):
        with pytest.raises(ValueError):
            _get_row_count_fast("not a table")


class TestIsEmptyData:
    def test_none(self):
        assert _is_empty_data(None) is True

    def test_empty_list(self):
        assert _is_empty_data([]) is True

    def test_non_empty_list(self):
        assert _is_empty_data([{"a": 1}]) is False

    def test_empty_dataframe(self):
        assert _is_empty_data(pd.DataFrame()) is True

    def test_non_empty_dataframe(self):
        assert _is_empty_data(pd.DataFrame({"a": [1]})) is False

    def test_empty_pyarrow(self):
        assert _is_empty_data(pa.table({"a": pa.array([], type=pa.int64())})) is True

    def test_non_empty_pyarrow(self):
        assert _is_empty_data(pa.table({"a": [1]})) is False


class TestGetDataDimensions:
    def test_list_of_dicts(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        rows, cols = _get_data_dimensions(data)
        assert rows == 2
        assert cols == ["a", "b"]

    def test_dataframe(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        rows, cols = _get_data_dimensions(df)
        assert rows == 2
        assert cols == ["x", "y"]

    def test_pyarrow_table(self):
        table = pa.table({"p": [1], "q": [2]})
        rows, cols = _get_data_dimensions(table)
        assert rows == 1
        assert cols == ["p", "q"]

    def test_empty_list(self):
        rows, cols = _get_data_dimensions([])
        assert rows == 0
        assert cols == []


class TestToDataframeEfficient:
    def test_from_dataframe(self):
        df = pd.DataFrame({"a": [1]})
        result = _to_dataframe_efficient(df)
        assert result is df

    def test_from_list(self):
        result = _to_dataframe_efficient([{"a": 1, "b": 2}])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_from_pyarrow(self):
        table = pa.table({"a": [1, 2]})
        result = _to_dataframe_efficient(table)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_unsupported_type(self):
        with pytest.raises(ValueError):
            _to_dataframe_efficient("string data")


class TestFormatAsCsv:
    def test_simple(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _format_as_csv(df)
        assert "a" in result
        assert "b" in result

    def test_with_compressed_indices(self):
        data = [{"a": i, "b": i * 10} for i in range(21)]
        df = pd.DataFrame(data)
        head_indices = list(range(10))
        tail_indices = list(range(11, 21))
        result = _format_as_csv(df, compressed_indices=(head_indices, tail_indices))
        assert "..." in result


class TestFormatAsTable:
    def test_simple(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _format_as_table(df)
        assert "a" in result
        assert "b" in result

    def test_with_compressed_indices(self):
        data = [{"a": i, "b": i * 10} for i in range(21)]
        df = pd.DataFrame(data)
        head_indices = list(range(10))
        tail_indices = list(range(11, 21))
        result = _format_as_table(df, compressed_indices=(head_indices, tail_indices))
        assert "..." in result


class TestDataCompressor:
    def test_empty_data(self):
        compressor = DataCompressor()
        result = compressor.compress([])
        assert result["original_rows"] == 0
        assert result["is_compressed"] is False

    def test_small_data_no_compression(self):
        compressor = DataCompressor(token_threshold=10000)
        data = [{"a": 1, "b": 2}]
        result = compressor.compress(data)
        assert result["is_compressed"] is False
        assert result["original_rows"] == 1

    def test_large_list_data_row_compression(self):
        compressor = DataCompressor(token_threshold=500)
        data = [{"user_id": i, "name": f"User_{i}", "score": i * 10} for i in range(100)]
        result = compressor.compress(data)
        assert result["is_compressed"] is True
        assert result["original_rows"] == 100
        assert "rows" in result["compression_type"]

    def test_dataframe_compression(self):
        compressor = DataCompressor(token_threshold=500)
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        result = compressor.compress(df)
        assert result["is_compressed"] is True
        assert result["original_rows"] == 100

    def test_pyarrow_compression(self):
        compressor = DataCompressor(token_threshold=500)
        # Use string columns to avoid pyarrow schema mismatch with "..." ellipsis rows
        table = pa.table({"a": [str(i) for i in range(100)], "b": [str(i) for i in range(100)]})
        result = compressor.compress(table)
        assert result["is_compressed"] is True
        assert result["original_rows"] == 100

    def test_table_output_format(self):
        compressor = DataCompressor(token_threshold=10000, output_format="table")
        data = [{"a": 1, "b": 2}]
        result = compressor.compress(data)
        assert result["is_compressed"] is False

    def test_quick_compress(self):
        data = [{"a": i} for i in range(5)]
        result = DataCompressor.quick_compress(data, token_threshold=10000)
        assert isinstance(result, str)
        assert "a" in result

    def test_count_tokens(self):
        compressor = DataCompressor()
        count = compressor.count_tokens("hello world this is a test")
        assert count > 0

    def test_count_tokens_fallback(self):
        compressor = DataCompressor(model_name="totally_fake_model_xyz")
        count = compressor.count_tokens("hello world")
        assert count > 0  # Should use fallback

    def test_compress_columns_with_many_columns(self):
        compressor = DataCompressor(token_threshold=100)
        data = [{f"col_{i}": f"value_{i}" for i in range(20)} for _ in range(5)]
        result = compressor.compress(data)
        # With very low threshold, columns should be compressed
        assert result["original_rows"] == 5

    def test_compress_none_data(self):
        compressor = DataCompressor()
        result = compressor.compress(None)
        assert result["original_rows"] == 0
        assert result["is_compressed"] is False

    def test_large_data_with_table_format(self):
        compressor = DataCompressor(token_threshold=500, output_format="table")
        data = [{"x": i, "y": i * 2} for i in range(50)]
        result = compressor.compress(data)
        assert result["is_compressed"] is True

    def test_small_data_high_tokens_column_compression(self):
        # Small data but many columns -> column compression only
        compressor = DataCompressor(token_threshold=50)
        data = [{f"column_{i}": f"a_very_long_value_{i}" for i in range(30)}]
        result = compressor.compress(data)
        # With 1 row and 30 columns, may trigger column compression
        assert result["original_rows"] == 1
