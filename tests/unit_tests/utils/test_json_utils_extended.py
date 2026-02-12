# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for datus/utils/json_utils.py to improve coverage."""

import json
import tempfile
from pathlib import Path

import pytest

from datus.utils.json_utils import (
    _dump_json,
    _normalize_for_json,
    extract_code_block_content,
    extract_json_array,
    extract_json_object,
    find_matching_bracket,
    json2csv,
    json_list2markdown_table,
    llm_result2json,
    llm_result2sql,
    load_jsonl,
    load_jsonl_dict,
    load_jsonl_iterator,
    strip_json_str,
    to_pretty_str,
    to_str,
)


# ===========================================================================
# find_matching_bracket
# ===========================================================================


class TestFindMatchingBracket:
    def test_simple_brackets(self):
        assert find_matching_bracket("[abc]", 0) == 4

    def test_nested_brackets(self):
        assert find_matching_bracket("[[a][b]]", 0) == 7

    def test_curly_braces(self):
        assert find_matching_bracket("{a: {b: c}}", 0, "{", "}") == 10

    def test_no_matching_bracket(self):
        assert find_matching_bracket("[abc", 0) == -1

    def test_empty_stack_close_first(self):
        assert find_matching_bracket("]abc", 0) == -1


# ===========================================================================
# extract_json_object
# ===========================================================================


class TestExtractJsonObject:
    def test_extract_simple_object(self):
        result = extract_json_object('some text {"key": "value"} more text')
        assert result == '{"key": "value"}'

    def test_no_object_found(self):
        assert extract_json_object("no json here") == ""

    def test_unmatched_brace(self):
        assert extract_json_object("text {unmatched") == ""

    def test_multiple_objects_returns_first(self):
        text = '{"first": 1} text {"second": 2}'
        result = extract_json_object(text)
        assert result == '{"first": 1}'

    def test_nested_objects(self):
        text = '{"outer": {"inner": 1}}'
        result = extract_json_object(text)
        assert result == '{"outer": {"inner": 1}}'


# ===========================================================================
# extract_json_array
# ===========================================================================


class TestExtractJsonArray:
    def test_extract_simple_array(self):
        result = extract_json_array("text [1, 2, 3] more")
        assert result == "[1, 2, 3]"

    def test_no_array(self):
        assert extract_json_array("no array here") == ""

    def test_unmatched_bracket(self):
        assert extract_json_array("text [unmatched") == ""

    def test_nested_arrays(self):
        result = extract_json_array("[[1, 2], [3, 4]]")
        assert result == "[[1, 2], [3, 4]]"

    def test_multiple_arrays_returns_first(self):
        text = "[1, 2] separate [3, 4]"
        result = extract_json_array(text)
        assert result == "[1, 2]"


# ===========================================================================
# extract_code_block_content
# ===========================================================================


class TestExtractCodeBlockContent:
    def test_no_code_block(self):
        assert extract_code_block_content("no code block") == ""

    def test_simple_code_block(self):
        text = "```\nsome code\n```"
        assert extract_code_block_content(text) == "some code"

    def test_json_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_code_block_content(text) == '{"key": "value"}'

    def test_no_closing_block(self):
        text = "```\nno closing"
        assert extract_code_block_content(text) == ""


# ===========================================================================
# json2csv
# ===========================================================================


class TestJson2Csv:
    def test_empty_input(self):
        assert json2csv(None) == ""
        assert json2csv("") == ""
        assert json2csv([]) == ""

    def test_json_string_input(self):
        result = json2csv('[{"a": 1, "b": 2}]')
        assert "a" in result
        assert "b" in result

    def test_dict_input(self):
        result = json2csv({"a": 1, "b": 2})
        assert "a" in result

    def test_list_input(self):
        result = json2csv([{"a": 1}, {"a": 2}])
        assert "a" in result

    def test_non_json_string(self):
        assert json2csv("plain text") == "plain text"

    def test_json_object_string(self):
        result = json2csv('{"a": 1, "b": 2}')
        assert "a" in result

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            json2csv(12345)

    def test_with_columns_filter(self):
        result = json2csv([{"a": 1, "b": 2, "c": 3}], columns=["a", "c"])
        assert "a" in result
        assert "c" in result


# ===========================================================================
# llm_result2json
# ===========================================================================


class TestLlmResult2Json:
    def test_valid_json_object_with_sql(self):
        result = llm_result2json('{"sql": "SELECT 1", "output": ""}')
        assert result is not None
        assert result["sql"] == "SELECT 1"

    def test_valid_json_object_with_output(self):
        result = llm_result2json('{"sql": "", "output": "some output"}')
        assert result is not None
        assert result["output"] == "some output"

    def test_empty_json_object(self):
        result = llm_result2json('{"sql": "", "output": ""}')
        assert result is None

    def test_empty_string(self):
        assert llm_result2json("") is None

    def test_none_like(self):
        assert llm_result2json("   ") is None

    def test_non_dict_non_list_result(self):
        # json_repair.loads("42") returns 42, not a dict/list
        assert llm_result2json("42") is None

    def test_code_block_json(self):
        text = '```json\n{"sql": "SELECT * FROM t", "output": ""}\n```'
        result = llm_result2json(text)
        assert result is not None
        assert result["sql"] == "SELECT * FROM t"

    def test_list_result(self):
        result = llm_result2json('[{"a": 1}]', expected_type=list)
        assert result is not None
        assert isinstance(result, list)

    def test_has_content_with_numeric(self):
        result = llm_result2json('{"sql": 0, "output": ""}')
        assert result is not None

    def test_has_content_with_none_values(self):
        result = llm_result2json('{"sql": null, "output": null}')
        assert result is None


# ===========================================================================
# llm_result2sql
# ===========================================================================


class TestLlmResult2Sql:
    def test_sql_code_block(self):
        text = "```sql\nSELECT * FROM users\n```"
        assert llm_result2sql(text) == "SELECT * FROM users"

    def test_sql_code_block_uppercase(self):
        text = "```SQL\nSELECT 1\n```"
        assert llm_result2sql(text) == "SELECT 1"

    def test_generic_code_block_with_sql(self):
        text = "```\nSELECT * FROM users WHERE id = 1\n```"
        assert llm_result2sql(text) == "SELECT * FROM users WHERE id = 1"

    def test_generic_code_block_without_sql(self):
        text = "```\nprint('hello world')\n```"
        assert llm_result2sql(text) is None

    def test_none_input(self):
        assert llm_result2sql(None) is None

    def test_empty_input(self):
        assert llm_result2sql("") is None

    def test_non_string_input(self):
        assert llm_result2sql(123) is None

    def test_no_code_block(self):
        assert llm_result2sql("just some text") is None

    def test_empty_sql_block(self):
        text = "```sql\n\n```"
        assert llm_result2sql(text) is None


# ===========================================================================
# json_list2markdown_table
# ===========================================================================


class TestJsonList2MarkdownTable:
    def test_empty_list(self):
        assert json_list2markdown_table([]) == ""

    def test_simple_list(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = json_list2markdown_table(data)
        assert result is not None
        assert "Alice" in result
        assert "Bob" in result


# ===========================================================================
# strip_json_str
# ===========================================================================


class TestStripJsonStr:
    def test_none_input(self):
        assert strip_json_str(None) == ""

    def test_empty_input(self):
        assert strip_json_str("") == ""
        assert strip_json_str("   ") == ""

    def test_code_block_json(self):
        result = strip_json_str('```json\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_code_block_generic(self):
        result = strip_json_str('```\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_text_before_json(self):
        result = strip_json_str('Here is the JSON: {"key": "value"}')
        assert result == '{"key": "value"}'

    def test_text_before_array(self):
        result = strip_json_str("Here is the JSON: [1, 2, 3]")
        assert result == "[1, 2, 3]"

    def test_truncated_json_object(self):
        result = strip_json_str('{"key": "value", "nested": {"inner": 1')
        assert result.endswith("}")

    def test_truncated_json_array(self):
        result = strip_json_str("[1, 2, [3, 4")
        assert result.endswith("]")

    def test_unclosed_string_after_colon(self):
        result = strip_json_str('{"key": "incomplete value')
        assert result.count('"') % 2 == 0

    def test_unclosed_string_after_comma(self):
        result = strip_json_str('{"a": 1, "b": "incomplete')
        assert result.count('"') % 2 == 0


# ===========================================================================
# load_jsonl / load_jsonl_iterator / load_jsonl_dict
# ===========================================================================


class TestJsonlFunctions:
    def test_load_jsonl(self, tmp_path):
        file = tmp_path / "test.jsonl"
        file.write_text('{"a": 1}\n{"a": 2}\n')
        result = load_jsonl(str(file))
        assert len(result) == 2
        assert result[0]["a"] == 1
        assert result[1]["a"] == 2

    def test_load_jsonl_iterator(self, tmp_path):
        file = tmp_path / "test.jsonl"
        file.write_text('{"x": 10}\n{"x": 20}\n')
        results = list(load_jsonl_iterator(str(file)))
        assert len(results) == 2
        assert results[0]["x"] == 10

    def test_load_jsonl_dict(self, tmp_path):
        file = tmp_path / "test.jsonl"
        file.write_text('{"instance_id": "a", "val": 1}\n{"instance_id": "b", "val": 2}\n')
        result = load_jsonl_dict(str(file))
        assert "a" in result
        assert "b" in result
        assert result["a"]["val"] == 1

    def test_load_jsonl_dict_custom_key(self, tmp_path):
        file = tmp_path / "test.jsonl"
        file.write_text('{"id": "x", "data": 42}\n')
        result = load_jsonl_dict(str(file), key_field="id")
        assert "x" in result


# ===========================================================================
# _normalize_for_json
# ===========================================================================


class TestNormalizeForJson:
    def test_primitives(self):
        assert _normalize_for_json(42) == 42
        assert _normalize_for_json("hello") == "hello"
        assert _normalize_for_json(True) is True
        assert _normalize_for_json(None) is None
        assert _normalize_for_json(3.14) == 3.14

    def test_bytes_utf8(self):
        assert _normalize_for_json(b"hello") == "hello"

    def test_bytes_non_utf8(self):
        result = _normalize_for_json(b"\xff\xfe")
        assert isinstance(result, str)

    def test_enum(self):
        from enum import Enum

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        assert _normalize_for_json(Color.RED) == "red"

    def test_set_and_frozenset(self):
        result = _normalize_for_json({1, 2, 3})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

        result = _normalize_for_json(frozenset([4, 5]))
        assert isinstance(result, list)

    def test_tuple(self):
        assert _normalize_for_json((1, 2, 3)) == [1, 2, 3]

    def test_object_with_to_dict(self):
        class MyObj:
            def to_dict(self):
                return {"x": 1}

        result = _normalize_for_json(MyObj())
        assert result == {"x": 1}

    def test_object_with_dict(self):
        class SimpleObj:
            def __init__(self):
                self.a = 1
                self.b = "hello"

        result = _normalize_for_json(SimpleObj())
        assert result == {"a": 1, "b": "hello"}

    def test_fallback_to_str(self):
        # An object that has no __dict__, no to_dict, etc.
        result = _normalize_for_json(object())
        assert isinstance(result, str)

    def test_iterable_non_standard(self):
        # A custom iterable
        class MyIter:
            def __iter__(self):
                return iter([1, 2, 3])

        result = _normalize_for_json(MyIter())
        assert result == [1, 2, 3]


# ===========================================================================
# _dump_json
# ===========================================================================


class TestDumpJson:
    def test_string_input_valid_json(self):
        result = _dump_json('{"a": 1}')
        assert json.loads(result) == {"a": 1}

    def test_string_input_non_json(self):
        assert _dump_json("plain text") == "plain text"

    def test_empty_string(self):
        assert _dump_json("") == ""

    def test_bytes_input(self):
        result = _dump_json(b'{"b": 2}')
        assert json.loads(result) == {"b": 2}

    def test_bytearray_input(self):
        result = _dump_json(bytearray(b'{"c": 3}'))
        assert json.loads(result) == {"c": 3}

    def test_dict_input(self):
        result = _dump_json({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_indent(self):
        result = _dump_json({"x": 1}, indent=2)
        assert "\n" in result


# ===========================================================================
# to_pretty_str / to_str edge cases
# ===========================================================================


class TestToStrEdgeCases:
    def test_to_str_none(self):
        assert to_str(None) is None

    def test_to_pretty_str_none(self):
        assert to_pretty_str(None) is None

    def test_to_str_dict(self):
        result = to_str({"a": 1})
        assert json.loads(result) == {"a": 1}

    def test_to_pretty_str_dict(self):
        result = to_pretty_str({"a": 1})
        assert "\n" in result
        assert json.loads(result) == {"a": 1}

    def test_to_str_list(self):
        result = to_str([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_to_str_non_json_string(self):
        assert to_str("not json") == "not json"

    def test_to_str_empty_bytes(self):
        assert to_str(b"") == ""
