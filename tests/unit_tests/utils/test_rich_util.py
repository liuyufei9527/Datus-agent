# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/rich_util.py."""

from unittest.mock import MagicMock

from rich.tree import Tree

from datus.utils.rich_util import dict_to_tree


class TestDictToTree:
    def test_basic_dict(self):
        data = {"key1": "value1", "key2": "value2"}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_nested_dict(self):
        data = {"outer": {"inner": "value"}}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_list_values(self):
        data = {"items": ["a", "b", "c"]}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_list_of_dicts(self):
        data = {"items": [{"name": "item1"}, {"name": "item2"}]}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_sql_query_key(self):
        data = {"sql_query": "SELECT * FROM users"}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_long_value_truncated(self):
        # Default max_length is screen_width * 3 = 100 * 3 = 300
        data = {"long_key": "x" * 500}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_with_existing_tree(self):
        existing = Tree("root")
        data = {"child": "value"}
        result = dict_to_tree(data, tree=existing)
        assert result is existing

    def test_with_console(self):
        console = MagicMock()
        console.size.width = 80
        data = {"key": "val"}
        tree = dict_to_tree(data, console=console)
        assert isinstance(tree, Tree)

    def test_empty_dict_in_nested(self):
        data = {"empty": {}, "nonempty": {"k": "v"}}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_empty_list_in_nested(self):
        data = {"empty_list": [], "items": ["a"]}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_none_value(self):
        data = {"key": None}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_long_list_item(self):
        data = {"items": ["x" * 500]}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)

    def test_sql_query_non_string(self):
        data = {"sql_query": 12345}
        tree = dict_to_tree(data)
        assert isinstance(tree, Tree)
