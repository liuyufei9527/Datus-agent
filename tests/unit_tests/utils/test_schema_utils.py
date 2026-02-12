# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/schema_utils.py."""

import json

import pytest

from datus.utils.schema_utils import table_metadata2markdown, table_metadata_struct


class TestTableMetadataStruct:
    def test_basic(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "users",
                "schema_text": "CREATE TABLE users (id INT, name VARCHAR(100));",
            }
        ]
        result = table_metadata_struct(metadata)
        assert result is not None
        data = json.loads(result)
        assert "public.users" in data
        assert "columns" in data["public.users"]

    def test_with_comment(self):
        metadata = [
            {
                "schema_name": "db",
                "table_name": "orders",
                "schema_text": "CREATE TABLE orders (id INT) COMMENT 'Order table';",
            }
        ]
        result = table_metadata_struct(metadata)
        data = json.loads(result)
        assert "db.orders" in data

    def test_multiple_tables(self):
        metadata = [
            {
                "schema_name": "s1",
                "table_name": "t1",
                "schema_text": "CREATE TABLE t1 (a INT);",
            },
            {
                "schema_name": "s2",
                "table_name": "t2",
                "schema_text": "CREATE TABLE t2 (b TEXT);",
            },
        ]
        result = table_metadata_struct(metadata)
        data = json.loads(result)
        assert "s1.t1" in data
        assert "s2.t2" in data


class TestTableMetadata2Markdown:
    def test_empty_input(self):
        assert table_metadata2markdown([]) == ""

    def test_basic_table(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "users",
                "schema_text": "CREATE TABLE users (id INT, name VARCHAR(100));",
            }
        ]
        result = table_metadata2markdown(metadata)
        assert "public.users" in result
        assert "id" in result
        assert "name" in result

    def test_table_with_schema_prefix(self):
        metadata = [
            {
                "schema_name": "mydb",
                "table_name": "mydb.orders",
                "schema_text": "CREATE TABLE orders (order_id INT, amount FLOAT);",
            }
        ]
        result = table_metadata2markdown(metadata)
        assert "mydb.orders" in result

    def test_multiple_tables(self):
        metadata = [
            {
                "schema_name": "s1",
                "table_name": "t1",
                "schema_text": "CREATE TABLE t1 (a INT);",
            },
            {
                "schema_name": "s2",
                "table_name": "t2",
                "schema_text": "CREATE TABLE t2 (b TEXT);",
            },
        ]
        result = table_metadata2markdown(metadata)
        assert "s1.t1" in result
        assert "s2.t2" in result
