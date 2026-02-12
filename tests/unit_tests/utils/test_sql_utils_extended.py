# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended tests for datus/utils/sql_utils.py to improve coverage."""

import pytest

from datus.utils.constants import DBType, SQLType
from datus.utils.sql_utils import (
    _fallback_sql_type,
    _first_statement,
    _identifier_name,
    _is_escaped,
    _match_dollar_tag,
    _metadata_pattern,
    _normalize_expression,
    _table_parts,
    extract_table_names,
    format_sql_to_pretty,
    metadata_identifier,
    normalize_sql,
    parse_context_switch,
    parse_metadata_from_ddl,
    parse_sql_type,
    parse_table_name_parts,
    parse_table_names_parts,
    strip_sql_comments,
)


# ===========================================================================
# parse_create_table
# ===========================================================================


class TestParseMetadataFromDdl:
    def test_basic(self):
        sql = "CREATE TABLE users (id INT, name VARCHAR(100));"
        result = parse_metadata_from_ddl(sql)
        assert result["table"]["name"] == "users"
        assert len(result["columns"]) == 2

    def test_with_schema(self):
        sql = "CREATE TABLE mydb.users (id INT, name VARCHAR(100));"
        result = parse_metadata_from_ddl(sql)
        assert result["table"]["name"] == "users"

    def test_with_comment(self):
        sql = "CREATE TABLE users (id INT COMMENT 'Primary key', name VARCHAR(100));"
        result = parse_metadata_from_ddl(sql)
        assert len(result["columns"]) == 2

    def test_invalid_sql(self):
        result = parse_metadata_from_ddl("not a create table statement")
        assert result["table"]["name"] == ""


# ===========================================================================
# extract_table_names
# ===========================================================================


class TestExtractTableNames:
    def test_simple_select(self):
        tables = extract_table_names("SELECT * FROM users", dialect=DBType.SNOWFLAKE)
        assert len(tables) > 0

    def test_join(self):
        tables = extract_table_names(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
            dialect=DBType.SNOWFLAKE,
        )
        assert len(tables) >= 2

    def test_cte_excluded(self):
        sql = "WITH cte AS (SELECT 1) SELECT * FROM cte"
        tables = extract_table_names(sql, dialect=DBType.SNOWFLAKE)
        # cte should be excluded
        assert all("cte" not in t.lower() for t in tables)

    def test_empty_sql(self):
        assert extract_table_names("", dialect=DBType.SNOWFLAKE) == []

    def test_invalid_sql(self):
        result = extract_table_names("NOT SQL AT ALL !!!", dialect=DBType.SNOWFLAKE)
        assert isinstance(result, list)

    def test_mysql_dialect(self):
        tables = extract_table_names("SELECT * FROM mydb.users", dialect=DBType.MYSQL)
        assert len(tables) > 0

    def test_sqlite_dialect(self):
        tables = extract_table_names("SELECT * FROM users", dialect=DBType.SQLITE)
        assert len(tables) > 0

    def test_ignore_empty_true(self):
        tables = extract_table_names("SELECT * FROM users", dialect=DBType.SNOWFLAKE, ignore_empty=True)
        assert len(tables) > 0


# ===========================================================================
# metadata_identifier
# ===========================================================================


class TestMetadataIdentifier:
    def test_sqlite(self):
        result = metadata_identifier(database_name="main", table_name="users", dialect=DBType.SQLITE)
        assert result == "main.users"

    def test_sqlite_no_db(self):
        result = metadata_identifier(table_name="users", dialect=DBType.SQLITE)
        assert result == "users"

    def test_duckdb(self):
        result = metadata_identifier(database_name="db", schema_name="public", table_name="t", dialect=DBType.DUCKDB)
        assert result == "db.public.t"

    def test_mysql_with_catalog(self):
        result = metadata_identifier(
            catalog_name="cat", database_name="db", table_name="t", dialect=DBType.MYSQL
        )
        assert result == "cat.db.t"

    def test_mysql_without_catalog(self):
        result = metadata_identifier(database_name="db", table_name="t", dialect=DBType.MYSQL)
        assert result == "db.t"

    def test_oracle(self):
        result = metadata_identifier(
            database_name="db", schema_name="public", table_name="t", dialect=DBType.ORACLE
        )
        assert result == "db.public.t"

    def test_snowflake_with_catalog(self):
        result = metadata_identifier(
            catalog_name="cat", database_name="db", schema_name="sc", table_name="t", dialect=DBType.SNOWFLAKE
        )
        assert result == "cat.db.sc.t"

    def test_snowflake_without_catalog(self):
        result = metadata_identifier(
            database_name="db", schema_name="sc", table_name="t", dialect=DBType.SNOWFLAKE
        )
        assert result == "db.sc.t"

    def test_databricks_with_catalog(self):
        result = metadata_identifier(
            catalog_name="cat", schema_name="sc", table_name="t", dialect="databricks"
        )
        assert result == "cat.sc.t"

    def test_databricks_without_catalog(self):
        result = metadata_identifier(schema_name="sc", table_name="t", dialect="databricks")
        assert result == "sc.t"

    def test_starrocks(self):
        result = metadata_identifier(
            catalog_name="cat", database_name="db", table_name="t", dialect=DBType.STARROCKS
        )
        assert result == "cat.db.t"


# ===========================================================================
# parse_table_name_parts
# ===========================================================================


class TestParseTableNameParts:
    def test_simple_name(self):
        result = parse_table_name_parts("users", dialect=DBType.DUCKDB)
        assert result["table_name"] == "users"

    def test_two_parts(self):
        result = parse_table_name_parts("schema.table", dialect=DBType.DUCKDB)
        assert result["schema_name"] == "schema"
        assert result["table_name"] == "table"

    def test_three_parts_duckdb(self):
        result = parse_table_name_parts("db.schema.table", dialect=DBType.DUCKDB)
        assert result["database_name"] == "db"
        assert result["schema_name"] == "schema"
        assert result["table_name"] == "table"

    def test_sqlite(self):
        result = parse_table_name_parts("main.users", dialect=DBType.SQLITE)
        assert result["database_name"] == "main"
        assert result["table_name"] == "users"

    def test_snowflake_four_parts(self):
        result = parse_table_name_parts("cat.db.schema.table", dialect=DBType.SNOWFLAKE)
        assert result["catalog_name"] == "cat"
        assert result["table_name"] == "table"

    def test_unknown_dialect(self):
        result = parse_table_name_parts("a.b.c.d", dialect="unknown")
        assert result["table_name"] == "d"
        assert result["schema_name"] == "c"
        assert result["database_name"] == "b"
        assert result["catalog_name"] == "a"


class TestParseTableNamesParts:
    def test_list(self):
        result = parse_table_names_parts(["users", "orders"], dialect=DBType.DUCKDB)
        assert len(result) == 2


# ===========================================================================
# strip_sql_comments
# ===========================================================================


class TestStripSqlComments:
    def test_block_comment(self):
        assert strip_sql_comments("SELECT /* comment */ 1") == "SELECT   1"

    def test_line_comment(self):
        result = strip_sql_comments("SELECT 1 -- comment\nFROM t")
        assert "comment" not in result

    def test_no_comments(self):
        assert strip_sql_comments("SELECT 1") == "SELECT 1"


# ===========================================================================
# _is_escaped
# ===========================================================================


class TestIsEscaped:
    def test_not_escaped(self):
        assert _is_escaped("abc'def", 3) is False

    def test_escaped(self):
        assert _is_escaped("abc\\'def", 4) is True

    def test_double_backslash(self):
        assert _is_escaped("abc\\\\'def", 5) is False


# ===========================================================================
# _match_dollar_tag
# ===========================================================================


class TestMatchDollarTag:
    def test_simple(self):
        assert _match_dollar_tag("$$body$$", 0) == "$$"

    def test_named_tag(self):
        assert _match_dollar_tag("$tag$body$tag$", 0) == "$tag$"

    def test_no_match(self):
        assert _match_dollar_tag("hello", 0) is None


# ===========================================================================
# _first_statement
# ===========================================================================


class TestFirstStatement:
    def test_single_statement(self):
        assert _first_statement("SELECT 1") == "SELECT 1"

    def test_multi_statement(self):
        assert _first_statement("SELECT 1; SELECT 2") == "SELECT 1"

    def test_empty(self):
        assert _first_statement("") == ""

    def test_with_string_semicolon(self):
        result = _first_statement("SELECT 'a;b'")
        assert "a;b" in result

    def test_with_double_quote_semicolon(self):
        result = _first_statement('SELECT "a;b"')
        assert "a;b" in result

    def test_with_backtick_semicolon(self):
        result = _first_statement("SELECT `a;b`")
        assert "a;b" in result

    def test_with_bracket(self):
        result = _first_statement("SELECT [a;b]")
        assert "a;b" in result

    def test_with_dollar_quoting(self):
        result = _first_statement("SELECT $$a;b$$; SELECT 2")
        assert "a;b" in result

    def test_escaped_single_quote(self):
        result = _first_statement("SELECT 'it''s'; SELECT 2")
        assert "it" in result

    def test_with_comments(self):
        result = _first_statement("/* comment */ SELECT 1; SELECT 2")
        assert result == "SELECT 1"


# ===========================================================================
# parse_sql_type
# ===========================================================================


class TestParseSqlType:
    def test_select(self):
        assert parse_sql_type("SELECT * FROM users", DBType.SNOWFLAKE) == SQLType.SELECT

    def test_insert(self):
        assert parse_sql_type("INSERT INTO users VALUES (1, 'a')", DBType.SNOWFLAKE) == SQLType.INSERT

    def test_update(self):
        assert parse_sql_type("UPDATE users SET name='b' WHERE id=1", DBType.SNOWFLAKE) == SQLType.UPDATE

    def test_delete(self):
        assert parse_sql_type("DELETE FROM users WHERE id=1", DBType.SNOWFLAKE) == SQLType.DELETE

    def test_create(self):
        assert parse_sql_type("CREATE TABLE t (id INT)", DBType.SNOWFLAKE) == SQLType.DDL

    def test_alter(self):
        assert parse_sql_type("ALTER TABLE t ADD COLUMN c INT", DBType.SNOWFLAKE) == SQLType.DDL

    def test_drop(self):
        assert parse_sql_type("DROP TABLE t", DBType.SNOWFLAKE) == SQLType.DDL

    def test_show(self):
        assert parse_sql_type("SHOW TABLES", DBType.SNOWFLAKE) == SQLType.METADATA_SHOW

    def test_describe(self):
        assert parse_sql_type("DESCRIBE TABLE t", DBType.SNOWFLAKE) == SQLType.METADATA_SHOW

    def test_use(self):
        assert parse_sql_type("USE DATABASE mydb", DBType.SNOWFLAKE) == SQLType.CONTENT_SET

    def test_set(self):
        assert parse_sql_type("SET x = 1", DBType.SNOWFLAKE) == SQLType.CONTENT_SET

    def test_with_select(self):
        result = parse_sql_type("WITH cte AS (SELECT 1) SELECT * FROM cte", DBType.SNOWFLAKE)
        assert result == SQLType.SELECT

    def test_empty(self):
        assert parse_sql_type("", DBType.SNOWFLAKE) == SQLType.UNKNOWN

    def test_none(self):
        assert parse_sql_type(None, DBType.SNOWFLAKE) == SQLType.UNKNOWN

    def test_whitespace_only(self):
        assert parse_sql_type("   ", DBType.SNOWFLAKE) == SQLType.UNKNOWN

    def test_values(self):
        result = parse_sql_type("VALUES (1, 2), (3, 4)", DBType.SNOWFLAKE)
        assert result == SQLType.SELECT

    def test_explain(self):
        result = parse_sql_type("EXPLAIN SELECT 1", DBType.SNOWFLAKE)
        assert result == SQLType.EXPLAIN

    def test_merge(self):
        sql = "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.x = s.x"
        result = parse_sql_type(sql, DBType.SNOWFLAKE)
        assert result == SQLType.MERGE


# ===========================================================================
# _fallback_sql_type
# ===========================================================================


class TestFallbackSqlType:
    def test_select(self):
        assert _fallback_sql_type("SELECT 1") == SQLType.SELECT

    def test_with_cte_select(self):
        assert _fallback_sql_type("WITH cte AS (SELECT 1) SELECT * FROM cte") == SQLType.SELECT

    def test_with_cte_insert(self):
        assert _fallback_sql_type("WITH cte AS (SELECT 1) INSERT INTO t SELECT * FROM cte") == SQLType.INSERT

    def test_empty(self):
        assert _fallback_sql_type("") is None

    def test_unknown_keyword(self):
        assert _fallback_sql_type("FOOBAR something") is None


# ===========================================================================
# normalize_sql / format_sql_to_pretty
# ===========================================================================


class TestNormalizeSql:
    def test_newlines(self):
        assert normalize_sql("SELECT\n  1\n  FROM t") == "SELECT 1 FROM t"

    def test_tabs(self):
        assert normalize_sql("SELECT\t1") == "SELECT 1"

    def test_multiple_spaces(self):
        assert normalize_sql("SELECT   1") == "SELECT 1"


class TestFormatSqlToPretty:
    def test_basic(self):
        result = format_sql_to_pretty("SELECT * FROM users WHERE id = 1", DBType.SNOWFLAKE)
        assert "SELECT" in result

    def test_empty(self):
        assert format_sql_to_pretty("", DBType.SNOWFLAKE) == ""

    def test_invalid_sql(self):
        result = format_sql_to_pretty("NOT SQL AT ALL", DBType.SNOWFLAKE)
        assert isinstance(result, str)


# ===========================================================================
# parse_context_switch
# ===========================================================================


class TestParseContextSwitch:
    def test_use_database_snowflake(self):
        result = parse_context_switch("USE DATABASE mydb", DBType.SNOWFLAKE)
        assert result is not None
        assert result["command"] == "USE"
        assert result["database_name"] == "mydb"
        assert result["target"] == "database"

    def test_use_schema_snowflake(self):
        result = parse_context_switch("USE SCHEMA myschema", DBType.SNOWFLAKE)
        assert result is not None
        assert result["target"] == "schema"

    def test_use_catalog_snowflake(self):
        result = parse_context_switch("USE CATALOG mycat", DBType.SNOWFLAKE)
        assert result is not None
        assert result["target"] == "catalog"

    def test_use_mysql(self):
        result = parse_context_switch("USE mydb", DBType.MYSQL)
        assert result is not None
        assert result["target"] == "database"

    def test_use_duckdb_bare(self):
        result = parse_context_switch("USE myschema", DBType.DUCKDB)
        assert result is not None
        assert result["fuzzy"] is True

    def test_use_duckdb_qualified(self):
        result = parse_context_switch("USE db.schema", DBType.DUCKDB)
        assert result is not None
        assert result["target"] == "schema"

    def test_set_catalog(self):
        result = parse_context_switch("SET CATALOG mycat", DBType.SNOWFLAKE)
        assert result is not None
        assert result["target"] == "catalog"

    def test_set_database(self):
        result = parse_context_switch("SET DATABASE mydb", DBType.SNOWFLAKE)
        assert result is not None
        assert result["target"] == "database"

    def test_set_schema(self):
        result = parse_context_switch("SET SCHEMA myschema", DBType.SNOWFLAKE)
        assert result is not None
        assert result["target"] == "schema"

    def test_set_schema_with_equals(self):
        result = parse_context_switch("SET SCHEMA = myschema", DBType.SNOWFLAKE)
        assert result is not None
        assert result["schema_name"] == "myschema"

    def test_set_schema_with_to(self):
        result = parse_context_switch("SET SCHEMA TO myschema", DBType.DUCKDB)
        assert result is not None
        assert result["schema_name"] == "myschema"

    def test_empty_input(self):
        assert parse_context_switch("", DBType.SNOWFLAKE) is None

    def test_none_input(self):
        assert parse_context_switch(None, DBType.SNOWFLAKE) is None

    def test_non_use_set(self):
        assert parse_context_switch("SELECT 1", DBType.SNOWFLAKE) is None

    def test_use_snowflake_qualified_schema(self):
        result = parse_context_switch("USE db.schema", DBType.SNOWFLAKE)
        assert result is not None
        assert result["target"] == "schema"
        assert result["database_name"] == "db"

    def test_use_snowflake_full_qualified(self):
        result = parse_context_switch("USE cat.db.schema", DBType.SNOWFLAKE)
        assert result is not None
        assert result["target"] == "schema"
        assert result["catalog_name"] == "cat"

    def test_use_starrocks(self):
        result = parse_context_switch("USE mydb", DBType.STARROCKS)
        assert result is not None
        assert result["target"] == "database"

    def test_set_empty_remainder(self):
        result = parse_context_switch("SET SCHEMA ;", DBType.SNOWFLAKE)
        assert result is None

    def test_set_session_schema(self):
        result = parse_context_switch("SET SESSION SCHEMA myschema", DBType.SNOWFLAKE)
        assert result is not None
        assert result["schema_name"] == "myschema"


# ===========================================================================
# _identifier_name / _table_parts
# ===========================================================================


class TestIdentifierName:
    def test_none(self):
        assert _identifier_name(None) == ""

    def test_string(self):
        assert _identifier_name('"my_col"') == "my_col"

    def test_plain_string(self):
        assert _identifier_name("col") == "col"


class TestTableParts:
    def test_none_input(self):
        result = _table_parts(None)
        assert result == {"catalog": "", "database": "", "identifier": ""}

    def test_string_input(self):
        result = _table_parts("not a table")
        assert result == {"catalog": "", "database": "", "identifier": ""}
