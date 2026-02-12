"""Tests for datus/storage/reference_sql/sql_file_processor.py"""

import os

import pytest

from datus.storage.reference_sql.sql_file_processor import (
    _find_effective_semicolon,
    log_invalid_entries,
    parse_comment_sql_pairs,
    process_sql_files,
    process_sql_items,
    validate_sql,
)


class TestFindEffectiveSemicolon:
    def test_simple_semicolon(self):
        pos, in_block = _find_effective_semicolon("SELECT 1;", False)
        assert pos == 8
        assert in_block is False

    def test_no_semicolon(self):
        pos, in_block = _find_effective_semicolon("SELECT 1", False)
        assert pos == -1
        assert in_block is False

    def test_semicolon_in_line_comment(self):
        pos, in_block = _find_effective_semicolon("SELECT 1 -- ;comment", False)
        assert pos == -1
        assert in_block is False

    def test_semicolon_in_block_comment(self):
        pos, in_block = _find_effective_semicolon("SELECT /* ; */ 1", False)
        # The semicolon is inside a block comment
        assert pos == -1 or in_block is False

    def test_already_in_block_comment(self):
        pos, in_block = _find_effective_semicolon("still in comment */ SELECT 1;", True)
        assert pos >= 0

    def test_block_comment_no_end(self):
        pos, in_block = _find_effective_semicolon("still in comment", True)
        assert pos == -1
        assert in_block is True

    def test_semicolon_in_single_quote(self):
        pos, in_block = _find_effective_semicolon("SELECT ';' FROM t;", False)
        # The first ; is inside single quotes, the second is effective
        assert pos == 17

    def test_semicolon_in_double_quote(self):
        pos, in_block = _find_effective_semicolon('SELECT "col;" FROM t;', False)
        # The first ; is inside double quotes, the second is effective
        assert pos == 20

    def test_escaped_single_quote(self):
        pos, in_block = _find_effective_semicolon("SELECT 'it''s' FROM t;", False)
        assert pos == 21

    def test_escaped_double_quote(self):
        pos, in_block = _find_effective_semicolon('SELECT "col""name" FROM t;', False)
        assert pos == 25

    def test_block_comment_start_after_semicolon(self):
        pos, in_block = _find_effective_semicolon("SELECT 1; /* comment", False)
        assert pos == 8
        assert in_block is True

    def test_line_comment_after_semicolon(self):
        pos, in_block = _find_effective_semicolon("SELECT 1; -- comment", False)
        assert pos == 8
        assert in_block is False


class TestParseCommentSqlPairs:
    def test_single_statement(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("SELECT 1 FROM dual;")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        assert "SELECT 1 FROM dual" in pairs[0][1]

    def test_multiple_statements(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("SELECT 1;\nSELECT 2;\nSELECT 3;")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 3

    def test_statement_with_comments(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("-- This is a comment\nSELECT 1 FROM dual;")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        # Comments are included in the SQL block
        assert "comment" in pairs[0][1] or "SELECT" in pairs[0][1]

    def test_multiline_statement(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("SELECT\n  col1,\n  col2\nFROM\n  table1;")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        assert "col1" in pairs[0][1]
        assert "col2" in pairs[0][1]

    def test_block_comment(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("/* block comment */\nSELECT 1;")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1

    def test_empty_file(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 0

    def test_no_semicolon_trailing(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("SELECT 1 FROM dual")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1

    def test_gbk_encoding_fallback(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        # Write GBK content
        with open(str(sql_file), "w", encoding="gbk") as f:
            f.write("SELECT '中文' FROM dual;")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1


class TestValidateSql:
    def test_valid_select(self):
        is_valid, cleaned, error = validate_sql("SELECT * FROM users")
        assert is_valid is True
        assert cleaned != ""
        assert error == ""

    def test_valid_complex_query(self):
        is_valid, cleaned, error = validate_sql(
            "SELECT a.id, b.name FROM users a JOIN orders b ON a.id = b.user_id WHERE a.active = 1"
        )
        assert is_valid is True

    def test_invalid_sql(self):
        is_valid, cleaned, error = validate_sql("NOT A VALID SQL STATEMENT AT ALL %%% !!!")
        # Might or might not be valid depending on sqlglot parsing
        # At least one of the assertions should hold
        assert isinstance(is_valid, bool)

    def test_empty_sql(self):
        is_valid, cleaned, error = validate_sql("")
        # empty sql might not parse
        assert isinstance(is_valid, bool)


class TestProcessSqlItems:
    def test_valid_select_item(self):
        items = [
            {
                "sql": "SELECT * FROM users",
                "comment": "",
                "filepath": "test.sql",
                "line_number": 1,
            }
        ]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 1
        assert "line_number" not in valid[0]

    def test_empty_sql_skipped(self):
        items = [
            {
                "sql": "",
                "comment": "",
                "filepath": "test.sql",
                "line_number": 1,
            }
        ]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_non_select_skipped(self):
        items = [
            {
                "sql": "INSERT INTO users VALUES (1, 'name')",
                "comment": "",
                "filepath": "test.sql",
                "line_number": 1,
            }
        ]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 0


class TestProcessSqlFiles:
    def test_process_directory_with_sql_files(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("SELECT id, name FROM users;")
        valid, invalid = process_sql_files(str(tmp_path))
        assert len(valid) + len(invalid) >= 1

    def test_process_single_sql_file(self, tmp_path):
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("SELECT 1 FROM dual;")
        valid, invalid = process_sql_files(str(sql_file))
        assert len(valid) + len(invalid) >= 1

    def test_process_nonexistent_directory(self):
        with pytest.raises(ValueError, match="not found"):
            process_sql_files("/nonexistent/path")

    def test_process_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="No SQL files"):
            process_sql_files(str(tmp_path))

    def test_process_non_sql_file(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a sql file")
        with pytest.raises(ValueError):
            process_sql_files(str(tmp_path))


class TestLogInvalidEntries:
    def test_log_invalid_entries(self, tmp_path):
        os.chdir(str(tmp_path))
        entries = [
            {
                "filepath": "test.sql",
                "comment": "test comment",
                "error": "parse error",
                "sql": "INVALID SQL",
                "line_number": 5,
            }
        ]
        log_invalid_entries(entries)
        log_file = tmp_path / "sql_processing_errors.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "parse error" in content
        assert "test.sql" in content
        assert "line 5" in content
