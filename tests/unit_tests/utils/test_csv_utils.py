# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/csv_utils.py."""

from pathlib import Path

import pytest

from datus.utils.csv_utils import file_encoding, read_csv_and_clean_text


class TestFileEncoding:
    def test_utf8_bom(self, tmp_path):
        f = tmp_path / "bom.csv"
        f.write_bytes(b"\xef\xbb\xbfname,age\nAlice,30\n")
        assert file_encoding(f) == "utf-8-sig"

    def test_regular_utf8(self, tmp_path):
        f = tmp_path / "utf8.csv"
        f.write_text("name,age\nAlice,30\n", encoding="utf-8")
        result = file_encoding(f)
        assert result in ("utf-8", "ascii")

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "nonexistent.csv"
        assert file_encoding(f) == ""

    def test_directory_instead_of_file(self, tmp_path):
        d = tmp_path / "somedir"
        d.mkdir()
        assert file_encoding(d) == ""


class TestReadCsvAndCleanText:
    def test_empty_path(self):
        assert read_csv_and_clean_text("") == []

    def test_none_path(self):
        assert read_csv_and_clean_text(None) == []

    def test_nonexistent_file(self, tmp_path):
        result = read_csv_and_clean_text(str(tmp_path / "nonexistent.csv"))
        assert result == []

    def test_directory_path(self, tmp_path):
        result = read_csv_and_clean_text(str(tmp_path))
        assert result == []

    def test_valid_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
        result = read_csv_and_clean_text(str(f))
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"

    def test_csv_with_special_chars(self, tmp_path):
        f = tmp_path / "special.csv"
        # Write CSV with NBSP character
        content = "name,desc\nAlice,hello\u00a0world\n"
        f.write_text(content, encoding="utf-8")
        result = read_csv_and_clean_text(str(f))
        assert len(result) == 1
        # NBSP should be replaced with space
        assert "\u00a0" not in result[0]["desc"]

    def test_csv_with_path_object(self, tmp_path):
        f = tmp_path / "pathobj.csv"
        f.write_text("col1,col2\na,b\n", encoding="utf-8")
        result = read_csv_and_clean_text(f)
        assert len(result) == 1

    def test_csv_with_nan_values(self, tmp_path):
        f = tmp_path / "nan.csv"
        f.write_text("name,value\nAlice,\nBob,42\n", encoding="utf-8")
        result = read_csv_and_clean_text(str(f))
        assert len(result) == 2
        assert result[0]["value"] is None
