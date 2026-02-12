# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/path_utils.py."""

import os
from pathlib import Path

import pytest

from datus.utils.constants import DBType
from datus.utils.path_utils import (
    get_file_fuzzy_matches,
    get_file_name,
    get_files_from_glob_pattern,
    has_glob_pattern,
    safe_rmtree,
)


class TestSafeRmtree:
    def test_force_delete(self, tmp_path):
        d = tmp_path / "to_delete"
        d.mkdir()
        (d / "file.txt").write_text("content")
        assert safe_rmtree(d, description="test dir", force=True)
        assert not d.exists()

    def test_nonexistent_path(self, tmp_path):
        result = safe_rmtree(tmp_path / "nonexistent", force=True)
        assert result is False

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        result = safe_rmtree(f, force=True)
        assert result is False

    def test_string_path(self, tmp_path):
        d = tmp_path / "strpath"
        d.mkdir()
        result = safe_rmtree(str(d), force=True)
        assert result is True
        assert not d.exists()

    def test_force_delete_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert safe_rmtree(d, force=True) is True

    def test_non_interactive_skips(self, tmp_path):
        """Test safe_rmtree in non-interactive mode (stdin not a tty) without force."""
        d = tmp_path / "nontty"
        d.mkdir()
        (d / "f.txt").write_text("x")
        # Without force and stdin is not a tty (pytest), should skip deletion
        result = safe_rmtree(d, description="test", force=False)
        assert result is False
        assert d.exists()

    def test_force_delete_oserror(self, tmp_path):
        """Test safe_rmtree when rmtree fails."""
        d = tmp_path / "bad"
        d.mkdir()
        from unittest.mock import patch

        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            result = safe_rmtree(d, force=True)
            assert result is False


class TestHasGlobPattern:
    def test_no_pattern(self):
        assert has_glob_pattern("/simple/path/file.txt") is False

    def test_star(self):
        assert has_glob_pattern("/path/*.txt") is True

    def test_question_mark(self):
        assert has_glob_pattern("/path/file?.txt") is True

    def test_brackets(self):
        assert has_glob_pattern("/path/file[0-9].txt") is True

    def test_double_star(self):
        assert has_glob_pattern("/path/**/*.txt") is True


class TestGetFilesFromGlobPattern:
    def test_no_glob_pattern(self):
        result = get_files_from_glob_pattern("/simple/path.txt")
        assert result == []

    def test_glob_pattern_matches_files(self, tmp_path):
        # Create test files
        (tmp_path / "test1.db").write_text("")
        (tmp_path / "test2.db").write_text("")
        (tmp_path / "readme.txt").write_text("")

        pattern = str(tmp_path / "*.db")
        result = get_files_from_glob_pattern(pattern, dialect=DBType.SQLITE)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert "test1" in names
        assert "test2" in names

    def test_glob_with_string_dialect(self, tmp_path):
        (tmp_path / "data.db").write_text("")
        pattern = str(tmp_path / "*.db")
        result = get_files_from_glob_pattern(pattern, dialect="sqlite")
        assert len(result) == 1

    def test_glob_with_dir_wildcard(self, tmp_path):
        subdir1 = tmp_path / "ns1"
        subdir1.mkdir()
        (subdir1 / "data.db").write_text("")

        subdir2 = tmp_path / "ns2"
        subdir2.mkdir()
        (subdir2 / "data.db").write_text("")

        pattern = str(tmp_path / "*" / "*.db")
        result = get_files_from_glob_pattern(pattern, dialect=DBType.SQLITE)
        assert len(result) == 2
        # With dir wildcard, logic_name should be parent dir name
        logic_names = {r["logic_name"] for r in result}
        assert "ns1" in logic_names
        assert "ns2" in logic_names

    def test_no_matches(self, tmp_path):
        pattern = str(tmp_path / "*.nonexistent")
        result = get_files_from_glob_pattern(pattern)
        assert result == []

    def test_glob_no_slash_pattern(self, tmp_path):
        """Test glob pattern without any slash (line 123)."""
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            (tmp_path / "test.db").write_text("")
            result = get_files_from_glob_pattern("*.db")
            assert len(result) == 1
            assert result[0]["logic_name"] == "test"
        finally:
            os.chdir(original_cwd)

    def test_glob_skip_directories(self, tmp_path):
        """Test that directories in glob results are skipped (line 132)."""
        (tmp_path / "subdir.db").mkdir()
        (tmp_path / "file.db").write_text("")
        pattern = str(tmp_path / "*.db")
        result = get_files_from_glob_pattern(pattern)
        assert len(result) == 1
        assert result[0]["name"] == "file"


class TestGetFileName:
    def test_with_extension(self):
        assert get_file_name("path/to/file.txt") == "file"

    def test_without_extension(self):
        assert get_file_name("path/to/noext") == "noext"

    def test_multiple_dots(self):
        assert get_file_name("path/file.tar.gz") == "file.tar"

    def test_just_name(self):
        assert get_file_name("simple.py") == "simple"


class TestGetFileFuzzyMatches:
    def test_basic_match(self, tmp_path):
        (tmp_path / "test_file.py").write_text("")
        (tmp_path / "other.txt").write_text("")
        results = get_file_fuzzy_matches("test", str(tmp_path))
        assert any("test_file.py" in r for r in results)

    def test_no_match(self, tmp_path):
        (tmp_path / "abc.py").write_text("")
        results = get_file_fuzzy_matches("xyz", str(tmp_path))
        assert results == []

    def test_exact_case_match(self, tmp_path):
        (tmp_path / "TestFile.py").write_text("")
        results = get_file_fuzzy_matches("TestFile", str(tmp_path))
        assert len(results) > 0

    def test_max_matches(self, tmp_path):
        for i in range(10):
            (tmp_path / f"match_{i}.py").write_text("")
        results = get_file_fuzzy_matches("match", str(tmp_path), max_matches=3)
        assert len(results) <= 3

    def test_nonexistent_path(self):
        results = get_file_fuzzy_matches("test", "/nonexistent/path")
        assert results == []

    def test_subdirectory_match(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "deep_file.py").write_text("")
        results = get_file_fuzzy_matches("deep_file", str(tmp_path))
        assert len(results) > 0
