# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for filesystem_tools.py – 21.1% coverage target."""

import json
import os
from pathlib import Path

import pytest

from datus.tools.func_tool.filesystem_tools import (
    EditOperation,
    FilesystemConfig,
    FilesystemFuncTool,
)


# ---------------------------------------------------------------------------
# FilesystemConfig
# ---------------------------------------------------------------------------


class TestFilesystemConfig:
    def test_defaults(self):
        config = FilesystemConfig()
        assert config.root_path == os.path.expanduser("~")
        assert ".py" in config.allowed_extensions
        assert config.max_file_size == 1024 * 1024

    def test_custom(self, tmp_path):
        config = FilesystemConfig(
            root_path=str(tmp_path),
            allowed_extensions=[".txt"],
            max_file_size=100,
        )
        assert config.root_path == str(tmp_path)
        assert config.allowed_extensions == [".txt"]
        assert config.max_file_size == 100


# ---------------------------------------------------------------------------
# EditOperation
# ---------------------------------------------------------------------------


class TestEditOperation:
    def test_model(self):
        op = EditOperation(oldText="hello", newText="world")
        assert op.oldText == "hello"
        assert op.newText == "world"


# ---------------------------------------------------------------------------
# FilesystemFuncTool
# ---------------------------------------------------------------------------


class TestFilesystemFuncTool:
    @pytest.fixture
    def tool(self, tmp_path):
        return FilesystemFuncTool(root_path=str(tmp_path))

    # --- _get_safe_path ---

    def test_get_safe_path_valid(self, tool, tmp_path):
        result = tool._get_safe_path("subdir/file.txt")
        assert result is not None
        assert str(result).startswith(str(Path(tool.config.root_path).resolve()))

    def test_get_safe_path_traversal(self, tool):
        result = tool._get_safe_path("../../etc/passwd")
        assert result is None

    # --- _is_allowed_file ---

    def test_is_allowed_file_yes(self, tool):
        assert tool._is_allowed_file(Path("test.py")) is True
        assert tool._is_allowed_file(Path("test.json")) is True

    def test_is_allowed_file_no(self, tool):
        assert tool._is_allowed_file(Path("test.exe")) is False

    def test_is_allowed_file_no_extensions(self, tmp_path):
        tool = FilesystemFuncTool(root_path=str(tmp_path))
        tool.config.allowed_extensions = None
        assert tool._is_allowed_file(Path("anything.xyz")) is True

    # --- read_file ---

    def test_read_file_success(self, tool, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello World", encoding="utf-8")
        result = tool.read_file("hello.txt")
        assert result.success == 1
        assert result.result == "Hello World"

    def test_read_file_not_found(self, tool):
        result = tool.read_file("nonexistent.txt")
        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_read_file_not_a_file(self, tool, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = tool.read_file("subdir")
        assert result.success == 0
        assert "not a file" in result.error.lower()

    def test_read_file_not_allowed_ext(self, tool, tmp_path):
        f = tmp_path / "test.exe"
        f.write_text("content", encoding="utf-8")
        result = tool.read_file("test.exe")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_read_file_too_large(self, tool, tmp_path):
        tool.config.max_file_size = 5
        f = tmp_path / "big.txt"
        f.write_text("A" * 100, encoding="utf-8")
        result = tool.read_file("big.txt")
        assert result.success == 0
        assert "too large" in result.error.lower()

    def test_read_file_binary(self, tool, tmp_path):
        f = tmp_path / "binary.py"
        f.write_bytes(b"\x80\x81\x82\x83" * 100)
        result = tool.read_file("binary.py")
        assert result.success == 0
        assert "binary" in result.error.lower()

    # --- read_multiple_files ---

    def test_read_multiple_files(self, tool, tmp_path):
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("B", encoding="utf-8")
        result = tool.read_multiple_files(["a.txt", "b.txt"])
        assert result.success == 1
        assert result.result["a.txt"] == "A"
        assert result.result["b.txt"] == "B"

    def test_read_multiple_files_with_errors(self, tool, tmp_path):
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        result = tool.read_multiple_files(["a.txt", "nonexistent.txt"])
        assert result.success == 1
        assert result.result["a.txt"] == "A"
        assert "not found" in result.result["nonexistent.txt"].lower()

    def test_read_multiple_files_not_a_file(self, tool, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = tool.read_multiple_files(["subdir"])
        assert result.success == 1
        assert "not a file" in result.result["subdir"].lower()

    def test_read_multiple_files_not_allowed(self, tool, tmp_path):
        (tmp_path / "test.exe").write_text("content", encoding="utf-8")
        result = tool.read_multiple_files(["test.exe"])
        assert result.success == 1
        assert "not allowed" in result.result["test.exe"].lower()

    # --- write_file ---

    def test_write_file_success(self, tool, tmp_path):
        result = tool.write_file("new.txt", "content")
        assert result.success == 1
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "content"

    def test_write_file_creates_parent(self, tool, tmp_path):
        result = tool.write_file("sub/dir/file.txt", "nested")
        assert result.success == 1
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text(encoding="utf-8") == "nested"

    def test_write_file_not_allowed(self, tool, tmp_path):
        result = tool.write_file("test.exe", "content")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_write_file_invalid_path(self, tool):
        result = tool.write_file("../../outside.txt", "content")
        assert result.success == 0

    # --- edit_file ---

    def test_edit_file_success(self, tool, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello world", encoding="utf-8")
        result = tool.edit_file("edit.txt", [EditOperation(oldText="hello", newText="goodbye")])
        assert result.success == 1
        assert "goodbye world" in f.read_text(encoding="utf-8")

    def test_edit_file_dict_edits(self, tool, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("foo bar", encoding="utf-8")
        result = tool.edit_file("edit.txt", [{"oldText": "foo", "newText": "baz"}])
        assert result.success == 1
        assert "baz bar" in f.read_text(encoding="utf-8")

    def test_edit_file_json_string_edits(self, tool, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("abc", encoding="utf-8")
        edits_json = json.dumps([{"oldText": "abc", "newText": "xyz"}])
        result = tool.edit_file("edit.txt", edits_json)
        assert result.success == 1
        assert f.read_text(encoding="utf-8") == "xyz"

    def test_edit_file_invalid_json(self, tool, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("abc", encoding="utf-8")
        result = tool.edit_file("edit.txt", "not valid json")
        assert result.success == 0
        assert "invalid json" in result.error.lower()

    def test_edit_file_text_not_found(self, tool, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello", encoding="utf-8")
        result = tool.edit_file("edit.txt", [{"oldText": "nonexistent", "newText": "x"}])
        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_edit_file_no_edits_applied(self, tool, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello", encoding="utf-8")
        result = tool.edit_file("edit.txt", [{"oldText": "", "newText": "x"}])
        assert result.success == 0
        assert "no edits" in result.error.lower()

    def test_edit_file_not_found(self, tool):
        result = tool.edit_file("nonexistent.txt", [{"oldText": "a", "newText": "b"}])
        assert result.success == 0

    def test_edit_file_not_allowed_ext(self, tool, tmp_path):
        f = tmp_path / "test.exe"
        f.write_text("content", encoding="utf-8")
        result = tool.edit_file("test.exe", [{"oldText": "c", "newText": "d"}])
        assert result.success == 0

    # --- create_directory ---

    def test_create_directory_success(self, tool, tmp_path):
        result = tool.create_directory("new_dir")
        assert result.success == 1
        assert (tmp_path / "new_dir").is_dir()

    def test_create_directory_nested(self, tool, tmp_path):
        result = tool.create_directory("a/b/c")
        assert result.success == 1
        assert (tmp_path / "a" / "b" / "c").is_dir()

    def test_create_directory_invalid(self, tool):
        result = tool.create_directory("../../outside")
        assert result.success == 0

    # --- list_directory ---

    def test_list_directory_success(self, tool, tmp_path):
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        result = tool.list_directory(".")
        assert result.success == 1
        names = [item["name"] for item in result.result]
        assert "file.txt" in names
        assert "subdir" in names

    def test_list_directory_not_found(self, tool):
        result = tool.list_directory("nonexistent")
        assert result.success == 0

    def test_list_directory_not_a_dir(self, tool, tmp_path):
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")
        result = tool.list_directory("file.txt")
        assert result.success == 0

    # --- directory_tree ---

    def test_directory_tree_success(self, tool, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b.txt").write_text("b", encoding="utf-8")
        result = tool.directory_tree(".", max_depth=2)
        assert result.success == 1
        assert "a/" in result.result
        assert "b.txt" in result.result

    def test_directory_tree_max_depth(self, tool, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "b" / "c.txt").write_text("c", encoding="utf-8")
        result = tool.directory_tree(".", max_depth=1)
        assert result.success == 1
        assert "max depth" in result.result.lower()

    def test_directory_tree_max_items(self, tool, tmp_path):
        for i in range(5):
            (tmp_path / f"file_{i}.txt").write_text("x", encoding="utf-8")
        result = tool.directory_tree(".", max_items=2)
        assert result.success == 1
        assert "truncated" in result.result.lower()

    def test_directory_tree_not_found(self, tool):
        result = tool.directory_tree("nonexistent")
        assert result.success == 0

    def test_directory_tree_not_a_dir(self, tool, tmp_path):
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")
        result = tool.directory_tree("file.txt")
        assert result.success == 0

    # --- move_file ---

    def test_move_file_success(self, tool, tmp_path):
        (tmp_path / "src.txt").write_text("move me", encoding="utf-8")
        result = tool.move_file("src.txt", "dest.txt")
        assert result.success == 1
        assert not (tmp_path / "src.txt").exists()
        assert (tmp_path / "dest.txt").read_text(encoding="utf-8") == "move me"

    def test_move_file_source_not_found(self, tool):
        result = tool.move_file("nonexistent.txt", "dest.txt")
        assert result.success == 0

    def test_move_file_invalid_dest(self, tool, tmp_path):
        (tmp_path / "src.txt").write_text("x", encoding="utf-8")
        result = tool.move_file("src.txt", "../../outside.txt")
        assert result.success == 0

    # --- search_files ---

    def test_search_files_success(self, tool, tmp_path):
        (tmp_path / "a.py").write_text("x", encoding="utf-8")
        (tmp_path / "b.txt").write_text("x", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("x", encoding="utf-8")
        result = tool.search_files(".", "*.py")
        assert result.success == 1
        matches = result.result
        py_files = [m for m in matches if m.endswith(".py")]
        assert len(py_files) >= 1

    def test_search_files_with_exclude(self, tool, tmp_path):
        (tmp_path / "a.py").write_text("x", encoding="utf-8")
        (tmp_path / "b.py").write_text("x", encoding="utf-8")
        result = tool.search_files(".", "*.py", exclude_patterns=["b.py"])
        assert result.success == 1
        matches = result.result
        assert not any("b.py" in m for m in matches)

    def test_search_files_dir_not_found(self, tool):
        result = tool.search_files("nonexistent", "*.py")
        assert result.success == 0

    def test_search_files_not_a_dir(self, tool, tmp_path):
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")
        result = tool.search_files("file.txt", "*.py")
        assert result.success == 0

    # --- available_tools ---

    def test_available_tools(self, tool):
        tools = tool.available_tools()
        assert len(tools) == 9
        names = {t.name for t in tools}
        assert "read_file" in names
        assert "write_file" in names
        assert "search_files" in names
