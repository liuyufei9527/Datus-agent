# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for ``datus.models.session``."""

import asyncio
import json
import os

import pytest

from datus.models.session import SQLiteSession, copy_jsonl, read_jsonl, write_jsonl


@pytest.fixture
def jsonl_session(tmp_path):
    return SQLiteSession(session_id="s1", db_path=str(tmp_path / "s1.jsonl"))


def test_legacy_db_path_translates_to_jsonl(tmp_path):
    legacy_path = str(tmp_path / "legacy.db")
    session = SQLiteSession(session_id="x", db_path=legacy_path)
    assert session.db_path.endswith(".jsonl")


def test_add_and_get_items_round_trip(jsonl_session):
    asyncio.run(jsonl_session.add_items([{"role": "user", "content": "hi"}]))
    items = asyncio.run(jsonl_session.get_items())
    assert items == [{"role": "user", "content": "hi"}]


def test_get_items_with_limit(jsonl_session):
    asyncio.run(jsonl_session.add_items([{"role": "user", "content": str(i)} for i in range(5)]))
    last_two = asyncio.run(jsonl_session.get_items(limit=2))
    assert [item["content"] for item in last_two] == ["3", "4"]


def test_clear_session_truncates_file(tmp_path, jsonl_session):
    asyncio.run(jsonl_session.add_items([{"role": "user", "content": "hi"}]))
    asyncio.run(jsonl_session.clear_session())
    assert asyncio.run(jsonl_session.get_items()) == []
    # File still exists but is empty.
    assert os.path.getsize(jsonl_session.db_path) == 0


def test_in_memory_session_is_ephemeral(tmp_path):
    session = SQLiteSession(session_id="m", db_path=":memory:")
    assert session._is_memory is True
    asyncio.run(session.add_items([{"role": "user", "content": "in-mem"}]))
    items = asyncio.run(session.get_items())
    assert items == [{"role": "user", "content": "in-mem"}]


def test_store_run_usage_accumulates(jsonl_session):
    asyncio.run(jsonl_session.store_run_usage(turn_number=1, input_tokens=10, output_tokens=5))
    asyncio.run(jsonl_session.store_run_usage(turn_number=1, input_tokens=2, output_tokens=3))
    usage = jsonl_session.get_turn_usage()
    assert usage[1]["input_tokens"] == 12
    assert usage[1]["output_tokens"] == 8


def test_extra_state_dict(jsonl_session):
    jsonl_session.extra_state["key"] = "value"
    assert jsonl_session.extra_state == {"key": "value"}


def test_read_jsonl_tolerates_malformed_lines(tmp_path):
    path = str(tmp_path / "f.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "user"}))
        fh.write("\n")
        fh.write("not-json\n")
        fh.write(json.dumps({"role": "assistant"}))
        fh.write("\n")
    items = read_jsonl(path)
    assert len(items) == 2


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    assert read_jsonl(str(tmp_path / "missing.jsonl")) == []


def test_read_jsonl_handles_binary_legacy_file(tmp_path):
    binary_path = tmp_path / "stale.jsonl"
    binary_path.write_bytes(b"\x00\x8aSQLite format 3\x00")
    # Should not raise; returns no parseable items.
    assert read_jsonl(str(binary_path)) == []


def test_write_and_copy_jsonl(tmp_path):
    src = str(tmp_path / "src.jsonl")
    dst = str(tmp_path / "dst.jsonl")
    write_jsonl(src, [{"a": 1}, {"b": 2}])
    copy_jsonl(src, dst)
    items = read_jsonl(dst)
    assert items == [{"a": 1}, {"b": 2}]
