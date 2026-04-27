# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""JSONL-backed conversation session.

Each session lives in a single ``<session_id>.jsonl`` file under
``{session_dir}/[{scope}/]``. One JSON object per line, persisted in
insertion order via :meth:`add_items`. Token usage (per-turn) is held
in memory only — historical aggregates live entirely in the active
``SQLiteSession`` instance and are dropped when the process exits.

Public API mirrors the previous ``AdvancedSQLiteSession`` shim so
:class:`AgentLoop` and the CLI's resume / rewind logic can keep their
call sites unchanged:

* ``add_items(items)`` — append items to the file
* ``get_items(limit=None)`` — read every item back (optionally truncated)
* ``clear_session()`` — wipe the file
* ``store_run_usage(...)`` — record token usage for one turn (memory only)
* ``extra_state`` — process-local key/value bag
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class SQLiteSession:
    """JSONL-backed session.

    Name retained from the previous SDK-shim era so caller imports stay
    stable; the underlying storage is a plain JSONL file (no SQLite).
    """

    def __init__(self, session_id: str, db_path: str, create_tables: bool = True) -> None:
        self.session_id = session_id
        self.db_path = str(db_path)
        # Translate ":memory:" / ".db" callsites into a JSONL on-disk file
        # transparently — old call sites picked the ``.db`` extension out of
        # historical habit; we keep the suffix but the bytes inside are JSONL.
        if self.db_path == ":memory:":
            # Truly in-memory mode: keep an in-process buffer instead of a file.
            self._is_memory = True
            self._memory_items: List[Dict[str, Any]] = []
        else:
            self._is_memory = False
            self._memory_items = []
            if not self.db_path.endswith(".jsonl"):
                # Translate legacy "<session>.db" path to "<session>.jsonl"
                # so the on-disk file always matches the new format. Caller
                # paths that already end in ``.jsonl`` are kept verbatim.
                if self.db_path.endswith(".db"):
                    self.db_path = self.db_path[:-3] + ".jsonl"
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            if create_tables:
                # ``touch`` so an empty session is detectable on disk.
                Path(self.db_path).touch(exist_ok=True)
        self._lock = asyncio.Lock()
        self._extra_state: Dict[str, Any] = {}
        # Per-session in-memory usage tally (turn_number → counters).
        self._turn_usage: Dict[int, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def extra_state(self) -> Dict[str, Any]:
        return self._extra_state

    async def add_items(self, items: Iterable[Dict[str, Any]]) -> None:
        items_list = [dict(item) for item in items if item is not None]
        if not items_list:
            return
        async with self._lock:
            if self._is_memory:
                self._memory_items.extend(items_list)
                return
            with open(self.db_path, "a", encoding="utf-8") as fh:
                for item in items_list:
                    fh.write(json.dumps(item, ensure_ascii=False, default=str))
                    fh.write("\n")

    async def get_items(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            if self._is_memory:
                items = list(self._memory_items)
            else:
                items = self._read_all()
        if limit is not None:
            items = items[-int(limit) :]
        return items

    async def clear_session(self) -> None:
        async with self._lock:
            if self._is_memory:
                self._memory_items.clear()
                self._turn_usage.clear()
                return
            try:
                with open(self.db_path, "w", encoding="utf-8") as fh:
                    fh.truncate(0)
            except FileNotFoundError:
                pass
            self._turn_usage.clear()

    async def store_run_usage(
        self,
        turn_number: int,
        *,
        requests: int = 1,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        """Record token usage for one turn (memory-only — not persisted)."""
        slot = self._turn_usage.setdefault(
            int(turn_number),
            {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
            },
        )
        slot["requests"] += int(requests)
        slot["input_tokens"] += int(input_tokens)
        slot["output_tokens"] += int(output_tokens)
        slot["total_tokens"] += int(total_tokens)
        slot["cached_tokens"] += int(cached_tokens)

    def get_turn_usage(self) -> Dict[int, Dict[str, int]]:
        """Return the in-memory turn-usage dictionary (caller may copy)."""
        return dict(self._turn_usage)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return []
        items: List[Dict[str, Any]] = []
        with open(self.db_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items


# ---------------------------------------------------------------------------
# Free helpers used by SessionManager
# ---------------------------------------------------------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read every JSON record from *path*; tolerate malformed / binary lines.

    Stale ``.db`` files left over from the openai-agents-sdk era are
    binary SQLite blobs; opening them as UTF-8 would raise. We swallow
    decode errors to keep the resume listing best-effort.
    """
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, default=str))
            fh.write("\n")


def copy_jsonl(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
