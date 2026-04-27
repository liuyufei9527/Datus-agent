# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Session lifecycle / metadata helper.

In-house replacement for the previous SDK-shim version.  Sessions are
JSONL files under ``{session_dir}/[{scope}/]<session_id>.jsonl``; token
usage is held in memory on the active :class:`SQLiteSession` instance
and is not persisted across processes.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datus.models.session import SQLiteSession, copy_jsonl, read_jsonl, write_jsonl
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.async_utils import run_async
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger
from datus.utils.message_utils import extract_user_input

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.utils.path_manager import DatusPathManager


DEFAULT_CHAT_AGENT = "chat"


def extract_agent_from_session_id(session_id: str) -> str:
    """Return the agent name encoded in *session_id*.

    Datus session IDs follow ``{agent_name}_session_{uuid}``; legacy
    IDs without ``_session_`` resolve to the default chat agent so the
    UI can still surface them.
    """
    if "_session_" in session_id:
        return session_id.rsplit("_session_", 1)[0]
    return DEFAULT_CHAT_AGENT


def session_matches_agent(session_id: str, agent_name: Optional[str]) -> bool:
    target = agent_name or DEFAULT_CHAT_AGENT
    return extract_agent_from_session_id(session_id) == target


class SessionManager:
    """Thin manager over per-session JSONL files."""

    _SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
    # On-disk extension. The legacy `.db` suffix is still recognised when
    # listing existing sessions so users transitioning from the SDK era
    # don't lose track of files (the contents themselves are not migrated).
    _EXT = ".jsonl"
    _LEGACY_EXTS = (".db",)

    def __init__(
        self,
        session_dir: Optional[str] = None,
        scope: Optional[str] = None,
        *,
        path_manager: Optional["DatusPathManager"] = None,
        agent_config: Optional[Any] = None,
    ) -> None:
        if session_dir and str(session_dir).strip():
            resolved_dir = str(session_dir)
        else:
            from datus.utils.path_manager import get_path_manager

            resolved_dir = str(get_path_manager(path_manager=path_manager, agent_config=agent_config).sessions_dir)

        if scope and scope.strip():
            resolved_scope = scope.strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", resolved_scope):
                raise DatusException(
                    ErrorCode.COMMON_VALIDATION_FAILED,
                    message=(
                        f"Invalid scope: {resolved_scope!r}. Scope may only contain "
                        "alphanumerics, hyphens, and underscores."
                    ),
                )
            resolved_dir = os.path.join(resolved_dir, resolved_scope)
        os.makedirs(resolved_dir, exist_ok=True)
        self.session_dir = resolved_dir
        self._sessions: Dict[str, SQLiteSession] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _validate_session_id(cls, session_id: str) -> str:
        if not cls._SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(
                f"Invalid session ID: {session_id!r}. Allowed characters: alphanumerics, "
                "hyphens, underscores, and dots."
            )
        return session_id

    def _path(self, session_id: str) -> str:
        return os.path.join(self.session_dir, f"{session_id}{self._EXT}")

    def _resolve_existing_path(self, session_id: str) -> Optional[str]:
        """Return the on-disk path for *session_id* if any (jsonl or legacy)."""
        primary = self._path(session_id)
        if os.path.exists(primary):
            return primary
        for ext in self._LEGACY_EXTS:
            legacy = os.path.join(self.session_dir, f"{session_id}{ext}")
            if os.path.exists(legacy):
                return legacy
        return None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> SQLiteSession:
        self._validate_session_id(session_id)
        cached = self._sessions.get(session_id)
        if cached is not None:
            return cached
        session = SQLiteSession(session_id=session_id, db_path=self._path(session_id))
        self._sessions[session_id] = session
        return session

    def create_session(self, session_id: str) -> SQLiteSession:
        return self.get_session(session_id)

    def clear_session(self, session_id: str) -> None:
        if not self.session_exists(session_id):
            logger.warning("Attempted to clear non-existent session: %s", session_id)
            return
        session = self.get_session(session_id)
        run_async(session.clear_session())
        logger.debug("Cleared session: %s", session_id)

    def delete_session(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        self._sessions.pop(session_id, None)
        path = self._resolve_existing_path(session_id)
        if path is None:
            logger.warning("Attempted to delete non-existent session: %s", session_id)
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        # Tolerate stray sqlite WAL/SHM files left over from the SDK era.
        for suffix in ("-shm", "-wal"):
            aux = path + suffix
            if os.path.exists(aux):
                try:
                    os.remove(aux)
                except OSError:
                    pass
        logger.debug("Deleted session: %s", session_id)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_sessions(self, limit: Optional[int] = None, sort_by_modified: bool = False) -> List[str]:
        if not os.path.exists(self.session_dir):
            return []
        entries: List[tuple[str, float]] = []
        for filename in os.listdir(self.session_dir):
            stem, ext = os.path.splitext(filename)
            if ext != self._EXT and ext not in self._LEGACY_EXTS:
                continue
            full = os.path.join(self.session_dir, filename)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            entries.append((stem, mtime))
        if sort_by_modified:
            entries.sort(key=lambda item: item[1], reverse=True)
        seen: set[str] = set()
        ids: List[str] = []
        for sid, _ in entries:
            if sid in seen:
                continue
            seen.add(sid)
            ids.append(sid)
        if limit is not None:
            ids = ids[: int(limit)]
        return ids

    def session_exists(self, session_id: str) -> bool:
        self._validate_session_id(session_id)
        return self._resolve_existing_path(session_id) is not None

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        self._validate_session_id(session_id)
        path = self._resolve_existing_path(session_id)
        if path is None:
            return {"exists": False}

        info: Dict[str, Any] = {"exists": True, "session_id": session_id, "db_path": path}
        try:
            stat = os.stat(path)
            info["file_size"] = stat.st_size
            info["file_modified"] = stat.st_mtime
            info["created_at"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            info["updated_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except OSError as exc:
            logger.debug("stat failed for %s: %s", path, exc)

        items = read_jsonl(path)
        info["message_count"] = len(items)
        info["item_count"] = len(items)
        info["latest_message_at"] = info.get("updated_at")

        first_user, first_user_at = self._first_user_message(items, descending=False)
        latest_user, latest_user_at = self._first_user_message(items, descending=True)
        info["first_user_message"] = first_user
        info["first_user_message_at"] = first_user_at or info.get("created_at")
        info["latest_user_message"] = latest_user
        info["latest_user_message_at"] = latest_user_at or info.get("updated_at")

        # Token usage is in-memory only — surface the tally for the cached
        # session if present; otherwise return zero so the CLI status bar
        # never blows up.
        cached = self._sessions.get(session_id)
        if cached is not None:
            usage = cached.get_turn_usage()
            info["total_tokens"] = sum(slot.get("total_tokens", 0) for slot in usage.values())
        else:
            info["total_tokens"] = 0
        return info

    @staticmethod
    def _first_user_message(items: List[Dict[str, Any]], *, descending: bool) -> tuple[Optional[str], Optional[str]]:
        seq = reversed(items) if descending else items
        for item in seq:
            if item.get("role") == "user":
                return extract_user_input(item.get("content", "")), item.get("created_at")
        return None, None

    def get_detailed_usage(self, session_id: str) -> Dict[str, Any]:
        """Return memory-only token usage; zeros for sessions not in cache."""
        empty = {
            "total": {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
            },
            "turns": [],
            "turn_count": 0,
        }
        cached = self._sessions.get(session_id)
        if cached is None:
            return empty
        usage = cached.get_turn_usage()
        if not usage:
            return empty
        total = dict(empty["total"])
        turns: List[Dict[str, Any]] = []
        for turn_number in sorted(usage):
            slot = usage[turn_number]
            total["requests"] += slot.get("requests", 0)
            total["input_tokens"] += slot.get("input_tokens", 0)
            total["output_tokens"] += slot.get("output_tokens", 0)
            total["total_tokens"] += slot.get("total_tokens", 0)
            total["cached_tokens"] += slot.get("cached_tokens", 0)
            turns.append({"turn_number": turn_number, **slot})
        return {"total": total, "turns": turns, "turn_count": len(turns)}

    # ------------------------------------------------------------------
    # Copy / rewind
    # ------------------------------------------------------------------

    def copy_session(self, source_session_id: str, target_node_name: str) -> str:
        self._validate_session_id(source_session_id)
        new_session_id = f"{target_node_name}_session_{uuid.uuid4().hex[:8]}"
        src = self._resolve_existing_path(source_session_id)
        if src is None:
            return new_session_id  # nothing to copy; caller will start fresh
        new_path = self._path(new_session_id)
        copy_jsonl(src, new_path)
        self._sessions[new_session_id] = SQLiteSession(session_id=new_session_id, db_path=new_path, create_tables=False)
        logger.info(
            "Copied session %s -> %s (%d bytes)",
            source_session_id,
            new_session_id,
            os.path.getsize(new_path),
        )
        return new_session_id

    def rewind_session(
        self,
        source_session_id: str,
        up_to_user_turn: int,
        include_assistant_response: bool = True,
    ) -> str:
        self._validate_session_id(source_session_id)
        if up_to_user_turn < 1:
            raise ValueError("up_to_user_turn must be >= 1")
        node_type = extract_agent_from_session_id(source_session_id)
        new_session_id = f"{node_type}_session_{uuid.uuid4().hex[:8]}"
        src = self._resolve_existing_path(source_session_id)
        if src is None:
            raise FileNotFoundError(f"Source session not found: {source_session_id}")

        items = read_jsonl(src)
        user_count = 0
        cutoff = len(items)
        for idx, item in enumerate(items):
            if item.get("role") == "user":
                user_count += 1
                if user_count > up_to_user_turn:
                    cutoff = idx
                    break

        if not include_assistant_response and user_count >= up_to_user_turn:
            target = 0
            for idx, item in enumerate(items):
                if item.get("role") == "user":
                    target += 1
                    if target == up_to_user_turn:
                        cutoff = idx + 1
                        break

        kept = items[:cutoff]
        if not kept:
            raise ValueError(f"No messages to keep for turn {up_to_user_turn}")

        new_path = self._path(new_session_id)
        write_jsonl(new_path, kept)
        self._sessions[new_session_id] = SQLiteSession(session_id=new_session_id, db_path=new_path, create_tables=False)
        logger.info(
            "Rewound %s @ turn %d -> %s (%d items kept)",
            source_session_id,
            up_to_user_turn,
            new_session_id,
            len(kept),
        )
        return new_session_id

    # ------------------------------------------------------------------
    # Resume rendering
    # ------------------------------------------------------------------

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if not self._SESSION_ID_RE.fullmatch(session_id):
            logger.warning("Invalid session_id format: %s", session_id)
            return []

        path = self._resolve_existing_path(session_id)
        if path is None:
            return []
        # Prevent path traversal: ensure the resolved file lives inside
        # ``session_dir`` (paranoid, since we built the path ourselves).
        sessions_root = Path(self.session_dir).resolve()
        try:
            Path(path).resolve().relative_to(sessions_root)
        except ValueError:
            logger.warning("Session path traversal attempt: %s", path)
            return []

        rows = read_jsonl(path)
        out: List[Dict[str, Any]] = []
        current_assistant: Optional[Dict[str, Any]] = None
        current_actions: List[ActionHistory] = []
        progress: List[str] = []

        for msg in rows:
            role = msg.get("role")
            msg_type = msg.get("type")
            created_at = msg.get("created_at") or msg.get("timestamp")

            if role == "user":
                if current_assistant is not None:
                    self._finalize_assistant(out, current_assistant, current_actions, progress)
                    current_assistant = None
                    current_actions = []
                    progress = []
                content = extract_user_input(msg.get("content", ""))
                out.append({"role": "user", "content": content, "timestamp": created_at, "created_at": created_at})
                continue

            if msg_type == "function_call":
                if current_assistant is None:
                    current_assistant = self._new_assistant_group(created_at)
                tool_name = msg.get("name", "unknown")
                arguments = msg.get("arguments", "{}")
                try:
                    args_dict = json.loads(arguments) if arguments else {}
                    progress.append(f"✓ Tool call: {tool_name}({str(args_dict)[:60]})")
                except (json.JSONDecodeError, TypeError, ValueError):
                    progress.append(f"✓ Tool call: {tool_name}")
                current_actions.append(
                    ActionHistory(
                        action_id=msg.get("call_id", str(uuid.uuid4())),
                        role=ActionRole.TOOL,
                        messages=f"Tool call: {tool_name}",
                        action_type=tool_name,
                        input={"function_name": tool_name, "arguments": arguments},
                        output=None,
                        status=ActionStatus.PROCESSING,
                        start_time=_parse_ts(created_at),
                    )
                )
                continue

            if msg_type == "function_call_output":
                if not current_actions:
                    continue
                target = self._match_pending_call(current_actions, msg.get("call_id"))
                output_text = msg.get("output", "")
                output_data: Any = {}
                if output_text:
                    try:
                        output_data = json.loads(output_text)
                    except (TypeError, json.JSONDecodeError):
                        output_data = {"result": output_text}
                target_id = target.action_id if target else str(uuid.uuid4())
                current_actions.append(
                    ActionHistory(
                        action_id="complete_" + target_id,
                        role=ActionRole.TOOL,
                        messages=f"Tool result: {target.action_type if target else 'tool'}",
                        action_type=target.action_type if target else "tool",
                        input=target.input if target else None,
                        output=output_data,
                        status=ActionStatus.SUCCESS,
                        start_time=target.start_time if target else _parse_ts(created_at),
                        end_time=_parse_ts(created_at),
                    )
                )
                if target is not None:
                    target.status = ActionStatus.SUCCESS
                    target.end_time = _parse_ts(created_at)
                continue

            if role == "assistant":
                if current_assistant is None:
                    current_assistant = self._new_assistant_group(created_at)
                content = msg.get("content")
                texts: List[str] = []
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text") or item.get("output_text")
                            if text:
                                texts.append(text)
                for text in texts:
                    progress.append(f"💭Thinking: {text}")
                    current_actions.append(
                        ActionHistory(
                            action_id=msg.get("id", str(uuid.uuid4())),
                            role=ActionRole.ASSISTANT,
                            messages=text,
                            action_type="thinking",
                            input=None,
                            output={"raw_output": text},
                            status=ActionStatus.SUCCESS,
                            start_time=_parse_ts(created_at),
                            end_time=_parse_ts(created_at),
                        )
                    )

        if current_assistant is not None:
            self._finalize_assistant(out, current_assistant, current_actions, progress)

        return out

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def close_all_sessions(self) -> None:
        for sid in list(self._sessions.keys()):
            self._sessions.pop(sid, None)
            logger.debug("Closed session: %s", sid)

    @staticmethod
    def _new_assistant_group(created_at: Optional[str]) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "timestamp": created_at,
            "created_at": created_at,
        }

    @staticmethod
    def _match_pending_call(actions: List[ActionHistory], call_id: Optional[str]) -> Optional[ActionHistory]:
        if call_id:
            for candidate in reversed(actions):
                if candidate.action_id == call_id and candidate.status == ActionStatus.PROCESSING:
                    return candidate
        for candidate in reversed(actions):
            if candidate.role == ActionRole.TOOL and candidate.status == ActionStatus.PROCESSING:
                return candidate
        return None

    @classmethod
    def _finalize_assistant(
        cls,
        out: List[Dict[str, Any]],
        group: Dict[str, Any],
        actions: List[ActionHistory],
        progress: List[str],
    ) -> None:
        final = cls._parse_final_output(actions, group)
        if final is not None:
            actions = list(actions) + [final]
        if not group.get("content"):
            group["content"] = "Processing completed"
        if progress:
            group["progress_messages"] = list(progress)
        if actions:
            group["actions"] = list(actions)
        out.append(group)

    @staticmethod
    def _parse_final_output(actions: List[ActionHistory], group: Dict[str, Any]) -> Optional[ActionHistory]:
        last_assistant = None
        for action in reversed(actions):
            if action.role == ActionRole.ASSISTANT:
                last_assistant = action
                break
        if last_assistant is None or not last_assistant.messages:
            return None
        result_json = llm_result2json(last_assistant.messages)
        if isinstance(result_json, str):
            group["content"] = result_json
            return None
        if isinstance(result_json, dict) and (
            "sql" in result_json or "output" in result_json or "response" in result_json
        ):
            content_value = result_json.get("response") or result_json.get("output", "")
            group["content"] = content_value
            group["sql"] = result_json.get("sql", "")
            return ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="chat_response",
                messages="Chat interaction completed successfully",
                input_data={},
                output_data={"sql": result_json.get("sql", ""), "response": content_value},
                status=ActionStatus.SUCCESS,
            )
        group["content"] = last_assistant.messages
        return None


def _parse_ts(raw: Optional[str]) -> datetime:
    if not raw:
        return datetime.now()
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now()
