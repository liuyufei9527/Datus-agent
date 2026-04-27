# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""SQL context extraction from a :class:`RunResult`.

Walks ``result.new_items`` looking for ``function_call`` items whose
name is registered for the active database type, then pairs each call
with its ``function_call_output`` to assemble a :class:`SQLContext`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from datus.models.result import RunResult
from datus.schemas.node_models import SQLContext
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

DB_QUERY_FUNCTIONS: Dict[str, Set[str]] = {
    "snowflake": {"read_query", "list_tables", "describe_table"},
    DBType.SQLITE: {"read_query", "write_query", "list_tables", "describe_table"},
    "starrocks": {"read_query", "write_query", "table_overview", "db_overview"},
    DBType.DUCKDB: {"query"},
}


def get_function_call_names(db_type: str) -> Set[str]:
    return DB_QUERY_FUNCTIONS.get(db_type, set())


def extract_sql_contexts(result: RunResult, db_type: str = "snowflake") -> List[SQLContext]:
    """Extract :class:`SQLContext` records from *result.new_items*.

    Order matters: the call → its matching output → the (optional)
    assistant reflection message immediately after the output.
    """
    valid = get_function_call_names(db_type)
    items = list(result.to_input_list()) if hasattr(result, "to_input_list") else []
    contexts: List[SQLContext] = []

    for i, item in enumerate(items):
        if item.get("type") != "function_call" or item.get("name") not in valid:
            continue
        function_name = item.get("name", "")
        arguments = item.get("arguments", "{}")
        call_id = item.get("call_id")
        output: Optional[str] = None
        reflection: Optional[str] = None

        for j in range(i + 1, len(items)):
            forward = items[j]
            if forward.get("type") == "function_call_output" and forward.get("call_id") == call_id:
                output = forward.get("output", "")
                if j + 1 < len(items):
                    after = items[j + 1]
                    if after.get("type") == "function_call":
                        break
                    is_assistant_message = after.get("role") == "assistant" and after.get("content") is not None
                    if is_assistant_message:
                        content = after.get("content")
                        if isinstance(content, str):
                            reflection = content
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("text"):
                                    reflection = block["text"]
                                    break
                break

        try:
            contexts.append(
                SQLContext(
                    sql_query=f"{function_name}:{arguments}",
                    sql_return=output,
                    row_count=None,
                    reflection_explanation=reflection,
                )
            )
        except Exception as exc:  # noqa: BLE001 — schema mismatch should not abort the run
            logger.error("Failed to build SQLContext: %s", exc)
    return contexts
