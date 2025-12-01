# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, Optional, Tuple

from datus.utils.loggings import get_logger

from .prompt_manager import prompt_manager

logger = get_logger(__name__)


def _convert_sql_result_to_string(result_value: Any) -> Optional[str]:
    """
    Convert SQL result to string format.
    Handles DataFrame, Arrow table, and other types.

    Args:
        result_value: SQL result value (can be str, DataFrame, Arrow table, etc.)

    Returns:
        String representation of the result, or None if empty/invalid
    """
    if result_value is None:
        return None

    # If already a string, return it
    if isinstance(result_value, str):
        return result_value.strip() if result_value.strip() else None

    # If it's a DataFrame or has to_csv method, convert to CSV
    if hasattr(result_value, "to_csv"):
        try:
            csv_str = result_value.to_csv(index=False)
            return csv_str.strip() if csv_str.strip() else None
        except Exception:
            pass

    # Try to convert to string
    try:
        str_value = str(result_value)
        return str_value.strip() if str_value.strip() else None
    except Exception:
        return None


def _extract_sql_result_from_candidate(candidate: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract SQL query and SQL result from a candidate result.
    Handles nested structures like SubworkflowResult.

    Returns:
        Tuple of (sql_query, sql_result) or (None, None) if not found
    """
    sql_query = None
    sql_result = None

    if not candidate:
        return None, None

    # Handle dict type candidate
    if isinstance(candidate, dict):
        result_obj = candidate.get("result")
        if result_obj:
            # Check if it's a SubworkflowResult with node_results
            if hasattr(result_obj, "node_results") and result_obj.node_results:
                # Look through node_results to find SQL result
                for node_id, node_result in result_obj.node_results.items():
                    if isinstance(node_result, dict):
                        node_result_obj = node_result.get("result")
                        if node_result_obj:
                            # Try to get sql_result_final, sql_result, or sql_return
                            for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                                if hasattr(node_result_obj, field_name):
                                    result_value = getattr(node_result_obj, field_name)
                                    if result_value is not None:
                                        converted_result = _convert_sql_result_to_string(result_value)
                                        if converted_result:
                                            sql_result = converted_result
                                            break
                                elif isinstance(node_result_obj, dict) and field_name in node_result_obj:
                                    result_value = node_result_obj[field_name]
                                    if result_value is not None:
                                        converted_result = _convert_sql_result_to_string(result_value)
                                        if converted_result:
                                            sql_result = converted_result
                                            break

                            # Try to get SQL query
                            for field_name in ["sql_query_final", "gen_sql_final", "sql_query", "gen_sql"]:
                                if hasattr(node_result_obj, field_name):
                                    query_value = getattr(node_result_obj, field_name)
                                    if isinstance(query_value, str) and query_value.strip():
                                        sql_query = query_value.strip()
                                        break
                                elif isinstance(node_result_obj, dict) and field_name in node_result_obj:
                                    query_value = node_result_obj[field_name]
                                    if isinstance(query_value, str) and query_value.strip():
                                        sql_query = query_value.strip()
                                        break

                            if sql_result:
                                break
                    elif hasattr(node_result, "sql_result_final") or hasattr(node_result, "sql_result") or hasattr(
                            node_result, "sql_return"):
                        for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                            if hasattr(node_result, field_name):
                                result_value = getattr(node_result, field_name)
                                if result_value is not None:
                                    converted_result = _convert_sql_result_to_string(result_value)
                                    if converted_result:
                                        sql_result = converted_result
                                        break
                        if sql_result:
                            break

            # Try direct attributes
            for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                if hasattr(result_obj, field_name):
                    result_value = getattr(result_obj, field_name)
                    if result_value is not None:
                        converted_result = _convert_sql_result_to_string(result_value)
                        if converted_result:
                            sql_result = converted_result
                            break

    # Handle object type candidate
    elif hasattr(candidate, "__dict__"):
        if hasattr(candidate, "result"):
            result_obj = getattr(candidate, "result")
            if result_obj and hasattr(result_obj, "node_results") and result_obj.node_results:
                # Similar logic as above for object type
                for node_id, node_result in result_obj.node_results.items():
                    if isinstance(node_result, dict):
                        node_result_obj = node_result.get("result")
                        if node_result_obj:
                            for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                                if hasattr(node_result_obj, field_name):
                                    result_value = getattr(node_result_obj, field_name)
                                    if result_value is not None:
                                        converted_result = _convert_sql_result_to_string(result_value)
                                        if converted_result:
                                            sql_result = converted_result
                                            break
                            if sql_result:
                                break
                    elif hasattr(node_result, "sql_result_final") or hasattr(node_result, "sql_result") or hasattr(
                            node_result, "sql_return"):
                        for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                            if hasattr(node_result, field_name):
                                result_value = getattr(node_result, field_name)
                                if result_value is not None:
                                    converted_result = _convert_sql_result_to_string(result_value)
                                    if converted_result:
                                        sql_result = converted_result
                                        break
                        if sql_result:
                            break

    return sql_query, sql_result


def create_selection_prompt(candidates: Dict[str, Any], prompt_version: str = "1.0", max_text_length: int = 500) -> str:
    """
    Create prompt for LLM-based candidate selection.

    Args:
        candidates: Dictionary of candidate results to analyze
        prompt_version: Version of the prompt template to use. If "1.0" (default),
                        automatically selects the latest version from user template directory.
        max_text_length: Maximum length for text fields to avoid overly long prompts

    Returns:
        Formatted prompt string for candidate selection
    """
    # Auto-select latest version if using default "1.0"
    # This allows users to put newer templates in their template directory
    # without needing to modify the schema
    if prompt_version == "1.0":
        try:
            # Check if there are newer versions in user template directory
            versions = prompt_manager.list_template_versions("selection_analysis")
            if versions:
                # Use the latest version (last in sorted list)
                prompt_version = None  # None means auto-select latest
                logger.info(f"Auto-selecting latest template version from available versions: {versions}")
        except Exception as e:
            logger.debug(f"Could not auto-select latest version, using default 1.0: {e}")
            prompt_version = "1.0"

    # Process candidates to extract SQL results and flatten structure
    processed_candidates = {}

    for candidate_id, candidate in candidates.items():
        processed_candidate = candidate.copy() if isinstance(candidate, dict) else {}

        if isinstance(candidate, dict):
            # Extract SQL query and result from nested structure
            sql_query, sql_result = _extract_sql_result_from_candidate(candidate)

            # Create a flattened structure for the template
            processed_candidate["success"] = candidate.get("success", False)
            processed_candidate["status"] = candidate.get("status", "unknown")
            processed_candidate["error"] = candidate.get("error")

            # Add extracted SQL info
            if sql_query:
                # Truncate if needed
                if len(sql_query) > max_text_length:
                    sql_query = sql_query[:max_text_length] + "\n... (truncated)"
                processed_candidate["sql_query"] = sql_query

            if sql_result:
                # Truncate if needed
                if len(sql_result) > max_text_length:
                    sql_result = sql_result[:max_text_length] + "\n... (truncated)"
                processed_candidate["sql_result"] = sql_result
                processed_candidate["sql_result_final"] = sql_result  # Also set for template compatibility

        processed_candidates[candidate_id] = processed_candidate

    # Render the template (prompt_version can be None for auto-select)
    prompt = prompt_manager.render_template(
        "selection_analysis", candidates=processed_candidates, version=prompt_version
    )

    logger.info(
        f"Generated selection prompt for {len(candidates)} candidates using version: {prompt_version or 'latest'}")
    return prompt
