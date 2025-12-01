# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.prompts.selection import create_selection_prompt
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.parallel_node_models import SelectionResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SelectionNode(Node):
    """Node that selects the best result from multiple parallel candidates"""

    def update_context(self, workflow: Workflow) -> Dict:
        """Update workflow context with selected result"""
        if self.result and self.result.success and self.result.selected_result:
            # Update context with the selected result
            workflow.context.update_selection_result(
                self.result.selected_result,
                {
                    "selected_source": self.result.selected_source,
                    "selection_reason": self.result.selection_reason,
                    "score_analysis": self.result.score_analysis,
                },
            )

            # Extract and add SQL context from the selected result
            selected_data = self.result.selected_result
            sql_query = None
            sql_result = None
            explanation = ""
            sql_error = None
            row_count = None

            if isinstance(selected_data, dict) and "result" in selected_data:
                # Get the actual result object from the child node result
                child_result = selected_data["result"]

                # Handle SubworkflowResult with node_results
                if hasattr(child_result, "node_results") and child_result.node_results:
                    # Look through node_results to find SQL query and result
                    for node_id, node_result in child_result.node_results.items():
                        if isinstance(node_result, dict):
                            node_result_obj = node_result.get("result")
                            if node_result_obj:
                                # Try to get SQL query from GenSQLNodeResult
                                if hasattr(node_result_obj, "sql_query") or hasattr(node_result_obj, "gen_sql"):
                                    sql_query = getattr(node_result_obj, "sql_query", None) or getattr(node_result_obj,
                                                                                                       "gen_sql", None)
                                    explanation = getattr(node_result_obj, "explanation", "") or ""

                                # Try to get SQL result from ExecuteSQLNodeResult or OutputNodeResult
                                if hasattr(node_result_obj, "sql_result_final"):
                                    sql_result = getattr(node_result_obj, "sql_result_final")
                                elif hasattr(node_result_obj, "sql_result"):
                                    sql_result = getattr(node_result_obj, "sql_result")
                                elif hasattr(node_result_obj, "sql_return"):
                                    sql_return = getattr(node_result_obj, "sql_return")
                                    # Convert to string if needed
                                    if sql_return is not None:
                                        if hasattr(sql_return, "to_csv"):
                                            sql_result = sql_return.to_csv(index=False)
                                        else:
                                            sql_result = str(sql_return)

                                # Get row_count and error if available
                                if hasattr(node_result_obj, "row_count"):
                                    row_count = getattr(node_result_obj, "row_count")
                                if hasattr(node_result_obj, "error"):
                                    sql_error = getattr(node_result_obj, "error")

                                # If we found both query and result, break
                                if sql_query and sql_result is not None:
                                    break
                        elif hasattr(node_result, "sql_query") or hasattr(node_result, "gen_sql"):
                            # Direct node result object
                            sql_query = getattr(node_result, "sql_query", None) or getattr(node_result, "gen_sql", None)
                            explanation = getattr(node_result, "explanation", "") or ""
                            if hasattr(node_result, "sql_result_final"):
                                sql_result = getattr(node_result, "sql_result_final")
                            elif hasattr(node_result, "sql_result"):
                                sql_result = getattr(node_result, "sql_result")
                            elif hasattr(node_result, "sql_return"):
                                sql_return = getattr(node_result, "sql_return")
                                if sql_return is not None:
                                    if hasattr(sql_return, "to_csv"):
                                        sql_result = sql_return.to_csv(index=False)
                                    else:
                                        sql_result = str(sql_return)
                            if hasattr(node_result, "row_count"):
                                row_count = getattr(node_result, "row_count")
                            if hasattr(node_result, "error"):
                                sql_error = getattr(node_result, "error")
                            if sql_query and sql_result is not None:
                                break

                # Fallback: Check if it's a direct SQL generation result
                elif hasattr(child_result, "sql_query") or hasattr(child_result, "gen_sql"):
                    sql_query = getattr(child_result, "sql_query", None) or getattr(child_result, "gen_sql", None)
                    explanation = getattr(child_result, "explanation", "") or ""
                    if hasattr(child_result, "sql_result_final"):
                        sql_result = getattr(child_result, "sql_result_final")
                    elif hasattr(child_result, "sql_result"):
                        sql_result = getattr(child_result, "sql_result")
                    elif hasattr(child_result, "sql_return"):
                        sql_return = getattr(child_result, "sql_return")
                        if sql_return is not None:
                            if hasattr(sql_return, "to_csv"):
                                sql_result = sql_return.to_csv(index=False)
                            else:
                                sql_result = str(sql_return)
                    if hasattr(child_result, "row_count"):
                        row_count = getattr(child_result, "row_count")
                    if hasattr(child_result, "error"):
                        sql_error = getattr(child_result, "error")

            # Create SQL context if we have SQL query
            if sql_query:
                from datus.schemas.node_models import SQLContext

                # Convert sql_result to string if it's not already
                sql_result_str = ""
                if sql_result is not None:
                    if isinstance(sql_result, str):
                        sql_result_str = sql_result
                    elif hasattr(sql_result, "to_csv"):
                        sql_result_str = sql_result.to_csv(index=False)
                    else:
                        sql_result_str = str(sql_result)

                # Create SQL context from the selected result
                sql_context = SQLContext(
                    sql_query=sql_query,
                    explanation=explanation,
                    sql_return=sql_result_str if sql_result_str else None,
                    sql_error=sql_error,
                    row_count=row_count,
                )

                # Add to workflow context
                workflow.context.sql_contexts.append(sql_context)
                logger.info(
                    f"Added selected SQL to context: {sql_query[:100]}..., "
                    f"result length: {len(sql_result_str) if sql_result_str else 0}"
                )

                return {"success": True, "message": "Selection node context updated with SQL"}

        return {"success": True, "message": "Selection node context updated"}

    def setup_input(self, workflow: Workflow) -> Dict:
        """Setup input for selection node"""
        logger.info(
            f"SelectionNode.setup_input called: current_index={workflow.current_node_index}, "
            f"node_order={workflow.node_order}"
        )

        # Get parallel results from workflow context
        parallel_results = workflow.context.parallel_results
        logger.info(f"SelectionNode.setup_input: parallel_results available = {parallel_results is not None}")

        if not parallel_results:
            # If no parallel results, look for any completed reasoning node
            reasoning_node = None
            # Check all nodes for a reasoning node that has completed successfully
            for i, node_id in enumerate(workflow.node_order):
                node = workflow.nodes.get(node_id)
                logger.info(
                    f"SelectionNode.setup_input: checking node at index {i}: node_id={node_id}, "
                    f"node_type={node.type if node else None}, status={node.status if node else None}"
                )
                if (
                        node
                        and node.type == "reasoning"
                        and node.status == "completed"
                        and node.result
                        and node.result.success
                ):
                    reasoning_node = node
                    logger.info(f"Found completed reasoning node: {node_id}")
                    break

            if reasoning_node:
                logger.info(f"No parallel results found, using reasoning node {reasoning_node.id} result as candidate")
                # Create a single candidate result from the reasoning node
                reasoning_result = {
                    "reasoning_node": {
                        "success": True,
                        "result": reasoning_node.result,
                        "node_id": reasoning_node.id,
                        "node_type": reasoning_node.type,
                    }
                }
                parallel_results = reasoning_result
                logger.info(
                    f"SelectionNode.setup_input: created reasoning_result candidate with {len(parallel_results)} "
                    f"entries"
                )
            else:
                logger.error("SelectionNode.setup_input: No completed reasoning node found with successful result")
                return {"success": False, "message": "No parallel results available in workflow context"}

        # Create or update the SelectionInput with parallel results
        from datus.schemas.parallel_node_models import SelectionInput

        # If input already exists and is SelectionInput, update it; otherwise create new
        if isinstance(self.input, SelectionInput):
            self.input.candidate_results = parallel_results
        else:
            self.input = SelectionInput(
                candidate_results=parallel_results, selection_criteria="best_quality"
            )

        logger.info(f"Setup selection input with {len(parallel_results)} candidates")
        return {"success": True, "message": "Selection node input setup complete"}

    def execute(self) -> SelectionResult:
        """Select the best result from candidates"""
        if not self.input or not self.input.candidate_results:
            result = SelectionResult(
                success=False,
                error="No candidate results provided for selection",
                selected_result=None,
                selected_source="",
                selection_reason="No candidates available",
                all_candidates={},
            )
            self.result = result
            return result

        candidates = self.input.candidate_results

        logger.info(f"Starting selection from {len(candidates)} candidates")

        try:
            # If only one candidate, select it directly
            if len(candidates) == 1:
                candidate_id = list(candidates.keys())[0]
                candidate_result = candidates[candidate_id]

                result = SelectionResult(
                    success=True,
                    selected_result=candidate_result,
                    selected_source=candidate_id,
                    selection_reason="Only one candidate available",
                    all_candidates=candidates,
                    score_analysis={candidate_id: {"score": 10, "reason": "Single candidate"}},
                )
                self.result = result
                return result

            # Try smart selection first (for multiple candidates)
            smart_result = self._smart_selection(candidates)
            if smart_result:
                logger.info("Smart selection succeeded, using smart selection result")
                self.result = smart_result
                return smart_result

            # If smart selection cannot decide, fall back to LLM or rule-based selection
            logger.info("Smart selection could not decide, falling back to LLM/rule-based selection")
            if self.model:
                result = self._llm_based_selection(candidates)
            else:
                # Fallback to rule-based selection
                result = self._rule_based_selection(candidates)

            self.result = result
            return result

        except Exception as e:
            logger.error(f"Selection failed: {str(e)}")
            result = SelectionResult(
                success=False,
                error=f"Selection failed: {str(e)}",
                selected_result=None,
                selected_source="",
                selection_reason="Selection process failed",
                all_candidates=candidates,
            )
            self.result = result
            return result

    def _extract_result_from_candidate(self, candidate: Any) -> Optional[str]:
        """
        Extract SQL result (CSV content) from a candidate result.

        Supports multiple possible field names and nested result objects.
        Also handles SubworkflowResult which contains node_results.
        Now also supports sql_return field (used by ExecuteSQLResult).

        Args:
            candidate: Candidate result (can be dict or object)

        Returns:
            Result string (CSV content) if found, None otherwise
        """
        if not candidate:
            return None

        # Helper function to convert result value to string
        def _convert_to_string(value):
            """Convert result value to string, handling DataFrame and other types"""
            if value is None:
                return None
            if isinstance(value, str):
                return value.strip() if value.strip() else None
            # Handle DataFrame or objects with to_csv method
            if hasattr(value, "to_csv"):
                try:
                    csv_str = value.to_csv(index=False)
                    return csv_str.strip() if csv_str.strip() else None
                except Exception:
                    pass
            # Try to convert to string
            try:
                str_value = str(value)
                return str_value.strip() if str_value.strip() else None
            except Exception:
                return None

        # Handle dict type candidate
        if isinstance(candidate, dict):
            # Try to get result from candidate directly
            for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                if field_name in candidate and candidate[field_name]:
                    result = _convert_to_string(candidate[field_name])
                    if result:
                        return result

            # Try to get result from nested result object
            result_obj = candidate.get("result")
            if result_obj:
                # Check if it's a SubworkflowResult with node_results
                if hasattr(result_obj, "node_results") and result_obj.node_results:
                    # Look through node_results to find SQL result
                    for node_id, node_result in result_obj.node_results.items():
                        # node_result might be a dict with 'result' key
                        if isinstance(node_result, dict):
                            node_result_obj = node_result.get("result")
                            if node_result_obj:
                                for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                                    if hasattr(node_result_obj, field_name):
                                        result = _convert_to_string(getattr(node_result_obj, field_name))
                                        if result:
                                            return result
                                    elif isinstance(node_result_obj, dict) and field_name in node_result_obj:
                                        result = _convert_to_string(node_result_obj[field_name])
                                        if result:
                                            return result
                        # Or node_result might be an object directly
                        elif hasattr(node_result, "sql_result_final") or hasattr(node_result, "sql_result") or hasattr(
                                node_result, "sql_return"):
                            for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                                if hasattr(node_result, field_name):
                                    result = _convert_to_string(getattr(node_result, field_name))
                                    if result:
                                        return result

                # If result is a dict
                if isinstance(result_obj, dict):
                    for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                        if field_name in result_obj and result_obj[field_name]:
                            result = _convert_to_string(result_obj[field_name])
                            if result:
                                return result
                # If result is an object with attributes
                elif hasattr(result_obj, "__dict__"):
                    for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                        if hasattr(result_obj, field_name):
                            result = _convert_to_string(getattr(result_obj, field_name))
                            if result:
                                return result
                # If result is an object, try direct attribute access
                else:
                    for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                        if hasattr(result_obj, field_name):
                            result = _convert_to_string(getattr(result_obj, field_name))
                            if result:
                                return result

        # Handle object type candidate
        elif hasattr(candidate, "__dict__"):
            # Check if it's a SubworkflowResult with node_results
            if hasattr(candidate, "node_results") and candidate.node_results:
                # Look through node_results to find SQL result
                for node_id, node_result in candidate.node_results.items():
                    # node_result might be a dict with 'result' key
                    if isinstance(node_result, dict):
                        node_result_obj = node_result.get("result")
                        if node_result_obj:
                            for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                                if hasattr(node_result_obj, field_name):
                                    result = _convert_to_string(getattr(node_result_obj, field_name))
                                    if result:
                                        return result
                                elif isinstance(node_result_obj, dict) and field_name in node_result_obj:
                                    result = _convert_to_string(node_result_obj[field_name])
                                    if result:
                                        return result
                    # Or node_result might be an object directly
                    elif hasattr(node_result, "sql_result_final") or hasattr(node_result, "sql_result") or hasattr(
                            node_result, "sql_return"):
                        for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                            if hasattr(node_result, field_name):
                                result = _convert_to_string(getattr(node_result, field_name))
                                if result:
                                    return result

            # Try direct attributes
            for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                if hasattr(candidate, field_name):
                    result = _convert_to_string(getattr(candidate, field_name))
                    if result:
                        return result

            # Try nested result attribute
            if hasattr(candidate, "result"):
                result_obj = getattr(candidate, "result")
                if result_obj:
                    for field_name in ["sql_result_final", "sql_result", "sql_return"]:
                        if hasattr(result_obj, field_name):
                            result = _convert_to_string(getattr(result_obj, field_name))
                            if result:
                                return result

        return None

    def _normalize_result(self, result: str) -> str:
        """
        Normalize result string (CSV content) for comparison.

        Removes extra whitespace, converts to lowercase, and sorts CSV rows
        to handle cases where rows are in different orders but contain the same data.

        IMPORTANT: Column names (header) are ignored in comparison - only data rows are compared.
        However, the number of columns must match. Results with different column names but same
        data and same number of columns will be considered identical.

        Args:
            result: Result string (CSV content) to normalize

        Returns:
            Normalized result string: "column_count|sorted_data_rows"
            Format ensures that:
            - Different column counts -> different results
            - Same column count, different data -> different results
            - Same column count, same data (different column names) -> same results
        """
        if not result:
            return ""

        # Split into lines
        lines = result.strip().split('\n')
        if not lines:
            return ""

        # First line is usually the header - we'll ignore it for comparison
        # Extract data lines (skip header)
        data_lines = lines[1:] if len(lines) > 1 else []

        # If there's no header line, treat all lines as data
        if len(lines) == 1 or not data_lines:
            # If only one line, treat it as data
            if len(lines) == 1:
                data_lines = lines
            else:
                # If we have multiple lines but no data after header, use empty
                data_lines = []

        if not data_lines:
            return "0|"  # No data rows

        # Determine column count from first data row
        # Try both comma and tab as separators
        first_line = data_lines[0].strip()
        if ',' in first_line:
            column_count = len([col.strip() for col in first_line.split(',') if col.strip()])
        elif '\t' in first_line:
            column_count = len([col.strip() for col in first_line.split('\t') if col.strip()])
        else:
            # If no separator found, treat as single column
            column_count = 1

        # Normalize and sort data lines (ignore column names/header)
        normalized_data_lines = []
        for line in data_lines:
            if line.strip():  # Skip empty lines
                # Remove extra whitespace and convert to lowercase
                normalized_line = re.sub(r'\s+', ' ', line.strip()).lower()
                normalized_data_lines.append(normalized_line)

        # Sort data lines to ensure consistent comparison regardless of row order
        normalized_data_lines.sort()

        # Return format: "column_count|sorted_data_rows"
        # This ensures column count is part of comparison, but column names are not
        normalized = f"{column_count}|" + '\n'.join(normalized_data_lines)

        return normalized

    def _smart_selection(self, candidates: Dict[str, Any]) -> Optional[SelectionResult]:
        """
        Smart selection logic with priority:
        1. If two or more candidates have the same result (CSV content), select the first successful one among them
        2. If only one candidate has actual results (CSV content), select it (even if others are marked as successful)
        3. If only one candidate has successful execution, select it
        4. Otherwise return None to let LLM decide

        Args:
            candidates: Dictionary of candidate results

        Returns:
            SelectionResult if smart selection can decide, None otherwise
        """
        logger.info("Attempting smart selection based on result similarity and success status")

        # Extract result content and success status for each candidate
        candidate_result_map = {}  # candidate_id -> normalized_result
        candidate_success_map = {}  # candidate_id -> success (bool)

        for candidate_id, candidate in candidates.items():
            result_content = self._extract_result_from_candidate(candidate)
            normalized_result = self._normalize_result(result_content) if result_content else None

            # Check success status
            success = False
            if isinstance(candidate, dict):
                success = candidate.get("success", False)
            elif hasattr(candidate, "success"):
                success = candidate.success

            candidate_result_map[candidate_id] = normalized_result
            candidate_success_map[candidate_id] = success

            # Log detailed information for debugging
            result_preview = ""
            if normalized_result:
                result_preview = normalized_result[:200] + "..." if len(normalized_result) > 200 else normalized_result
            else:
                result_preview = "None or empty"

            logger.info(
                f"Candidate {candidate_id}: Success={success}, "
                f"Result={'present' if normalized_result else 'missing'}, "
                f"Result length={len(normalized_result) if normalized_result else 0}, "
                f"Result preview={result_preview[:100]}"
            )

        # Count result occurrences
        result_count_map = {}  # normalized_result -> list of candidate_ids
        for candidate_id, normalized_result in candidate_result_map.items():
            if normalized_result:  # Only count non-empty results
                if normalized_result not in result_count_map:
                    result_count_map[normalized_result] = []
                result_count_map[normalized_result].append(candidate_id)

        # Log result count map for debugging
        logger.info(f"Result count map: {len(result_count_map)} unique results found")
        for result_hash, candidate_ids in result_count_map.items():
            logger.info(
                f"  Result (hash={hash(result_hash) % 10000}): "
                f"appears in {len(candidate_ids)} candidate(s): {candidate_ids}, "
                f"preview={result_hash[:100] if result_hash else 'empty'}"
            )

        # Find results that appear 2 or more times (same result)
        same_result_groups = {
            result: candidate_ids
            for result, candidate_ids in result_count_map.items()
            if len(candidate_ids) >= 2
        }

        logger.info(f"Same result groups found: {len(same_result_groups)}")
        for result_hash, candidate_ids in same_result_groups.items():
            logger.info(
                f"  Same result group: {len(candidate_ids)} candidates with same result: {candidate_ids}"
            )

        # Priority 1: If there are same results, select the first successful one
        if same_result_groups:
            logger.info(f"Found {len(same_result_groups)} result(s) with multiple occurrences")
            # Sort by count (descending) to prioritize most common result
            sorted_groups = sorted(
                same_result_groups.items(), key=lambda x: len(x[1]), reverse=True
            )

            for normalized_result, candidate_ids in sorted_groups:
                # Find the first successful candidate in this group
                for candidate_id in candidate_ids:
                    if candidate_success_map.get(candidate_id, False):
                        logger.info(
                            f"Smart selection: Selected candidate {candidate_id} "
                            f"(same result as {len(candidate_ids) - 1} other candidate(s))"
                        )
                        return SelectionResult(
                            success=True,
                            selected_result=candidates[candidate_id],
                            selected_source=candidate_id,
                            selection_reason=(
                                f"Selected candidate with same result as {len(candidate_ids) - 1} "
                                f"other candidate(s) (smart selection)"
                            ),
                            all_candidates=candidates,
                            score_analysis={
                                candidate_id: {
                                    "score": 10,
                                    "reason": f"Same result as {len(candidate_ids) - 1} other candidate(s)",
                                }
                            },
                        )

                # If no successful candidate in this group, try next group
                logger.debug(
                    f"No successful candidate found in group with result (length={len(normalized_result)})"
                )

        # Priority 2: Check for candidates with actual results
        # A candidate with actual results is more valuable than one without, even if both are marked as successful
        candidates_with_results = [
            candidate_id
            for candidate_id, normalized_result in candidate_result_map.items()
            if normalized_result  # Has actual result content
        ]

        candidates_without_results = [
            candidate_id
            for candidate_id, normalized_result in candidate_result_map.items()
            if not normalized_result  # No result content
        ]

        logger.info(f"Candidates with results: {len(candidates_with_results)} out of {len(candidates)}")
        logger.info(f"  Candidates with results: {candidates_with_results}")
        logger.info(f"  Candidates without results: {candidates_without_results}")

        # Priority 2a: If only one candidate has actual results, select it
        if len(candidates_with_results) == 1:
            candidate_id = candidates_with_results[0]
            logger.info(
                f"Smart selection: Selected candidate {candidate_id} "
                "(only one candidate with actual results)"
            )
            return SelectionResult(
                success=True,
                selected_result=candidates[candidate_id],
                selected_source=candidate_id,
                selection_reason="Selected the only candidate with actual SQL results (smart selection)",
                all_candidates=candidates,
                score_analysis={
                    candidate_id: {
                        "score": 10,
                        "reason": "Only candidate with actual results",
                    }
                },
            )

        # Priority 2b: If only one candidate is successful (fallback to original logic)
        successful_candidates = [
            candidate_id
            for candidate_id, success in candidate_success_map.items()
            if success
        ]

        logger.info(f"Successful candidates: {len(successful_candidates)} out of {len(candidates)}")
        logger.info(f"  Successful candidate IDs: {successful_candidates}")

        if len(successful_candidates) == 1:
            candidate_id = successful_candidates[0]
            logger.info(
                f"Smart selection: Selected candidate {candidate_id} "
                "(only one successful candidate)"
            )
            return SelectionResult(
                success=True,
                selected_result=candidates[candidate_id],
                selected_source=candidate_id,
                selection_reason="Selected the only successful candidate (smart selection)",
                all_candidates=candidates,
                score_analysis={
                    candidate_id: {
                        "score": 10,
                        "reason": "Only successful candidate",
                    }
                },
            )

        # Cannot decide with smart selection, return None
        logger.info(
            f"Smart selection cannot decide: "
            f"same_result_groups={len(same_result_groups)}, "
            f"candidates_with_results={len(candidates_with_results)}, "
            f"successful_candidates={len(successful_candidates)}, "
            f"total_candidates={len(candidates)}"
        )
        if len(same_result_groups) == 0:
            if len(candidates_with_results) > 1:
                logger.info(
                    f"  Reason: No same results found, and {len(candidates_with_results)} candidate(s) with results found"
                )
            elif len(successful_candidates) > 1:
                logger.info(
                    f"  Reason: No same results found, and {len(successful_candidates)} successful candidate(s) found"
                )
            else:
                logger.info(
                    "  Reason: No same results found, and no single candidate with results or successful candidate")
        return None

    def _llm_based_selection(self, candidates: Dict[str, Any]) -> SelectionResult:
        """Use LLM to select the best candidate"""
        # Use None to auto-select latest version from user template directory
        prompt_version = self.input.prompt_version if self.input else None
        prompt = create_selection_prompt(candidates, prompt_version=prompt_version)

        try:
            logger.info("Calling LLM for candidate selection...")
            logger.debug(f"Selection prompt length: {len(prompt)} characters")
            response = self.model.generate_with_json_output(prompt)

            best_candidate_id = response.get("best_candidate", "")
            selection_reason = response.get("reason", "LLM-based selection")
            decision_process = response.get("decision_process", {})
            score_analysis = response.get("score_analysis", {})

            logger.info(
                f"LLM selected candidate: {best_candidate_id}, "
                f"reason: {selection_reason if selection_reason else 'N/A'}"
            )
            if decision_process:
                logger.info(
                    f"LLM decision process: candidates_with_results={decision_process.get('candidates_with_results', [])}, "
                    f"candidates_without_results={decision_process.get('candidates_without_results', [])}, "
                    f"rule_applied={decision_process.get('rule_applied', 'N/A')}, "
                    f"why_selected={decision_process.get('why_selected', 'N/A')}"
                )
            if score_analysis:
                logger.info(f"LLM score analysis: {score_analysis}")

            # Log full LLM response for debugging
            logger.debug(f"Full LLM response: {response}")

            if best_candidate_id not in candidates:
                # Fallback if LLM returns invalid candidate
                logger.warning(f"LLM returned invalid candidate: {best_candidate_id}")
                return self._rule_based_selection(candidates)

            selected_result = candidates[best_candidate_id]

            return SelectionResult(
                success=True,
                selected_result=selected_result,
                selected_source=best_candidate_id,
                selection_reason=selection_reason,
                all_candidates=candidates,
                score_analysis=score_analysis,
            )

        except Exception as e:
            logger.error(f"LLM-based selection failed: {str(e)}")
            # Fallback to rule-based selection
            return self._rule_based_selection(candidates)

    def _rule_based_selection(self, candidates: Dict[str, Any]) -> SelectionResult:
        """Fallback rule-based selection"""
        logger.info("Using rule-based selection as fallback")

        # Score candidates based on simple rules
        scores = {}

        for candidate_id, candidate in candidates.items():
            score = 0
            reasons = []

            # Check if candidate has a successful result
            if isinstance(candidate, dict):
                if candidate.get("success", False):
                    score += 5
                    reasons.append("Successful execution")

                # Check if result exists and has content
                result = candidate.get("result")
                if result:
                    score += 3
                    reasons.append("Has result content")

                    # For SQL results, prefer results with data
                    if hasattr(result, "sql_result_final") and result.sql_result_final:
                        score += 2
                        reasons.append("Has SQL result")

                # Check error status
                if not candidate.get("error"):
                    score += 2
                    reasons.append("No errors")

            scores[candidate_id] = {"score": score, "reason": "; ".join(reasons) if reasons else "Basic scoring"}

        # Select candidate with highest score
        best_candidate_id = max(scores.keys(), key=lambda x: scores[x]["score"])
        best_score = scores[best_candidate_id]["score"]

        return SelectionResult(
            success=True,
            selected_result=candidates[best_candidate_id],
            selected_source=best_candidate_id,
            selection_reason=f"Rule-based selection (score: {best_score})",
            all_candidates=candidates,
            score_analysis=scores,
        )

    async def execute_stream(
            self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the selection node with streaming support.

        Args:
            action_history_manager: Manager for tracking action history

        Yields:
            ActionHistory: Progress updates during node execution
        """
        # Create initial action history
        action_id = f"{self.id}_stream"
        if action_history_manager:
            action_history = action_history_manager.create(
                action_id=action_id,
                action_type="selection",
                status="running",
                message="Selecting the best result from candidates",
            )
            yield action_history

        # Execute the main logic (non-streaming)
        result = self.execute()

        # Generate final action history
        if action_history_manager:
            status = "completed" if result.success else "failed"
            message = (
                f"Selection completed. Selected source: {result.selected_source}" if result.success else result.error
            )
            action_history = action_history_manager.update(
                action_id=action_id,
                status=status,
                message=message,
                result={
                    "success": result.success,
                    "selected_source": result.selected_source if result.success else None,
                    "reason": result.selection_reason if result.success else None,
                },
            )
            yield action_history
