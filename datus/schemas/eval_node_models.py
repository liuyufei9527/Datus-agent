# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, List, Optional, Tuple

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class EvalInput(BaseInput):
    """
    Input model for evaluation node.
    Validates the input for SQL evaluation against gold standard.
    """

    sql_task_id: str = Field(str, description="The SQL task id for gold data lookup")
    benchmark_platform:str = Field(str, description="Benchmark platform for gold data lookup")


class EvalResult(BaseResult):
    """
    Result model for evaluation node containing comparison metrics and gold data.
    Compares actual SQL execution results against gold standard.
    """

    # Comparison metrics (from compare_pandas_tables)
    match_rate: float = Field(default=0.0, description="Column match rate between 0.0 and 1.0")
    matched_columns: List[Tuple[str, str]] = Field(
        default_factory=list, description="List of (actual_col, gold_col) pairs that matched"
    )
    missing_columns: List[str] = Field(
        default_factory=list, description="Columns present in gold but missing in actual result"
    )
    extra_columns: List[str] = Field(
        default_factory=list, description="Columns present in actual but not in gold result"
    )

    # Gold data for downstream CompareNode
    gold_sql: Optional[str] = Field(None, description="Gold standard SQL query")
    gold_result: Optional[str] = Field(None, description="Gold standard result in CSV format")

    actual_sql: Optional[str] = Field(None, description="Actual SQL query")
    actual_result: Optional[str] = Field(None, description="Actual result in CSV format")

    # Shape and preview metadata
    actual_shape: Optional[Tuple[int, int]] = Field(None, description="(rows, cols) of actual result")
    expected_shape: Optional[Tuple[int, int]] = Field(None, description="(rows, cols) of gold result")
    actual_preview: Optional[str] = Field(None, description="Preview of actual result")
    expected_preview: Optional[str] = Field(None, description="Preview of gold result")
