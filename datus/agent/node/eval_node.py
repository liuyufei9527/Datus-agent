# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

import pandas as pd

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.eval_node_models import EvalInput, EvalResult
from datus.utils.benchmark_utils import (
    JsonMappingSqlProvider,
    SingleFileGoldProvider,
    compare_pandas_tables,
    csv_str_to_pands,
    preview_dataframe, build_benchmark_evaluator,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class EvalNode(Node):
    """
    Evaluation node that loads gold data and compares with actual results.

    This node:
    1. Loads gold SQL/result from benchmark configuration
    2. Compares actual vs gold results using compare_pandas_tables()
    3. Outputs comparison metrics + gold data for downstream CompareNode
    """

    def execute(self):
        """Execute evaluation synchronously."""
        self.result = self._execute_eval()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute evaluation with streaming support."""
        # Evaluation is mostly synchronous data loading and comparison
        # Wrap in a single action for streaming
        result = self._execute_eval()
        self.result = result

        # Yield evaluation completion action
        action = ActionHistory(
            action_id="eval_completion",
            role=ActionRole.WORKFLOW,
            messages=f"Evaluation completed: match_rate={result.match_rate:.2%}" if result.success else f"Evaluation failed: {result.error}",
            action_type="evaluation",
            input={"task_id": self.input.sql_task_id.id if self.input and hasattr(self.input.sql_task_id, 'id') else "unknown"},
            output={
                "success": result.success,
                "match_rate": result.match_rate,
                "matched_columns": result.matched_columns,
                "missing_columns": result.missing_columns,
                "extra_columns": result.extra_columns,
            } if result.success else {"error": result.error},
            status=ActionStatus.SUCCESS if result.success else ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        """Setup input from workflow context."""
        # Get actual SQL and result from last SQL context

        next_input = EvalInput(
            sql_task_id=workflow.task.id,
            benchmark_platform=workflow.task.benchmark_platform,
        )
        self.input = next_input
        return {"success": True, "message": "Eval input setup complete"}

    def update_context(self, workflow: Workflow) -> Dict:
        """Store eval results in workflow context for CompareNode."""
        result = self.result
        try:
            # Store gold data and metrics in workflow context for CompareNode access
            if not hasattr(workflow.context, 'eval_result'):
                workflow.context.eval_result = {}

            workflow.context.eval_result = result

            logger.info(f"Stored eval context with match_rate={result.match_rate:.2%}")
            return {"success": True, "message": "Eval context updated"}
        except Exception as e:
            logger.error(f"Failed to update eval context: {e}")
            return {"success": False, "message": str(e)}

    def _execute_eval(self) -> EvalResult:
        sql_task_id = self.input.sql_task_id
        evaluator, _ = build_benchmark_evaluator(self.agent_config, self.input.benchmark_platform,
                                              list(sql_task_id))
        """Execute evaluation logic."""
        try:
            if not sql_task_id:
                return EvalResult(
                    success=False,
                    error="Task ID not set in sql_task"
                )

            # 2. Load gold SQL
            gold_sql = evaluator.gold_sql_provider.fetch(sql_task_id).sql

            # 3. Load gold result
            gold_result_df = evaluator.gold_result_provider.fetch(sql_task_id).dataframe
            if gold_result_df is None:
                return EvalResult(
                    success=False,
                    error=f"Failed to load gold result for task_id={sql_task_id}",
                    gold_sql=gold_sql,
                )

            actual_sql = evaluator.result_sql_provider.fetch(sql_task_id).sql
            # 4. Parse actual result
            actual_result_df = evaluator.result_provider.fetch(sql_task_id).dataframe

            if actual_result_df is None:
                return EvalResult(
                    success=True,
                    gold_sql=gold_sql,
                    gold_result=gold_result_df.to_csv(index=False),
                    actual_sql=actual_sql,
                    actual_shape=(0, 0),
                    expected_shape=(len(gold_result_df), len(gold_result_df.columns)),
                    expected_preview=preview_dataframe(gold_result_df),
                )
            # 5. Compare using existing utility
            comparison = compare_pandas_tables(actual_result_df, gold_result_df)

            # 6. Build result
            return EvalResult(
                success=True,
                match_rate=comparison['match_rate'],
                matched_columns=comparison['matched_columns'],
                missing_columns=comparison['missing_columns'],
                extra_columns=comparison['extra_columns'],
                gold_sql=gold_sql,
                gold_result=gold_result_df.to_csv(index=False),
                actual_sql=actual_sql,
                actual_result=actual_result_df.to_csv(index=False),
                actual_shape=(len(actual_result_df), len(actual_result_df.columns)),
                expected_shape=(len(gold_result_df), len(gold_result_df.columns)),
                actual_preview=preview_dataframe(actual_result_df),
                expected_preview=preview_dataframe(gold_result_df),
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return EvalResult(
                success=False,
                error=str(e)
            )

    def _get_benchmark_config(self):
        """Get benchmark configuration for current namespace."""

        if not self.agent_config or not self.agent_config.benchmark_configs:
            logger.warning("No agent_config or benchmark configuration available")
            return None

        # Determine platform name from current namespace
        # Convention: namespace name matches benchmark platform name
        platform_name = self.agent_config.current_namespace

        if platform_name and platform_name in self.agent_config.benchmark:
            logger.info(f"Using benchmark config for platform: {platform_name}")
            return self.agent_config.benchmark[platform_name]

        # Fallback: use first available benchmark config
        if self.agent_config.benchmark:
            first_platform = list(self.agent_config.benchmark.keys())[0]
            logger.warning(f"Platform '{platform_name}' not found, using fallback benchmark config: {first_platform}")
            return self.agent_config.benchmark[first_platform]

        logger.warning("No benchmark configuration available")
        return None

    def _load_gold_sql(self, task_id: str, benchmark_config) -> Optional[str]:
        """Load gold SQL using existing providers."""
        try:
            # Build SQL provider (reuse logic from benchmark_utils)
            base_path = Path(self.agent_config.storage.base_path)
            question_file = benchmark_config.question_file
            question_path = base_path / question_file if question_file else None

            if not question_path or not question_path.exists():
                logger.warning(f"Question file not found: {question_path}")
                return None

            # Use JsonMappingSqlProvider for JSON/JSONL files
            sql_key = getattr(benchmark_config, 'gold_sql_key', None) or "sql"
            task_id_key = getattr(benchmark_config, 'question_id_key', None) or "question_id"
            dialect = self.input.sql_task_id.database_type

            provider = JsonMappingSqlProvider(
                json_path=str(question_path),
                task_id_key=task_id_key,
                sql_key=sql_key,
                dialect=dialect,
            )

            sql_data = provider.fetch(task_id)
            if sql_data and sql_data.available:
                logger.info(f"Loaded gold SQL for task_id={task_id}")
                return sql_data.sql
            else:
                logger.warning(f"Gold SQL not available for task_id={task_id}")
                return None

        except Exception as e:
            logger.warning(f"Failed to load gold SQL: {e}")
            return None

    def _load_gold_result(self, task_id: str, benchmark_config) -> Optional[pd.DataFrame]:
        """Load gold result using existing providers."""
        try:
            # Build result provider
            base_path = Path(self.agent_config.storage.base_path)
            question_file = benchmark_config.question_file
            question_path = base_path / question_file if question_file else None

            if not question_path or not question_path.exists():
                logger.warning(f"Question file not found: {question_path}")
                return None

            # Build connections for SQL execution (if gold result is SQL-based)
            connections = {}
            try:
                connector = self._sql_connector(self.input.sql_task_id.database_name)
                if connector:
                    connections[self.input.sql_task_id.database_name] = connector
            except Exception as e:
                logger.warning(f"Failed to get SQL connector: {e}")

            # Use SingleFileGoldProvider
            result_key = getattr(benchmark_config, 'gold_result_key', None) or "result"
            sql_key = getattr(benchmark_config, 'gold_sql_key', None) or "sql"
            task_id_key = getattr(benchmark_config, 'question_id_key', None) or "question_id"

            provider = SingleFileGoldProvider(
                result_file=str(question_path),
                connections=connections,
                task_id_key=task_id_key,
                query_result_key=result_key,
                sql_key=sql_key,
            )

            result_data = provider.fetch(task_id)
            if result_data and result_data.available:
                logger.info(f"Loaded gold result for task_id={task_id}, shape={result_data.dataframe.shape}")
                return result_data.dataframe
            else:
                logger.warning(f"Gold result not available for task_id={task_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to load gold result: {e}", exc_info=True)
            return None
