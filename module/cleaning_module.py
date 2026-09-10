from typing import Tuple
import pandas as pd

from core.schema import CleaningAction, ExecutionLog
from utils.logger import get_logger
from utils.preprocessing_utils import (
    drop_column,
    impute_missing,
    remove_duplicates,
    cap_outliers,
    standardize_string_column,
    cast_column_type,
)

logger = get_logger(__name__)


class CleaningModule:
    def __init__(self):
        self.operation_map = {
            "drop_column": self._handle_drop_column,
            "impute_missing": self._handle_impute_missing,
            "remove_duplicates": self._handle_remove_duplicates,
            "cap_outliers": self._handle_cap_outliers,
            "standardize_strings": self._handle_standardize_strings,
            "cast_type": self._handle_cast_type,
        }

    def execute_plan(
        self,
        df: pd.DataFrame,
        actions: list[CleaningAction],
    ) -> Tuple[pd.DataFrame, list[ExecutionLog]]:
        if not actions:
            logger.info("No cleaning actions. Returning copy of original DataFrame.")
            return df.copy(), []

        logger.info(f"CleaningModule: executing {len(actions)} action(s)...")
        df_working = df.copy()
        logs: list[ExecutionLog] = []

        for action in actions:
            handler = self.operation_map.get(action.operation)

            if handler is None:
                msg = f"Unknown operation '{action.operation}'"
                logger.error(msg)
                logs.append(ExecutionLog(action.column, action.operation, "error", msg))
                continue

            try:
                df_working, success_msg = handler(df_working, action)
                logs.append(
                    ExecutionLog(action.column, action.operation, "success", success_msg)
                )
            except Exception as e:
                msg = str(e)
                logger.error(f"Failed {action.operation} on '{action.column}': {msg}")
                logs.append(ExecutionLog(action.column, action.operation, "error", msg))

        n_ok = sum(1 for log in logs if log.status == "success")
        logger.info(f"Cleaning complete. {n_ok}/{len(actions)} succeeded.")
        return df_working, logs

    # ── handlers ────────────────────────────────────────────────────────────

    def _handle_drop_column(self, df: pd.DataFrame, action: CleaningAction):
        df_new = drop_column(df, action.column)
        return df_new, f"Dropped column '{action.column}'"

    def _handle_impute_missing(self, df: pd.DataFrame, action: CleaningAction):
        strategy = action.params.get("strategy", "median")
        fill_value = action.params.get("fill_value", None)
        kwargs = {"strategy": strategy}
        if strategy == "constant":
            kwargs["fill_value"] = fill_value
        df_new = impute_missing(df, action.column, **kwargs)
        return df_new, f"Imputed missing values using '{strategy}'"

    def _handle_remove_duplicates(self, df: pd.DataFrame, action: CleaningAction):
        n_before = len(df)
        keep = action.params.get("keep", "first")
        subset = action.params.get("subset", None)
        df_new = remove_duplicates(df, subset=subset, keep=keep)
        n_removed = n_before - len(df_new)
        return df_new, f"Removed {n_removed} duplicate row(s)"

    def _handle_cap_outliers(self, df: pd.DataFrame, action: CleaningAction):
        method = action.params.get("method", "iqr")
        iqr_multiplier = action.params.get("multiplier", action.params.get("iqr_multiplier", 1.5))
        z_threshold = action.params.get("z_threshold", 3.0)
        df_new = cap_outliers(
            df,
            action.column,
            method=method,
            iqr_multiplier=iqr_multiplier,
            z_threshold=z_threshold,
        )
        return df_new, f"Capped outliers using '{method}'"

    def _handle_standardize_strings(self, df: pd.DataFrame, action: CleaningAction):
        df_new = standardize_string_column(df, action.column)
        return df_new, "Standardized string formatting"

    def _handle_cast_type(self, df: pd.DataFrame, action: CleaningAction):
        target_type = action.params.get("target_type", "float")
        df_new = cast_column_type(df, action.column, target_type=target_type)
        return df_new, f"Cast column to '{target_type}'"


# Optional alias
CleaningEngine = CleaningModule