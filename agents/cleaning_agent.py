from __future__ import annotations
from typing import Any
import pandas as pd
from agents.decision_agent import CleaningAction
from utils.logger import get_logger, log_section
from utils.preprocessing_utils import (
    clip_outliers_iqr,
    coerce_numeric,
    impute_categorical,
    impute_numeric,
    normalize_text_column,
    remove_duplicates,
    strip_currency_symbols,
    try_parse_dates,
    winsorize_series,
)

logger = get_logger("cleaning_agent")


class CleaningAgent:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.cleaning_cfg = config.get("cleaning", {})
        # Actions whose confidence falls below this threshold are skipped.
        self.confidence_threshold: float = (
            config.get("pipeline", {}).get("confidence_threshold", 0.75)
        )
        self.change_log: list[dict[str, Any]] = []

    def clean(self, df: pd.DataFrame, actions: list[CleaningAction]) -> pd.DataFrame:
        """Apply selected cleaning actions and return a cleaned DataFrame."""
        log_section("Cleaning Agent", f"{len(actions)} actions to apply")

        self.change_log = []
        df = df.copy()  # Never mutate the caller's DataFrame.

        for action in actions:
            if action.confidence < self.confidence_threshold:
                logger.warning(
                    "Skipping '%s' on '%s' - confidence %.2f is below threshold %.2f",
                    action.action_type,
                    action.column,
                    action.confidence,
                    self.confidence_threshold,
                )
                continue

            df, log_entry = self._apply(df, action)
            if log_entry:
                self.change_log.append(log_entry)

        # Optional final pass: auto-parse remaining object date-like columns.
        if self.cleaning_cfg.get("date_inference", True):
            df = self._infer_dates(df)

        logger.info(
            "Cleaning complete - %d changes applied, %d rows retained.",
            len(self.change_log),
            len(df),
        )
        return df

    def _apply(
        self,
        df: pd.DataFrame,
        action: CleaningAction,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        action_type = action.action_type
        column = action.column
        params = action.params

        try:
            if action_type == "drop_column":
                return self._drop_column(df, column, action)

            if action_type == "drop_duplicates":
                return self._drop_duplicates(df, params, action)

            if action_type == "impute":
                return self._impute(df, column, params, action)

            if action_type in {"outlier_clip", "outlier_remove", "outlier_winsorize"}:
                return self._handle_outliers(df, column, action_type, params, action)

            if action_type == "normalize_text":
                return self._normalize_text(df, column, action)

            if action_type == "coerce_dtype":
                return self._coerce_dtype(df, column, action)

            logger.warning("Unknown action type '%s' - skipping.", action_type)
            return df, None

        except Exception as error:
            # Non-fatal: log the problem and let the pipeline continue.
            logger.error("Action '%s' on '%s' failed: %s", action_type, column, error)
            return df, None

    def _drop_column(
        self,
        df: pd.DataFrame,
        column: str,
        action: CleaningAction,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if column not in df.columns:
            return df, None

        shape_before = df.shape
        df = df.drop(columns=[column])
        return df, self._make_log_entry(
            action,
            description=f"Dropped column '{column}'",
            before_shape=shape_before,
            after_shape=df.shape,
        )

    def _drop_duplicates(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        action: CleaningAction,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows_before = len(df)
        keep = params.get("keep", "first")
        # The string "drop_all" is the UI's way of requesting keep=False.
        df = remove_duplicates(df, keep=keep if keep != "drop_all" else False)
        rows_removed = rows_before - len(df)
        return df, self._make_log_entry(
            action,
            description=f"Removed {rows_removed:,} duplicate rows",
            before_shape=(rows_before, df.shape[1]),
            after_shape=df.shape,
        )

    def _impute(
        self,
        df: pd.DataFrame,
        column: str,
        params: dict[str, Any],
        action: CleaningAction,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if column not in df.columns:
            return df, None

        missing_before = int(df[column].isnull().sum())
        if missing_before == 0:
            return df, None

        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            strategy = params.get("numeric_strategy", "median")
            df[column] = impute_numeric(series, strategy)
            description = f"Imputed '{column}' (numeric) with {strategy}"
        else:
            strategy = params.get("categorical_strategy", "mode")
            constant = params.get("constant", "Unknown")
            df[column] = impute_categorical(series, strategy, constant)
            description = f"Imputed '{column}' (categorical) with {strategy}"

        missing_after = int(df[column].isnull().sum())
        return df, self._make_log_entry(
            action,
            description=description,
            extra={"missing_before": missing_before, "missing_after": missing_after},
        )

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        action_type: str,
        params: dict[str, Any],
        action: CleaningAction,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df, None

        iqr_factor = params.get("iqr_factor", 1.5)
        series = df[column]
        stats_before = {"min": float(series.min()), "max": float(series.max())}

        if action_type == "outlier_clip":
            df[column] = clip_outliers_iqr(series, iqr_factor)

        elif action_type == "outlier_winsorize":
            df[column] = winsorize_series(series)

        elif action_type == "outlier_remove":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            within_bounds = (series >= q1 - iqr_factor * iqr) & (
                series <= q3 + iqr_factor * iqr
            )
            # Keep rows that are within bounds OR have a null value in this column.
            df = df[within_bounds | series.isnull()]

        stats_after = {"min": float(df[column].min()), "max": float(df[column].max())}
        return df, self._make_log_entry(
            action,
            description=f"Handled outliers in '{column}' via {action_type}",
            extra={"before": stats_before, "after": stats_after},
        )

    def _normalize_text(
        self,
        df: pd.DataFrame,
        column: str,
        action: CleaningAction,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if column not in df.columns:
            return df, None

        df[column] = normalize_text_column(df[column])
        return df, self._make_log_entry(
            action,
            description=f"Normalized text in '{column}'",
        )

    def _coerce_dtype(
        self,
        df: pd.DataFrame,
        column: str,
        action: CleaningAction,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if column not in df.columns:
            return df, None

        original_dtype = str(df[column].dtype)
        series = df[column]

        # Try numeric first; success means >85% of values parsed cleanly.
        numeric_series = coerce_numeric(strip_currency_symbols(series))
        parse_success_rate = numeric_series.notna().sum() / max(series.notna().sum(), 1)
        if parse_success_rate > 0.85:
            df[column] = numeric_series
            return df, self._make_log_entry(
                action,
                description=f"Coerced '{column}' from {original_dtype} to float64",
            )

        # Try datetime next.
        parsed_dates = try_parse_dates(series)
        if pd.api.types.is_datetime64_any_dtype(parsed_dates):
            df[column] = parsed_dates
            return df, self._make_log_entry(
                action,
                description=f"Coerced '{column}' from {original_dtype} to datetime64",
            )

        return df, None

    def _infer_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include="object").columns:
            parsed = try_parse_dates(df[column])
            if pd.api.types.is_datetime64_any_dtype(parsed):
                df[column] = parsed
                logger.debug("Auto-parsed '%s' as datetime.", column)
        return df

    @staticmethod
    def _make_log_entry(
        action: CleaningAction,
        description: str,
        before_shape: tuple[int, int] | None = None,
        after_shape: tuple[int, int] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "column": action.column,
            "description": description,
        }
        if before_shape:
            entry["before_shape"] = before_shape
        if after_shape:
            entry["after_shape"] = after_shape
        if extra:
            entry.update(extra)
        return entry
