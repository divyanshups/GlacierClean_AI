from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def compute_quality_metrics(df: pd.DataFrame) -> dict[str, Any]:
    row_count, col_count = df.shape
    if row_count == 0:
        return {"error": "Empty dataframe"}

    # ── Missing values ────────────────────────────────────────────────────────
    missing_total      = int(df.isnull().sum().sum())
    overall_miss_rate  = missing_total / (row_count * col_count) if col_count else 0.0
    missing_per_column = (df.isnull().mean() * 100).round(2).to_dict()

    # ── Duplicate rows ────────────────────────────────────────────────────────
    duplicate_count = int(df.duplicated().sum())
    duplicate_rate  = duplicate_count / row_count

    # ── Outliers (IQR method, numeric columns only) ───────────────────────────
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts_per_col: dict[str, int] = {}

    for col in numeric_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr     = q3 - q1
        if iqr == 0:
            continue
        outlier_count = int(
            ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        )
        if outlier_count:
            outlier_counts_per_col[col] = outlier_count

    total_outliers = sum(outlier_counts_per_col.values())
    outlier_rate   = total_outliers / (row_count * max(len(numeric_columns), 1))

    # ── Categorical column stats ──────────────────────────────────────────────
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    high_cardinality_cols = {
        col: int(df[col].nunique())
        for col in categorical_columns
        if df[col].nunique() > 50
    }

    # ── Schema consistency (proxy: what fraction of columns have uniform types) ─
    mixed_type_columns: list[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].dropna().map(type).nunique() > 1:
                mixed_type_columns.append(col)

    schema_consistency = 1.0 - len(mixed_type_columns) / max(col_count, 1)

    return {
        "n_rows":                  row_count,
        "n_cols":                  col_count,
        "missing_total":           missing_total,
        "missing_rate":            round(overall_miss_rate * 100, 4),     # percent
        "missing_per_column":      missing_per_column,
        "n_duplicates":            duplicate_count,
        "duplicate_rate":          round(duplicate_rate * 100, 4),        # percent
        "n_outlier_cells":         total_outliers,
        "outlier_rate":            round(outlier_rate * 100, 4),          # percent
        "outlier_counts_per_col":  outlier_counts_per_col,
        "high_cardinality_cols":   high_cardinality_cols,
        "n_numeric_cols":          len(numeric_columns),
        "n_categorical_cols":      len(categorical_columns),
        "mixed_type_cols":         mixed_type_columns,
        "schema_consistency":      round(schema_consistency * 100, 4),    # percent
        "row_retention":           100.0,   # always 100% on the unmodified frame
    }


def compare_metrics(
    before: dict[str, Any],
    after:  dict[str, Any],
) -> dict[str, Any]:
    scalar_keys = [
        "missing_rate", "duplicate_rate", "outlier_rate",
        "schema_consistency", "n_rows", "n_cols",
        "missing_total", "n_duplicates", "n_outlier_cells",
    ]

    delta: dict[str, Any] = {}
    for key in scalar_keys:
        before_val = before.get(key, 0) or 0
        after_val  = after.get(key,  0) or 0
        abs_change = round(after_val - before_val, 4)
        rel_change = (
            round((after_val - before_val) / before_val * 100, 2)
            if before_val else 0.0
        )
        delta[key] = {
            "before":              before_val,
            "after":               after_val,
            "absolute_change":     abs_change,
            "relative_change_pct": rel_change,
        }

    # Row retention is always computed relative to the original row count
    original_rows = before.get("n_rows", 1) or 1
    cleaned_rows  = after.get("n_rows", original_rows)
    delta["row_retention"] = {
        "before":       100.0,
        "after":        round(cleaned_rows / original_rows * 100, 4),
        "rows_removed": original_rows - cleaned_rows,
    }

    return delta


def quality_score(metrics: dict[str, Any]) -> float:
    weights = {
        "missing_rate":       0.30,
        "duplicate_rate":     0.20,
        "outlier_rate":       0.20,
        "schema_consistency": 0.20,
        "row_retention":      0.10,
    }

    # Each component is mapped to [0, 100]: a lower bad-rate means a higher score
    components = {
        "missing_rate":       max(0, 100 - metrics.get("missing_rate",   0)),
        "duplicate_rate":     max(0, 100 - metrics.get("duplicate_rate", 0)),
        "outlier_rate":       max(0, 100 - metrics.get("outlier_rate",   0)),
        "schema_consistency": metrics.get("schema_consistency",          100),
        "row_retention":      metrics.get("row_retention",               100),
    }

    return round(sum(weights[k] * components[k] for k in weights), 2)
