from __future__ import annotations
from typing import Any
import pandas as pd
from utils.logger import get_logger, log_section

logger = get_logger("profiler")


class ProfilerAgent:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        # Maximum number of rows to sample for per-column statistics.
        self.sample_size: int | None = config.get("profiler", {}).get("sample_size", 5000)
        # Columns with more unique values than this are flagged as high-cardinality.
        self.cardinality_threshold: int = config.get("profiler", {}).get(
            "cardinality_threshold", 50
        )
        # Absolute skewness above this threshold is flagged.
        self.skew_threshold: float = config.get("profiler", {}).get("skew_threshold", 1.0)

    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        log_section("Profiler Agent", f"Rows: {len(df):,}  Cols: {df.shape[1]}")

        # Use a sample for per-column stats to keep profiling fast on large datasets.
        sample = self._draw_sample(df)

        profile: dict[str, Any] = {
            "shape": {"rows": len(df), "cols": df.shape[1]},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "global": self._global_stats(df),
            "columns": {},
        }

        for col in df.columns:
            # Use sampled series for stats, full series for counts.
            col_sample = sample[col] if col in sample else df[col]
            profile["columns"][col] = self._column_profile(df[col], col_sample)

        logger.info("Profiling complete - %d columns analyzed.", df.shape[1])
        return profile

    def _draw_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sample_size and len(df) > self.sample_size:
            return df.sample(n=self.sample_size, random_state=42)
        return df

    def _global_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "total_cells": int(df.size),
            "missing_cells": int(df.isnull().sum().sum()),
            "missing_pct": round(df.isnull().mean().mean() * 100, 3),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_pct": round(df.duplicated().mean() * 100, 3),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "numeric_cols": int(
                sum(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
            ),
            "categorical_cols": int(
                sum(
                    pd.api.types.is_object_dtype(df[col])
                    or pd.api.types.is_categorical_dtype(df[col])
                    for col in df.columns
                )
            ),
            "datetime_cols": int(
                sum(pd.api.types.is_datetime64_any_dtype(df[col]) for col in df.columns)
            ),
        }

    def _column_profile(self, full: pd.Series, sample: pd.Series) -> dict[str, Any]:
        base: dict[str, Any] = {
            "dtype": str(full.dtype),
            "count": int(full.notna().sum()),
            "missing": int(full.isnull().sum()),
            "missing_pct": round(full.isnull().mean() * 100, 3),
            "unique": int(full.nunique(dropna=True)),
            "unique_pct": round(full.nunique(dropna=True) / max(len(full), 1) * 100, 3),
        }

        if pd.api.types.is_numeric_dtype(full):
            base.update(self._numeric_stats(sample))
        elif pd.api.types.is_datetime64_any_dtype(full):
            base.update(self._datetime_stats(sample))
        else:
            base.update(self._categorical_stats(sample))

        return base

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if clean.empty:
            return {}

        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        skew = float(clean.skew()) if len(clean) > 2 else 0.0

        return {
            "mean": round(float(clean.mean()), 6),
            "median": round(float(clean.median()), 6),
            "std": round(float(clean.std()), 6),
            "min": round(float(clean.min()), 6),
            "max": round(float(clean.max()), 6),
            "q1": round(float(q1), 6),
            "q3": round(float(q3), 6),
            "iqr": round(float(iqr), 6),
            "skewness": round(skew, 4),
            "is_skewed": abs(skew) > self.skew_threshold,
            "n_outliers_iqr": int(
                ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
            ),
            "is_constant": bool(clean.nunique() <= 1),
            "is_integer_like": bool((clean == clean.round()).all()),
        }

    def _categorical_stats(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna().astype(str)
        value_counts = clean.value_counts()
        top5 = value_counts.head(5).to_dict()

        return {
            "top_values": top5,
            "most_frequent": value_counts.index[0] if not value_counts.empty else None,
            "most_frequent_pct": (
                round(float(value_counts.iloc[0] / max(len(clean), 1)) * 100, 3)
                if not value_counts.empty
                else 0
            ),
            "is_high_cardinality": series.nunique() > self.cardinality_threshold,
            "avg_str_length": round(clean.str.len().mean(), 1) if not clean.empty else 0,
        }

    def _datetime_stats(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if clean.empty:
            return {}

        return {
            "min": str(clean.min()),
            "max": str(clean.max()),
            "range_days": (clean.max() - clean.min()).days,
        }
