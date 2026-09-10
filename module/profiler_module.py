from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import yaml

from utils.logger import get_logger
from utils.preprocessing_utils import detect_column_types

logger = get_logger(__name__)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    count: int
    null_count: int
    null_pct: float
    unique_count: int
    cardinality_ratio: float
    top_values: dict
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    sample_values: list = field(default_factory=list)


@dataclass
class ProfileReport:
    columns: dict[str, ColumnProfile]
    correlation_matrix: Optional[pd.DataFrame] = None
    n_rows: int = 0
    n_columns: int = 0


class ProfilerModule:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f).get("profiler", {})
        self.sample_size = self.config.get("sample_size", 1000)
        self.top_n = self.config.get("top_n_values", 5)

    def profile(self, df: pd.DataFrame) -> ProfileReport:
        if df is None or df.empty:
            raise ValueError("Cannot profile an empty DataFrame")

        logger.info(f"Profiling dataset: {df.shape[0]} rows × {df.shape[1]} cols")

        if len(df) > self.sample_size:
            df_sample = df.sample(n=self.sample_size, random_state=42)
        else:
            df_sample = df

        semantic_types = detect_column_types(df)
        columns_profile = {}

        for col in df.columns:
            series = df[col]
            n = len(series)
            null_count = int(series.isna().sum())
            unique_count = int(series.nunique())
            cardinality = unique_count / n if n > 0 else 0.0

            top_vals = series.dropna().value_counts().head(self.top_n).to_dict()
            top_vals = {str(k): int(v) for k, v in top_vals.items()}

            col_prof = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                semantic_type=semantic_types.get(col, "unknown"),
                count=n,
                null_count=null_count,
                null_pct=round(null_count / n, 4) if n > 0 else 0.0,
                unique_count=unique_count,
                cardinality_ratio=round(cardinality, 4),
                top_values=top_vals,
                sample_values=[str(x) for x in series.dropna().head(3).tolist()],
            )

            if pd.api.types.is_numeric_dtype(series):
                numeric_sample = pd.to_numeric(df_sample[col], errors="coerce").dropna()
                if len(numeric_sample) > 0:
                    col_prof.min_val = round(float(numeric_sample.min()), 4)
                    col_prof.max_val = round(float(numeric_sample.max()), 4)
                    col_prof.mean_val = round(float(numeric_sample.mean()), 4)
                    col_prof.median_val = round(float(numeric_sample.median()), 4)
                    col_prof.std_val = round(float(numeric_sample.std()), 4)

            columns_profile[col] = col_prof

        numeric_cols = df.select_dtypes(include="number").columns
        corr_matrix = None
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().round(3)

        report = ProfileReport(
            columns=columns_profile,
            correlation_matrix=corr_matrix,
            n_rows=len(df),
            n_columns=len(df.columns),
        )
        logger.info(f"Profile complete. {len(columns_profile)} columns analyzed.")
        return report