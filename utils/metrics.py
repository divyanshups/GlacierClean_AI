from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class QualityReport:
    completeness: float
    consistency: float
    uniqueness: float
    validity: float
    overall: float

    missing_by_column: dict = field(default_factory = dict)
    type_issues_by_column: dict = field(default_factory=dict)
    duplicate_count: int = 0
    duplicate_percentage: float = 0.0
    outlier_count_by_column: dict = field(default_factory=dict)

    n_rows: int = 0
    n_columns: int = 0

    def to_dict(self) -> dict:
        return{
            "scores": {
                "completeness": round(self.completeness, 4),
                "consistency": round(self.consistency, 4),
                "uniqueness": round(self.uniqueness, 4),
                "validity": round(self.validity, 4),
                "overall": round(self.overall, 4),
            },
            "details": {
                "missing_by_column": self.missing_by_column,
                "type_issues_by_column": self.type_issues_by_column,
                "duplicate_count": self.duplicate_count,
                "duplicate_percentage": round(self.duplicate_percentage, 4),
                "outlier_count_by_column": self.outlier_count_by_column,
            },
            "dataset_info": {
                "n_rows": self.n_rows,
                "n_columns": self.n_columns,
            }
        }
def compute_completeness(df: pd.DataFrame) -> tuple[float, dict]:
    total_cells = df.size
    total_missing = df.isna().sum().sum()

    score = 1.0 - (total_missing / total_cells) if total_cells > 0 else 1.0

    per_column = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) if len(df) > 0 else 0.0
        if missing_pct > 0:  # Only include columns that actually have missing values
            per_column[col] = round(missing_pct, 4)

    logger.debug(f"Completeness score: {score:.4f}")
    return round(score, 4), per_column

def compute_consistency(df: pd.DataFrame) -> tuple[float, dict]:
    per_column_issues = {}
    consistency_scores = []

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            consistency_scores.append(1.0)  # Empty columns get a pass
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            consistency_scores.append(1.0)
        elif pd.api.types.is_object_dtype(df[col]):
            sample_size = min(500, len(series))
            sample = series.sample(sample_size, random_state=42)

            n_consistent = sample.apply(lambda x: isinstance(x, str)).sum()
            col_consistency = n_consistent / len(sample)

            if col_consistency < 1.0:
                inconsistent_count = len(sample) - n_consistent
                per_column_issues[col] = f"{inconsistent_count} non-string values in string column"

            consistency_scores.append(col_consistency)
        else:
            consistency_scores.append(1.0)

    overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
    logger.debug(f"Consistency score: {overall_consistency:.4f}")
    return round(float(overall_consistency), 4), per_column_issues

def compute_uniqueness(df: pd.DataFrame) -> tuple[float, int, float]:
    n_total = len(df)
    if n_total == 0:
        return 1.0, 0, 0.0

    # duplicated() returns True for ALL occurrences of duplicates except the first
    n_duplicates = df.duplicated(keep="first").sum()
    duplicate_pct = n_duplicates / n_total
    score = 1.0 - duplicate_pct

    logger.debug(f"Uniqueness score: {score:.4f} ({n_duplicates} duplicate rows)")
    return round(score, 4), int(n_duplicates), round(duplicate_pct, 4)

def compute_validity(df: pd.DataFrame) -> tuple[float, dict]:
    outlier_counts = {}
    validity_scores = []

    for col in df.columns:
        series = df[col].dropna()

        if not pd.api.types.is_numeric_dtype(df[col]) or len(series) < 10:
            # Only check numeric columns with enough data
            validity_scores.append(1.0)
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            # No spread means no outliers by IQR definition
            validity_scores.append(1.0)
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((series < lower) | (series > upper)).sum())

        if n_outliers > 0:
            outlier_counts[col] = n_outliers

        col_validity = 1.0 - (n_outliers / len(series))
        validity_scores.append(col_validity)

    overall_validity = np.mean(validity_scores) if validity_scores else 1.0
    logger.debug(f"Validity score: {overall_validity:.4f}")
    return round(float(overall_validity), 4), outlier_counts

def compute_quality_report(
    df: pd.DataFrame,
    config_path: str = "config/config.yaml",
    ) -> QualityReport:
    logger.info("Computing data quality report...")

    # Load weights from config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    weights = config.get("evaluation", {}).get("weights", {
        "completeness": 0.30,
        "consistency": 0.25,
        "uniqueness": 0.20,
        "validity": 0.25,
    })

    # Compute all dimensions
    completeness_score, missing_by_col = compute_completeness(df)
    consistency_score, type_issues = compute_consistency(df)
    uniqueness_score, dup_count, dup_pct = compute_uniqueness(df)
    validity_score, outlier_counts = compute_validity(df)

    # Compute weighted overall score
    overall = (
        weights.get("completeness", 0.25) * completeness_score
        + weights.get("consistency", 0.25) * consistency_score
        + weights.get("uniqueness", 0.25) * uniqueness_score
        + weights.get("validity", 0.25) * validity_score
    )

    report = QualityReport(
        completeness=completeness_score,
        consistency=consistency_score,
        uniqueness=uniqueness_score,
        validity=validity_score,
        overall=round(overall, 4),
        missing_by_column=missing_by_col,
        type_issues_by_column=type_issues,
        duplicate_count=dup_count,
        duplicate_percentage=dup_pct,
        outlier_count_by_column=outlier_counts,
        n_rows=len(df),
        n_columns=len(df.columns),
    )

    logger.info(
        f"Quality Report — Overall: [bold]{report.overall:.2%}[/bold] | "
        f"Completeness: {report.completeness:.2%} | "
        f"Consistency: {report.consistency:.2%} | "
        f"Uniqueness: {report.uniqueness:.2%} | "
        f"Validity: {report.validity:.2%}"
    )

    return report

def compare_quality(
    before: QualityReport, after: QualityReport
    ) -> dict:
    delta = {
        "overall": round(after.overall - before.overall, 4),
        "completeness": round(after.completeness - before.completeness, 4),
        "consistency": round(after.consistency - before.consistency, 4),
        "uniqueness": round(after.uniqueness - before.uniqueness, 4),
        "validity": round(after.validity - before.validity, 4),
        "rows_removed": before.n_rows - after.n_rows,
        "improved": after.overall > before.overall,
    }

    if delta["improved"]:
        logger.info(
            f"Quality improved by [bold green]{delta['overall']:+.2%}[/bold green] "
            f"(from {before.overall:.2%} to {after.overall:.2%})"
        )
    else:
        logger.warning(
            f"Quality did not improve. Delta: {delta['overall']:+.2%}"
        )

    return delta