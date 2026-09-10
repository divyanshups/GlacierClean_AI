from dataclasses import dataclass
from typing import Literal
import pandas as pd
import yaml

from module.profiler_module import ProfileReport
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Issue:
    column: str
    issue_type: str
    severity: Literal["critical", "high", "medium", "low"]
    affected_rows: int
    affected_pct: float
    description: str
    recommended_action: str


class IssueDetectorModule:
    """Detects and ranks data quality issues based on configurable thresholds."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f).get("issue_detector", {})

    def detect(self, df: pd.DataFrame, profile: ProfileReport) -> list[Issue]:
        """Runs all detectors and returns issues sorted by severity then impact."""
        logger.info("Running issue detection...")
        issues = []

        issues.extend(self._check_missing_rates(profile))
        issues.extend(self._check_cardinality(profile))
        issues.extend(self._check_duplicates(df))
        issues.extend(self._check_outliers(df, profile))
        issues.extend(self._check_type_inconsistencies(df, profile))

        # Sort by affected percentage (highest percentage first)
        issues.sort(key=lambda x: x.affected_pct, reverse=True)

        # Sort by severity rank (critical -> high -> medium -> low)
        severity_rank = {"critical": 1, "high": 2, "medium": 3, "low": 4}
        issues.sort(key=lambda x: severity_rank.get(x.severity, 5))

        logger.info(f"Detected {len(issues)} issues.")
        return issues

    # checks missing rate
    def _check_missing_rates(self, profile: ProfileReport) -> list[Issue]:
        threshold = self.config.get("missing_rate_threshold", 0.05)
        issues = []
        for col, prof in profile.columns.items():
            if prof.null_pct > threshold:
                severity = (
                    "critical" if prof.null_pct > 0.5 
                    else "high" if prof.null_pct > 0.2 
                    else "medium"
                )
                issues.append(Issue(
                    column=col,
                    issue_type="high_missing_rate",
                    severity=severity,
                    affected_rows=prof.null_count,
                    affected_pct=prof.null_pct,
                    description=f"Column '{col}' has {prof.null_pct:.1%} missing values.",
                    recommended_action="impute or drop depending on semantic type",
                ))
        return issues

    # checks cardinality
    def _check_cardinality(self, profile: ProfileReport) -> list[Issue]:
        high_thresh = self.config.get("high_cardinality_threshold", 0.95)
        low_thresh = self.config.get("low_cardinality_threshold", 0.01)
        issues = []
        for col, prof in profile.columns.items():
            if prof.cardinality_ratio > high_thresh and prof.semantic_type not in ("identifier", "free_text"):
                issues.append(Issue(
                    column=col,
                    issue_type="unexpected_high_cardinality",
                    severity="medium",
                    affected_rows=prof.unique_count,
                    affected_pct=prof.cardinality_ratio,
                    description=f"'{col}' has {prof.cardinality_ratio:.1%} unique values but isn't an identifier.",
                    recommended_action="verify if column should be categorical or contains noisy data",
                ))
            elif prof.cardinality_ratio < low_thresh and prof.unique_count > 1:
                issues.append(Issue(
                    column=col,
                    issue_type="low_cardinality",
                    severity="low",
                    affected_rows=prof.count,
                    affected_pct=prof.cardinality_ratio,
                    description=f"'{col}' has very low cardinality ({prof.unique_count} unique).",
                    recommended_action="consider encoding as category or dropping if constant-like",
                ))
        return issues

    # check duplicates
    def _check_duplicates(self, df: pd.DataFrame) -> list[Issue]:
        threshold = self.config.get("duplicate_threshold", 0.01)
        n_dup = int(df.duplicated(keep="first").sum())
        dup_pct = n_dup / len(df) if len(df) > 0 else 0.0
        issues = []
        if dup_pct > threshold:
            issues.append(Issue(
                column="__dataset__",
                issue_type="duplicate_rows",
                severity="high",
                affected_rows=n_dup,
                affected_pct=dup_pct,
                description=f"Dataset contains {n_dup} duplicate rows ({dup_pct:.1%}).",
                recommended_action="remove duplicates keeping first occurrence",
            ))
        return issues

    # checks outliers
    def _check_outliers(self, df: pd.DataFrame, profile: ProfileReport) -> list[Issue]:
        iqr_mult = self.config.get("outlier_iqr_multiplier", 1.5)
        issues = []
        for col, prof in profile.columns.items():
            if prof.semantic_type == "continuous" and prof.std_val and prof.std_val > 0:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) < 10:
                    continue
                Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
                n_outliers = int(((series < lower) | (series > upper)).sum())
                out_pct = n_outliers / len(series)
                if out_pct > 0.02:  # flag if >2% are outliers
                    issues.append(Issue(
                        column=col,
                        issue_type="outliers",
                        severity="medium",
                        affected_rows=n_outliers,
                        affected_pct=out_pct,
                        description=f"'{col}' has {n_outliers} IQR outliers ({out_pct:.1%}).",
                        recommended_action="cap outliers using IQR fences",
                    ))
        return issues

    def _check_type_inconsistencies(self, df: pd.DataFrame, profile: ProfileReport) -> list[Issue]:
        """Checks if an object/string column contains predominantly numeric strings."""
        issues = []

        for col, prof in profile.columns.items():
            # Skip non-object columns or text/ID types
            if df[col].dtype != "object" or prof.semantic_type in ("identifier", "free_text"):
                continue

            sample = df[col].dropna().head(200)
            if sample.empty:
                continue

            # Directly calculate the ratio of values that can be numeric (0.0 to 1.0)
            numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean()

            # Flag if 70% to 99% of sample is numeric (dirty numeric column)
            if 0.70 <= numeric_ratio < 1.0:
                inconsistency_pct = round(1.0 - numeric_ratio, 4)
                issues.append(Issue(
                    column=col,
                    issue_type="type_inconsistency",
                    severity="high",
                    affected_rows=int(inconsistency_pct * prof.count),
                    affected_pct=inconsistency_pct,
                    description=f"'{col}' is an object column containing mixed text and numbers.",
                    recommended_action="coerce to numeric, handle parse failures",
                ))

        return issues


# Backward-compatible alias
IssueDetector = IssueDetectorModule