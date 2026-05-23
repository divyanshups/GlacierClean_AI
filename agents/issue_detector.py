from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd
from utils.logger import get_logger, log_section

logger = get_logger("issue_detector")


class Severity(str, Enum):
    """How urgently an issue needs to be addressed."""
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"


@dataclass
class Issue:
    issue_type:       str
    column:           str | None
    severity:         Severity
    description:      str
    affected_count:   int        = 0
    affected_pct:     float      = 0.0
    suggested_action: str        = ""
    metadata:         dict       = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value   # use the string value, not the enum
        return data


class IssueDetector:
    # Maps severity to a sort key so CRITICAL always comes first
    SEVERITY_ORDER = {
        Severity.CRITICAL: 0,
        Severity.HIGH:     1,
        Severity.MEDIUM:   2,
        Severity.LOW:      3,
        Severity.INFO:     4,
    }

    def __init__(self, config: dict[str, Any]):
        cfg = config.get("issue_detector", {})
        # Missing-value thresholds that determine severity
        self.missing_critical: float = cfg.get("missing_critical",         0.50)
        self.missing_high:     float = cfg.get("missing_high",             0.20)
        self.missing_low:      float = cfg.get("missing_low",              0.05)
        # Outlier detection parameters
        self.outlier_method:   str   = cfg.get("outlier_method",           "iqr")
        self.iqr_factor:       float = cfg.get("outlier_iqr_factor",       1.5)
        self.z_threshold:      float = cfg.get("outlier_zscore_threshold", 3.0)
        # Minimum variance below which a numeric column is considered constant
        self.min_variance:     float = cfg.get("min_variance_threshold",   0.01)

    # ── Public interface ──────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[Issue]:
        log_section("🔍 Issue Detector")
        issues: list[Issue] = []
        row_count = len(df)
        issues += self._check_missing(df, profile, row_count)
        issues += self._check_duplicates(df, row_count)
        issues += self._check_outliers(df)
        issues += self._check_constant_columns(df)
        issues += self._check_cardinality(profile)
        issues += self._check_mixed_types(df)
        issues += self._check_skewness(profile)
        issues += self._check_string_anomalies(df)

        # Sort so the most urgent issues appear first
        issues.sort(key=lambda issue: self.SEVERITY_ORDER[issue.severity])

        for issue in issues:
            logger.info(
                "[%s] %s — %s",
                issue.severity.value.upper(),
                issue.issue_type,
                issue.description[:80],
            )

        logger.info("Issue detection complete — %d issues found.", len(issues))
        return issues

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_missing(
        self, df: pd.DataFrame, profile: dict, row_count: int
    ) -> list[Issue]:
        issues = []
        for col, col_stats in profile.get("columns", {}).items():
            missing_pct = col_stats.get("missing_pct", 0) / 100
            missing_count = col_stats.get("missing", 0)
            if missing_count == 0:
                continue

            if missing_pct >= self.missing_critical:
                severity = Severity.CRITICAL
                action   = "Consider dropping this column (>50% missing)"
            elif missing_pct >= self.missing_high:
                severity = Severity.HIGH
                action   = "Impute or drop column"
            elif missing_pct >= self.missing_low:
                severity = Severity.MEDIUM
                action   = "Impute missing values"
            else:
                severity = Severity.LOW
                action   = "Impute missing values"

            issues.append(Issue(
                issue_type       = "missing_values",
                column           = col,
                severity         = severity,
                description      = (
                    f"'{col}' has {missing_pct * 100:.1f}% missing values "
                    f"({missing_count:,} rows)"
                ),
                affected_count   = missing_count,
                affected_pct     = round(missing_pct * 100, 2),
                suggested_action = action,
            ))
        return issues

    def _check_duplicates(self, df: pd.DataFrame, row_count: int) -> list[Issue]:
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count == 0:
            return []

        pct      = duplicate_count / row_count * 100
        severity = (
            Severity.HIGH   if pct > 10 else
            Severity.MEDIUM if pct > 2  else
            Severity.LOW
        )
        return [Issue(
            issue_type       = "duplicate_rows",
            column           = None,
            severity         = severity,
            description      = f"{duplicate_count:,} duplicate rows detected ({pct:.1f}%)",
            affected_count   = duplicate_count,
            affected_pct     = round(pct, 2),
            suggested_action = "Remove duplicate rows (keep first occurrence)",
        )]

    def _check_outliers(self, df: pd.DataFrame) -> list[Issue]:
        issues = []
        numeric_columns = df.select_dtypes(include=np.number).columns

        for col in numeric_columns:
            series = df[col].dropna()
            if len(series) < 4:
                continue  # too few values to calculate meaningful quartiles

            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr    = q3 - q1
            if iqr == 0:
                continue  # constant column — handled separately

            outlier_count = int(
                ((series < q1 - self.iqr_factor * iqr) |
                 (series > q3 + self.iqr_factor * iqr)).sum()
            )
            if outlier_count == 0:
                continue

            pct      = outlier_count / len(series) * 100
            severity = (
                Severity.HIGH   if pct > 10 else
                Severity.MEDIUM if pct > 3  else
                Severity.LOW
            )
            issues.append(Issue(
                issue_type       = "outliers",
                column           = col,
                severity         = severity,
                description      = (
                    f"'{col}' has {outlier_count:,} outliers ({pct:.1f}%) by IQR method"
                ),
                affected_count   = outlier_count,
                affected_pct     = round(pct, 2),
                suggested_action = "Clip or winsorize outliers",
                metadata         = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr)},
            ))
        return issues

    def _check_constant_columns(self, df: pd.DataFrame) -> list[Issue]:
        issues = []
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                issues.append(Issue(
                    issue_type       = "constant_column",
                    column           = col,
                    severity         = Severity.MEDIUM,
                    description      = (
                        f"'{col}' has only 1 unique value — carries no information"
                    ),
                    affected_count   = len(df),
                    affected_pct     = 100.0,
                    suggested_action = "Drop this column",
                ))
        return issues

    def _check_cardinality(self, profile: dict) -> list[Issue]:
        issues = []
        for col, col_stats in profile.get("columns", {}).items():
            if col_stats.get("is_high_cardinality"):
                unique_count = col_stats.get("unique", 0)
                issues.append(Issue(
                    issue_type       = "high_cardinality",
                    column           = col,
                    severity         = Severity.INFO,
                    description      = (
                        f"'{col}' has {unique_count} unique values — "
                        "may be an ID or free-text field"
                    ),
                    affected_count   = unique_count,
                    suggested_action = "Review — consider grouping, hashing, or dropping",
                ))
        return issues

    def _check_mixed_types(self, df: pd.DataFrame) -> list[Issue]:
        issues = []
        for col in df.select_dtypes(include="object").columns:
            types_found = df[col].dropna().map(type).unique()
            if len(types_found) > 1:
                type_names = [t.__name__ for t in types_found]
                issues.append(Issue(
                    issue_type       = "mixed_types",
                    column           = col,
                    severity         = Severity.MEDIUM,
                    description      = (
                        f"'{col}' contains mixed Python types: {type_names}"
                    ),
                    suggested_action = "Coerce to a single type or investigate the data",
                ))
        return issues

    def _check_skewness(self, profile: dict) -> list[Issue]:
        issues = []
        for col, col_stats in profile.get("columns", {}).items():
            if col_stats.get("is_skewed"):
                skew_value = col_stats.get("skewness", 0)
                issues.append(Issue(
                    issue_type       = "high_skewness",
                    column           = col,
                    severity         = Severity.LOW,
                    description      = (
                        f"'{col}' is highly skewed (skew={skew_value:.2f}) — "
                        "may benefit from a log or sqrt transform"
                    ),
                    metadata         = {"skewness": skew_value},
                    suggested_action = "Apply log or Box-Cox transform if using for ML",
                ))
        return issues

    def _check_string_anomalies(self, df: pd.DataFrame) -> list[Issue]:
        issues = []
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().astype(str).head(500)
            has_whitespace = sample.str.contains(r"^\s|\s$", regex=True).any()
            if has_whitespace:
                issues.append(Issue(
                    issue_type       = "string_whitespace",
                    column           = col,
                    severity         = Severity.LOW,
                    description      = (
                        f"'{col}' contains entries with leading/trailing whitespace"
                    ),
                    suggested_action = "Strip whitespace from string column",
                ))
        return issues
