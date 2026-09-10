from dataclasses import dataclass
import pandas as pd

from core.schema import ExecutionLog
from utils.metrics import compute_quality_report, compare_quality, QualityReport
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    before: QualityReport
    after: QualityReport
    delta: dict
    operations_applied: list[ExecutionLog]
    summary: str

    def to_score_box(self) -> dict:
        """Flat structure for Streamlit metrics / charts."""
        return {
            "before": {
                "completeness": self.before.completeness,
                "consistency": self.before.consistency,
                "uniqueness": self.before.uniqueness,
                "validity": self.before.validity,
                "overall": self.before.overall,
            },
            "after": {
                "completeness": self.after.completeness,
                "consistency": self.after.consistency,
                "uniqueness": self.after.uniqueness,
                "validity": self.after.validity,
                "overall": self.after.overall,
            },
            "delta": {
                "completeness": self.delta.get("completeness", 0.0),
                "consistency": self.delta.get("consistency", 0.0),
                "uniqueness": self.delta.get("uniqueness", 0.0),
                "validity": self.delta.get("validity", 0.0),
                "overall": self.delta.get("overall", 0.0),
            },
            "improved": self.delta.get("improved", False),
            "rows_removed": self.delta.get("rows_removed", 0),
            "columns_removed": self.delta.get("columns_removed", 0),
            "summary": self.summary,
            "n_rows_before": self.before.n_rows,
            "n_rows_after": self.after.n_rows,
            "n_cols_before": self.before.n_columns,
            "n_cols_after": self.after.n_columns,
        }


class EvaluationModule:
    def evaluate(
        self,
        df_original: pd.DataFrame,
        df_cleaned: pd.DataFrame,
        ops: list[ExecutionLog],
    ) -> EvaluationReport:
        logger.info("Running evaluation...")

        before = compute_quality_report(df_original)
        after = compute_quality_report(df_cleaned)
        delta = compare_quality(before, after)

        skip = {"improved", "rows_removed", "columns_removed"}
        improved_dims = [
            k for k, v in delta.items()
            if k not in skip and isinstance(v, (int, float)) and v > 0
        ]
        regressed_dims = [
            k for k, v in delta.items()
            if k not in skip and isinstance(v, (int, float)) and v < 0
        ]

        summary = f"Overall quality changed by {delta['overall']:+.2%}. "
        if improved_dims:
            summary += f"Improved: {', '.join(improved_dims)}. "
        if regressed_dims:
            summary += f"Regressed: {', '.join(regressed_dims)}. "

        rows_removed = delta.get("rows_removed", 0)
        if rows_removed:
            summary += f"{rows_removed} row(s) removed. "

        n_ok = sum(1 for op in ops if getattr(op, "status", "") == "success")
        n_fail = sum(1 for op in ops if getattr(op, "status", "") == "error")
        if ops:
            summary += f"Actions: {n_ok} succeeded"
            if n_fail:
                summary += f", {n_fail} failed"
            summary += "."

        logger.info(f"Evaluation complete. {summary}")

        return EvaluationReport(
            before=before,
            after=after,
            delta=delta,
            operations_applied=ops,
            summary=summary.strip(),
        )


# Optional alias
EvaluationAgent = EvaluationModule