from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
from utils.logger import get_logger, log_section
from utils.metrics import compare_metrics, compute_quality_metrics, quality_score

logger = get_logger("evaluation_agent")

class EvaluationAgent:
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        # Where to save the JSON report
        self.report_path: str = config.get("output", {}).get(
            "report_json", "outputs/reports/cleaning_report.json"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate(
        self,
        df_before:  pd.DataFrame,
        df_after:   pd.DataFrame,
        change_log: list[dict],
        actions:    list,
        issues:     list,
        profile:    dict,
    ) -> dict[str, Any]:
        log_section("📈 Evaluation Agent")

        metrics_before = compute_quality_metrics(df_before)
        metrics_after  = compute_quality_metrics(df_after)
        delta          = compare_metrics(metrics_before, metrics_after)

        score_before = quality_score(metrics_before)
        score_after  = quality_score(metrics_after)

        report: dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "input_shape": {
                "rows": int(df_before.shape[0]),
                "cols": int(df_before.shape[1]),
            },
            "output_shape": {
                "rows": int(df_after.shape[0]),
                "cols": int(df_after.shape[1]),
            },
            "quality_score": {
                "before":      score_before,
                "after":       score_after,
                "improvement": round(score_after - score_before, 2),
            },
            "metrics_before":   metrics_before,
            "metrics_after":    metrics_after,
            "delta":            delta,
            "issues_detected":  len(issues),
            "actions_planned":  len(actions),
            "actions_applied":  len(change_log),
            "change_log":       change_log,
            "issues": [
                issue.to_dict() if hasattr(issue, "to_dict") else dict(issue)
                for issue in issues
            ],
            "profile_summary": {
                "global": profile.get("global", {}),
            },
        }

        self._save_report(report)
        self._log_summary(report)
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_report(self, report: dict[str, Any]) -> None:
        """Write the report to disk as a prettily-indented JSON file."""
        path = Path(self.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Report saved → %s", path)

    def _log_summary(self, report: dict[str, Any]) -> None:
        """Print the key headline numbers to the log for quick review."""
        qs    = report["quality_score"]
        delta = report["delta"]

        logger.info(
            "Quality score: %.1f → %.1f (+%.1f pts)",
            qs["before"], qs["after"], qs["improvement"],
        )
        logger.info(
            "Missing: %.2f%% → %.2f%%  |  Duplicates: %.2f%% → %.2f%%  |  Outliers: %.2f%% → %.2f%%",
            delta["missing_rate"]["before"],   delta["missing_rate"]["after"],
            delta["duplicate_rate"]["before"], delta["duplicate_rate"]["after"],
            delta["outlier_rate"]["before"],   delta["outlier_rate"]["after"],
        )
        logger.info(
            "Rows retained: %d / %d (%.1f%%)",
            report["output_shape"]["rows"],
            report["input_shape"]["rows"],
            delta["row_retention"]["after"],
        )
