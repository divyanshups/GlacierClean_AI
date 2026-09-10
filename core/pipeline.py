import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml

from core.data_loader import load_dataset, LoadMetadata
from core.schema import CleaningAction, ExecutionLog
from module.profiler_module import ProfilerModule, ProfileReport
from module.issue_detector_module import IssueDetectorModule, Issue
from module.cleaning_module import CleaningModule
from module.evaluation_module import EvaluationModule, EvaluationReport
from utils.logger import get_logger, print_section

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    df_original: pd.DataFrame
    df_cleaned: pd.DataFrame
    metadata: LoadMetadata
    profile: ProfileReport
    issues: list[Issue]
    actions: list[CleaningAction]
    execution_logs: list[ExecutionLog]
    evaluation: Optional[EvaluationReport]
    output_paths: dict = field(default_factory=dict)

    def score_box(self) -> Optional[dict]:
        if self.evaluation is None:
            return None
        return self.evaluation.to_score_box()


class DataCleaningPipeline:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        project_cfg = self.config.get("project", {})

        # Decision log FILE path from config
        self.decision_log_file = project_cfg.get(
            "decision_log_file", "logs/decision_logs.json"
        )
        # If someone put a trailing slash by mistake, strip it
        self.decision_log_file = self.decision_log_file.rstrip("/\\")
        log_dir = os.path.dirname(self.decision_log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self.output_dir = project_cfg.get("output_dir", "outputs/")
        os.makedirs(self.output_dir, exist_ok=True)

        self.profiler = ProfilerModule(config_path)
        self.issue_detector = IssueDetectorModule(config_path)
        self.cleaner = CleaningModule()
        self.evaluator = EvaluationModule()

    # ── full run ────────────────────────────────────────────────────────────

    def run(
        self,
        file_path: str,
        actions: Optional[list[CleaningAction]] = None,
        save_outputs: bool = True,
        skip_evaluation: bool = False,
        source: str = "manual",
    ) -> PipelineResult:
        actions = actions or []

        print_section("1. Load")
        df, metadata = load_dataset(file_path, self.config_path)
        df_original = df.copy()

        print_section("2. Profile")
        profile = self.profiler.profile(df)

        print_section("3. Detect issues")
        issues = self.issue_detector.detect(df, profile)
        logger.info(f"{len(issues)} issue(s) found")

        print_section("4. Clean")
        if actions:
            df_cleaned, exec_logs = self.cleaner.execute_plan(df, actions)
        else:
            logger.info("No actions supplied — cleaned data equals original")
            df_cleaned = df.copy()
            exec_logs = []

        evaluation = None
        if not skip_evaluation and actions:
            print_section("5. Evaluate")
            evaluation = self.evaluator.evaluate(df_original, df_cleaned, exec_logs)
            logger.info(evaluation.summary)
        else:
            print_section("5. Evaluate")
            logger.info("Evaluation skipped")

        output_paths = {}
        if save_outputs:
            print_section("6. Save outputs")
            output_paths = self._save_outputs(
                df_cleaned=df_cleaned,
                metadata=metadata,
            )
            # Decision JSON log
            if actions:
                log_path = self.log_run_changes(
                    df_cleaned=df_cleaned,
                    metadata=metadata,
                    issues=issues,
                    actions=actions,
                    exec_logs=exec_logs,
                    evaluation=evaluation,
                    source=source,
                )
                output_paths["decision_logs"] = log_path

        result = PipelineResult(
            df_original=df_original,
            df_cleaned=df_cleaned,
            metadata=metadata,
            profile=profile,
            issues=issues,
            actions=actions,
            execution_logs=exec_logs,
            evaluation=evaluation,
            output_paths=output_paths,
        )
        logger.info("Pipeline run complete")
        return result

    # ── step helpers ────────────────────────────────────────────────────────

    def load(self, file_path: str) -> tuple[pd.DataFrame, LoadMetadata]:
        return load_dataset(file_path, self.config_path)

    def profile_only(self, df: pd.DataFrame) -> ProfileReport:
        return self.profiler.profile(df)

    def detect_only(self, df: pd.DataFrame, profile: ProfileReport) -> list[Issue]:
        return self.issue_detector.detect(df, profile)

    def clean_only(
        self,
        df: pd.DataFrame,
        actions: list[CleaningAction],
    ) -> tuple[pd.DataFrame, list[ExecutionLog]]:
        return self.cleaner.execute_plan(df, actions)

    def evaluate_only(
        self,
        df_original: pd.DataFrame,
        df_cleaned: pd.DataFrame,
        ops: list[ExecutionLog],
    ) -> EvaluationReport:
        return self.evaluator.evaluate(df_original, df_cleaned, ops)

            # ── clean + evaluate ───────────────────────────────────

    def run_clean_and_evaluate(
        self,
        df: pd.DataFrame,
        actions: list[CleaningAction],
        source: str = "manual",
        metadata: Optional[LoadMetadata] = None,
        issues: Optional[list[Issue]] = None,
        save_outputs: bool = False,
        save_decision_log: bool = True,
    ) -> tuple[pd.DataFrame, list[ExecutionLog], EvaluationReport]:
        """
        Used by Streamlit (manual bucket + LLM).
        Executes actions, evaluates, and logs run changes.
        """
        df_original = df.copy()
        df_cleaned, logs = self.cleaner.execute_plan(df, actions)
        report = self.evaluator.evaluate(df_original, df_cleaned, logs)

        if save_outputs and metadata is not None:
            self._save_outputs(df_cleaned=df_cleaned, metadata=metadata)

        if save_decision_log and metadata is not None:
            self.log_run_changes(
                df_cleaned=df_cleaned,
                metadata=metadata,
                issues=issues or [],
                actions=actions,
                exec_logs=logs,
                evaluation=report,
                source=source,
            )

        return df_cleaned, logs, report

    # ── persistence ─────────────────────────────────────────────────────────

    def _save_outputs(
        self,
        df_cleaned: pd.DataFrame,
        metadata: LoadMetadata,
    ) -> dict:
        """Save cleaned CSV only."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(metadata.file_name)[0]
        paths = {}

        csv_path = os.path.join(self.output_dir, f"{base}_cleaned_{stamp}.csv")
        df_cleaned.to_csv(csv_path, index=False)
        paths["cleaned_csv"] = csv_path
        logger.info(f"Saved cleaned data → {csv_path}")

        return paths

    def log_run_changes(
        self,
        df_cleaned: pd.DataFrame,
        metadata: LoadMetadata,
        issues: list[Issue],
        actions: list[CleaningAction],
        exec_logs: list[ExecutionLog],
        evaluation: Optional[EvaluationReport],
        source: str = "manual",
    ) -> str:
        """
        Appends decision / change log JSON entry to the history list in config path.
        Returns the file path written.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "source_file": metadata.file_name,
            "fingerprint": metadata.dataset_fingerprint,
            "shape_original": list(metadata.shape),
            "shape_cleaned": list(df_cleaned.shape),
            "source": source,
            "issues": [self._issue_to_dict(i) for i in issues],
            "actions": [self._action_to_dict(a) for a in actions],
            "execution_logs": [self._log_to_dict(l) for l in exec_logs],
            "evaluation": None,
        }

        if evaluation is not None:
            report["evaluation"] = {
                "summary": evaluation.summary,
                "delta": evaluation.delta,
                "before": evaluation.before.to_dict()
                if hasattr(evaluation.before, "to_dict")
                else {
                    "overall": evaluation.before.overall,
                    "completeness": evaluation.before.completeness,
                    "consistency": evaluation.before.consistency,
                    "uniqueness": evaluation.before.uniqueness,
                    "validity": evaluation.before.validity,
                },
                "after": evaluation.after.to_dict()
                if hasattr(evaluation.after, "to_dict")
                else {
                    "overall": evaluation.after.overall,
                    "completeness": evaluation.after.completeness,
                    "consistency": evaluation.after.consistency,
                    "uniqueness": evaluation.after.uniqueness,
                    "validity": evaluation.after.validity,
                },
                "score_box": evaluation.to_score_box()
                if hasattr(evaluation, "to_score_box")
                else None,
            }

        decision_log_path = self.decision_log_file
        parent = os.path.dirname(decision_log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # --- READ EXISTING LOGS & APPEND ---
        logs_history = []
        if os.path.exists(decision_log_path):
            try:
                with open(decision_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logs_history = data
                    elif isinstance(data, dict):
                        # Convert previous single-dict log to list format
                        logs_history = [data]
            except Exception:
                logs_history = []

        # Append new run report to the list
        logs_history.append(report)

        # Write full list back
        with open(decision_log_path, "w", encoding="utf-8") as f:
            json.dump(logs_history, f, indent=2, default=str)

        logger.info(f"Appended decision log → {decision_log_path} (Total entries: {len(logs_history)})")
        return decision_log_path

    # ── serializers ─────────────────────────────────────────────────────────

    @staticmethod
    def _issue_to_dict(issue: Issue) -> dict:
        return {
            "column": issue.column,
            "issue_type": issue.issue_type,
            "severity": issue.severity,
            "affected_rows": issue.affected_rows,
            "affected_pct": issue.affected_pct,
            "description": issue.description,
            "recommended_action": issue.recommended_action,
        }

    @staticmethod
    def _action_to_dict(action: CleaningAction) -> dict:
        return {
            "column": action.column,
            "operation": action.operation,
            "params": action.params,
            "source": action.source,
            "rationale": action.rationale,
        }

    @staticmethod
    def _log_to_dict(log: ExecutionLog) -> dict:
        return {
            "column": log.column,
            "operation": log.operation,
            "status": log.status,
            "message": log.message,
        }