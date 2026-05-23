from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Callable
import pandas as pd
from agents.cleaning_agent   import CleaningAgent
from agents.decision_agent   import DecisionAgent
from agents.evaluation_agent import EvaluationAgent
from agents.issue_detector   import IssueDetector
from agents.profiler_agent   import ProfilerAgent
from core.data_loader        import load_dataset, save_dataset
from utils.logger            import get_logger, log_section

logger = get_logger("pipeline")

DEFAULT_CONFIG_PATH = "config/config.yaml"

def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and return the YAML configuration file as a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class CleaningPipeline:
    """
    >>> pipeline = CleaningPipeline()
    >>> result   = pipeline.run("data/raw/my_file.csv")
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config    = load_config(config_path)
        self.profiler  = ProfilerAgent(self.config)
        self.detector  = IssueDetector(self.config)
        self.decider   = DecisionAgent(self.config)
        self.cleaner   = CleaningAgent(self.config)
        self.evaluator = EvaluationAgent(self.config)

        # Optional callback for live progress updates (registered by the Streamlit UI)
        self.on_progress: Callable[[str, float], None] | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        source,                  # file path, BytesIO, or StringIO
        file_name:   str  = "",
        save_output: bool = True,
    ) -> dict[str, Any]:
        log_section("Autonomous Data Cleaning Agent", "Pipeline starting…")
        self._progress("Loading dataset…", 0.05)

        # 1. Load
        df_raw = load_dataset(source, file_name=file_name)
        logger.info("Loaded dataset — shape: %s", df_raw.shape)
        self._progress("Dataset loaded ✓", 0.10)

        # 2. Profile
        self._progress("Profiling data…", 0.20)
        profile = self.profiler.profile(df_raw)
        self._progress("Profiling complete ✓", 0.35)

        # 3. Detect issues
        self._progress("Detecting issues…", 0.40)
        issues = self.detector.detect(df_raw, profile)
        self._progress(f"{len(issues)} issues detected ✓", 0.50)

        # 4. Plan cleaning actions
        self._progress("Planning cleaning actions…", 0.55)
        actions = self.decider.decide(issues, df_raw.shape)
        self._progress(f"{len(actions)} actions planned ✓", 0.65)

        # 5. Execute cleaning
        self._progress("Applying cleaning actions…", 0.70)
        df_cleaned = self.cleaner.clean(df_raw, actions)
        change_log = self.cleaner.change_log
        self._progress("Cleaning complete ✓", 0.85)

        # 6. Evaluate results
        self._progress("Evaluating results…", 0.90)
        report = self.evaluator.evaluate(
            df_before  = df_raw,
            df_after   = df_cleaned,
            change_log = change_log,
            actions    = actions,
            issues     = issues,
            profile    = profile,
        )
        self._progress("Evaluation complete ✓", 0.95)

        # 7. Optionally save the cleaned dataset
        if save_output:
            output_path = self.config.get("output", {}).get(
                "cleaned_csv", "outputs/cleaned_dataset.csv"
            )
            save_dataset(df_cleaned, output_path)
            logger.info("Cleaned dataset saved → %s", output_path)

        self._progress("✅ Pipeline complete!", 1.0)
        log_section(
            "✅ Done",
            f"Quality: {report['quality_score']['before']} → {report['quality_score']['after']}",
        )

        return {
            "df_raw":      df_raw,
            "df_cleaned":  df_cleaned,
            "report":      report,
            "change_log":  change_log,
            "issues":      issues,
            "actions":     actions,
            "profile":     profile,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _progress(self, message: str, fraction: float) -> None:
        """Forward a progress update to the registered callback, if any."""
        if self.on_progress:
            self.on_progress(message, fraction)
