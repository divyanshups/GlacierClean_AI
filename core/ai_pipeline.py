from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import pandas as pd
from agents.approval_layer     import ApprovalLayer
from agents.cleaning_agent     import CleaningAgent
from agents.evaluation_agent   import EvaluationAgent
from agents.issue_detector     import IssueDetector
from agents.llm_decision_agent import DEFAULT_MODEL, LLMDecisionAgent, LLMProposal
from agents.profiler_agent     import ProfilerAgent
from core.data_loader          import load_dataset, save_dataset
from core.pipeline             import load_config
from utils.logger              import get_logger, log_section

logger = get_logger("ai_pipeline")

DEFAULT_CONFIG_PATH = "config/config.yaml"


class AIPipeline:
    def __init__(
        self,
        config_path: str       = DEFAULT_CONFIG_PATH,
        config:      dict | None = None,
        llm_model:   str       = DEFAULT_MODEL,
    ):
        # Load config from file unless one was passed in directly
        self.config = config if config is not None else load_config(config_path)
        self.model  = llm_model

        # Instantiate all pipeline agents
        self.profiler  = ProfilerAgent(self.config)
        self.detector  = IssueDetector(self.config)
        self.llm_agent = LLMDecisionAgent(self.config, model=llm_model)
        self.approval  = ApprovalLayer()
        self.cleaner   = CleaningAgent(self.config)
        self.evaluator = EvaluationAgent(self.config)

        # Optional callback the UI registers to receive progress updates
        self.on_progress: Callable[[str, float], None] | None = None

        # State passed between the two pipeline phases
        self._df_raw:            pd.DataFrame | None = None
        self._profile:           dict[str, Any]      = {}
        self._issues:            list                 = []
        self._proposals:         list[LLMProposal]   = []
        self._executive_summary: str                 = ""

    # ── Progress reporting ────────────────────────────────────────────────────

    def _report_progress(self, message: str, fraction: float) -> None:
        """Send a progress update to the registered callback, if any."""
        if self.on_progress:
            self.on_progress(message, fraction)

    # ── Phase 1: Load → Profile → Detect → Propose ───────────────────────────

    def run_until_approval(
        self,
        source,
        file_name: str = "",
    ) -> dict[str, Any]:
        log_section("🚀 AI Pipeline — Phase 1", "Analysis & Proposal generation")

        # 1. Load
        self._report_progress("Loading dataset…", 0.05)
        self._df_raw = load_dataset(source, file_name=file_name)
        logger.info("Loaded — shape: %s", self._df_raw.shape)
        self._report_progress("Dataset loaded ✓", 0.12)

        # 2. Profile
        self._report_progress("Profiling dataset…", 0.18)
        self._profile = self.profiler.profile(self._df_raw)
        self._report_progress("Profiling complete ✓", 0.32)

        # 3. Detect issues
        self._report_progress("Detecting issues…", 0.38)
        self._issues = self.detector.detect(self._df_raw, self._profile)
        self._report_progress(f"{len(self._issues)} issues detected ✓", 0.50)

        # 4. Generate LLM proposals
        self._report_progress(f"AI analysing dataset with {self.model}…", 0.55)
        self._proposals, self._executive_summary = self.llm_agent.propose(
            self._issues, self._profile, self._df_raw.shape
        )
        self.approval.present(self._proposals)
        self._report_progress(
            f"{len(self._proposals)} proposals ready — awaiting your approval", 0.70
        )

        return {
            "df_raw":            self._df_raw,
            "profile":           self._profile,
            "issues":            self._issues,
            "proposals":         self._proposals,
            "executive_summary": self._executive_summary,
            "llm_available":     self.llm_agent.llm_available,
            "llm_error":         self.llm_agent.llm_error,
            "model":             self.model,
        }

    # ── Phase 2: Execute → Evaluate ───────────────────────────────────────────

    def run_after_approval(self, save_output: bool = False) -> dict[str, Any]:
        if self._df_raw is None:
            raise RuntimeError("run_until_approval() must be called before run_after_approval().")

        log_section("⚡ AI Pipeline — Phase 2", "Executing approved actions")

        # Convert approved and modified proposals to CleaningAction objects
        actions          = self.approval.get_cleaning_actions()
        approval_summary = self.approval.get_approval_summary()

        logger.info(
            "Approval summary: %d approved, %d rejected, %d modified, %d pending",
            approval_summary["approved"],
            approval_summary["rejected"],
            approval_summary["modified"],
            approval_summary["pending"],
        )

        if not actions:
            logger.warning("No actions approved — returning original dataset unchanged.")
            df_cleaned = self._df_raw.copy()
        else:
            self._report_progress("Executing approved cleaning actions…", 0.75)
            df_cleaned = self.cleaner.clean(self._df_raw, actions)
            self._report_progress("Cleaning complete ✓", 0.88)

        # Evaluate and generate the report
        self._report_progress("Evaluating results…", 0.92)
        report = self.evaluator.evaluate(
            df_before  = self._df_raw,
            df_after   = df_cleaned,
            change_log = self.cleaner.change_log,
            actions    = actions,
            issues     = self._issues,
            profile    = self._profile,
        )
        # Embed the approval audit trail in the report
        report["approval_summary"] = approval_summary

        if save_output:
            output_path = self.config.get("output", {}).get(
                "cleaned_csv", "outputs/cleaned_dataset.csv"
            )
            save_dataset(df_cleaned, output_path)
            logger.info("Saved → %s", output_path)

        self._report_progress("✅ Pipeline complete!", 1.0)
        log_section(
            "✅ Done",
            f"Quality: {report['quality_score']['before']} → {report['quality_score']['after']} "
            f"| Approved: {approval_summary['approved']} / {approval_summary['total']}",
        )

        return {
            "df_raw":            self._df_raw,
            "df_cleaned":        df_cleaned,
            "report":            report,
            "change_log":        self.cleaner.change_log,
            "issues":            self._issues,
            "actions":           actions,
            "profile":           self._profile,
            "proposals":         self._proposals,
            "executive_summary": self._executive_summary,
            "approval_summary":  approval_summary,
        }

    # ── LLM helpers the UI can call directly ─────────────────────────────────

    def explain_proposal(self, proposal: LLMProposal) -> str:
        """Ask the LLM for a deeper explanation of a specific proposal."""
        return self.llm_agent.explain_action(proposal)

    def suggest_modification(self, proposal: LLMProposal, feedback: str) -> dict:
        """Ask the LLM to suggest a revised version based on user feedback."""
        return self.llm_agent.suggest_modification(proposal, feedback)

    def switch_model(self, model_name: str) -> None:
        """Switch to a different Ollama model without restarting the pipeline."""
        self.model = model_name
        self.llm_agent.set_model(model_name)

    @property
    def available_models(self) -> list[str]:
        from agents.llm_decision_agent import _list_ollama_models
        return _list_ollama_models()

    @property
    def llm_available(self) -> bool:
        return self.llm_agent.llm_available

    @property
    def llm_error(self) -> str:
        return self.llm_agent.llm_error
