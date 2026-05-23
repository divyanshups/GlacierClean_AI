from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
from agents.issue_detector import Issue, Severity
from utils.logger import get_logger, log_decision, log_section

logger = get_logger("decision_agent")

@dataclass
class CleaningAction:
    
    action_id:   str
    action_type: str
    column:      str | None
    params:      dict  = field(default_factory=dict)
    confidence:  float = 1.0
    rationale:   str   = ""
    priority:    int   = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class DecisionAgent:

    def __init__(self, config: dict[str, Any]):
        self.config        = config
        self.cleaning_cfg  = config.get("cleaning", {})
        self.confidence_threshold: float = (
            config.get("pipeline", {}).get("confidence_threshold", 0.75)
        )
        self.decision_log_path: str = (
            config.get("logging", {}).get("decision_log", "memory/decision_logs.json")
        )
        self._action_counter = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def decide(
        self,
        issues:   list[Issue],
        df_shape: tuple[int, int],
    ) -> list[CleaningAction]:
        
        log_section("🧠 Decision Agent", f"Processing {len(issues)} issues")

        actions: list[CleaningAction] = []
        # Track which action types have already been planned for each column
        # so we never plan the same operation twice on the same column.
        actions_planned_per_column: dict[str | None, set[str]] = {}

        for issue in issues:
            column_actions_so_far = actions_planned_per_column.setdefault(
                issue.column, set()
            )
            new_actions = self._resolve(issue, column_actions_so_far, df_shape)
            for action in new_actions:
                column_actions_so_far.add(action.action_type)
                actions.append(action)
                self._record_decision(action, issue)

        # Sort for deterministic, priority-respecting execution order
        actions.sort(key=lambda a: (a.priority, a.column or ""))

        above_threshold = sum(
            1 for a in actions if a.confidence >= self.confidence_threshold
        )
        logger.info(
            "Decision plan ready — %d actions total, %d above confidence threshold.",
            len(actions),
            above_threshold,
        )
        return actions

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _next_id(self) -> str:
        """Generate the next sequential action ID."""
        self._action_counter += 1
        return f"ACT-{self._action_counter:04d}"

    def _resolve(
        self,
        issue:          Issue,
        existing_types: set[str],
        df_shape:       tuple[int, int],
    ) -> list[CleaningAction]:
        
        issue_type = issue.issue_type
        column     = issue.column
        severity   = issue.severity
        pct        = issue.affected_pct

        actions: list[CleaningAction] = []

        # ── Missing values ────────────────────────────────────────────────────
        if issue_type == "missing_values":
            if pct >= 70:
                # Imputing more than 70% of a column produces unreliable data —
                # it's better to drop it entirely.
                if "drop_column" not in existing_types:
                    actions.append(CleaningAction(
                        action_id   = self._next_id(),
                        action_type = "drop_column",
                        column      = column,
                        confidence  = 0.90,
                        rationale   = (
                            f"Column '{column}' is {pct:.1f}% missing. "
                            "Imputing this much data would introduce severe bias — "
                            "dropping the column is the safer option."
                        ),
                        priority    = 0,
                    ))
            else:
                # Choose strategy from config; boost confidence for severe cases
                num_strategy = self.cleaning_cfg.get("missing_numeric_strategy", "median")
                cat_strategy = self.cleaning_cfg.get("missing_categorical_strategy", "mode")
                confidence   = 0.95 if severity in (Severity.CRITICAL, Severity.HIGH) else 0.80

                if "impute" not in existing_types:
                    actions.append(CleaningAction(
                        action_id   = self._next_id(),
                        action_type = "impute",
                        column      = column,
                        params      = {
                            "numeric_strategy":     num_strategy,
                            "categorical_strategy": cat_strategy,
                            "constant":             self.cleaning_cfg.get(
                                "missing_constant_fill", "Unknown"
                            ),
                        },
                        confidence  = confidence,
                        rationale   = (
                            f"Impute '{column}' ({pct:.1f}% missing) using "
                            f"{num_strategy} for numeric values and "
                            f"{cat_strategy} for categorical values."
                        ),
                        priority    = 1,
                    ))

        # ── Duplicate rows ────────────────────────────────────────────────────
        elif issue_type == "duplicate_rows":
            keep_strategy = self.cleaning_cfg.get("duplicate_strategy", "keep_first")
            if "drop_duplicates" not in existing_types:
                actions.append(CleaningAction(
                    action_id   = self._next_id(),
                    action_type = "drop_duplicates",
                    column      = None,  # applies to the whole dataset
                    params      = {"keep": keep_strategy},
                    confidence  = 0.98,
                    rationale   = (
                        f"Remove {issue.affected_count:,} duplicate rows "
                        f"(strategy: {keep_strategy})."
                    ),
                    priority    = 0,
                ))

        # ── Outliers ──────────────────────────────────────────────────────────
        elif issue_type == "outliers":
            strategy = self.cleaning_cfg.get("outlier_strategy", "clip")
            if f"outlier_{strategy}" not in existing_types:
                actions.append(CleaningAction(
                    action_id   = self._next_id(),
                    action_type = f"outlier_{strategy}",
                    column      = column,
                    params      = {
                        "method":     self.config.get("issue_detector", {}).get(
                            "outlier_method", "iqr"
                        ),
                        "iqr_factor": self.config.get("issue_detector", {}).get(
                            "outlier_iqr_factor", 1.5
                        ),
                    },
                    confidence  = 0.85,
                    rationale   = (
                        f"'{column}' has {pct:.1f}% outliers — applying '{strategy}'."
                    ),
                    priority    = 2,
                ))

        # ── Constant column ───────────────────────────────────────────────────
        elif issue_type == "constant_column":
            # A column with only one unique value carries zero information
            if "drop_column" not in existing_types:
                actions.append(CleaningAction(
                    action_id   = self._next_id(),
                    action_type = "drop_column",
                    column      = column,
                    confidence  = 0.95,
                    rationale   = (
                        f"'{column}' contains only one unique value — "
                        "zero variance, no predictive power."
                    ),
                    priority    = 0,
                ))

        # ── String whitespace ─────────────────────────────────────────────────
        elif issue_type == "string_whitespace":
            if self.cleaning_cfg.get("text_normalization", True):
                if "normalize_text" not in existing_types:
                    actions.append(CleaningAction(
                        action_id   = self._next_id(),
                        action_type = "normalize_text",
                        column      = column,
                        params      = {"lowercase": True, "strip": True},
                        confidence  = 0.90,
                        rationale   = (
                            f"Strip whitespace and normalise text in '{column}'."
                        ),
                        priority    = 1,
                    ))

        # ── Mixed types ───────────────────────────────────────────────────────
        elif issue_type == "mixed_types":
            if "coerce_dtype" not in existing_types:
                actions.append(CleaningAction(
                    action_id   = self._next_id(),
                    action_type = "coerce_dtype",
                    column      = column,
                    confidence  = 0.75,
                    rationale   = (
                        f"Coerce mixed-type column '{column}' to a single uniform dtype."
                    ),
                    priority    = 1,
                ))

        # ── High skewness — informational only ────────────────────────────────
        elif issue_type == "high_skewness":
            # Applying a log or Box-Cox transform automatically is risky without
            # domain knowledge, so we log a note and leave it to the user.
            logger.info(
                "Skewness noted for '%s' — no automatic transform applied.", column
            )

        # ── High cardinality — informational only ─────────────────────────────
        elif issue_type == "high_cardinality":
            logger.info(
                "High cardinality noted for '%s' — no automatic action.", column
            )

        return actions

    def _record_decision(self, action: CleaningAction, issue: Issue) -> None:
        """Write the decision to the persistent JSON decision log."""
        log_decision(
            decision_log_path = self.decision_log_path,
            agent             = "DecisionAgent",
            action            = action.action_type,
            rationale         = action.rationale,
            params            = {"column": action.column, **action.params},
            confidence        = action.confidence,
        )
