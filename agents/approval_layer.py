from __future__ import annotations
import copy
import datetime
from dataclasses import dataclass, field
from typing import Any
from agents.decision_agent import CleaningAction
from agents.llm_decision_agent import LLMProposal
from utils.logger import get_logger

logger = get_logger("approval_layer")

# Human-readable decision states
APPROVED = "approved"
REJECTED = "rejected"
MODIFIED = "modified"
PENDING  = "pending"


@dataclass
class ApprovalRecord:
    """
    A single entry in the approval audit trail.

    Captures what the user decided for each proposal, any edits they made,
    and an optional free-text note explaining their reasoning.
    """
    proposal_id:  str
    original:     dict        # snapshot of the proposal before any edits
    decision:     str         # one of: approved | rejected | modified
    final_params: dict        = field(default_factory=dict)
    final_action: str         = ""
    final_column: str | None  = None
    user_note:    str         = ""
    timestamp:    str         = ""


class ApprovalLayer:
    """
    Manages the human review step between proposal generation and execution.

    Typical workflow
    ----------------
    1. present(proposals)
         Store the LLM's proposals and reset any previous decisions.

    2. decide(proposal_id, APPROVED / REJECTED / MODIFIED, ...)
         Record the user's choice for each proposal, one at a time.
         For MODIFIED decisions, pass the revised action type, column, or
         parameters that should override the LLM's originals.

    3. get_cleaning_actions()
         Convert every approved or modified proposal into a CleaningAction
         object ready for CleaningAgent.clean().  Rejected proposals are
         silently ignored.
    """

    def __init__(self) -> None:
        # Internal storage — all keyed by proposal_id
        self._proposals: dict[str, LLMProposal]   = {}
        self._records:   dict[str, ApprovalRecord] = {}
        self._order:     list[str]                 = []   # preserves insertion order

    # ── Presentation ──────────────────────────────────────────────────────────

    def present(self, proposals: list[LLMProposal]) -> list[LLMProposal]:
        """
        Store a new batch of proposals and mark them all as pending.

        Any decisions from a previous round are discarded so the layer
        starts clean for each pipeline run.
        """
        self._proposals = {}
        self._records   = {}
        self._order     = []

        for proposal in proposals:
            self._proposals[proposal.proposal_id] = proposal
            self._order.append(proposal.proposal_id)
            # Reset to a neutral state so nothing is pre-approved
            proposal.user_decision = PENDING
            proposal.user_note     = ""

        logger.info(
            "ApprovalLayer: %d proposals presented for review.",
            len(proposals),
        )
        return proposals

    # ── Decision recording ────────────────────────────────────────────────────

    def decide(
        self,
        proposal_id:     str,
        decision:        str,
        new_action_type: str | None  = None,
        new_column:      str | None  = None,
        new_params:      dict | None = None,
        user_note:       str         = "",
    ) -> LLMProposal:
        """
        Record the user's decision for a single proposal.

        Parameters
        ----------
        proposal_id     : the proposal to act on, e.g. "P-001"
        decision        : "approved", "rejected", or "modified"
        new_action_type : replaces the LLM's action (MODIFIED only)
        new_column      : replaces the target column (MODIFIED only)
        new_params      : replaces the parameters dict (MODIFIED only)
        user_note       : optional free-text comment from the user

        Returns the updated LLMProposal for convenience.
        """
        if proposal_id not in self._proposals:
            raise KeyError(f"Unknown proposal_id: {proposal_id!r}")

        proposal = self._proposals[proposal_id]
        proposal.user_decision = decision
        proposal.user_note     = user_note

        # Apply user overrides when the proposal is being modified
        if decision == MODIFIED:
            if new_action_type:
                proposal.action_type = new_action_type
            if new_column is not None:
                proposal.column = new_column
            if new_params is not None:
                proposal.params = new_params

        # Write the audit record immediately so nothing is lost
        self._records[proposal_id] = ApprovalRecord(
            proposal_id  = proposal_id,
            original     = copy.deepcopy(proposal.to_dict()),
            decision     = decision,
            final_params = proposal.params,
            final_action = proposal.action_type,
            final_column = proposal.column,
            user_note    = user_note,
            timestamp    = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

        # Use an emoji prefix in the log so decisions are easy to scan
        icon = {"approved": "✅", "rejected": "❌", "modified": "✏️"}.get(decision, "?")
        logger.info(
            "%s  %s · %s on %s",
            icon,
            proposal_id,
            proposal.action_type,
            proposal.column or "global",
        )
        return proposal

    def approve_all(self) -> None:
        """Bulk-approve every proposal that is still pending."""
        for pid, proposal in self._proposals.items():
            if proposal.user_decision == PENDING:
                self.decide(pid, APPROVED)

    def reject_all(self) -> None:
        """Bulk-reject every proposal that is still pending."""
        for pid, proposal in self._proposals.items():
            if proposal.user_decision == PENDING:
                self.decide(pid, REJECTED)

    # ── Status helpers ────────────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        """Number of proposals the user has not yet reviewed."""
        return sum(1 for p in self._proposals.values() if p.user_decision == PENDING)

    @property
    def approved_count(self) -> int:
        return sum(1 for p in self._proposals.values() if p.user_decision == APPROVED)

    @property
    def rejected_count(self) -> int:
        return sum(1 for p in self._proposals.values() if p.user_decision == REJECTED)

    @property
    def modified_count(self) -> int:
        return sum(1 for p in self._proposals.values() if p.user_decision == MODIFIED)

    def all_decided(self) -> bool:
        """Return True when every proposal has received a decision."""
        return self.pending_count == 0

    def get_proposals(self) -> list[LLMProposal]:
        """Return proposals in the order they were originally presented."""
        return [self._proposals[pid] for pid in self._order]

    def get_approval_summary(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable summary of all decisions made so far.
        Useful for embedding in the final report.
        """
        return {
            "total":    len(self._proposals),
            "approved": self.approved_count,
            "rejected": self.rejected_count,
            "modified": self.modified_count,
            "pending":  self.pending_count,
            "records": [
                {
                    "proposal_id":  r.proposal_id,
                    "decision":     r.decision,
                    "final_action": r.final_action,
                    "final_column": r.final_column,
                    "user_note":    r.user_note,
                    "timestamp":    r.timestamp,
                }
                for r in self._records.values()
            ],
        }

    # ── Conversion to cleaning actions ────────────────────────────────────────

    def get_cleaning_actions(self) -> list[CleaningAction]:
        """
        Convert every approved or modified proposal into a CleaningAction.

        Actions are sorted by priority so structural changes (e.g. dropping
        columns) happen before value-level fixes (e.g. imputation).

        Rejected and still-pending proposals are silently skipped.
        """
        # Lower number = higher priority = runs first
        PRIORITY_MAP = {
            "drop_column":       0,
            "drop_duplicates":   0,
            "impute":            1,
            "normalize_text":    1,
            "coerce_dtype":      1,
            "outlier_clip":      2,
            "outlier_winsorize": 2,
            "outlier_remove":    2,
        }

        actions: list[CleaningAction] = []

        for index, pid in enumerate(self._order):
            proposal = self._proposals[pid]
            if proposal.user_decision not in (APPROVED, MODIFIED):
                continue

            actions.append(CleaningAction(
                action_id   = f"ACT-{index + 1:04d}",
                action_type = proposal.action_type,
                column      = proposal.column,
                params      = proposal.params,
                confidence  = proposal.confidence,
                rationale   = proposal.rationale,
                priority    = PRIORITY_MAP.get(proposal.action_type, 3),
            ))

        # Sort so higher-priority (lower number) actions run first;
        # use column name as a tiebreaker for consistent ordering
        actions.sort(key=lambda action: (action.priority, action.column or ""))

        logger.info(
            "ApprovalLayer: %d / %d proposals converted to cleaning actions.",
            len(actions),
            len(self._proposals),
        )
        return actions
