from __future__ import annotations
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any
from utils.logger import get_logger, log_section

logger = get_logger("llm_decision_agent")

# Default connection settings — change if Ollama is running on another port
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "qwen2.5:3b"   # swap for llama3, phi3, gemma2, mistral, etc.
REQUEST_TIMEOUT = 120            # seconds before the request times out

# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class LLMProposal:
    
    proposal_id:   str
    action_type:   str
    column:        str | None
    params:        dict      = field(default_factory=dict)
    confidence:    float     = 0.85
    rationale:     str       = ""
    risk_level:    str       = "low"
    alternatives:  list[str] = field(default_factory=list)
    user_decision: str       = "pending"
    user_note:     str       = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Ollama connection helpers ─────────────────────────────────────────────────
def _is_ollama_running() -> bool:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags"), timeout=5
        ):
            return True
    except Exception:
        return False


def _list_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags"), timeout=5
        ) as response:
            data = json.loads(response.read())
            return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []


def _resolve_model_name(requested: str, available: list[str]) -> str:
    if not available:
        return requested
    if requested in available:
        return requested
    for full_name in available:
        if full_name.split(":")[0] == requested:
            return full_name
    requested_lower = requested.lower()
    for full_name in available:
        if full_name.lower().startswith(requested_lower):
            return full_name
    logger.warning(
        "Model %r not found in Ollama. Available: %s. Falling back to %r.",
        requested, available, available[0],
    )
    return available[0]


def _call_ollama(
    prompt: str,
    model:  str = DEFAULT_MODEL,
) -> str:
    
    # Resolve "mistral" → "mistral:latest" before the API call
    available     = _list_ollama_models()
    resolved_name = _resolve_model_name(model, available) if available else model
    logger.debug("Resolved model: %r → %r", model, resolved_name)

    payload = json.dumps({
        "model":  resolved_name,
        "prompt": prompt,
        "stream": True,          # streaming avoids timeouts on slow hardware
        "options": {
            "temperature": 0.2,  # low temperature = more deterministic, less creative
            "top_p":       0.9,
            "num_predict": 2048,
        },
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            # Reassemble the streamed NDJSON chunks into a single string
            full_text = ""
            for raw_line in response:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                full_text += chunk.get("response", "")
                if chunk.get("done"):
                    break
            return full_text

    except urllib.error.HTTPError as error:
        # Try to include the server's error message for easier debugging
        body = ""
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:
            pass

        if error.code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for model {resolved_name!r}. "
                f"The model is not installed locally. Run:  ollama pull {model}\n"
                f"Currently available models: {available}"
            ) from error

        raise RuntimeError(
            f"Ollama HTTP {error.code} error: {body[:300]}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Is 'ollama serve' running?  Detail: {error.reason}"
        ) from error


def _extract_json_block(text: str) -> Any:
    # Remove markdown code fences before attempting to parse
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try parsing the whole response directly first (the happy path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to finding the first [...] or {...} block
    for pattern in (r"(\[.*?\])", r"(\{.*?\})"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"No valid JSON found in LLM response:\n{text[:500]}")


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_analysis_prompt(
    profile_summary: dict[str, Any],
    issues:          list[dict[str, Any]],
    df_shape:        tuple[int, int],
) -> str:
    rows, cols = df_shape
    global_stats = profile_summary.get("global", {})

    # Format the top 20 issues as a bulleted list inside the prompt
    issue_lines = "\n".join(
        f"  - [{issue['severity'].upper()}] {issue['issue_type']} "
        f"{'on column ' + repr(issue['column']) if issue.get('column') else '(global)'}: "
        f"{issue['description']}"
        for issue in issues[:20]   # cap at 20 to stay within the model's token budget
    )

    return f"""You are an expert data scientist and data-cleaning AI assistant.

## Dataset Overview
- Shape: {rows:,} rows × {cols} columns
- Missing cells: {global_stats.get('missing_cells', 'N/A')} ({global_stats.get('missing_pct', 'N/A')}%)
- Duplicate rows: {global_stats.get('duplicate_rows', 'N/A')} ({global_stats.get('duplicate_pct', 'N/A')}%)
- Memory: {global_stats.get('memory_mb', 'N/A')} MB

## Detected Issues
{issue_lines}

## Your Task
Produce a JSON array of cleaning proposals.  Each proposal must follow this exact schema:

[
  {{
    "proposal_id": "P-001",
    "action_type": "<one of: impute | drop_column | drop_duplicates | outlier_clip | outlier_winsorize | outlier_remove | normalize_text | coerce_dtype>",
    "column": "<column name or null for global actions>",
    "params": {{}},
    "confidence": <float 0.0-1.0>,
    "rationale": "<2-3 sentence explanation citing specific numbers from the issues list>",
    "risk_level": "<low | medium | high>",
    "alternatives": ["<alternative 1>", "<alternative 2>"]
  }}
]

Rules:
- Output ONLY the JSON array. No prose before or after it.
- Do not wrap in markdown code fences.
- Each proposal must have a unique proposal_id (P-001, P-002, ...).
- confidence must be between 0.0 and 1.0.
- alternatives must contain 1-2 strings describing other options.
- rationale must cite specific numbers from the issues list.
- For impute actions, include params: {{"strategy": "median"}} or {{"strategy": "mode"}}.
- For outlier actions, include params: {{"iqr_factor": 1.5}}.
- Order proposals from highest to lowest priority.
- Limit to at most 15 proposals total.
"""
def _build_summary_prompt(proposals: list[dict], df_shape: tuple[int, int]) -> str:
    rows, cols = df_shape
    action_lines = "\n".join(
        f"  - {p['action_type']} on {p.get('column') or 'all rows'}: "
        f"{p['rationale'][:80]}..."
        for p in proposals
    )

    return f"""You are a data cleaning assistant summarising a cleaning plan for a business user.

Dataset: {rows:,} rows × {cols} columns.

Proposed cleaning actions:
{action_lines}"""

# ── Main Agent ────────────────────────────────────────────────────────────────

class LLMDecisionAgent:
    

    def __init__(
        self,
        config:      dict[str, Any],
        model:       str = DEFAULT_MODEL,
        ollama_url:  str = OLLAMA_BASE_URL,
    ):
        self.config      = config
        self.model       = model
        self.ollama_url  = ollama_url
        self._counter    = 0
        self.llm_available: bool = False
        self.llm_error:   str   = ""
        self._check_llm()

    # ── Connectivity check ────────────────────────────────────────────────────

    def _check_llm(self) -> None:
        
        if _is_ollama_running():
            available_models = _list_ollama_models()
            if available_models:
                self.llm_available = True
                resolved = _resolve_model_name(self.model, available_models)
                self.model = resolved
                logger.info(
                    "Ollama ready. Using model: %r. All models: %s",
                    resolved, available_models,
                )
            else:
                self.llm_available = False
                self.llm_error = (
                    "Ollama is running but no models are installed. "
                    "Run:  ollama pull mistral"
                )
                logger.warning(self.llm_error)
        else:
            self.llm_available = False
            self.llm_error = (
                "Ollama is not running. Start it with:  ollama serve  "
                "then pull a model:  ollama pull mistral"
            )
            logger.warning(self.llm_error)

    def set_model(self, model: str) -> None:
        self.model = model
        logger.info("LLM model switched to: %s", model)

    # ── Core interface ────────────────────────────────────────────────────────

    def propose(
        self,
        issues:   list,
        profile:  dict[str, Any],
        df_shape: tuple[int, int],
    ) -> tuple[list[LLMProposal], str]:
        
        log_section(
            "🤖 LLM Decision Agent",
            f"Model: {self.model}  |  Ollama: {'✓' if self.llm_available else '✗'}",
        )

        # Convert Issue dataclass objects to plain dicts for prompt building
        issue_dicts = [
            issue.to_dict() if hasattr(issue, "to_dict") else dict(issue)
            for issue in issues
        ]

        if self.llm_available:
            proposals, summary = self._llm_propose(issue_dicts, profile, df_shape)
        else:
            logger.warning("LLM unavailable — using rule-based fallback proposals.")
            proposals, summary = self._rule_based_propose(issue_dicts, profile, df_shape)

        logger.info("Generated %d proposals.", len(proposals))
        return proposals, summary

    # ── LLM proposal path ─────────────────────────────────────────────────────

    def _llm_propose(
        self,
        issue_dicts: list[dict],
        profile:     dict[str, Any],
        df_shape:    tuple[int, int],
    ) -> tuple[list[LLMProposal], str]:

        # Step 1 — Generate structured proposals
        analysis_prompt = _build_analysis_prompt(profile, issue_dicts, df_shape)
        try:
            raw_json       = _call_ollama(analysis_prompt, self.model)
            proposal_dicts = _extract_json_block(raw_json)
            if not isinstance(proposal_dicts, list):
                proposal_dicts = [proposal_dicts]
        except Exception as error:
            logger.error("LLM proposal generation failed: %s", error)
            # Fall back to rule-based proposals rather than returning nothing
            return self._rule_based_propose(issue_dicts, profile, df_shape)

        # Parse the returned JSON into typed LLMProposal objects
        proposals: list[LLMProposal] = []
        for index, item in enumerate(proposal_dicts):
            try:
                proposals.append(LLMProposal(
                    proposal_id  = item.get("proposal_id", f"P-{index + 1:03d}"),
                    action_type  = item.get("action_type", "impute"),
                    column       = item.get("column"),
                    params       = item.get("params", {}),
                    confidence   = float(item.get("confidence", 0.8)),
                    rationale    = item.get("rationale", ""),
                    risk_level   = item.get("risk_level", "low"),
                    alternatives = item.get("alternatives", []),
                ))
            except Exception as error:
                logger.warning("Skipping malformed proposal %d: %s", index, error)

        # Step 2 — Generate the executive summary from the same proposal list
        try:
            summary_prompt = _build_summary_prompt(proposal_dicts, df_shape)
            summary        = _call_ollama(summary_prompt, self.model).strip()
        except Exception:
            summary = f"The LLM has proposed {len(proposals)} cleaning actions for your dataset."

        return proposals, summary

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based_propose(
        self,
        issue_dicts: list[dict],
        profile:     dict[str, Any],
        df_shape:    tuple[int, int],
    ) -> tuple[list[LLMProposal], str]:
        proposals:    list[LLMProposal]              = []
        seen_actions: dict[str | None, set[str]]     = {}
        counter       = 0
        cleaning_cfg  = self.config.get("cleaning", {})

        for issue in issue_dicts:
            col         = issue.get("column")
            issue_type  = issue.get("issue_type", "")
            severity    = issue.get("severity", "low")
            pct         = float(issue.get("affected_pct") or 0)
            actions_for_col = seen_actions.setdefault(col, set())

            if issue_type == "missing_values":
                if pct >= 70 and "drop_column" not in actions_for_col:
                    counter += 1
                    proposals.append(LLMProposal(
                        proposal_id  = f"P-{counter:03d}",
                        action_type  = "drop_column",
                        column       = col,
                        confidence   = 0.90,
                        rationale    = (
                            f"Column '{col}' is {pct:.1f}% missing. "
                            "Imputing more than 70% of a column introduces severe bias "
                            "and unreliable signal — dropping is the safer choice."
                        ),
                        risk_level   = "medium",
                        alternatives = ["impute with median/mode", "impute with KNN"],
                    ))
                    actions_for_col.add("drop_column")

                elif pct < 70 and "impute" not in actions_for_col:
                    counter  += 1
                    strategy  = (
                        cleaning_cfg.get("missing_numeric_strategy", "median")
                        if severity in ("critical", "high") else "median"
                    )
                    proposals.append(LLMProposal(
                        proposal_id  = f"P-{counter:03d}",
                        action_type  = "impute",
                        column       = col,
                        params       = {"strategy": strategy},
                        confidence   = 0.88,
                        rationale    = (
                            f"Column '{col}' is {pct:.1f}% missing. "
                            f"{strategy.title()} imputation preserves the column's "
                            "distribution while filling gaps without introducing extreme bias."
                        ),
                        risk_level   = "low",
                        alternatives = [
                            "mean imputation",
                            "drop rows with missing values",
                            "constant fill",
                        ],
                    ))
                    actions_for_col.add("impute")

            elif issue_type == "duplicate_rows" and "drop_duplicates" not in actions_for_col:
                counter += 1
                proposals.append(LLMProposal(
                    proposal_id  = f"P-{counter:03d}",
                    action_type  = "drop_duplicates",
                    column       = None,
                    params       = {"keep": "first"},
                    confidence   = 0.98,
                    rationale    = (
                        f"{issue.get('affected_count', 0):,} exact duplicate rows detected "
                        f"({pct:.1f}%). Duplicates skew aggregations, distort ML training, "
                        "and inflate row counts — removing them is almost always correct."
                    ),
                    risk_level   = "low",
                    alternatives = ["keep last occurrence", "manual review before dropping"],
                ))
                actions_for_col.add("drop_duplicates")

            elif issue_type == "outliers" and "outlier_clip" not in actions_for_col:
                counter += 1
                proposals.append(LLMProposal(
                    proposal_id  = f"P-{counter:03d}",
                    action_type  = "outlier_clip",
                    column       = col,
                    params       = {"iqr_factor": 1.5},
                    confidence   = 0.82,
                    rationale    = (
                        f"Column '{col}' has {pct:.1f}% outliers by the IQR method. "
                        "Clipping to [Q1 − 1.5·IQR, Q3 + 1.5·IQR] caps extreme values "
                        "while keeping all rows, making it safer than deletion."
                    ),
                    risk_level   = "medium",
                    alternatives = [
                        "winsorize (5th–95th percentile)",
                        "remove outlier rows",
                        "leave unchanged",
                    ],
                ))
                actions_for_col.add("outlier_clip")

            elif issue_type == "constant_column" and "drop_column" not in actions_for_col:
                counter += 1
                proposals.append(LLMProposal(
                    proposal_id  = f"P-{counter:03d}",
                    action_type  = "drop_column",
                    column       = col,
                    confidence   = 0.95,
                    rationale    = (
                        f"Column '{col}' contains only one unique value across all rows. "
                        "Zero-variance columns carry no information and can cause errors "
                        "in variance-sensitive algorithms."
                    ),
                    risk_level   = "low",
                    alternatives = ["keep column (if used as a filter downstream)"],
                ))
                actions_for_col.add("drop_column")

            elif issue_type == "string_whitespace" and "normalize_text" not in actions_for_col:
                counter += 1
                proposals.append(LLMProposal(
                    proposal_id  = f"P-{counter:03d}",
                    action_type  = "normalize_text",
                    column       = col,
                    params       = {"lowercase": True, "strip": True},
                    confidence   = 0.92,
                    rationale    = (
                        f"Column '{col}' contains entries with leading/trailing whitespace. "
                        "Unstripped strings cause silent join failures and incorrect groupby "
                        "results — normalisation has essentially zero downside."
                    ),
                    risk_level   = "low",
                    alternatives = ["strip only (no lowercase)"],
                ))
                actions_for_col.add("normalize_text")

            elif issue_type == "mixed_types" and "coerce_dtype" not in actions_for_col:
                counter += 1
                proposals.append(LLMProposal(
                    proposal_id  = f"P-{counter:03d}",
                    action_type  = "coerce_dtype",
                    column       = col,
                    confidence   = 0.75,
                    rationale    = (
                        f"Column '{col}' contains mixed Python types. "
                        "Inconsistent types prevent reliable sorting, filtering, and "
                        "vectorised operations — coercing to a single dtype is necessary."
                    ),
                    risk_level   = "medium",
                    alternatives = ["cast to string (safest)", "cast to numeric", "manual review"],
                ))
                actions_for_col.add("coerce_dtype")

            if len(proposals) >= 15:
                break   # cap the proposal list to keep the UI manageable

        rows, num_cols = df_shape
        summary = (
            f"Your dataset ({rows:,} rows × {num_cols} columns) has "
            f"{len(issue_dicts)} detected quality issues. "
            f"The rule-based engine has proposed {len(proposals)} cleaning actions covering "
            "missing value imputation, duplicate removal, outlier handling, and text normalisation. "
            "Review each proposal carefully — moderate-risk actions like outlier clipping and "
            "dtype coercion may need adjustment based on your domain knowledge."
        )
        return proposals, summary

    # ── UI-facing helpers ─────────────────────────────────────────────────────

    def explain_action(self, proposal: LLMProposal) -> str:
        if not self.llm_available:
            return proposal.rationale

        prompt = f"""You are a data cleaning expert. Explain the following cleaning action
to a non-technical business user in 4-6 sentences. Be concrete, cite numbers, and mention risks.

Action: {proposal.action_type}
Column: {proposal.column or 'entire dataset'}
Parameters: {json.dumps(proposal.params)}
Initial rationale: {proposal.rationale}
Risk level: {proposal.risk_level}
Alternatives considered: {', '.join(proposal.alternatives) if proposal.alternatives else 'none'}

Output ONLY the explanation paragraph. No bullet points, no headers.
"""
        try:
            return _call_ollama(prompt, self.model).strip()
        except Exception as error:
            logger.warning("Explain call failed: %s", error)
            return proposal.rationale

    def suggest_modification(self, proposal: LLMProposal, user_feedback: str) -> dict[str, Any]:
        
        if not self.llm_available:
            return {}

        prompt = f"""A user reviewed this data cleaning proposal and provided feedback.
Suggest a modified version as a JSON object containing only the fields that should change
(action_type, column, params, rationale, confidence).

Original proposal:
{json.dumps(proposal.to_dict(), indent=2)}

User feedback: "{user_feedback}"

Return ONLY a JSON object with the changed fields. Do not include unchanged fields.
Do not wrap in markdown.
"""
        try:
            raw = _call_ollama(prompt, self.model).strip()
            return _extract_json_block(raw)
        except Exception as error:
            logger.warning("Modification suggestion failed: %s", error)
            return {}
