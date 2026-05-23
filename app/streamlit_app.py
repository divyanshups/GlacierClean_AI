from __future__ import annotations
import io
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st
from agents.approval_layer     import APPROVED, MODIFIED, PENDING, REJECTED, ApprovalLayer
from agents.llm_decision_agent import _is_ollama_running, _list_ollama_models
from core.ai_pipeline          import AIPipeline
from utils.visualization import (
    plot_before_after_comparison,
    plot_correlation_heatmap,
    plot_missing_bar,
    plot_missing_heatmap,
    plot_numeric_distributions,
    plot_outlier_boxplots,
    plot_quality_gauge,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title          = "Glacier Cleaning · AI Agent",
    page_icon           = "🧊",
    layout              = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Base typography and background */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0e1a !important;
    color: #e0e8f0 !important;
}
.main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

/* Slim, unobtrusive scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(125,211,252,0.2); border-radius: 10px; }

/* Glass-morphism card variants */
.glass    { background: rgba(15,21,36,0.6);  backdrop-filter: blur(16px); border: 1px solid rgba(125,211,252,0.10); border-radius: 1rem; }
.glass-hi { background: rgba(15,21,36,0.8);  backdrop-filter: blur(24px); border: 1px solid rgba(125,211,252,0.18); border-radius: 1rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95) !important;
    border-right: 1px solid rgba(125,211,252,0.1) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: #a0b4c4 !important; font-size: 0.78rem !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(30,40,56,0.5) !important;
    border: 1px solid rgba(125,211,252,0.2) !important;
    border-radius: 0.5rem !important;
    color: #e0e8f0 !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stTextInput input {
    background: rgba(30,40,56,0.5) !important;
    border: 1px solid rgba(125,211,252,0.2) !important;
    border-radius: 0.5rem !important;
    color: #e0e8f0 !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(125,211,252,0.1) !important; }

/* Buttons */
.stButton > button {
    background: rgba(125,211,252,0.12) !important;
    border: 1px solid rgba(125,211,252,0.3) !important;
    color: #7dd3fc !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 0.75rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(125,211,252,0.22) !important;
    box-shadow: 0 4px 20px rgba(125,211,252,0.15) !important;
    transform: translateY(-1px);
}

/* Progress bar — gradient from blue to purple */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7dd3fc, #c8a0f0) !important;
    box-shadow: 0 0 10px rgba(125,211,252,0.4);
}
.stProgress > div > div { background: rgba(30,40,56,0.6) !important; border-radius: 9999px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,21,36,0.6) !important;
    border: 1px solid rgba(125,211,252,0.1) !important;
    border-radius: 0.75rem !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    color: #64748b !important;
    border-radius: 0.5rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(125,211,252,0.1) !important;
    color: #7dd3fc !important;
    border: 1px solid rgba(125,211,252,0.2) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: rgba(125,211,252,0.08) !important;
    border: 1px solid rgba(125,211,252,0.2) !important;
    color: #7dd3fc !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    border-radius: 0.6rem !important;
}

/* Expandable sections */
.streamlit-expanderHeader {
    background: rgba(15,21,36,0.5) !important;
    border: 1px solid rgba(125,211,252,0.08) !important;
    border-radius: 0.6rem !important;
    color: #a0b4c4 !important;
    font-size: 0.82rem !important;
}
.streamlit-expanderContent {
    background: rgba(10,14,26,0.5) !important;
    border: 1px solid rgba(125,211,252,0.06) !important;
    border-top: none !important;
    border-radius: 0 0 0.6rem 0.6rem !important;
    font-size: 0.82rem !important;
    color: #a0b4c4 !important;
}

/* Pipeline stage indicator pills */
.stage-pill    { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
.stage-active  { background: rgba(125,211,252,0.15); color: #7dd3fc; border: 1px solid rgba(125,211,252,0.3); }
.stage-done    { background: rgba(52,211,153,0.12);  color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.stage-waiting { background: rgba(30,40,56,0.6);     color: #4a6070; border: 1px solid rgba(74,96,112,0.3); }

/* Proposal cards — border colour changes based on the user's decision */
.proposal-card          { background: rgba(15,21,36,0.55); border: 1px solid rgba(125,211,252,0.1); border-radius: 0.9rem; padding: 1.1rem 1.3rem; margin-bottom: 0.6rem; transition: border-color 0.2s; }
.proposal-card.approved { border-color: rgba(52,211,153,0.4);  background: rgba(52,211,153,0.04); }
.proposal-card.rejected { border-color: rgba(239,68,68,0.3);   background: rgba(239,68,68,0.03); }
.proposal-card.modified { border-color: rgba(251,191,36,0.4);  background: rgba(251,191,36,0.04); }

/* Proposal card sub-elements */
.p-id       { font-size: 0.65rem; font-family: monospace; color: #4a6070; margin-bottom: 0.15rem; }
.p-title    { font-size: 0.9rem; font-weight: 700; color: #e0e8f0; }
.p-col      { font-size: 0.7rem; color: #7dd3fc; font-family: monospace; }
.p-rationale{ font-size: 0.78rem; color: #64748b; line-height: 1.5; margin: 0.5rem 0; }
.p-alts     { font-size: 0.72rem; color: #4a6070; }

/* Risk-level badges */
.risk        { display: inline-block; padding: 0.12rem 0.5rem; border-radius: 9999px; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
.risk-low    { background: rgba(52,211,153,0.12);  color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.risk-medium { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.risk-high   { background: rgba(239,68,68,0.12);   color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

/* Confidence bar rendered below each proposal card */
.conf-bar-bg   { width: 100%; height: 3px; background: rgba(30,40,56,0.8); border-radius: 9999px; margin-top: 0.4rem; }
.conf-bar-fill { height: 100%; border-radius: 9999px; background: linear-gradient(90deg, #7dd3fc, #c8a0f0); }

/* Decision status badges (approved / rejected / modified / pending) */
.dec-badge    { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.18rem 0.6rem; border-radius: 9999px; font-size: 0.65rem; font-weight: 700; }
.dec-approved { background: rgba(52,211,153,0.15); color: #34d399; }
.dec-rejected { background: rgba(239,68,68,0.12);  color: #ef4444; }
.dec-modified { background: rgba(251,191,36,0.12); color: #fbbf24; }
.dec-pending  { background: rgba(74,96,112,0.3);   color: #64748b; }

/* Metric tiles used on the results screen */
.g-metric     { background: rgba(15,21,36,0.6); border: 1px solid rgba(125,211,252,0.1); border-radius: 1rem; padding: 1.1rem 1.3rem; text-align: center; }
.g-metric-val { font-size: 1.9rem; font-weight: 800; line-height: 1.1; }
.g-metric-lbl { font-size: 0.65rem; color: #4a6070; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.3rem; font-weight: 600; }

/* LLM status banners shown in the sidebar and approval screen */
.llm-ok   { background: rgba(52,211,153,0.08);  border: 1px solid rgba(52,211,153,0.2);  border-radius: 0.75rem; padding: 0.7rem 1rem; font-size: 0.8rem; color: #34d399; }
.llm-err  { background: rgba(239,68,68,0.08);   border: 1px solid rgba(239,68,68,0.2);   border-radius: 0.75rem; padding: 0.7rem 1rem; font-size: 0.8rem; color: #ef4444; }
.llm-warn { background: rgba(251,191,36,0.08);  border: 1px solid rgba(251,191,36,0.2);  border-radius: 0.75rem; padding: 0.7rem 1rem; font-size: 0.8rem; color: #fbbf24; }

/* Toast notification (fixed, bottom-right) shown when cleaning completes */
.glacier-toast { position: fixed; bottom: 1.5rem; right: 1.5rem; background: rgba(15,21,36,0.9); backdrop-filter: blur(24px); border: 1px solid rgba(125,211,252,0.3); border-radius: 9999px; padding: 0.35rem 1.25rem 0.35rem 0.35rem; display: flex; align-items: center; gap: 0.75rem; z-index: 9999; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
.toast-icon    { width: 2rem; height: 2rem; background: #7dd3fc; border-radius: 9999px; display: flex; align-items: center; justify-content: center; color: #0a0e1a; font-size: 0.85rem; font-weight: 900; }
.toast-text    { font-size: 0.75rem; font-weight: 500; color: #cbd5e1; }

/* Decorative background orbs (blurred circles, pointer-events disabled) */
.bg-orb-1 { position: fixed; top: 15%;   right: 10%; width: 400px; height: 400px; background: rgba(125,211,252,0.05);  border-radius: 9999px; filter: blur(120px); pointer-events: none; z-index: 0; }
.bg-orb-2 { position: fixed; bottom: 20%; left: 5%;  width: 280px; height: 280px; background: rgba(200,160,240,0.05); border-radius: 9999px; filter: blur(100px);  pointer-events: none; z-index: 0; }

/* Pulsing status dot used in the sidebar Ollama status banner */
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 9999px; background: #34d399; box-shadow: 0 0 6px #34d399; vertical-align: middle; margin-right: 5px; animation: pdot 2s infinite; }
@keyframes pdot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
<div class="bg-orb-1"></div>
<div class="bg-orb-2"></div>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
# All keys and their default values in one place so "Start Over" is trivial —
# we just write every default back and call st.rerun().
_DEFAULTS = {
    "phase":          "upload",  # current pipeline stage
    "phase1_result":  None,      # result dict from run_until_approval()
    "final_result":   None,      # result dict from run_after_approval()
    "pipeline":       None,      # the AIPipeline instance (kept alive across reruns)
    "df_preview":     None,      # a small preview DataFrame for the upload screen
    "uploaded_bytes": None,      # raw bytes of the uploaded file
    "uploaded_name":  "",        # original filename (used to infer the format)
    "decisions":      {},        # pid → {"decision": ..., "note": ..., ...}
    "explain_cache":  {},        # pid → LLM explanation string (cached to avoid re-calling)
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding header
    st.markdown(
        '<div style="padding:0 0.5rem 1rem">'
        '<div style="font-size:1.15rem;font-weight:800;color:#7dd3fc">Glacier Cleaning</div>'
        '<div style="font-size:0.65rem;color:#2a3a48;letter-spacing:0.1em;margin-top:2px">'
        'AI DATA PROCESSING ENGINE</div></div>',
        unsafe_allow_html=True,
    )

    # Ollama connectivity status
    ollama_running = _is_ollama_running()
    available_models = _list_ollama_models() if ollama_running else []

    if ollama_running and available_models:
        st.markdown(
            f'<div class="llm-ok">🟢 &nbsp;Ollama ready · {len(available_models)} model(s)</div>',
            unsafe_allow_html=True,
        )
    elif ollama_running:
        st.markdown(
            '<div class="llm-warn">🟡 &nbsp;Ollama running — no models installed.<br>'
            '<code>ollama pull mistral</code></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="llm-err">🔴 &nbsp;Ollama offline.<br>'
            '<code>ollama serve</code></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model selector — show full "name:tag" strings so the displayed name
    # exactly matches what will be sent to the Ollama API.
    if available_models:
        selected_model: str = st.selectbox("🤖  LLM Model", available_models, index=0)  # type: ignore
    else:
        selected_model = "mistral"
        st.text_input("🤖  LLM Model (offline)", value="mistral", disabled=True)

    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(125,211,252,0.08);margin:0.75rem 0"/>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.65rem;font-weight:700;color:#2a3a48;letter-spacing:0.15em;'
        'text-transform:uppercase">Cleaning Config</p>',
        unsafe_allow_html=True,
    )

    # All cleaning parameters are collected here and passed into _build_cfg()
    # so the pipeline always runs with whatever the user has set in the sidebar.
    num_strategy:   str   = st.selectbox("Numeric imputation",    ["median", "mean", "mode"],                    index=0)  # type: ignore
    cat_strategy:   str   = st.selectbox("Categorical imputation", ["mode", "constant"],                          index=0)  # type: ignore
    constant_fill:  str   = st.text_input("Constant fill", "Unknown")
    outlier_strat:  str   = st.selectbox("Outlier strategy",       ["clip", "winsorize", "remove", "none"],       index=0)  # type: ignore
    iqr_factor:     float = st.slider("IQR factor",         1.0, 3.0, 1.5, 0.1)
    conf_threshold: float = st.slider("Confidence threshold", 0.5, 1.0, 0.75, 0.05)
    text_norm:      bool  = st.toggle("Text normalisation", value=True)
    date_inf:       bool  = st.toggle("Date inference",     value=True)

    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(125,211,252,0.08);margin:0.75rem 0"/>',
        unsafe_allow_html=True,
    )
    if st.button("↺  Start Over", use_container_width=True):
        for key, default in _DEFAULTS.items():
            st.session_state[key] = default
        st.rerun()


def _build_config() -> dict:
    """
    Build the pipeline configuration dictionary from the current sidebar values.

    Called fresh each time a new pipeline run starts so the pipeline always
    reflects whatever the user last configured in the sidebar.
    """
    return {
        "pipeline": {
            "confidence_threshold": conf_threshold,
            "max_iterations":       1,
        },
        "profiler": {
            "sample_size":           5000,
            "cardinality_threshold": 50,
            "skew_threshold":        1.0,
        },
        "issue_detector": {
            "missing_critical":         0.50,
            "missing_high":             0.20,
            "missing_low":              0.05,
            "outlier_method":           "iqr",
            "outlier_iqr_factor":       iqr_factor,
            "outlier_zscore_threshold": 3.0,
            "min_variance_threshold":   0.01,
        },
        "cleaning": {
            "missing_numeric_strategy":     num_strategy,
            "missing_categorical_strategy": cat_strategy,
            "missing_constant_fill":        constant_fill,
            "outlier_strategy":             outlier_strat,
            "duplicate_strategy":           "keep_first",
            "text_normalization":           text_norm,
            "date_inference":               date_inf,
            "encode_categoricals":          False,
        },
        "evaluation": {},
        "logging": {
            "level":        "INFO",
            "log_file":     "memory/agent.log",
            "decision_log": "memory/decision_logs.json",
        },
        "output": {
            "cleaned_csv":  "outputs/cleaned_dataset.csv",
            "report_json":  "outputs/reports/cleaning_report.json",
        },
    }


# ── Pipeline stage header ─────────────────────────────────────────────────────
current_phase = st.session_state["phase"]


def _stage_css_class(stage_name: str) -> str:
    """
    Return the CSS class for a stage indicator pill based on whether the
    pipeline is currently on that stage, past it, or hasn't reached it yet.
    """
    stage_order = ["upload", "analysis", "approval", "results"]
    current_idx = stage_order.index(current_phase) if current_phase in stage_order else 0
    stage_idx   = stage_order.index(stage_name)
    if stage_idx < current_idx:
        return "stage-done"
    elif stage_idx == current_idx:
        return "stage-active"
    else:
        return "stage-waiting"


st.markdown(f"""
<div style="margin-bottom:1.5rem">
  <div style="font-size:2.2rem;font-weight:800;color:#e0e8f0;letter-spacing:-0.02em;margin-bottom:0.4rem">
    DataClean <span style="color:#7dd3fc">AI</span> Agent
  </div>
  <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center">
    <span class="stage-pill {_stage_css_class('upload')}">① Upload</span>
    <span style="color:#2a3a48">→</span>
    <span class="stage-pill {_stage_css_class('analysis')}">② AI Analysis</span>
    <span style="color:#2a3a48">→</span>
    <span class="stage-pill {_stage_css_class('approval')}">③ Your Approval</span>
    <span style="color:#2a3a48">→</span>
    <span class="stage-pill {_stage_css_class('results')}">④ Results</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 1 — UPLOAD
# ════════════════════════════════════════════════════════════════════════════════
if current_phase == "upload":
    upload_col, info_col = st.columns([2, 1], gap="large")

    with upload_col:
        # Decorative drop-zone panel — purely visual; the real uploader is below
        st.markdown("""
<div style="background:rgba(15,21,36,0.75);border:2px dashed rgba(125,211,252,0.3);
border-radius:1.25rem;padding:2.5rem 2rem;text-align:center">
  <div style="font-size:2.5rem;margin-bottom:0.75rem">📂</div>
  <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;color:#e0e8f0">Drop your dataset here</div>
  <div style="font-size:0.82rem;color:#64748b;margin-bottom:1.25rem">Drag and drop or use the uploader below</div>
  <span style="padding:0.25rem 0.6rem;background:rgba(15,21,36,0.6);border:1px solid rgba(125,211,252,0.1);color:rgba(125,211,252,0.8);font-size:0.65rem;border-radius:0.4rem;margin:0 2px">CSV</span>
  <span style="padding:0.25rem 0.6rem;background:rgba(15,21,36,0.6);border:1px solid rgba(125,211,252,0.1);color:rgba(125,211,252,0.8);font-size:0.65rem;border-radius:0.4rem;margin:0 2px">XLSX</span>
  <span style="padding:0.25rem 0.6rem;background:rgba(15,21,36,0.6);border:1px solid rgba(125,211,252,0.1);color:rgba(125,211,252,0.8);font-size:0.65rem;border-radius:0.4rem;margin:0 2px">JSON</span>
  <span style="padding:0.25rem 0.6rem;background:rgba(15,21,36,0.6);border:1px solid rgba(125,211,252,0.1);color:rgba(125,211,252,0.8);font-size:0.65rem;border-radius:0.4rem;margin:0 2px">PARQUET</span>
</div>
""", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "",
            type            = ["csv", "tsv", "xlsx", "xls", "json", "parquet"],
            label_visibility = "collapsed",
        )

    with info_col:
        st.markdown("""
<div class="glass" style="padding:1.3rem 1.5rem">
  <div style="font-size:0.72rem;font-weight:700;color:#a0b4c4;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.85rem">How It Works</div>
  <div style="font-size:0.78rem;color:#64748b;line-height:1.8">
    <div style="margin-bottom:0.5rem"><span style="color:#7dd3fc;font-weight:700">① Upload</span> your raw dataset</div>
    <div style="margin-bottom:0.5rem"><span style="color:#7dd3fc;font-weight:700">② AI analyses</span> issues &amp; proposes fixes with explanations</div>
    <div style="margin-bottom:0.5rem"><span style="color:#7dd3fc;font-weight:700">③ You review</span> each proposal — approve ✅, reject ❌, or modify ✏️</div>
    <div><span style="color:#7dd3fc;font-weight:700">④ Execute</span> only what you approved</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # When a file is uploaded, read the bytes into session state and generate a
    # quick preview DataFrame for the stats tiles below.
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        st.session_state["uploaded_bytes"] = file_bytes
        st.session_state["uploaded_name"]  = uploaded_file.name

        try:
            buf  = io.BytesIO(file_bytes)
            name = uploaded_file.name.lower()
            if name.endswith((".csv", ".tsv")):
                preview_df = pd.read_csv(buf)
            elif name.endswith((".xlsx", ".xls")):
                preview_df = pd.read_excel(buf)
            else:
                preview_df = pd.read_json(buf)
            st.session_state["df_preview"] = preview_df
        except Exception:
            pass   # preview failure is non-critical — the pipeline will load the file properly

    # Show a quick summary and preview if we have a DataFrame already
    if st.session_state["df_preview"] is not None:
        preview_df: pd.DataFrame = st.session_state["df_preview"]
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        for tile_col, label, value, colour in [
            (col1, "Rows",    f"{preview_df.shape[0]:,}",                          "#7dd3fc"),
            (col2, "Columns", f"{preview_df.shape[1]:,}",                          "#c8a0f0"),
            (col3, "Missing", f"{preview_df.isnull().mean().mean() * 100:.1f}%",   "#f97316"),
            (col4, "Dupes",   f"{preview_df.duplicated().mean() * 100:.1f}%",      "#f472b6"),
        ]:
            tile_col.markdown(
                f'<div class="g-metric">'
                f'<div class="g-metric-val" style="color:{colour}">{value}</div>'
                f'<div class="g-metric-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

        with st.expander("📋 Dataset Preview"):
            st.dataframe(preview_df.head(50), use_container_width=True, height=250)

        st.markdown("<br>", unsafe_allow_html=True)
        button_col, _ = st.columns([0.35, 0.65])
        with button_col:
            if st.button("🤖  Analyse with AI", use_container_width=True):
                st.session_state["phase"] = "analysis"
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 2 — AI ANALYSIS
# The pipeline's Phase 1 runs here. Streamlit re-renders the page while
# on_progress updates the progress bar, then transitions to the approval stage.
# ════════════════════════════════════════════════════════════════════════════════
elif current_phase == "analysis":
    st.markdown(
        '<div class="glass-hi" style="padding:1.1rem 1.4rem;margin-bottom:1.5rem">'
        '<div style="font-size:0.9rem;font-weight:700;color:#7dd3fc;margin-bottom:0.2rem">'
        '🤖 AI is analysing your dataset…</div>'
        '<div style="font-size:0.78rem;color:#64748b">'
        'Detecting issues and generating cleaning proposals. '
        'This may take 30–90 seconds depending on your model.</div></div>',
        unsafe_allow_html=True,
    )

    progress_bar  = st.progress(0.0)
    status_slot   = st.empty()

    def on_progress(message: str, fraction: float) -> None:
        """Update the progress bar and status text on each pipeline step."""
        progress_bar.progress(min(fraction, 1.0))
        status_slot.markdown(
            f'<div style="font-size:0.8rem;color:#7dd3fc;padding:0.3rem 0;font-family:monospace">'
            f'▶ {message}</div>',
            unsafe_allow_html=True,
        )

    config   = _build_config()
    pipeline = AIPipeline(config=config, llm_model=selected_model)
    pipeline.on_progress = on_progress
    st.session_state["pipeline"] = pipeline

    try:
        source = io.BytesIO(st.session_state["uploaded_bytes"])
        phase1_result = pipeline.run_until_approval(
            source,
            file_name=st.session_state["uploaded_name"],
        )
        st.session_state["phase1_result"] = phase1_result
        st.session_state["decisions"]     = {}
        st.session_state["phase"]         = "approval"
        st.rerun()

    except Exception as error:
        error_message = str(error)
        st.error("Analysis failed: " + error_message)

        # Provide actionable guidance for the two most common failure modes
        if "404" in error_message or "not installed" in error_message.lower():
            st.warning(
                "Model not found in Ollama. "
                "Run `ollama list` to see what is installed, "
                "then `ollama pull mistral` to add a model. "
                "Or pick a different model in the sidebar."
            )
        elif "Cannot connect" in error_message or "cannot reach" in error_message.lower():
            st.warning("Ollama is not reachable. Start it with: ollama serve")

        if st.button("Back to Upload"):
            st.session_state["phase"] = "upload"
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 3 — USER APPROVAL
# Renders every LLM proposal as a card. The user can approve, reject, or
# modify each one individually, or use the bulk-action buttons at the top.
# ════════════════════════════════════════════════════════════════════════════════
elif current_phase == "approval":
    phase1_result: dict       = st.session_state["phase1_result"]
    proposals:     list       = phase1_result["proposals"]
    pipeline:      AIPipeline = st.session_state["pipeline"]
    approval_layer: ApprovalLayer = pipeline.approval

    # Show which proposal source was used (LLM or rule-based fallback)
    if phase1_result["llm_available"]:
        st.markdown(
            f'<div class="llm-ok">🟢 &nbsp;AI proposals generated by '
            f'<strong>{phase1_result["model"]}</strong> via Ollama</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="llm-warn">🟡 &nbsp;Ollama offline — rule-based proposals used. '
            f'<em style="color:#64748b">{phase1_result["llm_error"]}</em></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive summary card
    st.markdown(
        f'<div class="glass" style="padding:1.3rem 1.5rem;margin-bottom:1.5rem">'
        f'<div style="font-size:0.65rem;font-weight:700;color:#4a6070;letter-spacing:0.15em;'
        f'text-transform:uppercase;margin-bottom:0.5rem">AI Executive Summary</div>'
        f'<div style="font-size:0.88rem;color:#a0b4c4;line-height:1.65">'
        f'{phase1_result["executive_summary"]}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Bulk-action bar ───────────────────────────────────────────────────────
    decisions = st.session_state["decisions"]
    st.markdown(
        '<div style="font-size:1rem;font-weight:700;color:#e0e8f0;margin-bottom:0.75rem">'
        'Review Proposals</div>',
        unsafe_allow_html=True,
    )

    bulk_col1, bulk_col2, bulk_col3, bulk_col4 = st.columns([1, 1, 1, 2])
    with bulk_col1:
        if st.button("✅  Approve All", use_container_width=True):
            for proposal in proposals:
                st.session_state["decisions"][proposal.proposal_id] = {
                    "decision": APPROVED, "note": "",
                }
    with bulk_col2:
        if st.button("❌  Reject All", use_container_width=True):
            for proposal in proposals:
                st.session_state["decisions"][proposal.proposal_id] = {
                    "decision": REJECTED, "note": "",
                }
    with bulk_col3:
        if st.button("↺  Reset All", use_container_width=True):
            st.session_state["decisions"] = {}

    # Live decision tally
    approved_count = sum(
        1 for p in proposals
        if decisions.get(p.proposal_id, {}).get("decision") == APPROVED
    )
    rejected_count = sum(
        1 for p in proposals
        if decisions.get(p.proposal_id, {}).get("decision") == REJECTED
    )
    modified_count = sum(
        1 for p in proposals
        if decisions.get(p.proposal_id, {}).get("decision") == MODIFIED
    )
    pending_count = sum(1 for p in proposals if p.proposal_id not in decisions)

    with bulk_col4:
        st.markdown(
            f'<div style="text-align:right;font-size:0.78rem;color:#64748b;padding:0.5rem 0">'
            f'<span style="color:#34d399;font-weight:700">{approved_count} ✅</span> · '
            f'<span style="color:#ef4444;font-weight:700">{rejected_count} ❌</span> · '
            f'<span style="color:#fbbf24;font-weight:700">{modified_count} ✏️</span> · '
            f'<span style="color:#64748b">{pending_count} ⏳ pending</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # CSS class / icon / badge mappings keyed by decision state
    CARD_CLASS   = {APPROVED: "approved", REJECTED: "rejected", MODIFIED: "modified", PENDING: ""}
    BADGE_CLASS  = {APPROVED: "dec-approved", REJECTED: "dec-rejected", MODIFIED: "dec-modified", PENDING: "dec-pending"}
    DECISION_ICON = {APPROVED: "✅", REJECTED: "❌", MODIFIED: "✏️", PENDING: "⏳"}
    RISK_CLASS   = {"low": "risk-low", "medium": "risk-medium", "high": "risk-high"}

    # ── Per-proposal cards ────────────────────────────────────────────────────
    for proposal in proposals:
        pid      = proposal.proposal_id
        decision = decisions.get(pid, {}).get("decision", PENDING)
        note     = decisions.get(pid, {}).get("note", "")
        conf_pct = int(proposal.confidence * 100)

        column_html = (
            f'&nbsp;<span class="p-col">· {proposal.column}</span>'
            if proposal.column else ""
        )
        alternatives_html = (
            f'<div class="p-alts">Alternatives: {" · ".join(proposal.alternatives)}</div>'
            if proposal.alternatives else ""
        )

        # Render the proposal card
        st.markdown(f"""
<div class="proposal-card {CARD_CLASS.get(decision, '')}">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:0.5rem">
    <div>
      <div class="p-id">{pid}</div>
      <div class="p-title">{proposal.action_type.replace("_", " ").title()}{column_html}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.3rem">
      <span class="dec-badge {BADGE_CLASS[decision]}">{DECISION_ICON[decision]} {decision.upper()}</span>
      <span class="risk {RISK_CLASS.get(proposal.risk_level, 'risk-low')}">{proposal.risk_level} risk</span>
    </div>
  </div>
  <div class="p-rationale">{proposal.rationale}</div>
  {alternatives_html}
  <div style="display:flex;justify-content:space-between;margin-top:0.5rem">
    <div style="font-size:0.7rem;color:#4a6070">Confidence {conf_pct}%</div>
  </div>
  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%"></div></div>
</div>
""", unsafe_allow_html=True)

        # Per-proposal action buttons
        btn_approve, btn_reject, btn_modify, btn_explain = st.columns([1, 1, 1, 2])

        with btn_approve:
            if st.button("✅ Approve", key=f"ap_{pid}", use_container_width=True):
                st.session_state["decisions"][pid] = {"decision": APPROVED, "note": note}
                st.rerun()

        with btn_reject:
            if st.button("❌ Reject", key=f"rj_{pid}", use_container_width=True):
                st.session_state["decisions"][pid] = {"decision": REJECTED, "note": note}
                st.rerun()

        with btn_modify:
            # Toggle the modify panel open/closed
            if f"show_modify_{pid}" not in st.session_state:
                st.session_state[f"show_modify_{pid}"] = False
            if st.button("✏️ Modify", key=f"mod_{pid}", use_container_width=True):
                st.session_state[f"show_modify_{pid}"] = not st.session_state[f"show_modify_{pid}"]
                st.rerun()

        with btn_explain:
            # Toggle the AI explanation panel open/closed
            if f"show_explain_{pid}" not in st.session_state:
                st.session_state[f"show_explain_{pid}"] = False
            if st.button("💡 Explain", key=f"ex_{pid}", use_container_width=True):
                st.session_state[f"show_explain_{pid}"] = not st.session_state[f"show_explain_{pid}"]
                st.rerun()

        # ── Modify panel ──────────────────────────────────────────────────────
        if st.session_state.get(f"show_modify_{pid}"):
            with st.container():
                st.markdown(
                    '<div style="background:rgba(251,191,36,0.05);border:1px solid rgba(251,191,36,0.2);'
                    'border-radius:0.75rem;padding:1rem;margin-top:0.25rem">',
                    unsafe_allow_html=True,
                )
                ACTION_OPTIONS = [
                    "impute", "drop_column", "drop_duplicates",
                    "outlier_clip", "outlier_winsorize", "outlier_remove",
                    "normalize_text", "coerce_dtype",
                ]

                # Pre-populate the form with the current decision values if any exist
                current_decision = decisions.get(pid, {})
                current_action   = current_decision.get("action",  proposal.action_type)
                current_column   = current_decision.get("column",  proposal.column or "")
                current_params   = dict(current_decision.get("params", proposal.params))

                modify_col1, modify_col2 = st.columns(2)
                with modify_col1:
                    new_action = st.selectbox(
                        "Action type",
                        ACTION_OPTIONS,
                        index=ACTION_OPTIONS.index(current_action)
                              if current_action in ACTION_OPTIONS else 0,
                        key=f"ma_{pid}",
                    )
                    new_column = st.text_input(
                        "Column (blank = global action)",
                        value=current_column,
                        key=f"mc_{pid}",
                    )
                with modify_col2:
                    # Show relevant parameter controls based on what's in the params dict
                    if "strategy" in current_params:
                        strategy_options = ["median", "mean", "mode", "constant", "drop"]
                        chosen_strategy  = st.selectbox(
                            "Strategy",
                            strategy_options,
                            index=strategy_options.index(current_params["strategy"])
                                  if current_params.get("strategy") in strategy_options else 0,
                            key=f"ms_{pid}",
                        )
                        current_params["strategy"] = chosen_strategy

                    if "iqr_factor" in current_params:
                        chosen_iqr = st.slider(
                            "IQR factor", 1.0, 3.0,
                            float(current_params.get("iqr_factor", 1.5)), 0.1,
                            key=f"mi_{pid}",
                        )
                        current_params["iqr_factor"] = chosen_iqr

                user_note_input = st.text_input("Note (optional)", value=note, key=f"mn_{pid}")

                # Optional: ask the LLM to suggest changes based on free-text feedback
                ai_feedback = st.text_input("Ask AI to suggest changes", key=f"maf_{pid}")
                if ai_feedback and st.button("🤖 Get AI suggestion", key=f"mag_{pid}"):
                    with st.spinner("Asking AI…"):
                        suggestion = pipeline.suggest_modification(proposal, ai_feedback)
                    if suggestion:
                        st.json(suggestion)
                        st.caption("Apply these suggested changes manually using the fields above.")

                save_col, _ = st.columns([0.3, 0.7])
                with save_col:
                    if st.button("💾 Save", key=f"msv_{pid}", use_container_width=True):
                        st.session_state["decisions"][pid] = {
                            "decision": MODIFIED,
                            "note":     user_note_input,
                            "action":   new_action,
                            "column":   new_column or None,
                            "params":   current_params,
                        }
                        st.session_state[f"show_modify_{pid}"] = False
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        # ── Explain panel ─────────────────────────────────────────────────────
        if st.session_state.get(f"show_explain_{pid}"):
            # Cache the explanation so we don't re-call the LLM every rerun
            if pid not in st.session_state["explain_cache"]:
                with st.spinner("🤖 Generating deeper explanation…"):
                    st.session_state["explain_cache"][pid] = pipeline.explain_proposal(proposal)

            st.markdown(
                f'<div style="background:rgba(125,211,252,0.05);border:1px solid rgba(125,211,252,0.15);'
                f'border-radius:0.75rem;padding:1rem;font-size:0.82rem;color:#a0b4c4;line-height:1.65;'
                f'margin-top:0.25rem"><strong style="color:#7dd3fc;font-size:0.7rem;letter-spacing:0.1em;'
                f'text-transform:uppercase">AI Deep Explanation</strong><br><br>'
                f'{st.session_state["explain_cache"][pid]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:0.2rem"></div>', unsafe_allow_html=True)

    # ── Execute bar ───────────────────────────────────────────────────────────
    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(125,211,252,0.08);margin:1.5rem 0"/>',
        unsafe_allow_html=True,
    )
    actions_to_execute = approved_count + modified_count
    execute_col, info_col = st.columns([0.35, 0.65])

    with execute_col:
        action_word = "Action" if actions_to_execute == 1 else "Actions"
        if st.button(
            f"⚡  Execute {actions_to_execute} Approved {action_word}",
            use_container_width=True,
            disabled=actions_to_execute == 0,
        ):
            # Record every decision on the ApprovalLayer before handing off
            # to Phase 2.  Any still-pending proposals are treated as rejected.
            for proposal in proposals:
                pid             = proposal.proposal_id
                decision_info   = decisions.get(pid, {})
                decision_type   = decision_info.get("decision", PENDING)
                if decision_type == PENDING:
                    decision_type = REJECTED
                approval_layer.decide(
                    pid,
                    decision_type,
                    decision_info.get("action"),
                    decision_info.get("column"),
                    decision_info.get("params"),
                    decision_info.get("note", ""),
                )
            st.session_state["phase"] = "running"
            st.rerun()

    with info_col:
        if actions_to_execute == 0:
            st.markdown(
                '<div style="padding:0.6rem 0;font-size:0.78rem;color:#64748b">'
                'Approve at least one proposal to enable execution.</div>',
                unsafe_allow_html=True,
            )
        elif pending_count > 0:
            st.markdown(
                f'<div style="padding:0.6rem 0;font-size:0.78rem;color:#fbbf24">'
                f'⚠️ {pending_count} pending proposal(s) will be treated as rejected.</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 3b — RUNNING (transient — user never stays on this screen)
# Executes Phase 2 of the pipeline while showing a progress bar, then
# transitions immediately to the results screen.
# ════════════════════════════════════════════════════════════════════════════════
elif current_phase == "running":
    st.markdown(
        '<div class="glass-hi" style="padding:1.1rem 1.4rem;margin-bottom:1rem">'
        '<div style="font-size:0.9rem;font-weight:700;color:#7dd3fc">'
        '⚡ Executing approved actions…</div></div>',
        unsafe_allow_html=True,
    )

    run_progress = st.progress(0.75)
    run_status   = st.empty()

    pipeline: AIPipeline = st.session_state["pipeline"]

    def _on_progress(msg: str, pct: float) -> None:
        run_progress.progress(min(pct, 1.0))
        run_status.markdown(
            f'<div style="font-size:0.8rem;color:#7dd3fc;font-family:monospace">▶ {msg}</div>',
            unsafe_allow_html=True,
        )

    pipeline.on_progress = _on_progress

    try:
        final_result = pipeline.run_after_approval(save_output=False)
        st.session_state["final_result"] = final_result
        st.session_state["phase"]        = "results"
        st.rerun()
    except Exception as error:
        st.error(f"Execution failed: {error}")
        if st.button("← Back to Approval"):
            st.session_state["phase"] = "approval"
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 4 — RESULTS
# Renders gauges, metric tiles, and five tabs:
# Summary | Changes | EDA | Cleaned Data | JSON Report
# ════════════════════════════════════════════════════════════════════════════════
elif current_phase == "results":
    final_result:    dict = st.session_state["final_result"]
    report:          dict = final_result["report"]
    quality_scores:  dict = report["quality_score"]
    delta:           dict = report["delta"]
    approval_summary:dict = report.get("approval_summary", {})
    change_log:      list = final_result["change_log"]
    detected_issues: list = final_result["issues"]

    # Toast notification in the corner
    st.markdown(
        f'<div class="glacier-toast">'
        f'<div class="toast-icon">✓</div>'
        f'<span class="toast-text">Cleaning complete · '
        f'Score {quality_scores["before"]} → {quality_scores["after"]}</span></div>',
        unsafe_allow_html=True,
    )

    # Three top-level charts: before gauge, after gauge, before/after bar
    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    with gauge_col1:
        st.plotly_chart(
            plot_quality_gauge(quality_scores["before"], "Quality Before"),
            use_container_width=True, key="gb",
        )
    with gauge_col2:
        st.plotly_chart(
            plot_quality_gauge(quality_scores["after"], "Quality After"),
            use_container_width=True, key="ga",
        )
    with gauge_col3:
        st.plotly_chart(
            plot_before_after_comparison(delta),
            use_container_width=True, key="bac",
        )

    # Five headline metric tiles
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    for tile_col, label, value, colour in [
        (metric_col1, "Issues",    str(len(detected_issues)),                    "#f97316"),
        (metric_col2, "Approved",  str(approval_summary.get("approved", 0)),     "#34d399"),
        (metric_col3, "Rejected",  str(approval_summary.get("rejected", 0)),     "#ef4444"),
        (metric_col4, "Score +",   f"+{quality_scores['improvement']}",          "#7dd3fc"),
        (metric_col5, "Rows Kept", f"{delta['row_retention']['after']:.1f}%",    "#c8a0f0"),
    ]:
        tile_col.markdown(
            f'<div class="g-metric">'
            f'<div class="g-metric-val" style="color:{colour}">{value}</div>'
            f'<div class="g-metric-lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result tabs ───────────────────────────────────────────────────────────
    (tab_summary, tab_changes, tab_eda,
     tab_data, tab_report) = st.tabs(
        ["📋 Summary", "🧹 Changes", "📈 EDA", "⬇ Cleaned Data", "📄 Report"]
    )

    with tab_summary:
        st.markdown(
            '<div style="font-size:0.9rem;font-weight:700;color:#e0e8f0;margin-bottom:0.75rem">'
            'Approval Log</div>',
            unsafe_allow_html=True,
        )
        for record in approval_summary.get("records", []):
            icon      = {"approved": "✅", "rejected": "❌", "modified": "✏️"}.get(record["decision"], "⏳")
            col_text  = (
                f' · <span style="color:#7dd3fc;font-family:monospace">{record["final_column"]}</span>'
                if record.get("final_column") else ""
            )
            note_text = (
                f' — <em style="color:#4a6070">{record["user_note"]}</em>'
                if record.get("user_note") else ""
            )
            st.markdown(
                f'<div style="padding:0.55rem 0.9rem;margin-bottom:0.3rem;border-radius:0.6rem;'
                f'background:rgba(15,21,36,0.5);border:1px solid rgba(125,211,252,0.07);'
                f'font-size:0.8rem;color:#a0b4c4">{icon} '
                f'<strong style="color:#e0e8f0">{record["proposal_id"]}</strong> · '
                f'{record["final_action"].replace("_", " ").title()}{col_text}{note_text}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.4rem">'
            f'<div style="font-size:0.65rem;font-weight:700;color:#4a6070;letter-spacing:0.15em;'
            f'text-transform:uppercase;margin-bottom:0.5rem">AI Summary</div>'
            f'<div style="font-size:0.85rem;color:#a0b4c4;line-height:1.65">'
            f'{final_result.get("executive_summary", "")}</div></div>',
            unsafe_allow_html=True,
        )

    with tab_changes:
        if change_log:
            # Show a tidy table — only the four most useful columns
            changes_df = pd.DataFrame(change_log)[["action_id", "action_type", "column", "description"]]
            changes_df.columns = pd.Index(["ID", "Type", "Column", "Description"])
            st.dataframe(changes_df, use_container_width=True, height=400)
        else:
            st.info("No changes were applied (all proposals were rejected).")

    with tab_eda:
        # All EDA charts operate on the raw (pre-cleaning) DataFrame so users
        # can see what the data looked like before the pipeline ran.
        raw_df = final_result["df_raw"]
        st.plotly_chart(plot_missing_bar(raw_df),            use_container_width=True, key="emb")
        st.plotly_chart(plot_missing_heatmap(raw_df),        use_container_width=True, key="emh")
        st.plotly_chart(plot_numeric_distributions(raw_df),  use_container_width=True, key="end")
        st.plotly_chart(plot_outlier_boxplots(raw_df),       use_container_width=True, key="eob")
        st.plotly_chart(plot_correlation_heatmap(raw_df),    use_container_width=True, key="ech")

    with tab_data:
        cleaned_df: pd.DataFrame = final_result["df_cleaned"]
        st.markdown(
            f'<div style="font-size:0.82rem;color:#64748b;margin-bottom:0.5rem">'
            f'{cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(cleaned_df.head(200), use_container_width=True, height=420)

        # Download buttons for both CSV and Excel
        dl_col1, dl_col2, _ = st.columns([1, 1, 3])
        with dl_col1:
            csv_buffer = io.BytesIO()
            cleaned_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇ CSV", csv_buffer.getvalue(),
                "cleaned_dataset.csv", "text/csv", key="dlc",
            )
        with dl_col2:
            xlsx_buffer = io.BytesIO()
            cleaned_df.to_excel(xlsx_buffer, index=False)
            st.download_button(
                "⬇ XLSX", xlsx_buffer.getvalue(),
                "cleaned_dataset.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dlx",
            )

    with tab_report:
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            "⬇ JSON Report", report_json,
            "cleaning_report.json", "application/json", key="dlj",
        )
        st.json(report)

    # Footer
    st.markdown(
        '<div style="margin-top:3rem;padding:1.25rem 0;border-top:1px solid rgba(125,211,252,0.05);'
        'display:flex;justify-content:space-between;opacity:0.4;font-size:0.7rem;color:#2a3a48">'
        '<div>© 2026 Glacier Cleaning Engine · All systems operational</div>'
        '<div style="display:flex;gap:1.5rem">'
        '<a style="color:#2a3a48;text-decoration:none" href="#">Privacy</a>'
        '<a style="color:#2a3a48;text-decoration:none" href="#">Terms</a>'
        '<a style="color:#2a3a48;text-decoration:none" href="#">API Docs</a>'
        '</div></div>',
        unsafe_allow_html=True,
    )
