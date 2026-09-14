import os
import tempfile
import streamlit as st
import pandas as pd

from core.pipeline import DataCleaningPipeline
from module.recommendation_module import RecommendationModule
from core.schema import CleaningAction

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glacier Clean",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS Design - Square corners & appropriate padding ─────────
st.markdown(
    """
    <style>
    /* Plain cornered rectangles */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 0 !important;
    }
    .stButton > button {
        border-radius: 0 !important;
    }
    .stSelectbox [data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    [data-baseweb="input"] {
        border-radius: 0 !important;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 0 !important;
    }
    /* Safe padding to prevent top clipping */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── SESSION INIT ───────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "page": "welcome",
        "pipeline": None,
        "recommender": None,
        "df": None,
        "meta": None,
        "profile": None,
        "issues": [],
        "cleaning_plan": [],
        "llm_response": None,
        "llm_status": None,
        "report": None,
        "df_clean": None,
        "exec_logs" : None,
        "_bootstrapped": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.pipeline is None:
        st.session_state.pipeline = DataCleaningPipeline()
    if st.session_state.recommender is None:
        st.session_state.recommender = RecommendationModule()
    st.session_state._bootstrapped = True


_init_state()


def require_data():
    if st.session_state.df is None or st.session_state.meta is None:
        st.session_state.page = "welcome"
        st.warning("No dataset loaded. Upload a file first.")
        st.stop()



# PAGE 1 — WELCOME
if st.session_state.page == "welcome":

    st.markdown("<br><br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    # Welcome Text
    with left:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## Hello!")
        st.markdown("# Welcome to Glacier Clean")
        st.caption("Intelligent Data Cleaning | Powered by LLM Recommendations")

    # Uploader syntax
    with right:
        with st.container(border=True, height=400):
            st.subheader("File import")
            uploaded = st.file_uploader(
                "Choose a Dataset",
                type=["csv", "json", "parquet", "xlsx"],
                key="uploader_welcome",
            )
            if uploaded is not None:
                if st.button("Upload & Analyze", type="primary", use_container_width=True, key="btn_upload"):
                    suffix = "." + uploaded.name.rsplit(".", 1)[-1]
                    tmp_path = None
                    try:
                        with st.spinner("Loading.."):
                            # assigning temporary names
                            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                            os.close(fd)
                            with open(tmp_path, "wb") as f:
                                f.write(uploaded.getvalue())

                            # loading file
                            df, meta = st.session_state.pipeline.load(tmp_path)
                            profile = st.session_state.pipeline.profile_only(df)
                            issues = st.session_state.pipeline.detect_only(df, profile)

                            st.session_state.df = df
                            st.session_state.meta = meta
                            st.session_state.profile = profile
                            st.session_state.issues = issues or []
                            st.session_state.cleaning_plan = []
                            st.session_state.report = None
                            st.session_state.df_clean = None
                            st.session_state.llm_response = None
                            st.session_state.llm_status = None
                            st.session_state.page = "profile"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass


# PAGE 2 — PROFILE
elif st.session_state.page == "profile":
    require_data()

    meta = st.session_state.meta
    df = st.session_state.df
    issues = st.session_state.issues or []

    file_name = getattr(meta, "file_name", "unknown")
    shape = getattr(meta, "shape", (len(df), df.shape[1] if df is not None else 0))
    size_kb = getattr(meta, "file_size_kb", "—")

    # Dataset information
    with st.container(border=True, height=100):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dataset Name**")
            st.write(file_name)
        with c2:
            st.markdown("**Dataset Information**")
            st.write(f"Rows: `{shape[0]}` · Columns: `{shape[1]}` · Size: `{size_kb} KB`")

    # Dataset view
    with st.container(border=True, height=260):
        st.markdown("**Dataset**")
        st.dataframe(df.head(200), use_container_width=True, height=190)

    # Dataset issues
    with st.container(border=True, height=250):
        st.markdown("**Detected Issues**")
        if not issues:
            st.success("No issues detected.")
        else:
            rows = []
            for i in issues:
                rows.append(
                    {
                        "Severity": str(getattr(i, "severity", "")).upper(),
                        "Column": str(getattr(i, "column", "")),
                        "Issue": str(getattr(i, "issue_type", "")),
                        "Affected": f"{float(getattr(i, 'affected_pct', 0)):.1%}",
                        "Description": str(getattr(i, "description", ""))[:200],
                    }
                )
            issues_df = pd.DataFrame(rows)
            st.dataframe(issues_df, use_container_width=True, height=180)

    b1, b2 = st.columns([1, 4])
    with b1:
        if st.button("← Back", key="profile_back"):
            st.session_state.page = "welcome"
            st.rerun()
    with b2:
        if st.button("Proceed to Cleaning Studio →", type="primary", use_container_width=True, key="profile_next"):
            st.session_state.page = "cleaning"
            st.rerun()


# PAGE 3 — CLEANING STUDIO
elif st.session_state.page == "cleaning":
    require_data()

    if st.button("← Back to Profile", key="clean_back"):
        st.session_state.page = "profile"
        st.rerun()

    top_l, top_r = st.columns([2, 1], gap="medium")

    with top_l:
        with st.container(border=True, height=330):
            st.markdown("**Options to select for each column**")
            cols = ["__dataset__"] + list(st.session_state.df.columns)
            c_col = st.selectbox(
                "Target column",
                options=cols,
                format_func=lambda x: "Full Dataset" if x == "__dataset__" else x,
                key="opt_col"
            )
            c_op = st.selectbox(
                "Operation",
                [
                    "remove_duplicates", "impute_missing", "drop_column",
                    "cap_outliers", "standardize_strings", "cast_type",
                ],
                key="opt_op",
            )
            params = {}
            if c_op == "impute_missing":
                params["strategy"] = st.selectbox("Strategy", ["median", "mean", "mode", "constant"], key="opt_strat")
            elif c_op == "remove_duplicates":
                params["keep"] = st.selectbox("Keep", ["first", "last", "none"], key="opt_keep")
            elif c_op == "cap_outliers":
                params["method"] = st.selectbox("Method", ["iqr", "zscore"], key="opt_method")
                params["multiplier"] = st.number_input("Multiplier / threshold", value=1.5, key="opt_mult")
            elif c_op == "cast_type":
                params["target_type"] = st.selectbox("Target type", ["float", "int", "str", "bool"], key="opt_cast")

            if st.button("Add to bucket", key="btn_add_bucket"):
                st.session_state.cleaning_plan.append(
                    CleaningAction(
                        column=c_col, operation=c_op, params=params,
                        source="manual", rationale="User manual selection",
                    )
                )
                st.rerun()

    with top_r:
        with st.container(border=True, height=330):
            st.markdown("**bucket and run button**")
            plan = st.session_state.cleaning_plan or []
            with st.container(height=180):
                if not plan:
                    st.caption("changes bucket to proceed")
                else:
                    for i, act in enumerate(plan, 1):
                        st.write(f"{i}. `{act.operation}` → `{act.column}`")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear", key="btn_clear_bucket"):
                    st.session_state.cleaning_plan = []
                    st.rerun()
            with c2:
                if st.button("Run manual", type="primary", key="btn_run_manual"):
                    if not st.session_state.cleaning_plan:
                        st.warning("Bucket empty.")
                    else:
                        with st.spinner("Running manual plan…"):
                            df_clean, logs, report = st.session_state.pipeline.run_clean_and_evaluate(
                            st.session_state.df,
                            st.session_state.cleaning_plan,
                            source="manual",
                            metadata=st.session_state.meta,
                            issues=st.session_state.issues,
                            save_outputs=True,
                            save_decision_log=True,
                        )
                            st.session_state.df_clean = df_clean
                            st.session_state.report = report
                            st.session_state.exec_logs = logs
                            st.session_state.llm_status = "MANUAL_SUCCESS"
                        st.session_state.page = "results"
                        st.rerun()

    with st.container(border=True):
        st.markdown("**LLM Section**")
        if st.button("Analyze & Clean using LLM", use_container_width=True, key="btn_llm"):
            with st.spinner("Calling LLM…"):
                chat_response, actions, success = st.session_state.recommender.recommend(st.session_state.issues)
                st.session_state.llm_response = chat_response
                if success and actions:
                    df_clean, logs, report = st.session_state.pipeline.run_clean_and_evaluate(
                    st.session_state.df,
                    actions,
                    source="llm",
                    metadata=st.session_state.meta,
                    issues=st.session_state.issues,
                    save_outputs=True,
                    save_decision_log=True,
                )
                    st.session_state.df_clean = df_clean
                    st.session_state.report = report
                    st.session_state.llm_status = "SUCCESS"
                else:
                    st.session_state.df_clean = None
                    st.session_state.report = None
                    st.session_state.exec_logs = logs
                    st.session_state.llm_status = "FAILED"
            st.rerun()

        L, M, R = st.columns(3)

        with L:
            st.caption("llm response")
            with st.container(border=True, height=220):
                if st.session_state.llm_status == "FAILED":
                    st.error("LLM is not able to produce a correct result.")
                elif st.session_state.llm_response:
                    st.markdown(st.session_state.llm_response)
                else:
                    st.caption("Awaiting LLM…")

        with M:
            st.caption("what changes were made")
            with st.container(border=True, height=220):
                if st.session_state.llm_status == "FAILED":
                    st.error("LLM not working")
                elif st.session_state.report is not None and st.session_state.llm_status == "SUCCESS":
                    for log in st.session_state.report.operations_applied:
                        icon = "OK" if log.status == "success" else "FAIL"
                        st.write(f"[{icon}] `{log.column}` · {log.operation}")
                else:
                    st.caption("No LLM changes yet.")

        with R:
            st.caption("changes after llm recommendation")
            with st.container(border=True, height=220):
                if st.session_state.llm_status == "FAILED":
                    st.error("LLM not working")
                elif st.session_state.report is not None and st.session_state.llm_status == "SUCCESS":
                    box = st.session_state.report.to_score_box()
                    st.write(f"Before: **{box['before']['overall']:.1%}**")
                    st.write(f"After: **{box['after']['overall']:.1%}**")
                    st.write(f"Delta: **{box['delta']['overall']:+.1%}**")
                    if st.button("View final data & export →", key="btn_llm_to_results"):
                        st.session_state.page = "results"
                        st.rerun()
                else:
                    st.caption("Awaiting evaluation…")



# PAGE 4 — RESULTS / EXPORT
elif st.session_state.page == "results":
    if st.session_state.df_clean is None:
        st.warning("No cleaned data. Run a plan first.")
        if st.button("← Cleaning studio"):
            st.session_state.page = "cleaning"
            st.rerun()
        st.stop()

    if st.button("← Back to Cleaning Studio", key="results_back"):
        st.session_state.page = "cleaning"
        st.rerun()

    left, right = st.columns([2, 1], gap="medium")

    with left:
        with st.container(border=True, height=520):
            st.markdown("**cleaned data**")
            failed_logs = [
                l for l in (st.session_state.exec_logs or []) if l.status == "error"
            ]
            if failed_logs:
                with st.expander(f"{len(failed_logs)} actions(s) failed", expanded = True):
                    for l in failed_logs:
                        st.error(f"`{l.column}' . {l.operation}: {l.message}")
            st.dataframe(st.session_state.df_clean, use_container_width=True, height=460)

    with right:
        with st.container(border=True, height=520):
            st.markdown("**score**")

            if st.session_state.report is None:
                st.warning("No score.")
            else:
                box = st.session_state.report.to_score_box()

                before = box.get("before") or {}
                after = box.get("after") or {}
                delta = box.get("delta") or {}

                def pct(x):
                    try:
                        return f"{float(x):.1%}"
                    except Exception:
                        return "—"

                def dlt(x):
                    try:
                        return f"{float(x):+.1%}"
                    except Exception:
                        return "—"

                rows = [
                    ("Overall", after.get("overall"), delta.get("overall"), before.get("overall")),
                    ("Completeness", after.get("completeness"), delta.get("completeness"), before.get("completeness")),
                    ("Consistency", after.get("consistency"), delta.get("consistency"), before.get("consistency")),
                    ("Validity", after.get("validity"), delta.get("validity"), before.get("validity")),
                    ("Uniqueness", after.get("uniqueness"), delta.get("uniqueness"), before.get("uniqueness")),
                ]

                with st.container(height=440):
                    for name, a, d, b in rows:
                        with st.container(border=True):
                            c1, c2 = st.columns([2, 1])
                            with c1:
                                st.markdown(f"**{name}**")
                                # FIX: Changed text color to a bright, readable green (#4CAF50)
                                st.markdown(
                                    f"<p style='font-size:1.6rem;margin:0;font-weight:700;color:#4CAF50'>{pct(a)}</p>",
                                    unsafe_allow_html=True,
                                )
                                st.caption(f"Before: {pct(b)}")
                            with c2:
                                try:
                                    dv = float(d)
                                    color = "#4CAF50" if dv >= 0 else "#ff4b4b"
                                    arrow = "↑" if dv >= 0 else "↓"
                                except Exception:
                                    color = "#888"
                                    arrow = "•"
                                st.markdown(
                                    f"<p style='font-size:1.1rem;margin-top:1.4rem;color:{color};font-weight:600'>{arrow} {dlt(d)}</p>",
                                    unsafe_allow_html=True,
                                )

    with st.container(border=True, height=120):
        st.markdown("**Export Cleaned Data**")
        csv_data = st.session_state.df_clean.to_csv(index=False).encode("utf-8")
        name = getattr(st.session_state.meta, "file_name", "data") if st.session_state.meta else "data"
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_data,
            file_name=f"cleaned_{name}",
            mime="text/csv",
            use_container_width=True,
            type="primary",
            key="btn_download",
        )