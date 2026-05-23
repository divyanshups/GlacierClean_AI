# 🧹 Intelligent Data Cleaning Agent

An end-to-end, AI-powered data cleaning pipeline that autonomously profiles, detects issues, plans fixes, applies cleaning operations, and evaluates data quality — with a polished Streamlit UI.

```
Input Dataset → Profiler → Issue Detector → Decision Engine → Cleaning Engine → Evaluator → Report
```

---

## ✨ Features

| Capability | Details |
|---|---|
| **Auto Profiling** | Shape, dtype, missing %, cardinality, skewness, correlation |
| **Issue Detection** | Missing values · Duplicates · Outliers · Constant cols · Mixed types · Skewness · String anomalies |
| **Decision Engine** | Confidence-scored action plan with full rationale log |
| **Cleaning Engine** | Imputation · Duplicate removal · Outlier clipping/winsorizing · Text normalisation · Date inference · Type coercion |
| **Evaluation** | Before/after quality score (0–100) across 5 dimensions |
| **Reporting** | JSON report + Plotly visualisations + downloadable cleaned CSV |
| **Streamlit UI** | Dark, modern interface — upload, configure, run, inspect, download |

---

## 🚀 Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2a · CLI
```bash
python main.py --input data/raw/my_dataset.csv
# Options:
#   --config  path/to/config.yaml   (default: config/config.yaml)
#   --no-save                        skip writing outputs to disk
```

### 2b · Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

### 2c · Python API
```python
from core.pipeline import CleaningPipeline

pipeline = CleaningPipeline()
result   = pipeline.run("data/raw/my_dataset.csv")

df_cleaned = result["df_cleaned"]
report     = result["report"]
print(f"Quality: {report['quality_score']['before']} → {report['quality_score']['after']}")
```

---

## 🗂️ Project Structure

```
autonomous-data-cleaning-agent/
├── agents/
│   ├── profiler_agent.py      # EDA — builds column-level profile
│   ├── issue_detector.py      # Detects and ranks data-quality issues
│   ├── decision_agent.py      # Maps issues → confidence-scored actions
│   ├── cleaning_agent.py      # Applies cleaning operations
│   └── evaluation_agent.py   # Before/after quality comparison
├── core/
│   ├── data_loader.py         # Multi-format dataset loader
│   └── pipeline.py            # Main orchestration loop
├── utils/
│   ├── preprocessing_utils.py # Low-level wrangling helpers
│   ├── visualization.py       # Plotly chart factories
│   ├── metrics.py             # Quality score computation
│   └── logger.py              # Rich console + file logging
├── app/
│   └── streamlit_app.py       # Streamlit UI
├── config/
│   └── config.yaml            # All tuneable parameters
├── memory/
│   └── decision_logs.json     # Persistent decision audit trail
├── notebooks/
│   └── eda_experiments.ipynb  # Experimentation notebook
├── tests/
│   ├── test_loader.py
│   └── test_pipeline.py
├── outputs/                   # Cleaned CSV + JSON reports
├── main.py                    # CLI entry point
└── requirements.txt
```

---

## ⚙️ Configuration (`config/config.yaml`)

| Section | Key | Default | Description |
|---|---|---|---|
| `pipeline` | `confidence_threshold` | `0.75` | Min confidence to auto-apply an action |
| `cleaning` | `missing_numeric_strategy` | `median` | `mean` / `median` / `mode` |
| `cleaning` | `missing_categorical_strategy` | `mode` | `mode` / `constant` |
| `cleaning` | `outlier_strategy` | `clip` | `clip` / `winsorize` / `remove` / `none` |
| `cleaning` | `text_normalization` | `true` | Strip, lowercase, collapse whitespace |
| `cleaning` | `date_inference` | `true` | Auto-parse string columns as dates |
| `issue_detector` | `missing_critical` | `0.50` | Threshold for CRITICAL missing severity |
| `issue_detector` | `outlier_method` | `iqr` | `iqr` / `zscore` |

---

## 📊 Quality Score

The composite quality score (0–100) is a weighted average of five dimensions:

| Dimension | Weight |
|---|---|
| Missing rate | 30% |
| Duplicate rate | 20% |
| Outlier rate | 20% |
| Schema consistency | 20% |
| Row retention | 10% |

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 📦 Supported File Formats

CSV · TSV · XLSX · XLS

---

## 🏗️ Architecture

```
┌──────────────────────┐
│    Input Dataset      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   ProfilerAgent       │  Column stats, dtypes, cardinality, skewness
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   IssueDetector       │  Ranked list of Issues with severity
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   DecisionAgent  🧠   │  Confidence-scored CleaningAction plan
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   CleaningAgent       │  Applies actions, records change log
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   EvaluationAgent     │  Before/after metrics, quality score, report
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  cleaned_dataset.csv  │
│  cleaning_report.json │
└──────────────────────┘
```

---

## 📄 License

MIT
