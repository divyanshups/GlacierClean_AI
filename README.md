# Glacier Clean

**Intelligent Data Cleaning System** — a multi-step Streamlit app that profiles datasets, detects quality issues, lets you clean data manually or with an LLM advisor (Gemini), and reports before/after quality scores.

Prototype for personal / academic use. 

Manual cleaning is the core path; the LLM recommends and can auto-apply a plan when it returns a valid response.

---

## Features

- **Multi-format load** — CSV, Excel (`.xlsx`), JSON, Parquet  
- **Profiling** — column types, nulls, cardinality, numeric stats  
- **Issue detection** — missing values, duplicates, outliers, type issues, cardinality  
- **Manual cleaning studio** — pick operations per column, fill a “bucket”, run  
- **LLM advisor** (Gemini via LangChain) — short JSON recommendations → mapped to real cleaning ops  
- **Evaluation** — completeness, consistency, uniqueness, validity, overall score  
- **Export** — download cleaned CSV  
- **Decision logs** — append run history to JSON (actions, logs, scores)

---

## App pages

| Page | Purpose |
|------|---------|
| **1. Welcome** | File import |
| **2. Profile** | Dataset name, metadata, preview, detected issues |
| **3. Cleaning Studio** | Manual options + bucket · LLM section (response / changes / scores) |
| **4. Results** | Cleaned data, score cards, export |

---

## Project structure

```text
intelligent-data-cleaner/
├── app/
│   └── streamlit_app.py     # UI
├── core/
│   ├── data_loader.py       # Load datasets
│   ├── pipeline.py          # Orchestration
│   └── schema.py            # CleaningAction, ExecutionLog
├──logs/
│   ├── decision_logs.json      # decision log
│   └── pipeline.log            # pipeline log                
├── module/
│   ├── profiler_module.py
│   ├── issue_detector_module.py
│   ├── cleaning_module.py
│   ├── evaluation_module.py
│   └── recommendation_module.py  # Gemini recommendation
├── utils/
│   ├── logger.py
│   ├── metrics.py
│   └── preprocessing_utils.py
├── config/
│   └── config.yaml
├── outputs/                      # cleaned CSVs
├── main.py                       # launches Streamlit
├── requirements.txt
└── README.md