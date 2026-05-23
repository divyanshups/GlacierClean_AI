from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent colour palette used across all charts
PALETTE = px.colors.qualitative.Bold

# ── Missing-value heatmap ─────────────────────────────────────────────────────

def plot_missing_heatmap(df: pd.DataFrame, max_cols: int = 40) -> go.Figure:

    columns       = df.columns[:max_cols]
    missing_matrix = df[columns].isnull().astype(int)

    fig = go.Figure(go.Heatmap(
        z             = missing_matrix.values,
        x             = columns.tolist(),
        y             = list(range(len(df))),
        colorscale    = [[0, "#1e293b"], [1, "#f97316"]],
        showscale     = False,
        hovertemplate = "Col: %{x}<br>Row: %{y}<br>Missing: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title        = "Missing Value Map (orange = missing)",
        xaxis_title  = "Column",
        yaxis_title  = "Row index",
        height       = 350,
        margin       = dict(l=40, r=20, t=50, b=40),
        plot_bgcolor  = "#0f172a",
        paper_bgcolor = "#0f172a",
        font_color    = "#e2e8f0",
    )
    return fig


# ── Missing % bar chart ───────────────────────────────────────────────────────

def plot_missing_bar(df: pd.DataFrame) -> go.Figure:
    
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        fig = go.Figure()
        fig.update_layout(title="No missing values detected 🎉", height=200)
        return fig

    fig = px.bar(
        x                    = missing_pct.values,
        y                    = missing_pct.index,
        orientation          = "h",
        color                = missing_pct.values,
        color_continuous_scale = "Oranges",
        labels               = {"x": "Missing %", "y": "Column"},
        title                = "Missing Values per Column (%)",
    )
    fig.update_layout(
        height               = max(250, 30 * len(missing_pct)),
        coloraxis_showscale  = False,
        margin               = dict(l=140, r=20, t=50, b=40),
        plot_bgcolor          = "#0f172a",
        paper_bgcolor         = "#0f172a",
        font_color            = "#e2e8f0",
    )
    return fig


# ── Distribution grid ─────────────────────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame, max_cols: int = 12) -> go.Figure:
    
    numeric_cols = df.select_dtypes(include=np.number).columns[:max_cols].tolist()
    if not numeric_cols:
        return go.Figure()

    num_columns = min(3, len(numeric_cols))
    num_rows    = (len(numeric_cols) + num_columns - 1) // num_columns
    fig = make_subplots(rows=num_rows, cols=num_columns, subplot_titles=numeric_cols)

    for index, col in enumerate(numeric_cols):
        row_idx, col_idx = divmod(index, num_columns)
        fig.add_trace(
            go.Histogram(
                x            = df[col].dropna(),
                name         = col,
                marker_color = PALETTE[index % len(PALETTE)],
                opacity      = 0.85,
                showlegend   = False,
            ),
            row = row_idx + 1,
            col = col_idx + 1,
        )

    fig.update_layout(
        title         = "Numeric Column Distributions",
        height        = 280 * num_rows,
        plot_bgcolor  = "#0f172a",
        paper_bgcolor = "#0f172a",
        font_color    = "#e2e8f0",
    )
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, max_cols: int = 20) -> go.Figure:
   
    numeric_df = df.select_dtypes(include=np.number).iloc[:, :max_cols]
    if numeric_df.shape[1] < 2:
        return go.Figure()

    corr = numeric_df.corr()
    fig = go.Figure(go.Heatmap(
        z             = corr.values,
        x             = corr.columns.tolist(),
        y             = corr.index.tolist(),
        colorscale    = "RdBu",
        zmin          = -1,
        zmax          = 1,
        text          = corr.round(2).values,
        texttemplate  = "%{text}",
        hovertemplate = "%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title         = "Correlation Matrix",
        height        = 500,
        margin        = dict(l=100, r=20, t=60, b=100),
        plot_bgcolor  = "#0f172a",
        paper_bgcolor = "#0f172a",
        font_color    = "#e2e8f0",
    )
    return fig


# ── Outlier box plots ─────────────────────────────────────────────────────────

def plot_outlier_boxplots(df: pd.DataFrame, max_cols: int = 12) -> go.Figure:
    
    numeric_cols = df.select_dtypes(include=np.number).columns[:max_cols].tolist()
    if not numeric_cols:
        return go.Figure()

    fig = go.Figure()
    for index, col in enumerate(numeric_cols):
        fig.add_trace(go.Box(
            y            = df[col].dropna(),
            name         = col,
            marker_color = PALETTE[index % len(PALETTE)],
            boxmean      = True,   # show the mean as a dashed line inside the box
        ))

    fig.update_layout(
        title         = "Outlier Detection — Box Plots",
        height        = 420,
        plot_bgcolor  = "#0f172a",
        paper_bgcolor = "#0f172a",
        font_color    = "#e2e8f0",
    )
    return fig


# ── Quality score gauge ───────────────────────────────────────────────────────

def plot_quality_gauge(score: float, title: str = "Quality Score") -> go.Figure:
    if score >= 80:
        colour = "#22c55e"   # green
    elif score >= 50:
        colour = "#f97316"   # orange
    else:
        colour = "#ef4444"   # red

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        title = {"text": title, "font": {"size": 16, "color": "#e2e8f0"}},
        gauge = {
            "axis":      {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar":       {"color": colour},
            "bgcolor":   "#1e293b",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0,   50],  "color": "#1e293b"},
                {"range": [50,  80],  "color": "#1e293b"},
                {"range": [80, 100],  "color": "#1e293b"},
            ],
            "threshold": {
                "line":      {"color": "#f8fafc", "width": 3},
                "thickness": 0.75,
                "value":     score,
            },
        },
        number = {"suffix": "/100", "font": {"color": colour, "size": 36}},
    ))
    fig.update_layout(
        height        = 280,
        margin        = dict(l=20, r=20, t=40, b=20),
        paper_bgcolor = "#0f172a",
        font_color    = "#e2e8f0",
    )
    return fig


# ── Before / after comparison ─────────────────────────────────────────────────

def plot_before_after_comparison(delta: dict[str, Any]) -> go.Figure:

    metric_keys    = ["missing_rate", "duplicate_rate", "outlier_rate"]
    metric_labels  = ["Missing %", "Duplicate %", "Outlier %"]
    before_values  = [delta.get(key, {}).get("before", 0) for key in metric_keys]
    after_values   = [delta.get(key, {}).get("after",  0) for key in metric_keys]

    fig = go.Figure(data=[
        go.Bar(name="Before", x=metric_labels, y=before_values, marker_color="#f97316"),
        go.Bar(name="After",  x=metric_labels, y=after_values,  marker_color="#22c55e"),
    ])
    fig.update_layout(
        barmode       = "group",
        title         = "Before vs After Cleaning",
        yaxis_title   = "Percentage",
        height        = 350,
        plot_bgcolor  = "#0f172a",
        paper_bgcolor = "#0f172a",
        font_color    = "#e2e8f0",
        legend        = dict(bgcolor="#1e293b"),
    )
    return fig
