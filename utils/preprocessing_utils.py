from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def impute_numeric(series: pd.Series, strategy: str = "median") -> pd.Series:
    if series.isnull().sum() == 0:
        return series

    if strategy == "mean":
        return series.fillna(series.mean())
    if strategy == "mode":
        mode_values = series.mode()
        return series.fillna(mode_values.iloc[0] if not mode_values.empty else 0)

    return series.fillna(series.median())


def impute_categorical(
    series: pd.Series,
    strategy: str = "mode",
    constant: str = "Unknown",
) -> pd.Series:
    if series.isnull().sum() == 0:
        return series

    if strategy == "mode":
        mode_values = series.mode()
        fill_value = mode_values.iloc[0] if not mode_values.empty else constant
        return series.fillna(fill_value)
    if strategy == "constant":
        return series.fillna(constant)

    return series


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    z_scores = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.loc[series.dropna().index] = z_scores > threshold
    return mask


def clip_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Clip values to the [Q1 - factor*IQR, Q3 + factor*IQR] range."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)


def winsorize_series(series: pd.Series, limits: tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    arr = stats.mstats.winsorize(series.dropna(), limits=limits)
    result = series.copy()
    result.loc[series.dropna().index] = arr
    return result


def normalize_text_column(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .replace("nan", np.nan)
    )


def try_parse_dates(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series

    sample = series.dropna().head(200)
    try:
        parsed_sample = pd.to_datetime(sample, errors="coerce")
        success_rate = parsed_sample.notna().mean()
        if success_rate >= 0.80:
            return pd.to_datetime(series, errors="coerce")
    except Exception:
        pass

    return series


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Try to convert values to numeric, setting failures to NaN."""
    return pd.to_numeric(series, errors="coerce")


def strip_currency_symbols(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    cleaned = cleaned.str.replace("\u20ac", "", regex=False)
    cleaned = cleaned.str.replace("\u00a3", "", regex=False)
    cleaned = cleaned.str.replace("\u00a5", "", regex=False)
    return cleaned.str.strip()


def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str | bool = "first",
) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep=keep)


def infer_and_cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col]
        unique_count = series.nunique(dropna=True)
        valid_count = series.notna().sum()

        numeric_series = coerce_numeric(strip_currency_symbols(series))
        parse_rate = numeric_series.notna().sum() / max(valid_count, 1)
        if parse_rate > 0.85:
            df[col] = numeric_series
            continue

        parsed_dates = try_parse_dates(series)
        if pd.api.types.is_datetime64_any_dtype(parsed_dates):
            df[col] = parsed_dates
            continue

        if valid_count > 0 and unique_count / valid_count < 0.20:
            df[col] = series.astype("category")

    return df
