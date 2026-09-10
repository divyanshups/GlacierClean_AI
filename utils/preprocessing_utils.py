from typing import Literal, Optional
import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

logger = get_logger(__name__)

# detect column types
def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    type_result = {}

    for col in df.columns:
        #Check if column is empty
        col_data = df[col].dropna()
        if len(col_data) == 0:
            type_result[col] = "unknown"
        
        n_unique = col_data.nunique()
        n_total = len(col_data)
        cardinality_ratio = n_unique / n_total

        # Check for constant columns (only 1 unique value)
        if n_unique == 1:
            type_result[col] = "constant"
            continue

        # Check for boolean (exactly 2 unique values)
        if n_unique == 2:
            type_result[col] = "boolean"
            continue

        # --- Numeric columns ---
        if pd.api.types.is_numeric_dtype(col_data):
            if cardinality_ratio > 0.05:
                type_result[col] = "continuous"
            else:
                type_result[col] = "categorical"  # numeric but few unique values
            continue

        # --- Datetime columns ---
        if pd.api.types.is_datetime64_any_dtype(col_data):
            type_result[col] = "datetime"
            continue

        # --- String columns ---
        if pd.api.types.is_object_dtype(col_data):
            # Try to detect if strings are actually datetimes
            sample = col_data.head(100)
            try:
                pd.to_datetime(sample, format="mixed")
                type_result[col] = "datetime"
                continue
            except (ValueError, TypeError):
                pass

            # Check if strings are long (free text)
            avg_word_count = col_data.head(100).astype(str).apply(
                lambda x: len(x.split())
            ).mean()
            if avg_word_count > 5:
                type_result[col] = "free_text"
                continue

            # High cardinality string = likely identifier
            if cardinality_ratio > 0.95:
                type_result[col] = "identifier"
                continue

            # Low-to-medium cardinality string = categorical
            type_result[col] = "categorical"
            continue

        type_result[col] = "unknown"

    logger.debug(f"Detected column types: {type_result}")
    return type_result

#--------------------------------------------------------DROP COLUMN----------------------------------------------------------------
def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df_copy = df.drop(columns=[column])
    logger.info(f"Dropped column '{column}'")
    return df_copy

#--------------------------------------------------------HANDLING MISSING VALUES-----------------------------------------------------------------------------
def impute_missing(
    df: pd.DataFrame,
    column: str,
    strategy: Literal["mean", "median", "mode", "constant", "drop"],
    fill_value: Optional[any] = None,
    ) -> pd.DataFrame:
    if column not in df.columns:
        logger.error("Column not found in dataset.")
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df_copy = df.copy()
    n_missing_before = df_copy[column].isna().sum()

    if n_missing_before == 0:
        logger.debug(f"Column '{column}' has no missing values. Skipping imputation.")
        return df_copy

    if strategy == "mean":
        if not pd.api.types.is_numeric_dtype(df_copy[column]):
            raise ValueError(f"Cannot use 'mean' strategy on non-numeric column '{column}'")
        fill_val = df_copy[column].mean()
        df_copy[column] = df_copy[column].fillna(fill_val)

    elif strategy == "median":
        if not pd.api.types.is_numeric_dtype(df_copy[column]):
            raise ValueError(f"Cannot use 'median' strategy on non-numeric column '{column}'")
        fill_val = df_copy[column].median()
        df_copy[column] = df_copy[column].fillna(fill_val)

    elif strategy == "mode":
        # Mode works for both numeric and categorical
        mode_result = df_copy[column].mode()
        if len(mode_result) == 0:
            logger.warning(f"Could not compute mode for '{column}'. Using 'unknown'.")
            df_copy[column] = df_copy[column].fillna("unknown")
        else:
            df_copy[column] = df_copy[column].fillna(mode_result[0])

    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='constant'")
        df_copy[column] = df_copy[column].fillna(fill_value)

    elif strategy == "drop":
        # Remove entire rows where this column is null
        df_copy = df_copy.dropna(subset=[column])

    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")

    n_missing_after = df_copy[column].isna().sum()
    n_filled = n_missing_before - n_missing_after

    logger.info(
        f"Imputed '{column}' using '{strategy}': "
        f"filled {n_filled} missing values"
    )

    return df_copy

#--------------------------------------------------------REMOVING DUPLICATE VALUES-----------------------------------------------------------------------------
def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    keep: Literal["first", "last", "none"] = "first",
    ) -> pd.DataFrame:
    df_copy = df.copy()
    n_before = len(df_copy)

    keep_param = False if keep == "none" else keep
    df_copy = df_copy.drop_duplicates(subset=subset, keep=keep_param)

    n_removed = n_before - len(df_copy)
    logger.info(f"Removed {n_removed} duplicate rows. Rows remaining: {len(df_copy)}")

    return df_copy

#--------------------------------------------------------REMOVING OUTLIERS-----------------------------------------------------------------------------
def detect_outliers_iqr(
    series: pd.Series, iqr_multiplier: float = 1.5
    ) -> pd.Series:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - iqr_multiplier * IQR
    upper_fence = Q3 + iqr_multiplier * IQR
    return (series < lower_fence) | (series > upper_fence)

def detect_outliers_zscore(
    series: pd.Series, threshold: float = 3.0
    ) -> pd.Series:
    z_scores = np.abs(stats.zscore(series.dropna()))
    # Re-index to original series shape, marking NaN positions as False
    outlier_mask = pd.Series(False, index=series.index)
    non_null_idx = series.dropna().index
    outlier_mask[non_null_idx] = z_scores > threshold
    return outlier_mask

def cap_outliers(
    df: pd.DataFrame,
    column: str,
    method: Literal["iqr", "zscore"] = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    ) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")

    df_copy = df.copy()
    series = df_copy[column]

    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - iqr_multiplier * IQR
        upper_fence = Q3 + iqr_multiplier * IQR

    elif method == "zscore":
        mean = series.mean()
        std = series.std()
        lower_fence = mean - z_threshold * std
        upper_fence = mean + z_threshold * std
    else:
        raise ValueError(f"Unknown method: '{method}'")

    n_outliers = ((series < lower_fence) | (series > upper_fence)).sum()
    df_copy[column] = series.clip(lower=lower_fence, upper=upper_fence)

    logger.info(
        f"Capped {n_outliers} outliers in '{column}' using {method} method. "
        f"Fences: [{lower_fence:.2f}, {upper_fence:.2f}]"
    )

    return df_copy

#----------------------------------------------------------STRING CLEANING-----------------------------------------------------------------------
def standardize_string_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    df_copy = df.copy()
    original = df_copy[column].copy()

    # Apply string cleaning only to non-null values
    df_copy[column] = (
        df_copy[column]
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.lower()
    )

    # Restore original NaN values to nan
    df_copy.loc[original.isna(), column] = np.nan

    logger.info(f"Standardized string formatting in column '{column}'")
    return df_copy

#--------------------------------------------------------TO DATETIME-----------------------------------------------------------------------------
def standardize_dates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    df_copy = df.copy()
    original_null_count = df_copy[column].isna().sum()

    df_copy[column] = pd.to_datetime(df_copy[column], format = "mixed", errors="coerce")

    new_null_count = df_copy[column].isna().sum()
    parse_failures = new_null_count - original_null_count

    if parse_failures > 0:
        logger.warning(
            f"Could not parse {parse_failures} date values in '{column}'. "
            f"There are Not A Time values."
        )
    else:
        logger.info(f"Successfully converted '{column}' to datetime.")

    return df_copy

#--------------------------------------------------------CAST COLUMN TYPES-----------------------------------------------------------------------------
def cast_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: Literal["int", "float", "str", "bool", "category"],
) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    df_copy = df.copy()

    type_map = {
        "int": "Int64",
        "float": "float64",
        "str": "string",
        "bool": "boolean",
        "category": "category",
    }

    target_dtype = type_map.get(target_type)
    if not target_dtype:
        raise ValueError(f"Unknown target type: '{target_type}'")

    try:
        df_copy[column] = df_copy[column].astype(target_dtype)
        logger.info(f"Cast column '{column}' to {target_dtype}")
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to cast '{column}' to {target_dtype}: {e}")
        raise

    return df_copy