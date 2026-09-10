import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd
import yaml

from utils.logger import get_logger
logger = get_logger(__name__)

@dataclass
class LoadMetadata:
    file_name: str
    file_format: str
    file_size_kb: str
    shape: tuple
    column_names: list
    dtypes: dict
    memory_usage_kb: float
    load_timestamp: str
    dataset_fingerprint: str
    warnings: list = field(default_factory=list)

def _compute_fingerprint(df: pd.DataFrame):
    structure_string = ",".join(
        [f"{col}:{str(dtype)}" for col, dtype in df.dtypes.items()])
    hash_object = hashlib.md5(structure_string.encode())
    return hash_object.hexdigest()[:8]

def _load_csv(file_path: str, config: dict) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        encoding=config.get("encoding", "utf-8"),
        low_memory=False
    )

def _load_json(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_json(file_path)
    except ValueError:
        return pd.read_json(file_path, lines=True)

def _load_parquet(file_path: str) -> pd.DataFrame:
    return pd.read_parquet(file_path)

def _load_excel(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

#--- Main Data Loader ---
def load_dataset(
    file_path: str,
    config_path: str = "config/config.yaml",
) -> tuple[pd.DataFrame, LoadMetadata]:
    logger.info(f"Attempting to load dataset from: [bold]{file_path}[/bold]")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"No file found at path: {file_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    loader_config = config.get("data_loader", {})

    file_extension = file_path.rsplit(".", 1)[-1].lower()
    supported_formats = loader_config.get("supported_formats", ["csv"])

    if file_extension not in supported_formats:
        logger.error(f"File in unsupported format: {file_path}")
        raise ValueError(
            f"Unsupported format: .{file_extension}. "
            f"Supported formats: {supported_formats}"
        )

    file_size_bytes = os.path.getsize(file_path)
    file_size_kb = file_size_bytes / 1024
    file_size_mb = file_size_kb / 1024
    max_size_mb = loader_config.get("max_file_size_mb", 100)

    if file_size_mb > max_size_mb:
        logger.error(f"File large in size: {file_path}")
        raise ValueError(
            f"File size {file_size_mb:.1f} MB exceeds limit of {max_size_mb} MB"
        )

    logger.info(f"File size: {file_size_kb:.1f} KB, Format: .{file_extension}")

    warnings = []
    try:
        if file_extension == "csv":
            df = _load_csv(file_path, loader_config)
        elif file_extension == "xlsx":
            df = _load_excel(file_path)
        elif file_extension == "json":
            df = _load_json(file_path)
        elif file_extension == "parquet":
            df = _load_parquet(file_path)
        else:
            raise ValueError(f"No loader implemented for: {file_extension}")

    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        raise RuntimeError(f"Loading failed: {str(e)}") from e

   # --- Sanity Checks ---
    if df.empty:
        raise ValueError("The loaded dataset is completely empty.")

    if len(df) == 0:
        raise ValueError("The dataset has no rows.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    # Warn about very small datasets
    if len(df) < 10:
        warning_msg = f"Dataset has only {len(df)} rows. Results may be unreliable."
        logger.warning(warning_msg)
        warnings.append(warning_msg)

    metadata = LoadMetadata(
        file_name=os.path.basename(file_path),
        file_format=file_extension,
        file_size_kb=round(file_size_kb, 2),
        shape=df.shape,
        column_names=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        memory_usage_kb=round(df.memory_usage(deep=True).sum() / 1024, 2),
        load_timestamp=datetime.now().isoformat(),
        dataset_fingerprint=_compute_fingerprint(df),
        warnings=warnings,
    )

    logger.info(
        f"Successfully loaded [bold green]{metadata.file_name}[/bold green] | "
        f"Shape: {metadata.shape[0]} rows × {metadata.shape[1]} cols | "
        f"Fingerprint: {metadata.dataset_fingerprint}"
    )

    if warnings:
        for w in warnings:
            logger.warning(w)

    return df, metadata