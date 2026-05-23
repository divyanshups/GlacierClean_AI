from __future__ import annotations
import io
from pathlib import Path
from typing import Union
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}

# Encoding candidates tried in order when auto-detection is unavailable
_ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


def detect_encoding(file_path: Union[str, Path]) -> str:
    
    file_path = Path(file_path)
    raw = file_path.read_bytes()[:100_000]

    # ── Try charset-normalizer (preferred, pure Python) ───────────────────────
    try:
        from charset_normalizer import from_bytes  # type: ignore[import]
        result = from_bytes(raw).best()
        if result is not None:
            return str(result.encoding)
    except Exception:
        pass

    # ── Fallback: probe common encodings via stdlib ───────────────────────────
    for enc in _ENCODING_FALLBACKS:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue

    return "utf-8"  # last resort


def load_dataset(
    source: Union[str, Path, io.BytesIO, io.StringIO],
    file_name: str = "",
    **kwargs,
) -> pd.DataFrame:
   
    ext = _infer_extension(source, file_name)

    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        if isinstance(source, (str, Path)):
            encoding = detect_encoding(source)
            return pd.read_csv(
                source, sep=sep, encoding=encoding, low_memory=False, **kwargs
            )
        # File-like object — let pandas detect encoding automatically
        return pd.read_csv(source, sep=sep, low_memory=False, **kwargs)

    elif ext in {".xlsx", ".xls"}:
        return pd.read_excel(source, **kwargs)

    elif ext == ".json":
        return pd.read_json(source, **kwargs)

    elif ext == ".parquet":
        if isinstance(source, io.StringIO):
            raise ValueError(
                "Parquet sources must be a file path or a bytes buffer, not a StringIO text buffer."
            )
        return pd.read_parquet(source, **kwargs)

    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def save_dataset(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save a DataFrame to disk in the format inferred from the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in {".csv", ""}:
        df.to_csv(path, index=False)
    elif ext in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        df.to_csv(path, index=False)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _infer_extension(
    source: Union[str, Path, io.BytesIO, io.StringIO],
    file_name: str,
) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower()
    if file_name:
        return Path(file_name).suffix.lower()
    return ".csv"  # sensible default for file-like objects
