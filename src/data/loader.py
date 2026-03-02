"""
src/data/loader.py
==================
Data loading, splitting, and preprocessing pipeline for:
  - Binary detection task  (label col: "Output (S)", values 0/1)
  - Multi-class fault task (label cols: G, C, B, A → merged into single string label)

Supports CSV, JSON, and Excel input files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

DETECTION_LABEL_COL = "Output (S)"
DETECTION_FEATURE_COLS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

MULTICLASS_LABEL_COLS = ["G", "C", "B", "A"]
MULTICLASS_FEATURE_COLS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_file(path: str | Path) -> pd.DataFrame:
    """Load a CSV, JSON, or Excel file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()

    loaders = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }

    if suffix not in loaders:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: {list(loaders)}"
        )

    df = loaders[suffix](path)
    logger.info("Loaded '%s'  shape=%s", path.name, df.shape)
    return df


def _handle_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Fill missing values.

    Parameters
    ----------
    strategy : {"mean", "median", "zero", "drop"}
    """
    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        logger.info("Dropped %d rows with missing values.", before - len(df))
    elif strategy == "zero":
        df = df.fillna(0)
    elif strategy in ("mean", "median"):
        numeric_cols = df.select_dtypes(include="number").columns
        fill_val = df[numeric_cols].mean() if strategy == "mean" else df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(fill_val)
    else:
        raise ValueError(f"Unknown missing-value strategy '{strategy}'.")
    return df


def _scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on train, transform both splits."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_detection_data(
    path: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    missing_strategy: str = "mean",
    scale: bool = True,
) -> dict:
    """
    Load the binary-detection dataset.

    Expected columns
    ----------------
    ``Output (S)``  – binary label (0 / 1)
    ``Ia Ib Ic Va Vb Vc`` – electrical features

    Returns
    -------
    dict with keys:
        X_train, X_test  – numpy arrays
        y_train, y_test  – numpy arrays (int)
        feature_names    – list[str]
        scaler           – fitted StandardScaler or None
    """
    df = _load_file(path)
    df = _handle_missing(df, strategy=missing_strategy)

    # --- validate columns ---
    required = DETECTION_FEATURE_COLS + [DETECTION_LABEL_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in detection file: {missing_cols}")

    X = df[DETECTION_FEATURE_COLS].values.astype(float)
    y = df[DETECTION_LABEL_COL].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Detection split → train=%d  test=%d  pos_rate=%.3f",
        len(y_train),
        len(y_test),
        y_train.mean(),
    )

    scaler = None
    if scale:
        X_train, X_test, scaler = _scale_features(X_train, X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": DETECTION_FEATURE_COLS,
        "scaler": scaler,
    }


def load_multiclass_data(
    path: str | Path,
    *,
    label_format: Literal["combined_string", "encoded_int"] = "encoded_int",
    test_size: float = 0.2,
    random_state: int = 42,
    missing_strategy: str = "mean",
    scale: bool = True,
) -> dict:
    """
    Load the multi-class fault-type dataset.

    Expected columns
    ----------------
    ``G C B A``         – binary fault-type flags (combined into a single label)
    ``Ia Ib Ic Va Vb Vc`` – electrical features

    Label construction
    ------------------
    The four binary columns are concatenated into a string, e.g. ``"0001"`` → class A fault.
    Unique strings are then integer-encoded with ``LabelEncoder`` when
    ``label_format="encoded_int"`` (required by XGBoost).

    Returns
    -------
    dict with keys:
        X_train, X_test   – numpy arrays
        y_train, y_test   – numpy arrays (int if encoded_int, str otherwise)
        feature_names     – list[str]
        label_encoder     – fitted LabelEncoder or None
        class_names       – list of string labels in encoder order
        scaler            – fitted StandardScaler or None
    """
    df = _load_file(path)
    df = _handle_missing(df, strategy=missing_strategy)

    # --- validate columns ---
    required = MULTICLASS_FEATURE_COLS + MULTICLASS_LABEL_COLS
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in multiclass file: {missing_cols}")

    # Build compound label: e.g. G=0,C=0,B=0,A=1 → "0001"
    label_strings = (
        df[MULTICLASS_LABEL_COLS]
        .astype(int)
        .astype(str)
        .apply("".join, axis=1)
    )

    X = df[MULTICLASS_FEATURE_COLS].values.astype(float)

    le = None
    class_names: list[str] = []

    if label_format == "encoded_int":
        le = LabelEncoder()
        y = le.fit_transform(label_strings)
        class_names = list(le.classes_)
        logger.info("Label classes (%d): %s", len(class_names), class_names)
    else:
        y = label_strings.values
        class_names = sorted(label_strings.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Multiclass split → train=%d  test=%d  n_classes=%d",
        len(y_train),
        len(y_test),
        len(class_names),
    )

    scaler = None
    if scale:
        X_train, X_test, scaler = _scale_features(X_train, X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": MULTICLASS_FEATURE_COLS,
        "label_encoder": le,
        "class_names": class_names,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Generic entry-point (auto-detects task from columns)
# ---------------------------------------------------------------------------


def load_data(
    path: str | Path,
    task: Optional[Literal["detection", "multiclass"]] = None,
    **kwargs,
) -> dict:
    """
    Convenience wrapper — auto-detects task type if not specified.

    Parameters
    ----------
    path   : path to CSV / JSON / Excel file
    task   : "detection" | "multiclass" | None (auto-detect)
    kwargs : forwarded to the task-specific loader
    """
    df_cols = set(_load_file(path).columns)

    if task is None:
        if DETECTION_LABEL_COL in df_cols:
            task = "detection"
        elif all(c in df_cols for c in MULTICLASS_LABEL_COLS):
            task = "multiclass"
        else:
            raise ValueError(
                "Cannot auto-detect task. "
                f"Provide task='detection' or task='multiclass'. "
                f"Found columns: {sorted(df_cols)}"
            )
        logger.info("Auto-detected task: %s", task)

    if task == "detection":
        return load_detection_data(path, **kwargs)
    elif task == "multiclass":
        return load_multiclass_data(path, **kwargs)
    else:
        raise ValueError(f"Unknown task '{task}'.")


# ---------------------------------------------------------------------------
# Quick smoke-test (python -m src.data.loader <file1> <file2>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for fp in sys.argv[1:]:
        result = load_data(fp)
        print(
            f"\n[{Path(fp).name}]  "
            f"X_train={result['X_train'].shape}  "
            f"X_test={result['X_test'].shape}  "
            f"y_train={result['y_train'].shape}"
        )
        if "class_names" in result:
            print("  classes:", result["class_names"])