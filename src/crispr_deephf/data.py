"""Load, clean, validate, and split the DeepHF guide-RNA dataset.

The functions here are a faithful, refactored version of the team's original
Colab cleaning code. They encode the biological contract of the data (an
``N20 NGG`` guide + PAM, efficiencies in ``[0, 1]``) as explicit, testable steps.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

__all__ = [
    "find_raw_excel",
    "load_raw",
    "clean_and_validate",
    "summary_table",
    "variant_valid_mask",
    "make_variant_split_indices",
    "add_split_labels",
    "load_clean",
    "get_split_arrays",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_raw_excel(search_dirs: list[Path | str]) -> Path:
    """Return the first existing path to the raw ``.xlsx`` across *search_dirs*.

    Raises
    ------
    FileNotFoundError
        If the file is not found in any of the candidate locations.
    """
    candidates: list[Path] = []
    for d in search_dirs:
        d = Path(d)
        candidates.append(d)                       # treat as a direct file path
        candidates.append(d / config.RAW_XLSX_NAME)  # ...or a directory holding it
    for cand in candidates:
        if cand.is_file():
            return cand
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find '{config.RAW_XLSX_NAME}'. Searched:\n  {searched}"
    )


def load_raw(filepath: Path | str) -> pd.DataFrame:
    """Read the raw DeepHF Excel sheet and standardise its schema.

    Strips the title banner (``header=1``), drops the empty leading index
    column, and renames columns to :data:`config.COLUMNS`.
    """
    filepath = Path(filepath)
    xls = pd.ExcelFile(filepath)
    df_raw = pd.read_excel(
        filepath, sheet_name=xls.sheet_names[0], header=config.RAW_HEADER_ROW
    )
    # Drop the empty first column ("Supplementary Data 2 ...") and rename.
    df = df_raw.iloc[:, 1:].copy()
    if df.shape[1] != len(config.COLUMNS):
        raise ValueError(
            f"Expected {len(config.COLUMNS)} data columns, got {df.shape[1]}. "
            "The raw file layout may have changed."
        )
    df.columns = config.COLUMNS
    return df


# --------------------------------------------------------------------------- #
# Cleaning + validation
# --------------------------------------------------------------------------- #
def clean_and_validate(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Clean strings, repair the known PAM typo, and assert biological rules.

    Steps
    -----
    1. Uppercase / strip the string columns.
    2. Repair any malformed (2-letter) PAM using the 21st nt of the ``21mer``.
    3. Coerce the efficiency columns to numeric (bad values -> ``NaN``).
    4. When *strict*, assert the ``N20 NGG`` sequence contract and that every
       non-null efficiency lies in ``[0, 1]``.

    Returns a cleaned copy; the input is not modified.
    """
    df = df.copy()

    # 1. Normalise string columns.
    for col in config.STR_COLS:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # 2. Repair the malformed PAM (e.g. gRNA-26033: 'GG' -> 'NGG').
    bad_pam_mask = df["PAM"].str.len() != 3
    n_bad = int(bad_pam_mask.sum())
    if n_bad:
        df.loc[bad_pam_mask, "PAM"] = df.loc[bad_pam_mask, "21mer"].str[-1] + "GG"

    # 3. Coerce targets to float (invalid parses become NaN).
    for col in config.TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Biological + range integrity checks.
    if strict:
        assert df["gRNA_Seq"].str.match(r"^[ACGT]{20}$").all(), "Invalid 20-mer spacer!"
        assert df["PAM"].str.match(r"^[ACGT]GG$").all(), "Invalid NGG PAM!"
        assert df["21mer"].str.match(r"^[ACGT]{21}$").all(), "Invalid 21-mer!"
        assert ((df["gRNA_Seq"] + df["PAM"].str[0]) == df["21mer"]).all(), (
            "21mer does not equal gRNA_Seq + PAM[0]!"
        )
        for col in config.TARGET_COLS:
            vals = df[col].dropna()
            assert vals.between(0.0, 1.0).all(), f"{col} has values outside [0, 1]!"

    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-target summary stats plus a missing-value count (for EDA)."""
    summary = df[config.TARGET_COLS].describe().T
    summary["median"] = df[config.TARGET_COLS].median()
    summary["missing_NaN"] = df[config.TARGET_COLS].isnull().sum()
    cols = ["count", "missing_NaN", "mean", "median", "std", "min", "25%", "75%", "max"]
    return summary[cols]


# --------------------------------------------------------------------------- #
# Splitting (76.5% / 8.5% / 15%, matching the paper)
# --------------------------------------------------------------------------- #
def variant_valid_mask(df: pd.DataFrame, target_col: str) -> np.ndarray:
    """Boolean mask of rows with a non-null measurement for *target_col*."""
    return df[target_col].notnull().to_numpy()


def make_variant_split_indices(
    n_valid: int, seed: int = config.RANDOM_SEED
) -> dict[str, np.ndarray]:
    """Split ``range(n_valid)`` into train/val/test index arrays.

    Returns positional indices *into the valid subset* for one Cas9 variant,
    using the paper's 76.5 / 8.5 / 15 ratios and a fixed seed. Because the
    ratios and seed are shared via :mod:`config`, both notebooks reproduce the
    exact same partition.
    """
    idx = np.arange(n_valid)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=config.TEST_SIZE, random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=config.VAL_SIZE_OF_REMAINDER, random_state=seed
    )
    return {"train": idx_train, "val": idx_val, "test": idx_test}


def add_split_labels(
    df: pd.DataFrame, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Add a ``<KEY>_split`` column per target with 'train'/'val'/'test'/NaN.

    This freezes the split into the cleaned dataset itself, so the modelling
    notebook simply *reads* the labels rather than re-deriving them -- the two
    notebooks can never disagree about which guide is in which split.
    """
    df = df.copy()
    for target_col, key in config.TARGET_KEYS.items():
        split_col = f"{key}_split"
        df[split_col] = pd.Series(pd.NA, index=df.index, dtype="object")

        mask = variant_valid_mask(df, target_col)
        valid_positions = np.where(mask)[0]
        parts = make_variant_split_indices(len(valid_positions), seed=seed)

        for name, local_idx in parts.items():
            row_positions = valid_positions[local_idx]
            df.iloc[row_positions, df.columns.get_loc(split_col)] = name
    return df


def load_clean(path: Path | str) -> pd.DataFrame:
    """Load a cleaned CSV, keeping the ``21mer`` column as a string."""
    return pd.read_csv(path, dtype={"21mer": str, "gRNA_Seq": str, "PAM": str})


def get_split_arrays(
    df: pd.DataFrame, X: np.ndarray, target_col: str
) -> dict[str, dict[str, np.ndarray]]:
    """Slice a feature matrix into train/val/test using the frozen split labels.

    Reads the ``<KEY>_split`` column written by :func:`add_split_labels` (so the
    partition is exactly the one the data cleaning notebook produced), and
    returns ``{split: {'X', 'y', 'id'}}`` for the given target.

    Parameters
    ----------
    df:
        Cleaned dataframe that already contains the split-label columns.
    X:
        Feature matrix aligned row-for-row with *df*.
    target_col:
        One of :data:`config.TARGET_COLS`.
    """
    key = config.TARGET_KEYS[target_col]
    split_col = f"{key}_split"
    if split_col not in df.columns:
        raise KeyError(
            f"'{split_col}' not found. Re-run the data cleaning notebook so the "
            "cleaned CSV includes frozen split labels."
        )
    labels = df[split_col].to_numpy()
    y = df[target_col].to_numpy()
    ids = df["gRNA_ID"].to_numpy()
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in ("train", "val", "test"):
        pos = np.where(labels == name)[0]
        out[name] = {"X": X[pos], "y": y[pos], "id": ids[pos]}
    return out
