"""Load, clean, validate, and split the DeepHF guide-RNA dataset.

The functions here are a faithful, refactored version of the team's original
Colab cleaning code. They encode the biological contract of the data (an
``N20 NGG`` guide + PAM, efficiencies in ``[0, 1]``) as explicit, testable steps.
"""

from __future__ import annotations

import http.client
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

__all__ = [
    "add_shared_split",
    "add_split_labels",
    "assert_unique_sequences",
    "clean_and_validate",
    "ensure_raw_excel",
    "filter_for_target",
    "find_raw_excel",
    "get_split_arrays",
    "load_clean",
    "load_raw",
    "make_variant_split_indices",
    "split_counts",
    "summary_table",
    "variant_valid_mask",
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


def ensure_raw_excel(paths: config.Paths, *, allow_download: bool = True) -> Path:
    """Return the raw ``.xlsx``, downloading it into ``paths.data_raw`` if needed.

    Searches ``paths.data_raw`` and ``paths.root`` first. If absent and
    *allow_download*, fetches :data:`config.RAW_XLSX_URL` to a ``.part`` file and
    atomically renames it on success, so an interrupted download never leaves a
    truncated ``.xlsx`` that a later run would treat as valid.

    Parameters
    ----------
    paths:
        Project paths from :func:`config.get_paths`.
    allow_download:
        If ``False``, never touch the network.

    Raises
    ------
    FileNotFoundError
        If the file is absent and cannot be downloaded; the message explains
        how to upload it manually.
    """
    try:
        return find_raw_excel([paths.data_raw, paths.root])
    except FileNotFoundError:
        pass

    dest = paths.data_raw / config.RAW_XLSX_NAME
    reason = "downloading is disabled"
    cause: Exception | None = None
    if allow_download:
        try:
            _download_xlsx(config.RAW_XLSX_URL, dest)
            return dest
        except (OSError, http.client.HTTPException, ValueError) as exc:
            reason = f"download failed ({exc!r})"
            cause = exc

    raise FileNotFoundError(
        f"Could not obtain '{config.RAW_XLSX_NAME}': {reason}. Upload it manually "
        f"to '{paths.data_raw}' (or the project root), or download it from "
        f"{config.RAW_XLSX_URL}."
    ) from cause


def _download_xlsx(url: str, dest: Path) -> None:
    """Download *url* to *dest* atomically, rejecting non-``.xlsx`` payloads.

    The bytes go to ``<dest>.part`` and are renamed only after the zip magic
    bytes (``PK``) check passes, so neither a truncated transfer nor an HTML
    error page served with HTTP 200 is ever installed as the dataset.

    Raises
    ------
    ValueError
        If the payload is not a zip container (i.e. not an ``.xlsx``).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        with open(part, "rb") as fh:
            if fh.read(2) != b"PK":
                raise ValueError("response is not an .xlsx (missing 'PK' zip header)")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


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

    Precondition: ``21mer`` values are unique (true of the real dataset, which
    has no duplicates). The split is index-based and this function does not
    check it, so duplicates would leak across splits; call
    :func:`assert_unique_sequences` first on untrusted input.
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


def assert_unique_sequences(df: pd.DataFrame, seq_col: str = "21mer") -> None:
    """Raise if *seq_col* holds duplicate sequences.

    The splits are index-based, so a duplicated sequence could land in both
    train and test and silently inflate every reported metric. Nothing upstream
    deduplicates, so splitting code must call this first.

    Raises
    ------
    ValueError
        Naming the number of duplicated rows and a few example sequences.
    """
    dup = df[seq_col].duplicated(keep=False)
    if dup.any():
        examples = ", ".join(df.loc[dup, seq_col].drop_duplicates().head(3))
        raise ValueError(
            f"{int(df[seq_col].duplicated().sum())} duplicate '{seq_col}' "
            f"sequence(s) would leak across splits (e.g. {examples}). "
            "Deduplicate before splitting."
        )


def add_shared_split(
    df: pd.DataFrame, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Add a ``shared_split`` column labelling *every* row train/val/test.

    Unlike the per-enzyme ``<KEY>_split`` columns (which only label rows measured
    for that enzyme), this is one partition over all rows, so a guide has the
    same split for every enzyme -- required by the multi-task model. It reuses
    :func:`make_variant_split_indices`, so the ratios live in one place.

    Raises
    ------
    ValueError
        If ``21mer`` is not unique (see :func:`assert_unique_sequences`); the
        split is index-based and duplicates would leak across splits.
    """
    assert_unique_sequences(df)
    df = df.copy()
    labels = np.empty(len(df), dtype=object)
    for name, idx in make_variant_split_indices(len(df), seed=seed).items():
        labels[idx] = name
    df[config.SHARED_SPLIT_COL] = labels
    return df


def split_counts(df: pd.DataFrame, split_col: str) -> dict[str, int]:
    """Return ``{label: n_rows}`` for the non-null labels of *split_col*."""
    return {str(k): int(v) for k, v in df[split_col].value_counts().items()}


def filter_for_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Keep rows measured for *target_col* that carry a split label for it.

    This is the single place rows are dropped per enzyme, so a missing
    measurement can never reach a model as if it were real. The index is reset.
    """
    split_col = f"{config.TARGET_KEYS[target_col]}_split"
    keep = df[target_col].notna() & df[split_col].notna()
    return df.loc[keep].reset_index(drop=True)


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
        # .eq().fillna(False): an in-memory label column holds pd.NA, which breaks ``==``
        pos = np.flatnonzero(pd.Series(labels).eq(name).fillna(False).to_numpy(dtype=bool))
        out[name] = {"X": X[pos], "y": y[pos], "id": ids[pos]}
    return out
