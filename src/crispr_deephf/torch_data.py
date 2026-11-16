"""Torch data layer: sequence encoding, integrity checks and masked split loaders.

This is the single place a cleaned dataframe becomes tensors. Ported from the
team's ``models/Train_CNN.py`` (``check_integrity``, ``one_hot_encode_sequences``,
``build_loaders``) and generalised to several targets:

* splits are always *read* from a frozen label column, never re-derived;
* the bio-feature standardizer is fit on **train rows only**;
* unmeasured targets are zero-filled in ``y`` (never NaN, which would poison
  autograd) and flagged by a 0/1 ``mask`` so the loss can ignore them.

Importing this module requires torch; the package ``__init__`` does not import
it, so callers import it explicitly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from crispr_deephf.bio_features import SEQ_LEN, Standardizer, bio_matrix, flatten_bio

__all__ = [
    "SplitTensors",
    "build_splits",
    "check_integrity",
    "make_loader",
    "one_hot_encode",
]

SPLIT_NAMES = ("train", "val", "test")
_NUCLEOTIDE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def one_hot_encode(seqs: Sequence[str]) -> np.ndarray:
    """One-hot encode 21-mers as ``(N, 4, 21)`` float32, channel-first (A, C, G, T).

    Raises
    ------
    AssertionError
        If a sequence is not ``SEQ_LEN`` long or contains a non-ACGT character.
    """
    out = np.zeros((len(seqs), 4, SEQ_LEN), dtype=np.float32)
    for i, seq in enumerate(seqs):
        assert len(seq) == SEQ_LEN, f"Sequence length must be {SEQ_LEN}, but got {len(seq)}"
        for j, nt in enumerate(seq):
            assert nt in _NUCLEOTIDE_INDEX, f"non-ACGT character {nt!r} in {seq!r}"
            out[i, _NUCLEOTIDE_INDEX[nt], j] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


def check_integrity(
    df: pd.DataFrame,
    seq_col: str,
    split_col: str,
    target_cols: list[str],
    *,
    allow_missing: bool = False,
) -> str:
    """Fail loudly on anything that could leak or corrupt; return a 12-char data digest.

    Checks valid split labels (all of train/val/test present), ``[ACGT]{21}``
    sequences, finite targets in ``[0, 1]`` and that no sequence appears in more
    than one split.

    Parameters
    ----------
    allow_missing
        Permit NaN targets (unmeasured entries of a multi-task table, which the
        mask handles). Infinite or out-of-range values are still rejected.
    """
    seqs = df[seq_col].astype(str)

    bad = set(df[split_col].unique()) - set(SPLIT_NAMES)
    assert not bad, f"unexpected split labels: {bad}"
    assert df[split_col].nunique() == len(SPLIT_NAMES), "need all of train/val/test"

    assert (seqs.str.len() == SEQ_LEN).all(), f"sequences must be {SEQ_LEN} long"
    assert seqs.str.fullmatch("[ACGT]+").all(), "sequences must be only A/C/G/T"

    for target in target_cols:
        y = df[target].to_numpy(dtype=np.float64)
        if allow_missing:
            y = y[~np.isnan(y)]
        assert np.isfinite(y).all(), f"non-finite values in {target}"
        if y.size:
            assert y.min() >= 0.0 and y.max() <= 1.0, (
                f"{target} outside [0, 1]: {y.min()}..{y.max()}"
            )

    n_splits = df.groupby(seqs)[split_col].nunique()
    leaked = n_splits[n_splits > 1]
    assert leaked.empty, (
        f"{len(leaked)} sequences appear in more than one split, e.g. {list(leaked.index[:3])}"
    )

    digest = hashlib.sha256(
        "\n".join(f"{s},{p}" for s, p in zip(seqs, df[split_col], strict=True)).encode()
    ).hexdigest()[:12]
    print(f"integrity ok: {len(df)} rows, data hash {digest}")
    return digest


# --------------------------------------------------------------------------- #
# Split tensors and loaders
# --------------------------------------------------------------------------- #


@dataclass
class SplitTensors:
    """Numpy arrays for one split.

    Attributes
    ----------
    x : ``(N, 4, 21)`` float32 one-hot sequences.
    bio : ``(N, 100)`` standardized bio features, or ``(N, 0)`` when disabled.
    y : ``(N, T)`` float32 targets, zero where unmeasured (never NaN).
    mask : ``(N, T)`` float32, 1.0 exactly where a real measurement exists.
    ids : ``(N,)`` guide identifiers (``gRNA_ID`` when present, else the row label).
    """

    x: np.ndarray
    bio: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    ids: np.ndarray


def build_splits(
    df: pd.DataFrame,
    *,
    target_cols: Sequence[str],
    split_col: str,
    seq_col: str = "21mer",
    bio: bool = True,
    cache_path: Path | str | None = None,
    allow_missing: bool = False,
) -> tuple[dict[str, SplitTensors], Standardizer | None]:
    """Turn a cleaned dataframe into ``{"train"/"val"/"test": SplitTensors}``.

    The split is read from *split_col* (a frozen ``<KEY>_split`` or
    ``shared_split`` column); it is never re-derived here. The standardizer is
    fit on the train rows only and then applied to all three splits.

    With ``bio=False`` the ``bio`` arrays are zero-width ``(N, 0)`` and the
    returned standardizer is ``None``, so sequence-only models need no branch.
    Unmeasured (NaN) targets are zero-filled in ``y`` and marked 0 in ``mask``.

    Parameters
    ----------
    df : cleaned dataframe carrying *split_col*, *seq_col* and the targets.
    target_cols : target columns, in the column order of ``y`` and ``mask``.
    split_col : frozen split-label column to read.
    seq_col : 21-mer column.
    bio : compute and standardize bio features (``False`` gives ``(N, 0)`` and ``None``).
    cache_path : optional bio-feature cache file.
    allow_missing : permit NaN targets, for the multi-task table where the mask
        handles them. Default ``False`` fails loudly; single-target runs should
        instead pass a frame already filtered by ``data.filter_for_target``.
    """
    target_cols = list(target_cols)
    check_integrity(df, seq_col, split_col, target_cols, allow_missing=allow_missing)

    seqs = df[seq_col].astype(str).tolist()
    x = one_hot_encode(seqs)

    raw = df[target_cols].to_numpy(dtype=np.float64)
    measured = ~np.isnan(raw)
    y = np.where(measured, raw, 0.0).astype(np.float32)
    mask = measured.astype(np.float32)

    id_series = df["gRNA_ID"] if "gRNA_ID" in df.columns else pd.Series(df.index, index=df.index)
    ids = id_series.to_numpy()

    labels = df[split_col].to_numpy()
    idx = {name: np.where(labels == name)[0] for name in SPLIT_NAMES}

    scaler: Standardizer | None = None
    if bio:
        glob, chans = bio_matrix(seqs, cache_path=cache_path)
        raw_bio = flatten_bio(glob, chans)
        scaler = Standardizer().fit(raw_bio[idx["train"]])  # train rows only
        bio_all = scaler.transform(raw_bio)
    else:
        bio_all = np.zeros((len(seqs), 0), dtype=np.float32)

    splits = {
        name: SplitTensors(x=x[i], bio=bio_all[i], y=y[i], mask=mask[i], ids=ids[i])
        for name, i in idx.items()
    }
    print(", ".join(f"{n}: {len(i)}" for n, i in idx.items()))
    return splits, scaler


def make_loader(
    split: SplitTensors, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    """Return a loader yielding ``(x, bio, y, mask)``; *seed* only controls shuffling."""
    ds = TensorDataset(
        torch.from_numpy(split.x),
        torch.from_numpy(split.bio),
        torch.from_numpy(split.y),
        torch.from_numpy(split.mask),
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=g)
