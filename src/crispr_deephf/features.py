"""Sequence encodings and hand-crafted biological features.

Two representations are produced from the 21-mer (20-nt spacer + 1st nt of PAM):

* :func:`encode_sequences` -- integer tokens and a one-hot tensor, for the
  Tier-2 / Tier-3 deep-learning models (CNN / BiLSTM).
* :func:`extract_crispr_features` -- the 439 tabular features used by the
  Tier-1 gradient-boosted-tree baselines.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import config

__all__ = ["encode_sequences", "extract_crispr_features", "N_FEATURES"]

#: Number of tabular features produced by :func:`extract_crispr_features`.
#  84 positional one-hot + 320 positional dinucleotide + 8 global counts/freqs
#  + 16 global dinucleotide counts + 7 GC + 4 motif = 439.
N_FEATURES = 439

_NTS = ["A", "C", "G", "T"]
_DNTS = [a + b for a in _NTS for b in _NTS]


def encode_sequences(
    seq_series: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode 21-mers as integer tokens and a one-hot tensor.

    Returns
    -------
    int_tokens : ndarray, shape (N, 21), int64
        Embedding-layer input.
    one_hot : ndarray, shape (N, 21, 4), float32
        The paper's ``x in R^(21x4)`` tensor for 1D-CNN layers.
    """
    seqs = list(seq_series)
    int_tokens = np.array(
        [[config.NT_TO_INT[nt] for nt in seq] for seq in seqs], dtype=np.int64
    )
    one_hot = np.eye(4, dtype=np.float32)[int_tokens]
    return int_tokens, one_hot


def extract_crispr_features(
    sequences: Sequence[str], *, verbose: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Vectorised extraction of 439 biological features from 21-mer guides.

    Feature groups
    --------------
    1. Positional single-nucleotide one-hot ....... 21 x 4 = 84
    2. Positional adjacent dinucleotides .......... 20 x 16 = 320
    3. Global single-nt counts & frequencies ...... 8
    4. Global dinucleotide counts (20-mer) ........ 16
    5. Total + regional GC content ................ 7
    6. Biological motifs (poly-T / poly-G / purine) 4

    Parameters
    ----------
    sequences:
        21-mer strings (20-nt spacer + 1st nt of the PAM).

    Returns
    -------
    X : ndarray, shape (N, 439), float32
    feature_names : list[str]
    """
    t0 = time.time()
    seqs = list(sequences)
    n = len(seqs)
    arr = np.array([list(s) for s in seqs])  # (N, 21) character matrix

    feature_names: list[str] = []
    feature_arrays: list[np.ndarray] = []

    # 1. Positional single-nucleotide one-hot (21 * 4 = 84).
    for pos in range(21):
        for nt in _NTS:
            feature_names.append(f"pos_{pos + 1}_{nt}")
            feature_arrays.append((arr[:, pos] == nt).astype(np.float32))

    # 2. Positional adjacent dinucleotides (20 * 16 = 320).
    for pos in range(20):
        di_col = np.char.add(arr[:, pos], arr[:, pos + 1])
        for dnt in _DNTS:
            feature_names.append(f"di_{pos + 1}_{pos + 2}_{dnt}")
            feature_arrays.append((di_col == dnt).astype(np.float32))

    # 3. Global single-nt counts & frequencies over the 20-mer spacer (8).
    spacer = arr[:, :20]
    for nt in _NTS:
        cnt = np.sum(spacer == nt, axis=1).astype(np.float32)
        feature_names.append(f"count_{nt}")
        feature_arrays.append(cnt)
        feature_names.append(f"freq_{nt}")
        feature_arrays.append(cnt / 20.0)

    # 4. Global dinucleotide counts in the 20-mer spacer (16).
    di_all = [np.char.add(arr[:, pos], arr[:, pos + 1]) for pos in range(19)]
    for dnt in _DNTS:
        di_cnt = np.zeros(n, dtype=np.float32)
        for di_c in di_all:
            di_cnt += di_c == dnt
        feature_names.append(f"count_{dnt}")
        feature_arrays.append(di_cnt)

    # 5. Total + regional GC content (7).
    gc_spacer = np.sum((spacer == "G") | (spacer == "C"), axis=1).astype(np.float32)
    feature_names += ["gc_count_spacer", "gc_content_spacer"]
    feature_arrays += [gc_spacer, gc_spacer / 20.0]

    seed = arr[:, 10:20]  # PAM-proximal seed region (positions 11-20)
    gc_seed = np.sum((seed == "G") | (seed == "C"), axis=1).astype(np.float32)
    feature_names += ["gc_count_seed", "gc_content_seed"]
    feature_arrays += [gc_seed, gc_seed / 10.0]

    nonseed = arr[:, 0:10]  # PAM-distal region (positions 1-10)
    gc_nonseed = np.sum((nonseed == "G") | (nonseed == "C"), axis=1).astype(np.float32)
    feature_names += ["gc_count_nonseed", "gc_content_nonseed"]
    feature_arrays += [gc_nonseed, gc_nonseed / 10.0]

    feature_names.append("gc_diff_seed_nonseed")
    feature_arrays.append((gc_seed / 10.0) - (gc_nonseed / 10.0))

    # 6. Biological motifs (4): Pol III terminator, G-quadruplex risk, purines.
    poly_t4 = np.array(["TTTT" in s[:20] for s in seqs], dtype=np.float32)
    poly_t3 = np.array(["TTT" in s[:20] for s in seqs], dtype=np.float32)
    poly_g4 = np.array(["GGGG" in s[:20] for s in seqs], dtype=np.float32)
    purine_cnt = np.sum((spacer == "A") | (spacer == "G"), axis=1).astype(np.float32)
    feature_names += ["poly_T_ge4", "poly_T_ge3", "poly_G_ge4", "purine_ratio"]
    feature_arrays += [poly_t4, poly_t3, poly_g4, purine_cnt / 20.0]

    X = np.column_stack(feature_arrays).astype(np.float32)
    if verbose:
        print(
            f"Extracted {X.shape[1]} features for {X.shape[0]:,} sequences "
            f"in {time.time() - t0:.2f}s ({X.nbytes / 1024**2:.1f} MB)."
        )
    if X.shape[1] != N_FEATURES:
        raise AssertionError(f"Expected {N_FEATURES} features, built {X.shape[1]}.")
    return X, feature_names
