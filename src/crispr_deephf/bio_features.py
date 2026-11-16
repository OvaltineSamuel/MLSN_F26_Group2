"""ViennaRNA / Biopython biophysical features for a 21-nt guide (spacer + first PAM nt).

Ported from the team's ``models/Bio_Features.py`` (the scientific content is
unchanged). Each guide yields:

* a vector of global features (composition, melting temperatures, folding
  energies, scaffold integrity, mean accessibility), and
* three per-position channels (``p_unpaired``, ``p_intra``, ``p_scaffold``) over
  the 21 positions (the PAM-nt position is zero-padded).

:func:`flatten_bio` is the single layout every model consumes. This module does
not import torch. ViennaRNA and Biopython are optional: check
:data:`VIENNA_AVAILABLE` / :data:`BIOPYTHON_AVAILABLE`; computing features
without them raises ``ImportError``.

Differences from the original, both defect fixes: the feature cache is keyed by
:func:`sequence_digest` rather than row count, and :func:`melting_temp` no longer
raises on degenerate input.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "BIOPYTHON_AVAILABLE",
    "GLOBAL_FEATURE_NAMES",
    "N_BIO_FEATURES",
    "POSITIONAL_CHANNEL_NAMES",
    "SCAFFOLD",
    "SEQ_LEN",
    "SPACER_LEN",
    "VIENNA_AVAILABLE",
    "Standardizer",
    "bio_matrix",
    "compute_bio_features",
    "flatten_bio",
    "melting_temp",
    "sequence_digest",
]

# --------------------------------------------------------------------------- #
# Optional dependency detection
# --------------------------------------------------------------------------- #
try:
    import RNA

    VIENNA_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    RNA = None
    VIENNA_AVAILABLE = False

try:
    from Bio.SeqUtils import MeltingTemp as mt

    BIOPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    mt = None
    BIOPYTHON_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Constants and feature layout
# --------------------------------------------------------------------------- #
SEQ_LEN = 21  # 20-nt spacer plus first nt in PAM
SPACER_LEN = 20

SCAFFOLD = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTT"

# Dot-bracket structure of the bare scaffold, the reference for "scaffold integrity".
SCAFFOLD_REF = RNA.fold(SCAFFOLD.replace("T", "U"))[0] if VIENNA_AVAILABLE else ""

BASES = "ACGT"
DINUCS = [a + b for a in BASES for b in BASES]

GLOBAL_FEATURE_NAMES = (
    ["gc_content", "gc_count"]
    + [f"count_{b}" for b in BASES]
    + [f"count_{d}" for d in DINUCS]
    + [
        "tm_full",
        "tm_1_4",
        "tm_16_20",
        "tm_5_13",
        "mfe_spacer",
        "mfe_full",
        "mfe_difference",
        "ensemble_energy",
        "ensemble_diversity",
        "scaffold_integrity",
        "scaffold_bp_lost",
        "accessibility_mean",
        "accessibility_pam_prox",
        "intra_pairing_mean",
        "scaffold_pairing_mean",
    ]
)
# p_intra is hairpin within spacer, p_scaffold is pairing with the scaffold
POSITIONAL_CHANNEL_NAMES = ["p_unpaired", "p_intra", "p_scaffold"]

N_BIO_FEATURES = len(GLOBAL_FEATURE_NAMES) + len(POSITIONAL_CHANNEL_NAMES) * SEQ_LEN


def _require_vienna() -> None:
    if not (VIENNA_AVAILABLE and BIOPYTHON_AVAILABLE):
        raise ImportError(
            "Bio features need ViennaRNA and Biopython "
            "(pip install ViennaRNA biopython)."
        )


# --------------------------------------------------------------------------- #
# Per-sequence features
# --------------------------------------------------------------------------- #
def melting_temp(seq: str) -> float:
    """Nearest-neighbour melting temperature of ``seq``.

    Biopython's ``Tm_NN`` raises on empty or fully degenerate slices; rather than
    abort a whole run, such a slice returns ``0.0``. Real spacers (20 nt of ACGT)
    never hit that path.
    """
    try:
        return float(mt.Tm_NN(seq))
    except (IndexError, ValueError, KeyError):
        return 0.0


def fold(rna: str) -> tuple[str, float]:
    """Minimum-free-energy structure (dot-bracket) and energy of ``rna``."""
    structure, mfe = RNA.fold(rna)
    return structure, mfe


def unpaired_probabilities(
    rna: str, spacer_len: int = SPACER_LEN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-position pairing probabilities for the first ``spacer_len`` nt.

    Used to encode position accessibility: what are the odds a spacer nucleotide
    is free to bind the target DNA.

    Returns
    -------
    tuple of np.ndarray
        ``(p_unpaired, p_intra, p_scaffold)``, each of length ``spacer_len``.
    """
    fc = RNA.fold_compound(rna)  # ViennaRNA object holding sequence and energy rules
    _, mfe = fc.mfe()  # most stable fold and its free energy
    fc.exp_params_rescale(mfe)  # avoid floating point overflow in the partition function
    fc.pf()  # partition function: weighs all folds by stability
    bpp = fc.bpp()  # bpp[i][j] is the probability position i pairs with position j

    n = len(rna)
    intra = np.zeros(spacer_len, dtype=np.float64)
    scaffold = np.zeros(spacer_len, dtype=np.float64)
    # Loop over every (i, j) and add the pair probability to P(i paired) and
    # P(j paired), split by whether the partner is in the spacer or the scaffold.
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            for a, b in ((i, j), (j, i)):
                if a <= spacer_len:
                    if b <= spacer_len:
                        intra[a - 1] += p
                    else:
                        scaffold[a - 1] += p
    unpaired = np.clip(1.0 - intra - scaffold, 0.0, 1.0)

    return unpaired, np.clip(intra, 0, 1), np.clip(scaffold, 0, 1)


def _scaffold_integrity(full_structure: str, spacer_len: int = SPACER_LEN) -> tuple[float, float]:
    """Fraction of scaffold positions whose fold matches the bare scaffold, and the count lost."""
    ref = SCAFFOLD_REF
    tail = full_structure[spacer_len:]
    kept = sum(1 for a, b in zip(tail, ref, strict=False) if a == b)
    return kept / len(ref), float(len(ref) - kept)


def compute_bio_features(seq21: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute features for one 21-mer.

    Returns
    -------
    tuple of np.ndarray
        ``(global_features (F,), positional_channels (3, 21))``, both float32.
    """
    _require_vienna()
    seq21 = seq21.strip().upper().replace("U", "T")
    spacer = seq21[:20]

    gc_count = sum(1 for b in spacer if b in "GC")
    mono = [float(spacer.count(b)) for b in BASES]
    di = [float(sum(1 for i in range(len(spacer) - 1) if spacer[i : i + 2] == d)) for d in DINUCS]

    tm_full = melting_temp(spacer)
    tm_pam_distal_5 = melting_temp(spacer[0:5])
    tm_mid_8 = melting_temp(spacer[5:13])
    tm_pam_proximal_5 = melting_temp(spacer[15:20])

    spacer_rna = spacer.replace("T", "U")
    full_rna = (spacer + SCAFFOLD).replace("T", "U")

    _, mfe_spacer = fold(spacer_rna)
    struct_full, mfe_full = fold(full_rna)

    fc = RNA.fold_compound(full_rna)
    _, m = fc.mfe()
    fc.exp_params_rescale(m)
    ens_energy = float(fc.pf()[1])
    ens_div = float(fc.mean_bp_distance())
    # fold diversity: high means many folds with reasonable likelihood

    unpaired, intra, scaf = unpaired_probabilities(full_rna)

    integrity, bp_lost = _scaffold_integrity(struct_full)

    glob = np.array(
        [
            gc_count / SPACER_LEN,
            float(gc_count),
            *mono,
            *di,
            tm_full,
            tm_pam_distal_5,
            tm_mid_8,
            tm_pam_proximal_5,
            float(mfe_spacer),
            float(mfe_full),
            float(mfe_full - mfe_spacer),
            ens_energy,
            ens_div,
            integrity,
            bp_lost,
            float(unpaired.mean()),
            float(unpaired[15:20].mean()),
            float(intra.mean()),
            float(scaf.mean()),
        ],
        dtype=np.float32,
    )

    chans = np.zeros((3, SEQ_LEN), dtype=np.float32)
    chans[0, :SPACER_LEN] = unpaired
    chans[1, :SPACER_LEN] = intra
    chans[2, :SPACER_LEN] = scaf

    return glob, chans


# --------------------------------------------------------------------------- #
# Batch computation and caching
# --------------------------------------------------------------------------- #
def sequence_digest(seqs: Sequence[str]) -> str:
    """Content key for a sequence list: first 16 hex chars of the sha256 of the joined sequences."""
    return hashlib.sha256("\n".join(seqs).encode("utf-8")).hexdigest()[:16]


def bio_matrix(
    seqs: Sequence[str],
    cache_path: Path | str | None = None,
    *,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute features for many sequences, optionally cached on disk.

    The cache stores :func:`sequence_digest` of ``seqs`` and is reused only when
    the digest matches, so a different dataset of the same length is recomputed.

    Returns
    -------
    tuple of np.ndarray
        ``(glob (N, F), chans (N, 3, 21))``.
    """
    seqs = list(seqs)
    digest = sequence_digest(seqs)

    path: Path | None = None
    if cache_path:
        path = Path(cache_path)
        # np.savez_compressed silently appends ".npz"; normalise so save and load agree
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        if path.exists():
            try:
                with np.load(path) as cached:
                    if "digest" in cached.files and str(cached["digest"]) == digest:
                        if verbose:
                            print(f"loaded cached bio features from {path}")
                        return cached["glob"], cached["chans"]
            # A half-written .npz does not raise OSError/ValueError: a truncated zip
            # raises zipfile.BadZipFile (which subclasses Exception directly) and a
            # zero-byte file raises EOFError. Both were reproduced; neither was caught,
            # so a cache damaged by a killed Colab session took the whole run down
            # instead of being recomputed.
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                if verbose:
                    print(f"unreadable bio-feature cache at {path}: recomputing")
                # fall through: recompute and overwrite below

    glob = np.zeros((len(seqs), len(GLOBAL_FEATURE_NAMES)), dtype=np.float32)
    chans = np.zeros((len(seqs), len(POSITIONAL_CHANNEL_NAMES), SEQ_LEN), dtype=np.float32)
    for i, seq in enumerate(seqs):
        glob[i], chans[i] = compute_bio_features(seq)
        if verbose and (i + 1) % 2000 == 0:
            print(f"bio features: {i + 1}/{len(seqs)}")
    if path is not None:
        _save_cache_atomically(path, glob=glob, chans=chans, digest=digest)
        if verbose:
            print(f"saved bio features to {path}")
    return glob, chans


def _save_cache_atomically(path: Path, *, glob: np.ndarray, chans: np.ndarray, digest: str) -> None:
    """Write the ``.npz`` cache to a temp file beside *path*, then :func:`os.replace` it.

    This is the project's one expensive artifact (~37 min of folding per file) and it is
    written to a shared Drive folder, so a session killed mid-write must not be able to
    leave a truncated zip where the next run expects a cache. The temp file is in the
    **same directory**, which keeps ``os.replace`` on one filesystem and therefore atomic;
    the same discipline ``runner.save_result``, ``tuning.save_best_params`` and
    ``data._download_xlsx`` already use. Temp names do not end in ``.npz``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, glob=glob, chans=chans, digest=np.array(digest))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def flatten_bio(glob: np.ndarray, chans: np.ndarray) -> np.ndarray:
    """Concatenate globals with channels reshaped to ``(N, 3*21)`` -> ``(N, N_BIO_FEATURES)``."""
    return np.concatenate([glob, chans.reshape(len(chans), -1)], axis=1).astype(np.float32)


class Standardizer:
    """Z-score scaler; biofeatures are on very different scales.

    Fit on the training split only, then call ``transform`` on train, validation
    and test. Zero-variance columns are left unscaled (std set to 1).
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> Standardizer:
        """Learn per-column mean and std from ``x``."""
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return ``(x - mean) / std`` as float32."""
        return ((x - self.mean) / self.std).astype(np.float32)
