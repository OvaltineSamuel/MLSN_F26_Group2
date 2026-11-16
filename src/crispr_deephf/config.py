"""Project-wide configuration: paths, schema, split ratios, and constants.

Everything that the two notebooks need to agree on lives here, so the data
cleaning notebook and the baseline modelling notebook can never drift apart.

Path handling works the same locally and on Google Colab: call
:func:`get_paths` with the project root and it returns a small object with every
directory the pipeline uses, creating them on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Raw dataset
# --------------------------------------------------------------------------- #
#: File name of the DeepHF supplementary table (Wang et al., 2019).
RAW_XLSX_NAME = "41467_2019_12281_MOESM3_ESM.xlsx"

#: Nature Communications supplementary data URL for ``RAW_XLSX_NAME``.
RAW_XLSX_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-12281-8"
    "/MediaObjects/41467_2019_12281_MOESM3_ESM.xlsx"
)

#: Header row to use when reading the Excel sheet (row 0 is a title banner).
RAW_HEADER_ROW = 1

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
#: Standardised column names, in order, after dropping the empty index column.
COLUMNS = [
    "gRNA_ID",
    "gRNA_Seq",
    "PAM",
    "21mer",
    "WT_Efficiency",
    "eSpCas9_Efficiency",
    "SpCas9_HF1_Efficiency",
]

#: Columns holding string / identifier data.
STR_COLS = ["gRNA_ID", "gRNA_Seq", "PAM", "21mer"]

#: The three Cas9 nuclease efficiency targets we predict.
TARGET_COLS = ["WT_Efficiency", "eSpCas9_Efficiency", "SpCas9_HF1_Efficiency"]

#: Short machine-friendly keys for each target (used in split-label columns).
TARGET_KEYS = {
    "WT_Efficiency": "WT",
    "eSpCas9_Efficiency": "eSp",
    "SpCas9_HF1_Efficiency": "HF1",
}

#: Human-readable labels for plots and tables.
TARGET_LABELS = {
    "WT_Efficiency": "WT-SpCas9",
    "eSpCas9_Efficiency": "eSpCas9(1.1)",
    "SpCas9_HF1_Efficiency": "SpCas9-HF1",
}

#: Consistent colours for each target across every figure.
TARGET_COLORS = {
    "WT_Efficiency": "#d95f5f",
    "eSpCas9_Efficiency": "#5f87d9",
    "SpCas9_HF1_Efficiency": "#5fd987",
}

# --------------------------------------------------------------------------- #
# Train / validation / test split (matches the DeepHF paper)
# --------------------------------------------------------------------------- #
# The paper holds out 15% for test, then takes 10% of the remaining 85% for
# validation -> 76.5% train / 8.5% val / 15% test.
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE_OF_REMAINDER = 0.10

#: Column holding the shared train/val/test split label across all targets.
SHARED_SPLIT_COL = "shared_split"

#: Cleaned dataset file names produced by the data cleaning notebook.
CLEAN_ALL_CSV = "deephf_clean_all.csv"          # all rows + per-variant split labels
CLEAN_MULTITASK_CSV = "deephf_clean_multitask.csv"  # rows valid for all 3 variants
RAW_BACKUP_CSV = "deephf_raw_seq.csv"

#: Nucleotide -> integer token mapping used by every encoder.
NT_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}

# --------------------------------------------------------------------------- #
# Paper reference metrics (Wang et al., 2019)
# --------------------------------------------------------------------------- #
#: Paper's test-set Spearman rho per model, keyed by short target key (see
#: ``TARGET_KEYS``). Exact values from the article text, p.4-5 (Fig. 5c-e).
PAPER_SPEARMAN: dict[str, dict[str, float]] = {
    "XGBoost": {"WT": 0.8454, "eSp": 0.8310, "HF1": 0.8184},
    "MLP": {"WT": 0.8416, "eSp": 0.8457, "HF1": 0.8440},
    "CNN": {"WT": 0.8455, "eSp": 0.8313, "HF1": 0.8343},
    "RNN": {"WT": 0.8555, "eSp": 0.8491, "HF1": 0.8512},
    "RNN+biofeature": {"WT": 0.8670, "eSp": 0.8624, "HF1": 0.8603},
}

#: Linear and Ridge appear only as bars in Fig. 5c-e, so these are figure-read
#: APPROXIMATIONS (about +/-0.01). Never mix them with ``PAPER_SPEARMAN``.
PAPER_SPEARMAN_APPROX: dict[str, dict[str, float]] = {
    "Linear": {"WT": 0.78, "eSp": 0.77, "HF1": 0.75},
    "Ridge": {"WT": 0.78, "eSp": 0.77, "HF1": 0.76},
}

#: Comparability caveat to show wherever our numbers are set against the paper.
#:
#: It deliberately does **not** state a seed count: this text used to say "averaged over 3
#: model-init seeds" unconditionally, which described a method that was not used whenever the
#: notebook ran in ``quick`` mode (1 seed) -- printed right beside a table reading
#: ``± n/a``. The real count is appended by :func:`benchmark.format_caveat`, which takes it
#: from the results actually on disk.
PAPER_CAVEAT = (
    "The paper reports mean ± s.d. over tenfold shuffled validation (n=10); "
    "we report a single fixed split (seed 42) averaged over model-init seeds, "
    "so the two are not a like-for-like significance test."
)

#: Guide counts in the paper's per-target datasets (article text).
PAPER_N_GUIDES = {"WT": 55604, "eSp": 58167, "HF1": 56888}

# --------------------------------------------------------------------------- #
# Run modes (used by the notebook's quick/full switch)
# --------------------------------------------------------------------------- #
#: Training budget per run mode. ``quick`` checks the pipeline end to end;
#: ``full`` is the reported configuration.
RUN_MODES: dict[str, dict] = {
    "quick": {"epochs": 3, "n_seeds": 1, "patience": 2, "batch_size": 256},
    "full": {"epochs": 60, "n_seeds": 3, "patience": 10, "batch_size": 256},
}


def get_mode(name: str) -> dict:
    """Return a copy of the run-mode settings for *name*.

    Raises
    ------
    KeyError
        If *name* is not one of the ``RUN_MODES`` keys.
    """
    if name not in RUN_MODES:
        raise KeyError(f"unknown run mode {name!r}; valid modes: {sorted(RUN_MODES)}")
    return dict(RUN_MODES[name])


@dataclass(frozen=True)
class Paths:
    """Resolved project directories. All are created on construction."""

    root: Path
    data_raw: Path
    data_processed: Path
    cache: Path
    models: Path
    results: Path
    figures: Path

    def mkdirs(self) -> Paths:
        for p in (
            self.data_raw,
            self.data_processed,
            self.cache,
            self.models,
            self.results,
            self.figures,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def find_project_root(start: Path | str | None = None) -> Path:
    """Walk upwards from *start* until a repo marker is found.

    Looks for a ``src/crispr_deephf`` directory (or a ``.git`` folder). Falls
    back to the current working directory if no marker is found -- handy on
    Colab where the layout is flat.
    """
    here = Path(start).resolve() if start else Path.cwd().resolve()
    candidates = [here, *here.parents]
    for cand in candidates:
        if (cand / "src" / "crispr_deephf").is_dir() or (cand / ".git").is_dir():
            return cand
    return here


def get_paths(root: Path | str | None = None) -> Paths:
    """Return (and create) the standard project directories under *root*.

    Parameters
    ----------
    root:
        Project root. If ``None``, it is auto-detected with
        :func:`find_project_root`. On Colab, pass the Drive project folder.
    """
    root_path = Path(root).resolve() if root is not None else find_project_root()
    return Paths(
        root=root_path,
        data_raw=root_path / "data" / "raw",
        data_processed=root_path / "data" / "processed",
        cache=root_path / "data" / "cache",
        models=root_path / "models",
        results=root_path / "results",
        figures=root_path / "results" / "figures",
    ).mkdirs()
