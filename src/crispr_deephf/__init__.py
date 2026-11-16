"""crispr_deephf -- reusable pipeline for the MLSN Group 2 CRISPR project.

This package holds the *reusable logic* behind the project's Jupyter notebooks,
so the notebooks stay short, readable, and presentation-friendly while the heavy
lifting lives in tested, importable modules.

Modules
-------
config       : paths, column names, split ratios, colours, and other constants.
data         : load the raw DeepHF Excel file, clean/validate it, and build splits.
features     : sequence encodings and the 439 hand-crafted biological features.
bio_features : ViennaRNA/Biopython biophysical features, cached per dataset.
models       : baseline model trainers (Ridge, LightGBM, XGBoost, CatBoost) + metrics.
plots        : matplotlib/seaborn helpers for the figures shown in the notebook.
benchmark    : ours-vs-paper scoreboard tables and the caveats that go with them.

Torch-dependent modules (``torch_data``, ``architectures``, ``train_torch``,
``runner``, ``tuning``) are deliberately NOT imported here, so ``import
crispr_deephf`` works in an environment with no torch (as CI's is). Import them
explicitly: ``from crispr_deephf import runner``.

Reference
---------
Wang et al., "Optimized CRISPR guide RNA design for two high-fidelity Cas9
variants by deep learning", Nature Communications 10, 4284 (2019).
https://www.nature.com/articles/s41467-019-12281-8
"""

from __future__ import annotations

__version__ = "0.1.0"

# Only torch-free modules are imported eagerly so `import crispr_deephf` works with just the
# [dev] extra (as in CI). Deep-learning modules are imported explicitly by callers.
from . import benchmark, bio_features, config, data, features, models, plots

__all__ = [
    "__version__",
    "benchmark",
    "bio_features",
    "config",
    "data",
    "features",
    "models",
    "plots",
]
