"""crispr_deephf -- reusable pipeline for the MLSN Group 2 CRISPR project.

This package holds the *reusable logic* behind the project's Jupyter notebooks,
so the notebooks stay short, readable, and presentation-friendly while the heavy
lifting lives in tested, importable modules.

Modules
-------
config   : paths, column names, split ratios, colours, and other constants.
data     : load the raw DeepHF Excel file, clean/validate it, and build splits.
features : sequence encodings and the 439 hand-crafted biological features.
models   : baseline model trainers (Ridge, LightGBM, XGBoost, CatBoost) + metrics.
plots    : matplotlib/seaborn helpers for the figures shown in the notebooks.

Reference
---------
Wang et al., "Optimized CRISPR guide RNA design for two high-fidelity Cas9
variants by deep learning", Nature Communications 10, 5286 (2019).
https://www.nature.com/articles/s41467-019-12281-8
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import config, data, features, models, plots

__all__ = ["config", "data", "features", "models", "plots", "__version__"]
