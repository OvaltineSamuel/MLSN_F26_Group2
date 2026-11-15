"""Shared pytest fixtures: a synthetic cleaned DeepHF-style table (no raw data, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_N_ROWS = 400
_NAN_FRACTION = 0.10
_EFFICIENCY_COLUMNS = ("WT_Efficiency", "eSp_Efficiency", "HF1_Efficiency")


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Return a seeded synthetic table with the full cleaned schema.

    400 random N20 + NGG guides; ``21mer`` is ``gRNA_Seq + PAM[0]`` so the data
    contract holds. About 10% of each efficiency column is NaN; the rest is
    drawn from ``uniform(0, 1)``.
    """
    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    seqs = ["".join(rng.choice(bases, size=20)) for _ in range(_N_ROWS)]
    pams = [rng.choice(bases) + "GG" for _ in range(_N_ROWS)]
    df = pd.DataFrame(
        {
            "gRNA_ID": [f"g{i:04d}" for i in range(_N_ROWS)],
            "gRNA_Seq": seqs,
            "PAM": pams,
        }
    )
    df["21mer"] = df["gRNA_Seq"] + df["PAM"].str[0]
    for col in _EFFICIENCY_COLUMNS:
        values = rng.uniform(0, 1, size=_N_ROWS)
        values[rng.random(_N_ROWS) < _NAN_FRACTION] = np.nan
        df[col] = values
    return df


@pytest.fixture
def synthetic_seqs(synthetic_df: pd.DataFrame) -> list[str]:
    """Return the 21-mer sequences of ``synthetic_df``."""
    return synthetic_df["21mer"].tolist()
