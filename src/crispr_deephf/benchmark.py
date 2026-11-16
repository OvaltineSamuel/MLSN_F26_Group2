"""Ours-vs-paper benchmark tables (Wang et al., 2019).

Turns the runner's aggregated results into the scoreboard shown on the slides.
Honesty rules enforced here:

* only ``Split == "Test"`` rows are ever scored (a validation number is never a headline);
* a model with no paper counterpart has a blank paper column and never "beats" the paper;
* paper numbers read off a figure are labelled ``"figure (approx)"`` and must clear their
  ~0.01 reading tolerance before a win is claimed;
* a missing s.d. (single-seed runs) stays NaN / is shown as ``n/a``, never ``0``.

This module must stay torch-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "APPROX_TOLERANCE",
    "PAPER_EQUIVALENT",
    "SCOREBOARD_COLUMNS",
    "TIERS",
    "best_per_target",
    "format_caveat",
    "format_scoreboard",
    "scoreboard",
    "tier_best",
]

#: Our run name -> the paper's row. Anything absent has no paper counterpart. The paper's
#: MLP is deliberately unmapped: we do not build one.
PAPER_EQUIVALENT: dict[str, str] = {
    "XGBoost": "XGBoost",
    "CNN": "CNN",
    "BiLSTM": "RNN",
    "BiLSTM+bio": "RNN+biofeature",
    "Ridge": "Ridge",
    "Linear": "Linear",
}

#: Reading error of the figure-read paper numbers; a win over one must exceed it.
APPROX_TOLERANCE = 0.01

#: Tier membership by model name (mirrors ``architectures`` without importing torch).
TIERS: dict[str, list[str]] = {
    "Tier 1": ["Ridge", "LightGBM", "XGBoost", "CatBoost", "Ensemble", "Stack"],
    "Tier 2": ["LSTM", "BiLSTM", "CNN"],
    "Tier 3": ["BiLSTM+bio", "CNN+bio"],
    "Extras": ["CNN-BiLSTM", "Transformer", "MultiTask-BiLSTM+bio"],
}

SCOREBOARD_COLUMNS = [
    "Model", "Target", "Ours_rho", "Ours_sd", "Paper_rho", "Paper_source", "Delta", "Beats_paper",
]
_BEST_COLUMNS = [
    "Target", "Best_model", "Ours_rho", "Ours_sd", "Paper_model", "Paper_rho", "Delta",
    "Beats_paper",
]
_PAPER_BEST_MODEL = "RNN+biofeature"


def format_caveat() -> str:
    """Return the comparability caveat to print under every table and figure."""
    return config.PAPER_CAVEAT


def _test_rows(agg: pd.DataFrame) -> pd.DataFrame:
    """Keep only held-out test rows; a frame without ``Split`` is taken as test-only."""
    if "Split" not in agg.columns:
        return agg
    out = agg[agg["Split"] == "Test"]
    if out.empty:
        raise ValueError("aggregate frame has a Split column but no 'Test' rows")
    return out


def _paper_lookup(model: str, target: str) -> tuple[float, str, str]:
    """Return ``(rho, source, paper_row)`` for one of our models; NaN/blank if none."""
    paper_row = PAPER_EQUIVALENT.get(model)
    key = config.TARGET_KEYS.get(target)
    if paper_row is None or key is None:
        return np.nan, "", ""
    if paper_row in config.PAPER_SPEARMAN:
        return config.PAPER_SPEARMAN[paper_row][key], "text", paper_row
    if paper_row in config.PAPER_SPEARMAN_APPROX:
        return config.PAPER_SPEARMAN_APPROX[paper_row][key], "figure (approx)", paper_row
    return np.nan, "", ""


def _beats(ours: float, paper: float, source: str) -> bool:
    """True only against a real paper number, and past the tolerance if it is approximate."""
    if source == "" or np.isnan(paper) or np.isnan(ours):
        return False
    margin = APPROX_TOLERANCE if source == "figure (approx)" else 0.0
    return bool(ours - paper > margin)


def scoreboard(agg: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-enzyme test Spearman rho beside the paper's number.

    Parameters
    ----------
    agg : DataFrame from ``runner.aggregate``. Rows with ``Split != "Test"`` are dropped.

    Returns
    -------
    DataFrame with columns ``SCOREBOARD_COLUMNS``. ``Ours_sd`` is NaN for single-seed
    models. ``Paper_rho`` is NaN and ``Paper_source`` is ``""`` where the paper has no
    counterpart. ``Delta`` = ours - paper (NaN if none). ``Beats_paper`` is a plain bool,
    False when there is nothing to beat.
    """
    rows = []
    for rec in _test_rows(agg).itertuples(index=False):
        paper, source, _ = _paper_lookup(rec.Model, rec.Target)
        ours = float(rec.Spearman_rho_mean)
        sd = getattr(rec, "Spearman_rho_std", np.nan)
        rows.append({
            "Model": rec.Model,
            "Target": rec.Target,
            "Ours_rho": ours,
            "Ours_sd": np.nan if sd is None else float(sd),
            "Paper_rho": paper,
            "Paper_source": source,
            "Delta": ours - paper,
            "Beats_paper": _beats(ours, paper, source),
        })
    return pd.DataFrame(rows, columns=SCOREBOARD_COLUMNS)


def best_per_target(agg: pd.DataFrame) -> pd.DataFrame:
    """Our best test model per enzyme against the paper's best (``RNN+biofeature``)."""
    test = _test_rows(agg)
    rows = []
    for target in config.TARGET_COLS:
        sub = test[test["Target"] == target]
        if sub.empty:
            continue
        best = sub.loc[sub["Spearman_rho_mean"].idxmax()]
        paper = config.PAPER_SPEARMAN[_PAPER_BEST_MODEL][config.TARGET_KEYS[target]]
        ours = float(best["Spearman_rho_mean"])
        sd = best.get("Spearman_rho_std", np.nan)
        rows.append({
            "Target": target,
            "Best_model": best["Model"],
            "Ours_rho": ours,
            "Ours_sd": float(sd),
            "Paper_model": _PAPER_BEST_MODEL,
            "Paper_rho": paper,
            "Delta": ours - paper,
            "Beats_paper": _beats(ours, paper, "text"),
        })
    return pd.DataFrame(rows, columns=_BEST_COLUMNS)


def tier_best(agg: pd.DataFrame) -> pd.DataFrame:
    """Best test model per tier and enzyme: ``Tier, Target, Model, Ours_rho``."""
    test = _test_rows(agg)
    rows = []
    for tier, names in TIERS.items():
        for target in config.TARGET_COLS:
            sub = test[test["Model"].isin(names) & (test["Target"] == target)]
            if sub.empty:
                continue
            best = sub.loc[sub["Spearman_rho_mean"].idxmax()]
            rows.append({"Tier": tier, "Target": target, "Model": best["Model"],
                         "Ours_rho": float(best["Spearman_rho_mean"])})
    return pd.DataFrame(rows, columns=["Tier", "Target", "Model", "Ours_rho"])


def format_scoreboard(sb: pd.DataFrame) -> pd.DataFrame:
    """Presentation strings for a scoreboard: ``0.8712 ± 0.0011``, ``n/a`` for no s.d.

    Approximate paper numbers keep their ``(approx)`` label in the paper column.
    """
    def ours(r: pd.Series) -> str:
        sd = "n/a" if pd.isna(r["Ours_sd"]) else f"{r['Ours_sd']:.4f}"
        return f"{r['Ours_rho']:.4f} ± {sd}"

    def paper(r: pd.Series) -> str:
        if pd.isna(r["Paper_rho"]):
            return ""
        if r["Paper_source"] == "figure (approx)":
            return f"~{r['Paper_rho']:.2f} (approx)"
        return f"{r['Paper_rho']:.4f}"

    out = pd.DataFrame({"Model": sb["Model"], "Target": sb["Target"]})
    out["Ours (rho ± sd)"] = sb.apply(ours, axis=1)
    out["Paper rho"] = sb.apply(paper, axis=1)
    out["Paper source"] = sb["Paper_source"]
    out["Delta"] = sb["Delta"].map(lambda d: "" if pd.isna(d) else f"{d:+.4f}")
    out["Beats paper"] = [
        ("yes" if b else "no") if s else "" for b, s in zip(sb["Beats_paper"], sb["Paper_source"])
    ]
    return out
