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
    "format_selection_note",
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
    "Target", "Best_model", "Best_has_paper_counterpart", "Ours_rho", "Ours_sd", "Paper_model",
    "Paper_rho", "Delta", "Ours_best_exceeds_paper_best",
]
_PAPER_BEST_MODEL = "RNN+biofeature"


def format_caveat() -> str:
    """Return the comparability caveat to print under every table and figure."""
    return config.PAPER_CAVEAT


def format_selection_note(n_candidates: int | None = None) -> str:
    """Selection-bias note that must accompany a best-of-N-vs-paper comparison."""
    n = "all" if n_candidates is None else f"all {n_candidates}"
    return (
        f"Our best model is selected post hoc from {n} candidate models scored on the test "
        "set, whereas the paper's figure is its single reported model; this comparison is "
        "therefore optimistically biased in our favour."
    )


def _test_rows(agg: pd.DataFrame, assume_test: bool = False) -> pd.DataFrame:
    """Keep only held-out test rows.

    ``runner.results_frame`` always emits ``Split``; a frame without it was hand-built, so
    it is rejected unless the caller vouches for it with ``assume_test=True``.
    """
    if "Split" not in agg.columns:
        if not assume_test:
            raise ValueError(
                "aggregate frame has no 'Split' column, so test rows cannot be told from "
                "validation rows; pass assume_test=True only if every row is a test result"
            )
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


def _guarded_sd(sd: object, n_seeds: object) -> float:
    """Return the s.d., forced to NaN when fewer than two seeds back it."""
    if sd is None or pd.isna(sd) or (n_seeds is not None and n_seeds <= 1):
        return float("nan")
    return float(sd)


def scoreboard(agg: pd.DataFrame, assume_test: bool = False) -> pd.DataFrame:
    """Per-model, per-enzyme test Spearman rho beside the paper's number.

    Parameters
    ----------
    agg : DataFrame from ``runner.aggregate``. Rows with ``Split != "Test"`` are dropped.
    assume_test : accept a frame with no ``Split`` column as all-test (else ValueError).

    Returns
    -------
    DataFrame with columns ``SCOREBOARD_COLUMNS``. ``Ours_sd`` is NaN for single-seed
    models. ``Paper_rho`` is NaN and ``Paper_source`` is ``""`` where the paper has no
    counterpart. ``Delta`` = ours - paper (NaN if none). ``Beats_paper`` is a plain bool,
    False when there is nothing to beat.
    """
    rows = []
    for rec in _test_rows(agg, assume_test).itertuples(index=False):
        paper, source, _ = _paper_lookup(rec.Model, rec.Target)
        ours = float(rec.Spearman_rho_mean)
        sd = _guarded_sd(getattr(rec, "Spearman_rho_std", np.nan), getattr(rec, "n_seeds", None))
        rows.append({
            "Model": rec.Model,
            "Target": rec.Target,
            "Ours_rho": ours,
            "Ours_sd": sd,
            "Paper_rho": paper,
            "Paper_source": source,
            "Delta": ours - paper,
            "Beats_paper": _beats(ours, paper, source),
        })
    return pd.DataFrame(rows, columns=SCOREBOARD_COLUMNS)


def best_per_target(agg: pd.DataFrame, assume_test: bool = False) -> pd.DataFrame:
    """Our best test model per enzyme against the paper's best (``RNN+biofeature``).

    ``Best_has_paper_counterpart`` says whether the winning model has a paper row of its own;
    ``Ours_best_exceeds_paper_best`` is a best-of-N vs single-number comparison, not a
    like-for-like win. The selection-bias note is in ``result.attrs["selection_note"]``;
    print it beside the table, together with ``format_caveat()``.
    """
    test = _test_rows(agg, assume_test)
    rows = []
    for target in config.TARGET_COLS:
        sub = test[test["Target"] == target]
        if sub.empty:
            continue
        best = sub.loc[sub["Spearman_rho_mean"].idxmax()]
        paper = config.PAPER_SPEARMAN[_PAPER_BEST_MODEL][config.TARGET_KEYS[target]]
        ours = float(best["Spearman_rho_mean"])
        sd = _guarded_sd(best.get("Spearman_rho_std", np.nan), best.get("n_seeds"))
        rows.append({
            "Target": target,
            "Best_model": best["Model"],
            "Best_has_paper_counterpart": best["Model"] in PAPER_EQUIVALENT,
            "Ours_rho": ours,
            "Ours_sd": sd,
            "Paper_model": _PAPER_BEST_MODEL,
            "Paper_rho": paper,
            "Delta": ours - paper,
            "Ours_best_exceeds_paper_best": _beats(ours, paper, "text"),
        })
    out = pd.DataFrame(rows, columns=_BEST_COLUMNS)
    out.attrs["selection_note"] = format_selection_note(int(test["Model"].nunique()))
    return out


def tier_best(agg: pd.DataFrame, assume_test: bool = False) -> pd.DataFrame:
    """Best test model per tier and enzyme: ``Tier, Target, Model, Ours_rho``."""
    test = _test_rows(agg, assume_test)
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
    out["Delta"] = [_fmt_delta(d, s) for d, s in zip(sb["Delta"], sb["Paper_source"])]
    out["Beats paper"] = [
        _fmt_verdict(b, d, s)
        for b, d, s in zip(sb["Beats_paper"], sb["Delta"], sb["Paper_source"])
    ]
    return out


def _fmt_delta(delta: float, source: str) -> str:
    """Delta at the precision of the paper number: 2 decimals and ``~`` if figure-read."""
    if pd.isna(delta):
        return ""
    return f"~{delta:+.2f}" if source == "figure (approx)" else f"{delta:+.4f}"


def _fmt_verdict(beats: bool, delta: float, source: str) -> str:
    """yes / no / within reading error (approx paper numbers only) / blank (no counterpart)."""
    if not source:
        return ""
    if beats:
        return "yes"
    if source == "figure (approx)" and abs(delta) <= APPROX_TOLERANCE:
        return "within reading error"
    return "no"
