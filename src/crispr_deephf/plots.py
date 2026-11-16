"""Plotting helpers for the notebooks.

Keeping the figure code here lets the notebooks stay focused on narrative and
results. Every function draws onto the current matplotlib figure/axes and
returns the Figure so the caller can save it.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr

from . import benchmark, config

__all__ = [
    "plot_efficiency_distributions",
    "plot_feature_importance",
    "plot_model_comparison",
    "plot_pred_vs_actual",
    "plot_residuals",
    "plot_tier_progression",
    "plot_training_curves",
    "plot_vs_paper",
    "set_style",
]


def set_style() -> None:
    """Apply a consistent, presentation-friendly matplotlib style."""
    style = "seaborn-v0_8-whitegrid"
    plt.style.use(style if style in plt.style.available else "default")
    plt.rcParams["figure.dpi"] = 110


def plot_efficiency_distributions(df: pd.DataFrame) -> plt.Figure:
    """KDE + histogram of indel efficiency per Cas9 variant (paper Fig. 2d)."""
    fig = plt.figure(figsize=(8, 5))
    for col in config.TARGET_COLS:
        sns.histplot(
            df[col].dropna(),
            bins=50,
            kde=True,
            stat="density",
            color=config.TARGET_COLORS[col],
            label=config.TARGET_LABELS[col],
            alpha=0.35,
        )
    plt.title("Indel Efficiency Distribution Across Cas9 Variants (Fig. 2d)",
              fontsize=12, fontweight="bold")
    plt.xlabel("Indel Cleavage Efficiency")
    plt.ylabel("Density")
    plt.xlim(0, 1.0)
    plt.legend(frameon=True)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_comparison(metrics: pd.DataFrame) -> plt.Figure:
    """Side-by-side Pearson r and Spearman rho bars across models/targets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = ["#4A90E2", "#50E3C2", "#F5A623", "#D0021B", "#9013FE"]
    for ax, metric, title in (
        (axes[0], "Pearson_r", "Test Pearson Correlation (r)"),
        (axes[1], "Spearman_rho", "Test Spearman Rank Correlation (rho)"),
    ):
        sns.barplot(data=metrics, x="Target", y=metric, hue="Model", ax=ax, palette=palette)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_pred_vs_actual(
    y_true_by_target: dict[str, np.ndarray],
    y_pred_by_target: dict[str, np.ndarray],
    model_label: str = "Ensemble",
) -> plt.Figure:
    """Hexbin predicted-vs-actual panel, one axis per Cas9 variant."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, tgt in zip(axes, config.TARGET_COLS):
        y_true = np.asarray(y_true_by_target[tgt])
        y_pred = np.asarray(y_pred_by_target[tgt])
        r = pearsonr(y_true, y_pred)[0]
        rho = spearmanr(y_true, y_pred)[0]
        hb = ax.hexbin(y_true, y_pred, gridsize=40, cmap="Blues", mincnt=1)
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        ax.plot([0, 1], [intercept, slope + intercept], "r--", lw=2, label="Fit")
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5, label="Ideal y=x")
        ax.set_title(f"{config.TARGET_LABELS[tgt]}\nr = {r:.4f} | rho = {rho:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Indel Efficiency")
        ax.set_ylabel(f"Predicted ({model_label})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", frameon=True)
        fig.colorbar(hb, ax=ax, label="Guide RNA Density")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: list[str], importances: np.ndarray, title: str, top_n: int = 25
) -> plt.Figure:
    """Horizontal bar chart of the top-N most important features."""
    imp = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )
    # Drawn with matplotlib rather than ``sns.barplot(palette=...)``: seaborn >= 0.13
    # raises a FutureWarning for a palette with no ``hue``, and the ``hue`` +
    # ``legend=False`` replacement does not exist in seaborn 0.12, which pyproject
    # still allows. One colour per bar needs no categorical machinery anyway.
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(imp)))
    ax.barh(imp["Feature"].tolist(), imp["Importance"].to_numpy(), color=colors)
    ax.invert_yaxis()  # most important feature on top
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Engineered Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_residuals(
    residuals_by_target: dict[str, np.ndarray],
) -> plt.Figure:
    """Residual (actual - predicted) histograms, one per Cas9 variant."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, tgt in zip(axes, config.TARGET_COLS):
        res = np.asarray(residuals_by_target[tgt])
        sns.histplot(res, bins=50, kde=True, ax=ax, color="#3F51B5")
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_title(f"Residuals: {config.TARGET_LABELS[tgt]}\n"
                     f"Mean: {res.mean():.4f} | Std: {res.std():.4f}")
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Benchmark / presentation figures
# --------------------------------------------------------------------------- #
_PAPER_GREY = "#7f7f7f"
_SLIDE_FONT = 14


#: Why the tier-progression figure needs a note of its own: each point is a per-tier
#: *maximum*, and the tiers do not hold equally many candidates, so the larger pool wins
#: more often for reasons that have nothing to do with the architecture.
_TIER_POOL_NOTE = (
    "Each point is the best (maximum) model in that tier, and the tiers offer unequal "
    "candidate pools - Tier 1 up to 6 models, Tier 2 three, Tier 3 two - which favours the "
    "larger pool, so part of any step between tiers is best-of-N selection rather than "
    "architecture."
)


def _caption(fig: plt.Figure, note: str | None = None) -> None:
    """Print the paper-comparability caveat, and any extra *note*, along a figure's bottom.

    Each paragraph is wrapped separately: ``textwrap.wrap`` would otherwise collapse the
    blank line between the caveat and the note into one run-on block.
    """
    text = benchmark.format_caveat() if note is None else f"{benchmark.format_caveat()}\n{note}"
    caption = "\n".join(w for part in text.split("\n") for w in textwrap.wrap(part, 110))
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=10, style="italic")


def plot_vs_paper(scoreboard_df: pd.DataFrame) -> plt.Figure:
    """Grouped bars, ours vs paper, one panel per enzyme (models with a paper row only).

    Our bar carries a +/- s.d. error bar only when a s.d. exists (multi-seed models).
    Paper bars from the article text are solid grey; figure-read approximations are
    hatched. The comparability caveat is the figure caption.
    """
    sb = scoreboard_df[scoreboard_df["Paper_source"] != ""]
    sd = sb["Ours_sd"].fillna(0.0)
    lows = pd.concat([sb["Ours_rho"] - sd, sb["Paper_rho"]])
    highs = pd.concat([sb["Ours_rho"] + sd, sb["Paper_rho"]])
    ylo = min(0.60, float(lows.min()) - 0.02) if len(sb) else 0.60
    yhi = max(0.95, float(highs.max()) + 0.02) if len(sb) else 0.95
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharey=True)
    for ax, tgt in zip(axes, config.TARGET_COLS):
        sub = sb[sb["Target"] == tgt].reset_index(drop=True)
        x = np.arange(len(sub))
        w = 0.38
        for i, row in sub.iterrows():
            err = None if pd.isna(row["Ours_sd"]) else row["Ours_sd"]
            ax.bar(x[i] - w / 2, row["Ours_rho"], w, color=config.TARGET_COLORS[tgt],
                   yerr=err, capsize=4, ecolor="black")
            top = row["Ours_rho"] + (err or 0.0)
            ax.text(x[i] - w / 2, top + 0.004, f"{row['Ours_rho']:.3f}", ha="center",
                    va="bottom", fontsize=9, rotation=90)
            approx = row["Paper_source"] == "figure (approx)"
            ax.bar(x[i] + w / 2, row["Paper_rho"], w, color="white" if approx else _PAPER_GREY,
                   edgecolor=_PAPER_GREY, hatch="//" if approx else None)
            ax.text(x[i] + w / 2, row["Paper_rho"] + 0.004,
                    f"~{row['Paper_rho']:.2f}" if approx else f"{row['Paper_rho']:.3f}",
                    ha="center", va="bottom", fontsize=9, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Model"], rotation=30, ha="right", fontsize=_SLIDE_FONT - 2)
        ax.set_title(config.TARGET_LABELS[tgt], fontsize=_SLIDE_FONT, fontweight="bold")
        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].set_ylabel(f"Test Spearman rho (axis starts at {ylo:.2f})", fontsize=_SLIDE_FONT)
    fig.legend(
        handles=[
            Patch(facecolor="#bbbbbb", label="Ours (colour = enzyme; ± s.d., none if 1 seed)"),
            Patch(facecolor=_PAPER_GREY, label="Paper (article text)"),
            Patch(facecolor="white", edgecolor=_PAPER_GREY, hatch="//",
                  label="Paper (figure, approx)"),
        ],
        loc="lower center", ncol=3, fontsize=11, frameon=False,
        bbox_to_anchor=(0.5, 0.075),
    )
    fig.suptitle("Ours vs paper: test Spearman rho", fontsize=_SLIDE_FONT + 3,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    _caption(fig)
    return fig


def plot_training_curves(history_frame: pd.DataFrame) -> plt.Figure:
    """Loss curves (left) and validation Spearman rho (right) for one training run.

    ``history_frame`` has ``epoch, train_loss, val_loss, val_rho`` and optionally
    ``val_rho_per_target`` (dict keyed by target column). Targets absent from a row's
    dict (no measured validation entries) are left as gaps, never assumed present.
    """
    hist = pd.DataFrame(history_frame)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    ax_l.plot(hist["epoch"], hist["train_loss"], marker="o", color="#0072B2", label="Train")
    ax_l.plot(hist["epoch"], hist["val_loss"], marker="s", color="#D55E00", label="Validation")
    ax_l.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax_l.legend(fontsize=_SLIDE_FONT - 2)
    if "val_rho_per_target" in hist.columns:
        for tgt in config.TARGET_COLS:
            vals = [d.get(tgt, np.nan) if isinstance(d, dict) else np.nan
                    for d in hist["val_rho_per_target"]]
            if not np.all(np.isnan(vals)):
                ax_r.plot(hist["epoch"], vals, marker="o", lw=1.5, alpha=0.8,
                          color=config.TARGET_COLORS[tgt], label=config.TARGET_LABELS[tgt])
    ax_r.plot(hist["epoch"], hist["val_rho"], color="black", lw=2.5, ls="--", label="Mean")
    ax_r.set(xlabel="Epoch", ylabel="Validation Spearman rho", title="Validation rho")
    ax_r.legend(fontsize=_SLIDE_FONT - 4)
    for ax in (ax_l, ax_r):
        ax.title.set_fontsize(_SLIDE_FONT)
        ax.grid(alpha=0.4)
    fig.tight_layout()
    return fig


def plot_tier_progression(agg: pd.DataFrame, assume_test: bool = False) -> plt.Figure:
    """Best test Spearman rho per tier (Tier 1 -> 2 -> 3 -> extras), one line per enzyme.

    Each point is labelled with the winning model, so the reader can see whether a tier
    buys anything over the previous one.

    This is the figure that answers "did Tier 3 earn its complexity?", so it carries the
    paper-comparability caveat *and* :data:`_TIER_POOL_NOTE`: the points are per-tier
    maxima over unequal candidate pools, which biases the comparison toward the bigger
    tier. Shipping it bare invited exactly the reading it cannot support.
    """
    best = benchmark.tier_best(agg, assume_test)
    tiers = [t for t in benchmark.TIERS if t in set(best["Tier"])]
    fig, ax = plt.subplots(figsize=(11, 6))
    for tgt in config.TARGET_COLS:
        sub = best[best["Target"] == tgt].set_index("Tier").reindex(tiers)
        xs = np.arange(len(tiers))
        ax.plot(xs, sub["Ours_rho"], marker="o", lw=2.5, ms=9,
                color=config.TARGET_COLORS[tgt], label=config.TARGET_LABELS[tgt])
        for xi, (rho, name) in enumerate(zip(sub["Ours_rho"], sub["Model"])):
            if pd.notna(rho):
                ax.annotate(name, (xi, rho), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(tiers)))
    ax.set_xticklabels(tiers, fontsize=_SLIDE_FONT)
    ax.set_ylabel("Best test Spearman rho in tier", fontsize=_SLIDE_FONT)
    ax.set_title("Does each tier buy anything?", fontsize=_SLIDE_FONT + 2, fontweight="bold")
    ax.legend(fontsize=_SLIDE_FONT - 2, loc="lower right")
    ax.grid(alpha=0.4)
    fig.tight_layout(rect=(0, 0.17, 1, 1))  # leave room for the caption below the axes
    _caption(fig, note=_TIER_POOL_NOTE)
    return fig
