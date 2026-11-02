"""Plotting helpers for the notebooks.

Keeping the figure code here lets the notebooks stay focused on narrative and
results. Every function draws onto the current matplotlib figure/axes and
returns the Figure so the caller can save it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from . import config

__all__ = [
    "set_style",
    "plot_efficiency_distributions",
    "plot_model_comparison",
    "plot_pred_vs_actual",
    "plot_feature_importance",
    "plot_residuals",
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
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(data=imp, x="Importance", y="Feature", palette="viridis")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Feature Importance")
    plt.ylabel("Engineered Feature")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
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
