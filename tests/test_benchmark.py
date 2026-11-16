"""Tests for the ours-vs-paper benchmark and its figures (torch-free)."""

from __future__ import annotations

import ast
import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from crispr_deephf import benchmark, config, plots

WT, ESP, HF1 = config.TARGET_COLS


def _agg(rows):
    """rows: (Model, Target, Split, rho, sd, n_seeds)."""
    return pd.DataFrame(rows, columns=["Model", "Target", "Split", "Spearman_rho_mean",
                                       "Spearman_rho_std", "n_seeds"])


@pytest.fixture
def agg():
    rows = []
    for model, rho, sd, n in [
        ("BiLSTM+bio", 0.87, 0.001, 3), ("BiLSTM", 0.85, 0.002, 3), ("CNN", 0.84, 0.003, 3),
        ("Ridge", 0.785, np.nan, 1), ("XGBoost", 0.86, np.nan, 1),
        ("Transformer", 0.88, 0.002, 3), ("LSTM", 0.83, 0.002, 3), ("Stack", 0.862, np.nan, 1),
    ]:
        for tgt in config.TARGET_COLS:
            rows.append((model, tgt, "Test", rho, sd, n))
            rows.append((model, tgt, "Val", 0.99, sd, n))  # decoy: must never surface
    return _agg(rows)


def test_scoreboard_marks_source_of_paper_number():
    agg = pd.DataFrame({"Model": ["BiLSTM+bio", "Ridge", "Transformer"],
                        "Target": [WT] * 3,
                        "Spearman_rho_mean": [0.87, 0.78, 0.86],
                        "Spearman_rho_std": [0.001, 0.0, 0.002], "n_seeds": [3, 1, 3]})
    sb = benchmark.scoreboard(agg, assume_test=True).set_index("Model")
    assert sb.loc["BiLSTM+bio", "Paper_rho"] == 0.8670
    assert sb.loc["BiLSTM+bio", "Paper_source"] == "text"
    assert sb.loc["Ridge", "Paper_source"] == "figure (approx)"
    assert pd.isna(sb.loc["Transformer", "Paper_rho"])
    assert sb.loc["Transformer", "Paper_source"] == ""


def test_scoreboard_columns_exact(agg):
    assert list(benchmark.scoreboard(agg).columns) == [
        "Model", "Target", "Ours_rho", "Ours_sd", "Paper_rho", "Paper_source", "Delta",
        "Beats_paper"]


def test_val_rows_never_leak_into_scoreboard(agg):
    sb = benchmark.scoreboard(agg)
    assert len(sb) == 8 * 3  # one row per model-target, not two
    assert not sb.duplicated(["Model", "Target"]).any()
    assert (sb["Ours_rho"] < 0.95).all()  # the 0.99 Val decoys are absent
    assert benchmark.best_per_target(agg)["Ours_rho"].max() < 0.95
    assert benchmark.tier_best(agg)["Ours_rho"].max() < 0.95


def test_frame_with_split_but_no_test_rows_raises():
    val_only = _agg([("CNN", WT, "Val", 0.9, 0.0, 3)])
    with pytest.raises(ValueError):
        benchmark.scoreboard(val_only)


def test_beats_paper_flag(agg):
    sb = benchmark.scoreboard(agg).set_index(["Model", "Target"])
    assert bool(sb.loc[("BiLSTM+bio", WT), "Beats_paper"]) is True  # 0.87 > 0.8670
    assert bool(sb.loc[("BiLSTM", WT), "Beats_paper"]) is False  # 0.85 < 0.8555


def test_model_without_paper_counterpart_never_beats(agg):
    sb = benchmark.scoreboard(agg)
    for model in ("Transformer", "Stack", "LSTM"):
        sub = sb[sb["Model"] == model]
        assert sub["Paper_rho"].isna().all()
        assert (sub["Paper_source"] == "").all()
        assert sub["Delta"].isna().all()
        assert not sub["Beats_paper"].any()  # even at 0.88 > every paper number


def test_mlp_is_not_mapped_and_equivalents_are_exact():
    assert "MLP" not in benchmark.PAPER_EQUIVALENT
    assert benchmark.PAPER_EQUIVALENT["BiLSTM"] == "RNN"
    assert benchmark.PAPER_EQUIVALENT["BiLSTM+bio"] == "RNN+biofeature"
    for name in ("CNN+bio", "CNN-BiLSTM", "Transformer", "MultiTask-BiLSTM+bio", "LightGBM",
                 "CatBoost", "Ensemble", "Stack"):
        assert name not in benchmark.PAPER_EQUIVALENT


def test_approx_paper_number_is_labelled_and_needs_margin(agg):
    sb = benchmark.scoreboard(agg).set_index(["Model", "Target"])
    # Ridge 0.785 vs figure-read ~0.78: inside the +/-0.01 reading error, so no win.
    assert sb.loc[("Ridge", WT), "Paper_source"] == "figure (approx)"
    assert sb.loc[("Ridge", WT), "Paper_rho"] == config.PAPER_SPEARMAN_APPROX["Ridge"]["WT"]
    assert bool(sb.loc[("Ridge", WT), "Beats_paper"]) is False
    clear = _agg([("Ridge", WT, "Test", 0.80, np.nan, 1)])
    assert bool(benchmark.scoreboard(clear).loc[0, "Beats_paper"]) is True
    shown = benchmark.format_scoreboard(benchmark.scoreboard(agg))
    ridge = shown[shown["Model"] == "Ridge"]
    assert ridge["Paper rho"].str.contains("approx").all()
    assert (ridge["Paper source"] == "figure (approx)").all()


def test_nan_sd_stays_nan_and_is_never_zero(agg):
    sb = benchmark.scoreboard(agg)
    single = sb[sb["Model"].isin(["Ridge", "XGBoost", "Stack"])]
    assert single["Ours_sd"].isna().all()
    shown = benchmark.format_scoreboard(sb)
    cell = shown.loc[shown["Model"] == "Ridge", "Ours (rho ± sd)"].iloc[0]
    assert "n/a" in cell and "0.0000" not in cell
    multi = shown.loc[shown["Model"] == "BiLSTM+bio", "Ours (rho ± sd)"].iloc[0]
    assert multi == "0.8700 ± 0.0010"


def test_best_per_target(agg):
    best = benchmark.best_per_target(agg).set_index("Target")
    assert list(best.index) == config.TARGET_COLS
    assert (best["Best_model"] == "Transformer").all()
    assert best.loc[WT, "Paper_model"] == "RNN+biofeature"
    assert best.loc[WT, "Paper_rho"] == 0.8670
    assert best.loc[WT, "Delta"] == pytest.approx(0.88 - 0.8670)
    assert bool(best.loc[WT, "Ours_best_exceeds_paper_best"]) is True


def test_caveat_is_non_empty():
    assert "tenfold" in benchmark.format_caveat()
    assert benchmark.format_caveat() == config.PAPER_CAVEAT


def test_plot_vs_paper_returns_figure_and_carries_caveat(agg):
    fig = plots.plot_vs_paper(benchmark.scoreboard(agg))
    assert isinstance(fig, plt.Figure)
    text = " ".join(t.get_text() for t in fig.texts).replace("\n", " ")
    assert "tenfold" in text
    plt.close(fig)


def test_plot_vs_paper_draws_no_error_bar_for_nan_sd(agg):
    sb = benchmark.scoreboard(agg)
    sb = sb[sb["Model"].isin(["Ridge", "XGBoost"])]  # single-seed only
    fig = plots.plot_vs_paper(sb)
    assert all(len(ax.collections) == 0 for ax in fig.axes)  # no errorbar line collections
    plt.close(fig)


def test_plot_vs_paper_draws_error_bar_when_sd_present(agg):
    sb = benchmark.scoreboard(agg)
    sb = sb[sb["Model"] == "BiLSTM+bio"]
    fig = plots.plot_vs_paper(sb)
    assert any(len(ax.collections) > 0 for ax in fig.axes)
    plt.close(fig)


def test_plot_training_curves_tolerates_missing_targets():
    hist = pd.DataFrame({
        "epoch": [1, 2, 3], "train_loss": [0.3, 0.2, 0.1], "val_loss": [0.3, 0.25, 0.2],
        "val_rho": [0.5, 0.6, 0.7],
        "val_rho_per_target": [{WT: 0.5, ESP: 0.4}, {WT: 0.6}, {}],  # HF1 never measured
    })
    fig = plots.plot_training_curves(hist)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_tier_progression_returns_figure(agg):
    fig = plots.plot_tier_progression(agg)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_benchmark_and_plots_do_not_import_torch():
    for mod in (benchmark, plots):
        tree = ast.parse(inspect.getsource(mod))
        names = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
        names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(str(n).split(".")[0] == "torch" for n in names)


def test_tiers_match_architectures():
    architectures = pytest.importorskip("crispr_deephf.architectures")
    assert benchmark.TIERS["Tier 2"] == architectures.TIER2_MODELS
    assert benchmark.TIERS["Tier 3"] == architectures.TIER3_MODELS
    assert benchmark.TIERS["Extras"] == architectures.EXTRA_MODELS


def test_plot_vs_paper_ylim_adapts_to_low_values_and_label_reports_floor():
    low = _agg([("CNN", HF1, "Test", 0.42, 0.03, 3), ("BiLSTM", HF1, "Test", 0.84, 0.01, 3)])
    fig = plots.plot_vs_paper(benchmark.scoreboard(low))
    lo, hi = fig.axes[0].get_ylim()
    assert lo < 0.42 - 0.03  # lowest bar and its error bar sit inside the frame
    assert hi >= 0.95
    label = fig.axes[0].get_ylabel()
    assert f"{lo:.2f}" in label and "0.60" not in label
    plt.close(fig)


def test_plot_vs_paper_default_window_when_data_is_high(agg):
    fig = plots.plot_vs_paper(benchmark.scoreboard(agg))
    lo, hi = fig.axes[0].get_ylim()
    assert lo == pytest.approx(0.60) and hi == pytest.approx(0.95)
    plt.close(fig)


def test_split_less_frame_raises_unless_assume_test():
    no_split = _agg([("CNN", WT, "Test", 0.84, 0.01, 3)]).drop(columns="Split")
    for fn in (benchmark.scoreboard, benchmark.best_per_target, benchmark.tier_best):
        with pytest.raises(ValueError, match="Split"):
            fn(no_split)
        assert len(fn(no_split, assume_test=True)) == 1
    with pytest.raises(ValueError, match="Split"):
        plots.plot_tier_progression(no_split)


def test_best_per_target_distinguishes_counterpart_and_names_comparison(agg):
    best = benchmark.best_per_target(agg)
    assert "Beats_paper" not in best.columns
    assert list(best.columns) == [
        "Target", "Best_model", "Best_has_paper_counterpart", "Ours_rho", "Ours_sd",
        "Paper_model", "Paper_rho", "Delta", "Ours_best_exceeds_paper_best"]
    assert not best["Best_has_paper_counterpart"].any()  # Transformer has no paper row
    assert best["Ours_best_exceeds_paper_best"].all()
    paired = _agg([("BiLSTM+bio", WT, "Test", 0.87, 0.001, 3)])
    assert bool(benchmark.best_per_target(paired).loc[0, "Best_has_paper_counterpart"])


def test_best_per_target_carries_selection_bias_note(agg):
    best = benchmark.best_per_target(agg)
    note = best.attrs["selection_note"]
    assert note == benchmark.format_selection_note(8)
    assert "post hoc" in note and "optimistically biased" in note and "8" in note
    assert "tenfold" not in note  # separate from the paper caveat


def test_sd_forced_nan_when_single_seed():
    one = _agg([("CNN", WT, "Test", 0.84, 0.0, 1), ("CNN", ESP, "Test", 0.84, 0.0, 3)])
    sb = benchmark.scoreboard(one)
    assert pd.isna(sb.loc[0, "Ours_sd"]) and sb.loc[1, "Ours_sd"] == 0.0


def test_format_scoreboard_approx_rows_not_falsely_precise():
    rows = _agg([("Ridge", WT, "Test", 0.785, np.nan, 1),
                 ("Ridge", ESP, "Test", 0.80, np.nan, 1),
                 ("Ridge", HF1, "Test", 0.70, np.nan, 1),
                 ("BiLSTM+bio", WT, "Test", 0.8700, 0.001, 3)])
    shown = benchmark.format_scoreboard(benchmark.scoreboard(rows)).set_index(["Model", "Target"])
    assert shown.loc[("Ridge", WT), "Delta"] == "~+0.01"
    assert shown.loc[("Ridge", WT), "Beats paper"] == "within reading error"
    assert shown.loc[("Ridge", ESP), "Beats paper"] == "yes"
    assert shown.loc[("Ridge", HF1), "Beats paper"] == "no"
    assert shown.loc[("BiLSTM+bio", WT), "Delta"] == "+0.0030"


def test_plot_vs_paper_has_value_labels(agg):
    fig = plots.plot_vs_paper(benchmark.scoreboard(agg))
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert "0.870" in labels and "0.867" in labels
    plt.close(fig)


def test_caption_and_legend_do_not_overlap(agg):
    fig = plots.plot_vs_paper(benchmark.scoreboard(agg))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(r)
    caption = fig.texts[-1].get_window_extent(r)
    assert caption.y1 < legend.y0
    plt.close(fig)
