"""End-to-end smoke test: the whole pipeline on synthetic data, in quick mode, on CPU.

Every other test file exercises one module. This one is the only place where the
*pieces meet*: split -> features -> Tier 1 -> one deep model -> stack -> scoreboard
-> figures, in the same order and through the same calls the notebook makes.

It deliberately asserts on shape and wiring, not on quality: three epochs on 400
random guides cannot produce a meaningful rho, and a threshold on one would be a
test that fails for the wrong reason.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from crispr_deephf import benchmark, config, data, features, models

pytest.importorskip("torch")

from crispr_deephf import runner  # importing runner needs torch, hence after the skip

_QUICK = config.get_mode("quick")


def _prepared(synthetic_df: pd.DataFrame):
    """Split-labelled frame plus the 439-feature matrix, as the notebook builds them."""
    df = data.add_shared_split(data.add_split_labels(synthetic_df))
    X, names = features.extract_crispr_features(df["21mer"].tolist(), verbose=False)
    return df, X, names


def test_quick_pipeline_end_to_end(tmp_path, synthetic_df):
    """Clean -> split -> tier1 -> one deep model -> scoreboard, on synthetic data."""
    paths = config.get_paths(tmp_path)
    df, X, names = _prepared(synthetic_df)

    r1 = runner.run_tier1(df, X, names, paths=paths)
    r2 = runner.run_deep_model(
        df,
        model_name="CNN",
        target_cols=["WT_Efficiency"],
        split_col="WT_split",
        paths=paths,
        mode=_QUICK,
        seeds=[0],
        device="cpu",
    )

    sb = benchmark.scoreboard(runner.aggregate(runner.results_frame(r1 + r2)))
    assert len(sb) >= 2
    assert sb["Ours_rho"].notna().all()

    # Ridge is the only Tier-1 model guaranteed to be installed; the boosters (and so
    # the booster mean-Ensemble) are optional, so assert only on what actually ran.
    ran = set(sb["Model"])
    assert {"Ridge", "CNN"} <= ran
    assert ran <= {"Ridge", "LightGBM", "XGBoost", "CatBoost", "Ensemble", "CNN"}

    # Every installed Tier-1 model is scored on all three enzymes; the deep run was
    # asked for WT only.
    assert set(sb.loc[sb["Model"] == "Ridge", "Target"]) == set(config.TARGET_COLS)
    assert set(sb.loc[sb["Model"] == "CNN", "Target"]) == {"WT_Efficiency"}

    # The honesty guards survive the round trip: a single-seed run reports no s.d.,
    # and CNN is one of the rows the paper does report in its text.
    assert sb["Ours_sd"].isna().all()
    cnn = sb[sb["Model"] == "CNN"].iloc[0]
    assert cnn["Paper_source"] == "text"
    assert cnn["Paper_rho"] == pytest.approx(config.PAPER_SPEARMAN["CNN"]["WT"])


def test_quick_pipeline_stack_and_tier_tables(tmp_path, synthetic_df):
    """The notebook's section 10c + 12c/12d path: stack, then the tier/best tables."""
    paths = config.get_paths(tmp_path)
    df, X, names = _prepared(synthetic_df)

    results = runner.run_tier1(df, X, names, paths=paths)
    results += runner.run_deep_model(
        df,
        model_name="CNN",
        target_cols=["WT_Efficiency"],
        split_col="WT_split",
        paths=paths,
        mode=_QUICK,
        seeds=[0],
        device="cpu",
    )
    stacked = runner.run_stack(
        runner.load_results(paths.results),
        df,
        target_col="WT_Efficiency",
        paths=paths,
        mode_name="quick",
    )
    assert stacked["model_name"].startswith("Stack(")
    assert stacked["meta_fit_split"] == "val"

    agg = runner.aggregate(runner.results_frame(results + [stacked]))
    # The stack must reach the tier tables: its run name is "Stack(<tree>+<deep>)",
    # not the bare "Stack" that benchmark.TIERS lists.
    tiers = benchmark.tier_best(agg)
    assert "Tier 1" in set(tiers["Tier"])
    assert stacked["model_name"] in set(benchmark.scoreboard(agg)["Model"])
    assert benchmark.tier_of(stacked["model_name"]) == "Tier 1"
    wt_tier1 = tiers[(tiers["Tier"] == "Tier 1") & (tiers["Target"] == "WT_Efficiency")]
    assert len(wt_tier1) == 1

    best = benchmark.best_per_target(agg)
    assert set(best["Target"]) == set(config.TARGET_COLS)
    assert best["Paper_rho"].notna().all()


def test_quick_pipeline_resumes_without_retraining(tmp_path, synthetic_df):
    """A second identical call reuses the saved JSON instead of training again."""
    paths = config.get_paths(tmp_path)
    df = data.add_shared_split(data.add_split_labels(synthetic_df))
    kwargs = {
        "model_name": "CNN",
        "target_cols": ["WT_Efficiency"],
        "split_col": "WT_split",
        "paths": paths,
        "mode": _QUICK,
        "seeds": [0],
        "device": "cpu",
    }

    first = runner.run_deep_model(df, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a stale-result warning would mean no resume
        second = runner.run_deep_model(df, **kwargs)

    assert second[0]["duration_s"] == first[0]["duration_s"]
    assert second[0]["test"] == first[0]["test"]
    assert len(runner.load_results(paths.results)) == 1


def test_quick_pipeline_figures_are_warning_free(tmp_path, synthetic_df):
    """The notebook's figure calls must be silent (seaborn palette/hue, pandas, ...)."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from crispr_deephf import plots

    paths = config.get_paths(tmp_path)
    df, X, names = _prepared(synthetic_df)
    long = runner.results_frame(runner.run_tier1(df, X, names, paths=paths))
    agg = runner.aggregate(long)
    sb = benchmark.scoreboard(agg)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plots.set_style()
        plots.plot_efficiency_distributions(df)
        plots.plot_feature_importance(list(names), X.sum(axis=0), "smoke", top_n=10)
        plots.plot_model_comparison(long[long["Split"] == "Test"])
        plots.plot_vs_paper(sb)
        plots.plot_tier_progression(agg)
    plt.close("all")


def test_scoreboard_handles_a_run_without_the_ensemble(tmp_path, synthetic_df, monkeypatch):
    """Ridge alone (no boosters) still yields a scoreboard and no 'Ensemble' row.

    Section 12e of the notebook reads one model's saved ``predictions`` by name, and
    the ``Ensemble`` exists only when at least two boosters are installed, so the
    no-Ensemble case has to be a supported state rather than an accident of whichever
    machine happens to run this.
    """
    paths = config.get_paths(tmp_path)
    df, X, names = _prepared(synthetic_df)
    monkeypatch.setattr(models, "available_trainers", lambda: {"Ridge": models.train_ridge})

    results = runner.run_tier1(df, X, names, paths=paths)

    assert {r["model_name"] for r in results} == {"Ridge"}
    sb = benchmark.scoreboard(runner.aggregate(runner.results_frame(results)))
    assert "Ensemble" not in set(sb["Model"])
    saved = runner.load_results(paths.results)
    assert not [r for r in saved if r["model_name"] == "Ensemble"]
    assert all(isinstance(r.get("predictions"), dict) for r in saved)


def test_results_frame_is_empty_without_results(tmp_path):
    """The notebook's ``HAVE_RESULTS`` guard: an empty results dir is not an error."""
    paths = config.get_paths(tmp_path)
    assert runner.load_results(paths.results) == []
    long = runner.results_frame([])
    assert long.empty
    assert isinstance(long, pd.DataFrame)
