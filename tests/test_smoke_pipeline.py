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
from pathlib import Path

import numpy as np
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
    """Split -> features -> tier1 -> one deep model -> scoreboard, on synthetic data.

    The fixture is already clean, so cleaning itself is not in this path; the data contract
    and PAM repair are covered by ``tests/test_data.py``.
    """
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
    stack_name = stacked["model_name"]
    assert stack_name in set(benchmark.scoreboard(agg)["Model"])
    assert benchmark.tier_of(stack_name) == "Tier 1"

    # The stack must *reach* the tier tables, not merely be in a tier: its run name is
    # "Stack(<tree>+<deep>)", not the bare "Stack" that benchmark.TIERS lists, and literal
    # matching dropped it from tier_best entirely. Restricting the frame to the stack alone
    # makes the assertion independent of whether it happens to beat the trees on this data.
    stack_only = benchmark.tier_best(agg[agg["Model"] == stack_name])
    assert list(stack_only["Tier"]) == ["Tier 1"]
    assert list(stack_only["Model"]) == [stack_name]

    tiers = benchmark.tier_best(agg)
    wt_tier1 = tiers[(tiers["Tier"] == "Tier 1") & (tiers["Target"] == "WT_Efficiency")]
    assert len(wt_tier1) == 1
    # Whoever wins Tier 1 must be the best of a candidate pool that includes the stack.
    pool = {m for m in set(agg["Model"]) if benchmark.tier_of(m) == "Tier 1"}
    assert stack_name in pool
    wt_test = agg[(agg["Split"] == "Test") & (agg["Target"] == "WT_Efficiency")]
    candidates = wt_test[wt_test["Model"].isin(pool)]
    best = candidates.loc[candidates["Spearman_rho_mean"].idxmax(), "Model"]
    assert wt_tier1["Model"].iloc[0] == best

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


# --------------------------------------------------------------------------- #
# Section 12e: predictions are paired with the dataframe by guide id
# --------------------------------------------------------------------------- #
def _notebook_pairing_helper(namespace: dict):
    """Exec only the section-12e helper *definition* from the notebook and return it.

    The function is run as the notebook defines it rather than re-implemented here, so this
    test cannot pass against a copy that has drifted from the cell the team actually runs.
    Everything below the definition (the plotting and saving) is cut off.
    """
    nbformat = pytest.importorskip("nbformat")
    path = Path(__file__).resolve().parents[1] / "notebooks" / "crispr_deephf_pipeline.ipynb"
    nb = nbformat.read(path, as_version=4)
    src = next(
        c.source
        for c in nb.cells
        if c.cell_type == "code" and "def saved_test_predictions" in c.source
    )
    head = src.split("y_true, y_pred, seeds = saved_test_predictions(")[0]
    exec(compile(head, "<notebook section 12e>", "exec"), namespace)  # noqa: S102
    return namespace["saved_test_predictions"]


@pytest.fixture
def pairing_namespace(tmp_path, synthetic_df):
    """Tier-1 results on disk plus the globals the notebook's helper closes over."""
    paths = config.get_paths(tmp_path)
    df, X, names = _prepared(synthetic_df)
    runner.run_tier1(df, X, names, paths=paths)
    return {
        "config": config,
        "data": data,
        "runner": runner,
        "np": np,
        "df": df,
        "ACTIVE_TARGETS": list(config.TARGET_COLS),
        "saved": runner.load_results(paths.results),
        "split_col_for": lambda t: f"{config.TARGET_KEYS[t]}_split",
    }


def test_notebook_diagnostics_pairs_predictions_by_guide_id(pairing_namespace):
    """The happy path: ids match, so the panel gets the frame's own test rows."""
    fn = _notebook_pairing_helper(pairing_namespace)
    y_true, y_pred, seeds = fn("Ridge")
    assert set(y_true) == set(config.TARGET_COLS)
    assert set(seeds) == set(config.TARGET_COLS)
    for target in config.TARGET_COLS:
        frame = data.filter_for_target(pairing_namespace["df"], target)
        want = frame.loc[frame[f"{config.TARGET_KEYS[target]}_split"] == "test", target]
        assert len(y_true[target]) == len(y_pred[target]) == len(want)
        assert y_true[target] == pytest.approx(want.to_numpy())


def test_notebook_diagnostics_refuses_reordered_ids(pairing_namespace, capsys):
    """The hazard: same length, wrong order.

    Zipping predictions against ``frame.loc[labels == "test"]`` positionally is a silent
    wrong number -- ``plots.plot_pred_vs_actual`` would compute r and rho from the shuffled
    pairing and print both in the panel titles of a figure saved for the slides. Task 8
    fixed this class of bug in ``run_stack`` by comparing ids; the cell must do the same.
    """
    fn = _notebook_pairing_helper(pairing_namespace)
    for payload in pairing_namespace["saved"]:
        if payload["model_name"] == "Ridge" and payload["target"] == "WT":
            ids = payload["predictions"]["ids_test"]
            payload["predictions"]["ids_test"] = list(reversed(ids))
            assert len(payload["predictions"]["ids_test"]) == len(ids)  # length still matches
    assert fn("Ridge") == (None, None, None)
    assert "REFUSING" in capsys.readouterr().out


def test_notebook_diagnostics_refuses_a_foreign_data_digest(pairing_namespace, capsys):
    """A payload trained on another dataset must not be plotted against this frame."""
    fn = _notebook_pairing_helper(pairing_namespace)
    for payload in pairing_namespace["saved"]:
        if payload["model_name"] == "Ridge":
            payload["data_digest"] = "deadbeef1234"
    assert fn("Ridge") == (None, None, None)
    assert "different data" in capsys.readouterr().out


def test_notebook_diagnostics_refuses_a_payload_with_no_ids(pairing_namespace, capsys):
    """An older payload saved before ids were recorded must say so, not pair by position."""
    fn = _notebook_pairing_helper(pairing_namespace)
    for payload in pairing_namespace["saved"]:
        payload["predictions"].pop("ids_test", None)
    assert fn("Ridge") == (None, None, None)
    out = capsys.readouterr().out
    assert "REFUSING" in out and "FORCE_RERUN" in out


def test_notebook_diagnostics_follows_the_enzyme_switch(pairing_namespace):
    """With WT only active the helper reads, pairs and returns WT alone."""
    pairing_namespace["ACTIVE_TARGETS"] = ["WT_Efficiency"]
    fn = _notebook_pairing_helper(pairing_namespace)
    y_true, y_pred, seeds = fn("Ridge")
    assert set(y_true) == set(y_pred) == set(seeds) == {"WT_Efficiency"}


# --------------------------------------------------------------------------- #
# One enzyme end to end: the WT-only switch
# --------------------------------------------------------------------------- #
def test_single_enzyme_pipeline_reports_and_plots_only_that_enzyme(tmp_path, synthetic_df):
    """Tier 1 -> deep -> stack -> scoreboard/best/tier tables -> every figure, WT only."""
    import matplotlib

    matplotlib.use("Agg")
    from crispr_deephf import plots

    wt = ["WT_Efficiency"]
    paths = config.get_paths(tmp_path)
    df, X, names = _prepared(synthetic_df)
    results = runner.run_tier1(df, X, names, paths=paths, targets=wt)
    assert {r["target"] for r in results} == {"WT"}, "an unselected enzyme was trained"
    assert not [p for p in paths.results.glob("*.json") if "_eSp_" in p.name or "_HF1_" in p.name]
    results += runner.run_deep_model(
        df, model_name="CNN", target_cols=wt, split_col="WT_split", paths=paths,
        mode=_QUICK, seeds=[0], device="cpu",
    )
    stacked = runner.run_stack(
        runner.load_results(paths.results), df, target_col=wt[0], paths=paths, mode_name="quick"
    )
    agg = runner.aggregate(runner.results_frame(results + [stacked]))

    sb = benchmark.scoreboard(agg)
    assert set(sb["Target"]) == set(wt)
    assert list(benchmark.best_per_target(agg)["Target"]) == wt
    assert set(benchmark.tier_best(agg)["Target"]) == set(wt)
    assert benchmark.format_caveat(1) and benchmark.format_selection_note(len(set(sb["Model"])))

    assert len(plots.plot_vs_paper(sb, targets=wt).axes) == 1
    assert len(plots.plot_tier_progression(agg, targets=wt).axes) == 1
    y = df["WT_Efficiency"].dropna().to_numpy()[:50]
    one_panel = plots.plot_pred_vs_actual({wt[0]: y}, {wt[0]: y[::-1]}, targets=wt)
    assert len([a for a in one_panel.axes if a.get_xlabel()]) == 1   # (+ a colorbar axis)
    assert len(plots.plot_residuals({wt[0]: y - y[::-1]}, targets=wt).axes) == 1
    assert len(plots.plot_vs_paper(sb).axes) == 3  # the default is unchanged


def test_run_tier1_and_the_plots_reject_a_target_that_is_not_one():
    from crispr_deephf import plots

    with pytest.raises(ValueError, match="subset"):
        plots._resolve_targets(["WT"])           # a key, not a column
    with pytest.raises(ValueError, match="subset"):
        plots._resolve_targets([])
    with pytest.raises(ValueError, match="subset"):
        runner.run_tier1(None, None, [], paths=None, targets=["nope"])  # type: ignore[arg-type]
