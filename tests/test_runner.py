"""Tests for the resumable experiment runner: atomic saves, resume, leak guard, stacking."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from crispr_deephf import architectures, config, data, models, runner

WT = "WT_Efficiency"
ALL_TARGETS = list(config.TARGET_COLS)


@pytest.fixture
def split_df(synthetic_df):
    """Synthetic table with the per-enzyme split columns and ``shared_split``."""
    return data.add_shared_split(data.add_split_labels(synthetic_df))


@pytest.fixture
def paths(tmp_path):
    return config.get_paths(tmp_path)


def _quick_cnn(df, paths, **overrides):
    kwargs = {
        "model_name": "CNN",
        "target_cols": [WT],
        "split_col": "WT_split",
        "paths": paths,
        "mode": "quick",
        "seeds": [0],
        "device": "cpu",
    }
    return runner.run_deep_model(df, **{**kwargs, **overrides})


@pytest.fixture
def no_training(monkeypatch):
    """Make any call to train_model fail: proves a rerun retrained nothing."""

    def boom(*args, **kwargs):
        raise AssertionError("train_model was called but nothing should be retrained")

    monkeypatch.setattr(runner, "train_model", boom)


# --------------------------------------------------------------------------- #
# Keys and atomic persistence
# --------------------------------------------------------------------------- #
def test_result_key_format_and_filesystem_safety():
    assert runner.result_key("CNN", "WT", 2) == "CNN__WT__seed2"
    assert runner.result_key("MultiTask-BiLSTM+bio", "MULTI", 0) == (
        "MultiTask-BiLSTM+bio__MULTI__seed0"
    )
    key = runner.result_key("a/b c", "WT", 1)
    assert not set(key) & set('/\\ :*?"<>|')


def test_save_is_atomic_and_roundtrips(tmp_path):
    p = runner.save_result(tmp_path, "m__WT__seed0", {"model_name": "m", "Spearman_rho": 0.5})
    assert p.exists() and not list(tmp_path.glob("*.tmp*"))
    assert runner.load_results(tmp_path)[0]["Spearman_rho"] == 0.5


def test_save_handles_numpy_values(tmp_path):
    payload = {"model_name": "m", "x": np.float32(0.25), "n": np.int64(3), "a": np.arange(2)}
    runner.save_result(tmp_path, "k", payload)
    assert runner.load_results(tmp_path) == [{"model_name": "m", "x": 0.25, "n": 3, "a": [0, 1]}]


def test_corrupt_result_is_skipped_not_fatal(tmp_path):
    runner.save_result(tmp_path, "good__WT__seed0", {"model_name": "good"})
    (tmp_path / "broken__WT__seed0.json").write_text('{"model_name": "bro')
    with pytest.warns(UserWarning, match="broken__WT__seed0"):
        loaded = runner.load_results(tmp_path)
    assert [r["model_name"] for r in loaded] == ["good"]


def test_killed_write_leaves_previous_result_intact_and_no_temp(tmp_path, monkeypatch):
    runner.save_result(tmp_path, "k", {"model_name": "old"})

    def killed(src, dst):
        raise KeyboardInterrupt  # the process dies after the temp file is written

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        runner.save_result(tmp_path, "k", {"model_name": "new"})
    monkeypatch.undo()
    assert not list(tmp_path.glob("*.tmp*"))
    assert [r["model_name"] for r in runner.load_results(tmp_path)] == ["old"]


def test_stale_temp_file_from_a_killed_session_is_never_loaded(tmp_path):
    (tmp_path / "k.abc123.tmp").write_text('{"model_name": "half')
    assert runner.load_results(tmp_path) == []


# --------------------------------------------------------------------------- #
# The data-leak guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split_col", ["WT_split", "eSp_split", "HF1_split"])
def test_multi_target_on_per_enzyme_split_is_refused(split_df, paths, no_training, split_col):
    with pytest.raises(ValueError, match="data leak"):
        runner.run_deep_model(
            split_df,
            model_name="MultiTask-BiLSTM+bio",
            target_cols=ALL_TARGETS,
            split_col=split_col,
            paths=paths,
            mode="quick",
            seeds=[0],
            device="cpu",
        )
    assert not list(paths.results.glob("*.json"))  # refused before anything was written


def test_two_targets_on_per_enzyme_split_is_refused_for_any_model(split_df, paths, no_training):
    with pytest.raises(ValueError, match="shared_split"):
        _quick_cnn(split_df, paths, target_cols=[WT, "eSpCas9_Efficiency"], split_col="WT_split")


def test_multi_output_model_on_per_enzyme_split_is_refused_even_with_one_column(
    split_df, paths, no_training
):
    # n_targets=3 head, one target column, per-enzyme split: still the leak configuration
    with pytest.raises(ValueError, match="data leak"):
        _quick_cnn(split_df, paths, model_name="MultiTask-BiLSTM+bio", split_col="WT_split")


def test_multi_target_runs_on_shared_split(split_df, paths, monkeypatch):
    # A bio-free multi-task entry keeps the test off ViennaRNA.
    entry = (architectures.GuideBiLSTM, {"use_bio": False, "n_targets": 3, "rnn_layers": 1})
    monkeypatch.setitem(architectures.MODEL_REGISTRY, "MT-test", entry)
    out = _quick_cnn(
        split_df,
        paths,
        model_name="MT-test",
        target_cols=ALL_TARGETS,
        split_col=config.SHARED_SPLIT_COL,
    )
    assert len(out) == 1
    payload = out[0]
    assert payload["target"] == "MULTI" and payload["targets"] == ALL_TARGETS
    assert set(payload["test"]) == set(ALL_TARGETS)
    assert all(m["n_measured"] > 2 for m in payload["test"].values())
    assert (paths.results / "MT-test__MULTI__seed0.json").exists()
    # real target names in the history, not t0/t1/t2
    assert set(payload["history"][0]["val_rho_per_target"]) <= set(ALL_TARGETS)


# --------------------------------------------------------------------------- #
# Deep-model runs and resume
# --------------------------------------------------------------------------- #
def test_run_deep_model_payload_contents(split_df, paths):
    (payload,) = _quick_cnn(split_df, paths)
    for field in (
        "model_name", "target", "seed", "mode", "test", "val", "best_epoch", "history",
        "duration_s", "data_digest",
    ):  # fmt: skip
        assert field in payload
    assert payload["model_name"] == "CNN" and payload["target"] == "WT"
    assert len(payload["data_digest"]) == 12
    assert payload["mode"]["epochs"] == config.get_mode("quick")["epochs"]
    assert set(payload["test"]) == {WT}
    assert np.isfinite(payload["test"][WT]["Spearman_rho"])
    assert len(payload["history"]) >= 1 and WT in payload["history"][0]["val_rho_per_target"]
    # saved on disk as it finished, and the saved copy matches
    assert runner.load_results(paths.results)[0]["data_digest"] == payload["data_digest"]


def test_completed_seed_is_skipped_on_rerun(split_df, paths, monkeypatch):
    first = _quick_cnn(split_df, paths, seeds=[0, 1])
    assert [p["seed"] for p in first] == [0, 1]

    monkeypatch.setattr(
        runner,
        "train_model",
        lambda *a, **k: pytest.fail("a completed seed was retrained"),
    )
    second = _quick_cnn(split_df, paths, seeds=[0, 1])
    assert second == first  # identical payloads, loaded from disk


def test_resume_trains_only_the_missing_seed(split_df, paths, monkeypatch):
    _quick_cnn(split_df, paths, seeds=[0])
    calls = []
    real = runner.train_model
    monkeypatch.setattr(runner, "train_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    out = _quick_cnn(split_df, paths, seeds=[0, 1])
    assert len(calls) == 1 and [p["seed"] for p in out] == [0, 1]


def test_force_retrains(split_df, paths, monkeypatch):
    _quick_cnn(split_df, paths)
    calls = []
    real = runner.train_model
    monkeypatch.setattr(runner, "train_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    _quick_cnn(split_df, paths, force=True)
    assert len(calls) == 1


def test_corrupt_saved_result_is_retrained_with_a_warning(split_df, paths):
    _quick_cnn(split_df, paths)
    (paths.results / "CNN__WT__seed0.json").write_text('{"model_name": "CN')
    with pytest.warns(UserWarning, match="CNN__WT__seed0"):
        (payload,) = _quick_cnn(split_df, paths)
    assert np.isfinite(payload["test"][WT]["Spearman_rho"])
    assert json.loads((paths.results / "CNN__WT__seed0.json").read_text())["seed"] == 0


def test_result_from_another_mode_is_not_reused(split_df, paths, monkeypatch):
    _quick_cnn(split_df, paths)
    calls = []
    real = runner.train_model
    monkeypatch.setattr(runner, "train_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    custom = {**config.get_mode("quick"), "epochs": 2}
    with pytest.warns(UserWarning, match="retraining"):
        (payload,) = _quick_cnn(split_df, paths, mode=custom)
    assert len(calls) == 1 and payload["mode"]["epochs"] == 2


def test_bilstm_full_mode_uses_grid_searched_hyperparameters():
    s = runner._run_settings("BiLSTM", "full")
    assert (s["lr"], s["batch_size"], s["epochs"]) == (0.0005, 128, 20)
    assert s["patience"] == config.get_mode("full")["patience"]
    cnn = runner._run_settings("CNN", "full")
    assert (cnn["lr"], cnn["batch_size"], cnn["epochs"]) == (1e-3, 256, 60)
    assert runner._run_settings("BiLSTM", "quick")["epochs"] == 3  # quick stays quick


# --------------------------------------------------------------------------- #
# Tier-1 and stacking
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_boosters(monkeypatch):
    """Ridge plus two fake 'boosters' so the ensemble path runs without lightgbm."""
    trainers = {
        "Ridge": models.train_ridge,
        "LightGBM": lambda *a: models.train_ridge(*a, alpha=10.0),
        "XGBoost": lambda *a: models.train_ridge(*a, alpha=100.0),
    }
    monkeypatch.setattr(models, "available_trainers", lambda: dict(trainers))
    return trainers


@pytest.fixture
def features(split_df):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(len(split_df), 12))
    y = split_df[config.TARGET_COLS].fillna(0.5).mean(axis=1).to_numpy()
    X[:, 0] += 3 * y  # something learnable
    return X, [f"f{i}" for i in range(12)]


def test_run_tier1_saves_every_model_and_ensemble_and_resumes(
    split_df, paths, features, two_boosters, monkeypatch
):
    X, names = features
    out = runner.run_tier1(split_df, X, names, paths=paths)
    labels = {(r["model_name"], r["target"]) for r in out}
    assert labels == {
        (m, k) for m in ("Ridge", "LightGBM", "XGBoost", "Ensemble") for k in ("WT", "eSp", "HF1")
    }
    assert len(list(paths.results.glob("*.json"))) == 12

    ens = next(r for r in out if r["model_name"] == "Ensemble" and r["target"] == "WT")
    lgb = next(r for r in out if r["model_name"] == "LightGBM" and r["target"] == "WT")
    xgb = next(r for r in out if r["model_name"] == "XGBoost" and r["target"] == "WT")
    expected = np.mean([lgb["predictions"]["test"], xgb["predictions"]["test"]], axis=0)
    assert np.allclose(ens["predictions"]["test"], expected)

    def fail(*a, **k):
        pytest.fail("a completed Tier-1 model was retrained")

    monkeypatch.setitem(two_boosters, "Ridge", fail)
    monkeypatch.setitem(two_boosters, "LightGBM", fail)
    monkeypatch.setitem(two_boosters, "XGBoost", fail)
    again = runner.run_tier1(split_df, X, names, paths=paths)
    assert again == out


def test_run_stack_fits_meta_learner_on_validation_not_test(split_df, paths):
    frame = data.filter_for_target(split_df, WT)
    y_val = frame.loc[frame["WT_split"] == "val", WT].to_numpy()
    y_test = frame.loc[frame["WT_split"] == "test", WT].to_numpy()
    rng = np.random.default_rng(0)

    def payload(name, family, p_val, p_test, rho):
        return {
            "model_name": name, "family": family, "targets": [WT], "split_col": "WT_split",
            "data_digest": "abc", "val": {WT: {"Spearman_rho": rho}},
            "predictions": {"val": list(p_val), "test": list(p_test)},
        }  # fmt: skip

    # The tree is a perfect predictor on val but *anti*-correlated on test. A meta-learner
    # fitted on val learns a positive weight, so its test rho is strongly negative; one
    # fitted on test would learn the flipped weight and score positive.
    tree = payload("LightGBM", "tree", y_val, -y_test, 0.99)
    deep = payload("CNN", "deep", rng.random(len(y_val)), rng.random(len(y_test)), 0.1)
    out = runner.run_stack([tree, deep], split_df, target_col=WT, paths=paths)

    assert out["meta_fit_split"] == "val"
    assert out["components"] == ["LightGBM", "CNN"]
    assert out["test"][WT]["Spearman_rho"] < -0.5
    assert (paths.results / f"{runner.result_key(out['model_name'], 'WT', 0)}.json").exists()


def test_run_stack_chooses_the_best_model_by_validation_rho(split_df, paths):
    frame = data.filter_for_target(split_df, WT)
    n_val = int((frame["WT_split"] == "val").sum())
    n_test = int((frame["WT_split"] == "test").sum())

    def payload(name, family, rho):
        return {
            "model_name": name, "family": family, "targets": [WT], "split_col": "WT_split",
            "data_digest": "abc", "val": {WT: {"Spearman_rho": rho}},
            "predictions": {"val": [0.5] * n_val, "test": [0.5] * n_test},
        }  # fmt: skip

    runs = [
        payload("Ridge", "tree", 0.2),
        payload("LightGBM", "tree", 0.7),
        payload("CNN", "deep", 0.5),
        payload("BiLSTM", "deep", 0.6),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant predictions make rho NaN; irrelevant here
        out = runner.run_stack(runs, split_df, target_col=WT, paths=paths)
    assert out["components"] == ["LightGBM", "BiLSTM"]


def test_run_stack_rejects_results_from_different_data(split_df, paths):
    def payload(family, digest):
        return {
            "model_name": family, "family": family, "targets": [WT], "split_col": "WT_split",
            "data_digest": digest, "val": {WT: {"Spearman_rho": 0.5}},
            "predictions": {"val": [0.1], "test": [0.1]},
        }  # fmt: skip

    with pytest.raises(ValueError, match="different data"):
        runner.run_stack(
            [payload("tree", "aaa"), payload("deep", "bbb")], split_df, target_col=WT, paths=paths
        )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_results_frame_is_long_form():
    results = [
        {
            "model_name": "CNN", "seed": 1,
            "val": {WT: {"Spearman_rho": 0.7, "RMSE": 0.2}},
            "test": {WT: {"Spearman_rho": 0.8, "RMSE": 0.1}},
        },
        {"model_name": "no-metrics"},  # e.g. a payload without scores is ignored
    ]  # fmt: skip
    frame = runner.results_frame(results)
    assert list(frame["Split"]) == ["Val", "Test"]
    assert set(frame["Model"]) == {"CNN"} and set(frame["Target"]) == {WT}
    assert frame.loc[frame["Split"] == "Test", "Spearman_rho"].item() == 0.8
    assert {"Model", "Target", "Seed", "Split"} <= set(frame.columns)


def test_aggregate_reports_mean_and_sd():
    frame = pd.DataFrame(
        {
            "Model": ["CNN"] * 3,
            "Target": ["WT_Efficiency"] * 3,
            "Seed": [0, 1, 2],
            "Spearman_rho": [0.80, 0.82, 0.84],
        }
    )
    agg = runner.aggregate(frame)
    assert agg.loc[0, "Spearman_rho_mean"] == pytest.approx(0.82)
    assert agg.loc[0, "Spearman_rho_std"] == pytest.approx(0.02)
    assert agg.loc[0, "n_seeds"] == 3


def test_aggregate_column_names_are_the_contract_task_9_reads():
    frame = pd.DataFrame(
        {
            "Model": ["CNN"] * 2,
            "Target": [WT] * 2,
            "Seed": [0, 1],
            "Split": ["Test"] * 2,
            **{m: [0.5, 0.6] for m in runner.METRIC_COLS},
        }
    )
    expected = ["Model", "Target", "Split", "n_seeds"]
    for m in ("Pearson_r", "Spearman_rho", "RMSE", "MAE", "R2"):
        expected += [f"{m}_mean", f"{m}_std"]
    assert list(runner.aggregate(frame).columns) == expected


def test_aggregate_does_not_mix_val_and_test_rows():
    frame = pd.DataFrame(
        {
            "Model": ["CNN"] * 4,
            "Target": [WT] * 4,
            "Seed": [0, 1, 0, 1],
            "Split": ["Val", "Val", "Test", "Test"],
            "Spearman_rho": [0.1, 0.1, 0.9, 0.9],
        }
    )
    agg = runner.aggregate(frame).set_index("Split")
    assert agg.loc["Val", "Spearman_rho_mean"] == pytest.approx(0.1)
    assert agg.loc["Test", "Spearman_rho_mean"] == pytest.approx(0.9)
    assert (agg["n_seeds"] == 2).all()
