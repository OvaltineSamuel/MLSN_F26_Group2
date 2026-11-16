"""Tests for the masked multi-target training loop (CPU, tiny, never a real schedule)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from crispr_deephf import architectures, config, data, models, torch_data, train_torch


def _loaders(df, target_cols, batch_size=32):
    df = data.add_shared_split(df)
    splits, _ = torch_data.build_splits(
        df, target_cols=target_cols, split_col="shared_split", bio=False, allow_missing=True
    )
    return {
        name: torch_data.make_loader(sp, batch_size=batch_size, shuffle=name == "train", seed=0)
        for name, sp in splits.items()
    }


# --------------------------------------------------------------------------- #
# safe_spearman
# --------------------------------------------------------------------------- #
def test_safe_spearman_on_constant_prediction():
    assert models.safe_spearman(np.array([1.0, 2, 3]), np.array([0.5, 0.5, 0.5])) == 0.0


def test_safe_spearman_on_constant_truth():
    assert models.safe_spearman(np.array([0.5, 0.5, 0.5]), np.array([1.0, 2, 3])) == 0.0


def test_safe_spearman_matches_scipy_when_defined():
    a, b = np.array([1.0, 2, 3, 4]), np.array([1.0, 3, 2, 4])
    assert models.safe_spearman(a, b) == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# masked_mse
# --------------------------------------------------------------------------- #
def test_masked_mse_ignores_unmeasured_entries():
    pred = torch.tensor([[0.0, 9.9]])
    y = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert train_torch.masked_mse(pred, y, mask).item() == pytest.approx(0.0)


def test_masked_mse_divides_by_measured_count_not_numel():
    # 4 entries, 2 measured with squared errors 1 and 9 -> 10 / 2 = 5 (numel would give 2.5)
    pred = torch.tensor([[1.0, 5.0], [3.0, 7.0]])
    y = torch.zeros(2, 2)
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert train_torch.masked_mse(pred, y, mask).item() == pytest.approx(5.0)


def test_masked_mse_all_unmeasured_is_zero_with_grad():
    pred = torch.rand(3, 2, requires_grad=True)
    loss = train_torch.masked_mse(pred, torch.zeros(3, 2), torch.zeros(3, 2))
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad == 0).all()


# --------------------------------------------------------------------------- #
# set_seed
# --------------------------------------------------------------------------- #
def test_set_seed_makes_init_reproducible():
    train_torch.set_seed(0)
    a = architectures.build_model("CNN")
    train_torch.set_seed(0)
    b = architectures.build_model("CNN")
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(p, q)


def _three_target_model():
    """Sequence-only 3-target CNN (skips the slow ViennaRNA bio features)."""
    return architectures.build_model("CNN", n_targets=3)


# --------------------------------------------------------------------------- #
# train_model / predict / evaluate_split
# --------------------------------------------------------------------------- #
def _train(
    df,
    tmp_path,
    name="CNN",
    target_cols=("WT_Efficiency",),
    epochs=2,
    seed=0,
    target_names=None,
    **overrides,
):
    loaders = _loaders(df, list(target_cols))
    train_torch.set_seed(seed)
    model = architectures.build_model(name, **overrides)
    res = train_torch.train_model(
        model,
        loaders["train"],
        loaders["val"],
        epochs=epochs,
        patience=1,
        device="cpu",
        checkpoint_path=tmp_path / "best.pt",
        target_names=target_names,
        verbose=False,
    )
    return res, loaders


def test_training_runs_and_records_history(synthetic_df, tmp_path):
    res, _ = _train(synthetic_df, tmp_path)
    assert 1 <= len(res.history) <= 2
    assert {"epoch", "train_loss", "val_loss", "val_rho"} <= set(res.history[0])
    assert np.isfinite(res.best_val_rho)
    assert 1 <= res.best_epoch <= len(res.history)


def test_best_weights_are_restored_in_memory(synthetic_df, tmp_path):
    res, loaders = _train(synthetic_df, tmp_path, epochs=3)
    # the returned model's val rho equals the recorded best, not the last epoch's
    y_pred, y_true, mask = train_torch.predict(res.model, loaders["val"], "cpu")
    rho = models.safe_spearman(y_true[mask[:, 0] == 1, 0], y_pred[mask[:, 0] == 1, 0])
    assert rho == pytest.approx(res.best_val_rho, abs=1e-6)
    saved = torch.load(tmp_path / "best.pt")
    for k, v in res.model.state_dict().items():
        assert torch.equal(v, saved[k])


def test_training_is_reproducible(synthetic_df, tmp_path):
    a, _ = _train(synthetic_df, tmp_path, seed=3)
    b, _ = _train(synthetic_df, tmp_path, seed=3)
    assert [h["train_loss"] for h in a.history] == [h["train_loss"] for h in b.history]


def test_multitask_training_runs(synthetic_df, tmp_path):
    res, _ = _train(
        synthetic_df, tmp_path, name="CNN", target_cols=tuple(config.TARGET_COLS), n_targets=3
    )
    assert np.isfinite(res.best_val_rho)


def test_predict_shapes_and_mask(synthetic_df):
    loaders = _loaders(synthetic_df, list(config.TARGET_COLS))
    model = _three_target_model()
    y_pred, y_true, mask = train_torch.predict(model, loaders["test"], "cpu")
    assert y_pred.shape == y_true.shape == mask.shape
    assert y_pred.shape[1] == 3
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_evaluate_split_returns_one_row_per_target(synthetic_df):
    loaders = _loaders(synthetic_df, list(config.TARGET_COLS))
    model = _three_target_model()
    rows = train_torch.evaluate_split(
        model, loaders["test"], "cpu", list(config.TARGET_COLS), model_name="MT", split_name="Test"
    )
    assert len(rows) == 3
    assert all("Spearman_rho" in r for r in rows)
    assert [r["Target"] for r in rows] == list(config.TARGET_COLS)


def test_evaluate_split_scores_only_measured_entries(synthetic_df):
    loaders = _loaders(synthetic_df, list(config.TARGET_COLS))
    model = _three_target_model()
    y_pred, y_true, mask = train_torch.predict(model, loaders["test"], "cpu")
    rows = train_torch.evaluate_split(model, loaders["test"], "cpu", list(config.TARGET_COLS))
    for t, row in enumerate(rows):
        keep = mask[:, t] == 1.0
        expected = models.compute_metrics(y_true[keep, t], y_pred[keep, t])
        assert row["RMSE"] == pytest.approx(expected["RMSE"])


@pytest.mark.parametrize("n_keep", [0, 1])
def test_evaluate_split_tiny_target_gives_nan_row_and_warns(synthetic_df, n_keep):
    loaders = _loaders(synthetic_df, list(config.TARGET_COLS))
    loader = loaders["test"]
    ds = loader.dataset
    mask = ds.tensors[3].clone()
    mask[:, 1] = 0.0
    mask[:n_keep, 1] = 1.0  # exactly n_keep measured entries for target 1
    ds.tensors = (*ds.tensors[:3], mask)
    model = _three_target_model()
    with pytest.warns(UserWarning, match="eSpCas9_Efficiency"):
        rows = train_torch.evaluate_split(model, loader, "cpu", list(config.TARGET_COLS))
    assert len(rows) == 3
    assert rows[1]["n_measured"] == n_keep
    assert all(np.isnan(rows[1][k]) for k in ("Pearson_r", "Spearman_rho", "RMSE", "MAE", "R2"))
    assert rows[0]["n_measured"] == int(mask[:, 0].sum())
    assert np.isfinite(rows[0]["Spearman_rho"])


def test_history_records_per_target_rho(synthetic_df, tmp_path):
    names = tuple(config.TARGET_COLS)
    res, _ = _train(synthetic_df, tmp_path, target_cols=names, n_targets=3, target_names=names)
    row = res.history[0]
    assert set(row["val_rho_per_target"]) == set(names)
    assert row["val_rho"] == pytest.approx(np.mean(list(row["val_rho_per_target"].values())))


def test_history_per_target_single_target(synthetic_df, tmp_path):
    res, _ = _train(synthetic_df, tmp_path, target_names=["WT_Efficiency"])
    assert list(res.history[0]["val_rho_per_target"]) == ["WT_Efficiency"]
