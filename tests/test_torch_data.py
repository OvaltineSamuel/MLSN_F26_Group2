"""Tests for the torch data layer (CPU only; bio folding is kept to a handful of guides)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from crispr_deephf import bio_features, config, data, torch_data


def _fake_bio_matrix(seqs, cache_path=None, *, verbose=True):
    """Cheap deterministic stand-in for ViennaRNA features, distinct per row."""
    n = len(seqs)
    rng = np.random.default_rng(7)
    glob = rng.normal(size=(n, 37)).astype(np.float32) + np.arange(n)[:, None] * 0.1
    chans = rng.normal(size=(n, 3, 21)).astype(np.float32)
    return glob, chans


def test_one_hot_shape_and_sum(synthetic_seqs):
    x = torch_data.one_hot_encode(synthetic_seqs[:5])
    assert x.shape == (5, 4, 21)
    assert x.dtype == np.float32
    assert np.allclose(x.sum(axis=1), 1.0)


def test_one_hot_orientation_is_acgt_channel_first():
    x = torch_data.one_hot_encode(["ACGT" * 5 + "A"])
    assert x[0, 0, 0] == 1.0 and x[0, 1, 1] == 1.0
    assert x[0, 2, 2] == 1.0 and x[0, 3, 3] == 1.0


def test_one_hot_rejects_bad_input():
    with pytest.raises(AssertionError):
        torch_data.one_hot_encode(["ACGT"])
    with pytest.raises(AssertionError):
        torch_data.one_hot_encode(["N" * 21])


def test_check_integrity_returns_digest(synthetic_df):
    df = data.filter_for_target(data.add_split_labels(synthetic_df), "WT_Efficiency")
    digest = torch_data.check_integrity(df, "21mer", "WT_split", ["WT_Efficiency"])
    assert isinstance(digest, str) and len(digest) == 12


def test_check_integrity_rejects_leaked_sequence(synthetic_df):
    df = data.add_split_labels(synthetic_df)
    df = data.filter_for_target(df, "WT_Efficiency")
    df.loc[0, "21mer"] = df.loc[len(df) - 1, "21mer"]
    df.loc[0, "WT_split"], df.loc[len(df) - 1, "WT_split"] = "train", "test"
    with pytest.raises(AssertionError, match="more than one split"):
        torch_data.check_integrity(df, "21mer", "WT_split", ["WT_Efficiency"])


def test_check_integrity_rejects_out_of_range_target(synthetic_df):
    df = data.filter_for_target(data.add_split_labels(synthetic_df), "WT_Efficiency")
    df.loc[0, "WT_Efficiency"] = 1.5
    with pytest.raises(AssertionError, match="outside"):
        torch_data.check_integrity(df, "21mer", "WT_split", ["WT_Efficiency"])


def test_check_integrity_rejects_bad_split_label(synthetic_df):
    df = data.filter_for_target(data.add_split_labels(synthetic_df), "WT_Efficiency")
    df.loc[0, "WT_split"] = "holdout"
    with pytest.raises(AssertionError, match="split labels"):
        torch_data.check_integrity(df, "21mer", "WT_split", ["WT_Efficiency"])


def test_check_integrity_rejects_nan_target_unless_multitask(synthetic_df):
    df = data.add_shared_split(synthetic_df)
    cols = list(config.TARGET_COLS)
    with pytest.raises(AssertionError, match="non-finite"):
        torch_data.check_integrity(df, "21mer", "shared_split", cols)
    torch_data.check_integrity(df, "21mer", "shared_split", cols, allow_missing=True)


def test_mask_marks_missing_targets(synthetic_df):
    df = data.add_shared_split(synthetic_df)
    splits, _ = torch_data.build_splits(
        df, target_cols=config.TARGET_COLS, split_col="shared_split", bio=False
    )
    tr = splits["train"]
    assert tr.mask.shape == tr.y.shape
    assert set(np.unique(tr.mask)) <= {0.0, 1.0}
    assert np.isfinite(tr.y).all()  # NaN targets zero-filled, not NaN
    assert tr.mask.sum() < tr.mask.size  # synthetic data has missing values


def test_mask_matches_source_and_unmeasured_y_is_zero(synthetic_df):
    df = data.add_shared_split(synthetic_df)
    splits, _ = torch_data.build_splits(
        df, target_cols=config.TARGET_COLS, split_col="shared_split", bio=False
    )
    by_id = df.set_index("gRNA_ID")
    for sp in splits.values():
        assert np.isfinite(sp.y).all()
        src = by_id.loc[sp.ids, list(config.TARGET_COLS)].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(sp.mask == 1.0, ~np.isnan(src))
        assert (sp.y[sp.mask == 0.0] == 0.0).all()
        np.testing.assert_allclose(sp.y[sp.mask == 1.0], src[~np.isnan(src)], rtol=1e-6)


def test_split_sizes_follow_frozen_labels(synthetic_df):
    df = data.add_shared_split(synthetic_df)
    splits, scaler = torch_data.build_splits(
        df, target_cols=["WT_Efficiency"], split_col="shared_split", bio=False
    )
    for name in ("train", "val", "test"):
        assert splits[name].x.shape[0] == int((df["shared_split"] == name).sum())
    assert splits["train"].x.shape[0] > splits["test"].x.shape[0]
    assert scaler is None
    assert splits["train"].bio.shape == (splits["train"].x.shape[0], 0)


def test_standardizer_fit_on_train_only(synthetic_df, monkeypatch):
    monkeypatch.setattr(torch_data, "bio_matrix", _fake_bio_matrix)
    df = data.add_shared_split(synthetic_df)
    splits, scaler = torch_data.build_splits(
        df, target_cols=["WT_Efficiency"], split_col="shared_split", bio=True
    )
    glob, chans = _fake_bio_matrix(df["21mer"].tolist())
    raw = bio_features.flatten_bio(glob, chans)
    is_train = (df["shared_split"] == "train").to_numpy()
    np.testing.assert_allclose(scaler.mean, raw[is_train].mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(scaler.std, raw[is_train].std(axis=0), rtol=1e-5)
    # a fit that had seen val/test rows would give a different mean
    assert not np.allclose(scaler.mean, raw.mean(axis=0), rtol=1e-5)
    tr = splits["train"].bio
    assert tr.shape[1] == bio_features.N_BIO_FEATURES
    np.testing.assert_allclose(tr.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(tr.std(axis=0), 1.0, atol=1e-4)
    assert np.abs(splits["test"].bio.mean(axis=0)).max() > 1e-3  # not re-centred on test


def test_loader_yields_x_bio_y_mask_in_order(synthetic_df):
    df = data.add_shared_split(synthetic_df)
    splits, _ = torch_data.build_splits(
        df, target_cols=config.TARGET_COLS, split_col="shared_split", bio=False
    )
    loader = torch_data.make_loader(splits["train"], batch_size=32, shuffle=True, seed=1)
    x, bio, y, mask = next(iter(loader))
    assert x.shape == (32, 4, 21) and x.dtype == torch.float32
    assert bio.shape == (32, 0) and bio.dtype == torch.float32
    assert y.shape == (32, 3) and mask.shape == (32, 3)
    assert y.dtype == torch.float32 and mask.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_loader_shuffle_is_seeded(synthetic_df):
    df = data.add_shared_split(synthetic_df)
    splits, _ = torch_data.build_splits(
        df, target_cols=["WT_Efficiency"], split_col="shared_split", bio=False
    )

    def first(seed):
        loader = torch_data.make_loader(splits["train"], batch_size=16, shuffle=True, seed=seed)
        return next(iter(loader))[0]

    assert torch.equal(first(3), first(3))
    assert not torch.equal(first(3), first(4))


@pytest.mark.skipif(
    not (bio_features.VIENNA_AVAILABLE and bio_features.BIOPYTHON_AVAILABLE),
    reason="ViennaRNA / Biopython not installed",
)
def test_real_bio_features_on_a_few_guides(synthetic_df, tmp_path):
    df = synthetic_df.head(16).copy()
    df["shared_split"] = ["train"] * 8 + ["val"] * 4 + ["test"] * 4
    splits, scaler = torch_data.build_splits(
        df,
        target_cols=config.TARGET_COLS,
        split_col="shared_split",
        bio=True,
        cache_path=tmp_path / "bio.npz",
    )
    assert splits["train"].bio.shape == (8, bio_features.N_BIO_FEATURES)
    assert scaler is not None and scaler.mean.shape == (bio_features.N_BIO_FEATURES,)
    assert np.isfinite(splits["test"].bio).all()
