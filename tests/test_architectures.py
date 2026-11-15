"""Tests for the PyTorch architectures (CPU only, batches of 2-4, no training loops)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from crispr_deephf import architectures, bio_features

REGISTRY_NAMES = list(architectures.MODEL_REGISTRY)


def _batch(model, n: int = 4):
    """Return ``(x, bio)`` shaped for ``model``; ``bio`` is None for sequence-only models."""
    x = torch.rand(n, 4, bio_features.SEQ_LEN)
    bio = torch.rand(n, model.cfg.n_bio_features) if model.cfg.use_bio else None
    return x, bio


# --------------------------------------------------------------------------- #
# Contract shared by every registered model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", REGISTRY_NAMES)
def test_forward_shape_and_range(name):
    model = architectures.build_model(name)
    x, bio = _batch(model)
    out = model(x, bio)
    assert out.shape == (4, model.cfg.n_targets)
    assert ((out >= 0) & (out <= 1)).all()


@pytest.mark.parametrize("name", REGISTRY_NAMES)
def test_backward_reaches_every_parameter(name):
    model = architectures.build_model(name)
    x, bio = _batch(model)
    model(x, bio).sum().backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused


@pytest.mark.parametrize("name", REGISTRY_NAMES)
def test_forward_is_finite_on_batch_of_two(name):
    model = architectures.build_model(name)
    x, bio = _batch(model, n=2)
    assert torch.isfinite(model(x, bio)).all()


# --------------------------------------------------------------------------- #
# Registry, grouping and configuration
# --------------------------------------------------------------------------- #
def test_multitask_has_three_outputs():
    assert architectures.build_model("MultiTask-BiLSTM+bio").cfg.n_targets == 3


def test_single_task_entries_have_one_output():
    for name in REGISTRY_NAMES:
        if name != "MultiTask-BiLSTM+bio":
            assert architectures.build_model(name).cfg.n_targets == 1


def test_registry_bio_flags():
    expected = {
        "LSTM": False,
        "BiLSTM": False,
        "CNN": False,
        "BiLSTM+bio": True,
        "CNN+bio": True,
        "CNN-BiLSTM": True,
        "Transformer": True,
        "MultiTask-BiLSTM+bio": True,
    }
    actual = {name: architectures.build_model(name).cfg.use_bio for name in REGISTRY_NAMES}
    assert actual == expected


def test_groups_cover_registry():
    grouped = set(
        architectures.TIER2_MODELS + architectures.TIER3_MODELS + architectures.EXTRA_MODELS
    )
    assert grouped == set(architectures.MODEL_REGISTRY)


def test_groups_are_disjoint():
    grouped = (
        architectures.TIER2_MODELS + architectures.TIER3_MODELS + architectures.EXTRA_MODELS
    )
    assert len(grouped) == len(set(grouped)) == len(architectures.MODEL_REGISTRY)


def test_build_model_rejects_unknown_name():
    with pytest.raises(KeyError):
        architectures.build_model("NotAModel")


def test_build_model_overrides_reach_config():
    model = architectures.build_model("CNN", dropout=0.1, hidden_dims=(8,))
    assert model.cfg.dropout == 0.1
    assert model.cfg.hidden_dims == (8,)


def test_default_bio_width_matches_bio_features():
    assert architectures.ModelConfig().n_bio_features == bio_features.N_BIO_FEATURES


# --------------------------------------------------------------------------- #
# Bio-branch behaviour
# --------------------------------------------------------------------------- #
def test_bio_model_rejects_missing_bio():
    with pytest.raises(ValueError):
        architectures.build_model("CNN+bio")(torch.rand(2, 4, 21), None)


@pytest.mark.parametrize("name", ["BiLSTM+bio", "CNN-BiLSTM", "Transformer"])
def test_every_bio_model_rejects_missing_bio(name):
    with pytest.raises(ValueError):
        architectures.build_model(name)(torch.rand(2, 4, 21), None)


def test_sequence_only_model_has_no_bio_branch():
    model = architectures.build_model("CNN")
    assert model.bio_mlp is None
    assert torch.isfinite(model(torch.rand(2, 4, 21))).all()


# --------------------------------------------------------------------------- #
# The teammate-derived BiLSTM and CNN
# --------------------------------------------------------------------------- #
def test_bilstm_defaults_match_tuned_keras_architecture():
    cfg = architectures.build_model("BiLSTM").cfg
    assert (cfg.embed_dim, cfg.rnn_hidden, cfg.rnn_merge) == (64, 128, "sum")


def test_bilstm_merge_modes_both_run():
    for merge in ("sum", "concat"):
        model = architectures.build_model("BiLSTM", rnn_merge=merge)
        assert model(torch.rand(2, 4, 21)).shape == (2, 1)
    with pytest.raises(ValueError):
        architectures.build_model("BiLSTM", rnn_merge="average")


def test_cnn_pam_proximal_skip_can_be_disabled():
    model = architectures.build_model("CNN", pam_proximal_skip=0)
    assert model(torch.rand(2, 4, 21)).shape == (2, 1)


def test_cnn_matches_team_module_field_names():
    team_fields = {
        "kernel_widths",
        "channels_per_width",
        "bottleneck_channels",
        "bio_embed_dim",
        "hidden_dims",
        "dropout",
        "n_bio_features",
        "n_targets",
        "use_bio",
        "pam_proximal_skip",
    }
    assert team_fields <= set(architectures.ModelConfig.__dataclass_fields__)
