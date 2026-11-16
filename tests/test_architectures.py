"""Tests for the PyTorch architectures (CPU only, batches of 2-4, no training loops)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from crispr_deephf import architectures, bio_features

REGISTRY_NAMES = list(architectures.MODEL_REGISTRY)

# Derived from the registry, never hardcoded: a name added later is checked too.
BIO_NAMES = [n for n in REGISTRY_NAMES if architectures.build_model(n).cfg.use_bio]
RECURRENT_NAMES = [
    n
    for n in REGISTRY_NAMES
    if any(isinstance(m, torch.nn.LSTM) for m in architectures.build_model(n).modules())
]


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


@pytest.mark.parametrize("name", BIO_NAMES)
def test_every_bio_model_rejects_missing_bio(name):
    with pytest.raises(ValueError):
        architectures.build_model(name)(torch.rand(2, 4, 21), None)


def test_derived_name_lists_are_not_silently_empty():
    """Guard the derivations themselves, so a parametrization cannot vacuously pass."""
    assert len(BIO_NAMES) == 5
    assert len(RECURRENT_NAMES) == 5
    assert set(BIO_NAMES) | {"LSTM", "BiLSTM", "CNN"} == set(REGISTRY_NAMES)


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


def test_recurrent_entries_are_single_layer():
    """The registry must pin the tuned depth, whatever ``ModelConfig`` defaults to."""
    depths = {name: architectures.build_model(name).cfg.rnn_layers for name in RECURRENT_NAMES}
    assert depths == dict.fromkeys(RECURRENT_NAMES, 1)
    assert architectures.ModelConfig().rnn_layers == 2  # depth stays available to experiment with


def test_caller_overrides_beat_registry_overrides():
    assert architectures.build_model("BiLSTM", rnn_layers=2).cfg.rnn_layers == 2
    assert architectures.build_model("BiLSTM", use_bio=True).cfg.use_bio is True
    assert architectures.build_model("MultiTask-BiLSTM+bio", n_targets=1).cfg.n_targets == 1


def test_bilstm_merge_modes_both_run():
    for merge in ("sum", "concat"):
        model = architectures.build_model("BiLSTM", rnn_merge=merge)
        assert model(torch.rand(2, 4, 21)).shape == (2, 1)
    with pytest.raises(ValueError):
        architectures.build_model("BiLSTM", rnn_merge="average")


@pytest.mark.parametrize(("merge", "factor"), [("sum", 1), ("concat", 2)])
def test_head_width_follows_merge_mode(merge, factor):
    """A silent concat would double the hidden width and diverge from merge_mode='sum'."""
    model = architectures.build_model("BiLSTM", rnn_merge=merge)
    assert model.head[0].in_features == factor * model.cfg.rnn_hidden


def test_sum_merge_adds_the_two_directions():
    """``rnn_merge='sum'`` must really add the final states, not concatenate them."""
    cfg = architectures.ModelConfig(rnn_hidden=3, rnn_layers=1)
    h_n = torch.tensor([[[1.0, 2.0, 3.0]], [[10.0, 20.0, 30.0]]])  # (dirs, N=1, H=3)
    summed = architectures._final_state(h_n, cfg, bidirectional=True)
    assert torch.equal(summed, torch.tensor([[11.0, 22.0, 33.0]]))
    concat_cfg = architectures.ModelConfig(rnn_hidden=3, rnn_layers=1, rnn_merge="concat")
    concat = architectures._final_state(h_n, concat_cfg, bidirectional=True)
    assert torch.equal(concat, torch.tensor([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]]))


def test_cnn_pam_proximal_skip_can_be_disabled():
    model = architectures.build_model("CNN", pam_proximal_skip=0)
    assert model(torch.rand(2, 4, 21)).shape == (2, 1)


def test_multitask_leak_warning_is_documented():
    """The team's multi-output leak warning must stay findable from this module."""
    doc = architectures.ModelConfig.__doc__
    assert "data leak" in doc
    assert "shared_split" in doc


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
