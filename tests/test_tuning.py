"""Tests for the optional Optuna search: valid config keys, val-only objective, best-params file."""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")

from crispr_deephf import architectures, config, data, tuning

pytestmark = pytest.mark.skipif(not tuning.OPTUNA_AVAILABLE, reason="optuna not installed")

WT = "WT_Efficiency"
_VALID_KEYS = {f.name for f in dataclasses.fields(architectures.ModelConfig)} | {
    "lr",
    "weight_decay",
}


def _fresh_trial(seed: int = 0):
    """A real trial from an in-memory study (``FixedTrial({})`` cannot suggest anything)."""
    import optuna

    sampler = optuna.samplers.RandomSampler(seed=seed)
    return optuna.create_study(direction="maximize", sampler=sampler).ask()


@pytest.fixture
def split_df(synthetic_df):
    # 60 rows keeps training instant; bio=False models never fold anything.
    return data.add_split_labels(synthetic_df.head(60).copy())


@pytest.fixture
def paths(tmp_path):
    return config.get_paths(tmp_path)


# --------------------------------------------------------------------------- #
# suggest_config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", sorted(architectures.MODEL_REGISTRY))
@pytest.mark.parametrize("seed", range(3))
def test_suggest_config_keys_are_valid_model_config_fields(model_name, seed):
    params = tuning.suggest_config(_fresh_trial(seed), model_name)
    assert {"lr", "weight_decay"} <= set(params)
    assert set(params) <= _VALID_KEYS
    overrides = {k: v for k, v in params.items() if k not in ("lr", "weight_decay")}
    architectures.build_model(model_name, **overrides)  # must construct without error


def test_suggest_config_ranges_follow_architecture_family():
    conv = set(tuning.suggest_config(_fresh_trial(), "CNN"))
    rnn = set(tuning.suggest_config(_fresh_trial(), "BiLSTM"))
    tf = set(tuning.suggest_config(_fresh_trial(), "Transformer"))
    conv_only = {"kernel_widths", "channels_per_width", "bottleneck_channels"}
    rnn_only = {"rnn_hidden", "rnn_layers", "rnn_merge"}
    tf_only = {"n_heads", "n_transformer_layers", "embed_dim"}
    assert conv_only <= conv and not (conv & (rnn_only | tf_only))
    assert rnn_only <= rnn and not (rnn & (conv_only | tf_only))
    assert tf_only <= tf and not (tf & (conv_only | rnn_only))
    assert "bio_embed_dim" not in conv  # CNN is sequence-only
    assert "bio_embed_dim" in set(tuning.suggest_config(_fresh_trial(), "CNN+bio"))


def test_suggest_config_unidirectional_lstm_has_no_merge_knob():
    assert "rnn_merge" not in tuning.suggest_config(_fresh_trial(), "LSTM")


def test_suggest_config_unknown_model_raises():
    with pytest.raises(KeyError):
        tuning.suggest_config(_fresh_trial(), "NoSuchModel")


def test_transformer_heads_divide_every_embed_dim():
    assert all(e % h == 0 for e in tuning._EMBED_DIMS for h in tuning._N_HEADS)


# --------------------------------------------------------------------------- #
# best_params.json
# --------------------------------------------------------------------------- #
def test_best_params_roundtrip(tmp_path):
    p = tmp_path / "best_params.json"
    tuning.save_best_params(p, "CNN", {"lr": 0.001})
    tuning.save_best_params(p, "BiLSTM", {"lr": 0.002})
    assert tuning.load_best_params(p, "CNN") == {"lr": 0.001}
    assert tuning.load_best_params(p, "BiLSTM") == {"lr": 0.002}  # first entry not clobbered


def test_save_overwrites_same_model_and_restores_tuples(tmp_path):
    p = tmp_path / "best_params.json"
    tuning.save_best_params(p, "CNN", {"lr": 0.1, "kernel_widths": (3, 5)})
    tuning.save_best_params(p, "CNN", {"lr": 0.2, "hidden_dims": (256, 128)})
    assert tuning.load_best_params(p, "CNN") == {"lr": 0.2, "hidden_dims": (256, 128)}


def test_load_missing_file_or_model_returns_empty(tmp_path):
    p = tmp_path / "nested" / "best_params.json"
    assert tuning.load_best_params(p, "CNN") == {}
    tuning.save_best_params(p, "BiLSTM", {"lr": 0.1})  # creates the parent directory
    assert tuning.load_best_params(p, "CNN") == {}


def test_failed_save_leaves_existing_file_intact(tmp_path, monkeypatch):
    p = tmp_path / "best_params.json"
    tuning.save_best_params(p, "CNN", {"lr": 0.001})
    before = p.read_text()

    def killed(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", killed)  # the process dies just before the swap
    with pytest.raises(KeyboardInterrupt):
        tuning.save_best_params(p, "BiLSTM", {"lr": 0.002})
    assert p.read_text() == before
    assert json.loads(before) == {"CNN": {"lr": 0.001}}
    assert [f.name for f in tmp_path.iterdir()] == ["best_params.json"]  # no stray temp file


def test_corrupt_best_params_file_is_not_silently_overwritten(tmp_path):
    p = tmp_path / "best_params.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="best_params"):
        tuning.save_best_params(p, "CNN", {"lr": 0.1})
    assert p.read_text() == "{not json"


# --------------------------------------------------------------------------- #
# search_deep
# --------------------------------------------------------------------------- #
class _Poison:
    """Stands in for the test split: any use of it fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"the search touched the test split ({name!r}): tuning leakage")


def _spy(monkeypatch):
    """Replace the test split with poison; record the real splits and train_model's val loader."""
    seen = {}
    real_build, real_train = tuning.build_splits, tuning.train_model

    def build(*args, **kwargs):
        splits, scaler = real_build(*args, **kwargs)
        seen["splits"] = dict(splits)
        return {**splits, "test": _Poison()}, scaler

    def train(model, train_loader, val_loader, **kwargs):
        seen.setdefault("val_x", []).append(val_loader.dataset.tensors[0].numpy())
        return real_train(model, train_loader, val_loader, **kwargs)

    monkeypatch.setattr(tuning, "build_splits", build)
    monkeypatch.setattr(tuning, "train_model", train)
    return seen


def _search(split_df, paths, **overrides):
    kwargs = {
        "model_name": "CNN",
        "target_col": WT,
        "split_col": "WT_split",
        "paths": paths,
        "n_trials": 2,
        "epochs": 1,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return tuning.search_deep(split_df, **kwargs)


def test_search_deep_returns_best_params_and_study(split_df, paths):
    out = _search(split_df, paths)
    assert set(out) == {"best_params", "best_value", "study"}
    assert set(out["best_params"]) <= _VALID_KEYS
    assert len(out["study"].trials) == 2
    assert -1.0 <= out["best_value"] <= 1.0
    assert out["study"].direction.name == "MAXIMIZE"
    assert not list(paths.models.glob("tune_*"))  # trial checkpoints are cleaned up


def test_objective_reads_validation_never_test(split_df, paths, monkeypatch):
    """The search must be scored on the validation split; the test split is never touched."""
    seen = _spy(monkeypatch)
    out = _search(split_df, paths)  # a poisoned test split raises AssertionError if used
    assert len(out["study"].trials) == 2
    val_x = seen["splits"]["val"].x
    assert len(seen["val_x"]) == 2
    for x in seen["val_x"]:
        np.testing.assert_array_equal(x, val_x)
    assert not np.array_equal(val_x, seen["splits"]["test"].x)  # the two splits really differ


def test_search_deep_refuses_multi_output_model(split_df, paths):
    with pytest.raises(ValueError, match="single-target"):
        _search(split_df, paths, model_name="MultiTask-BiLSTM+bio")


def test_search_deep_best_params_feed_back_into_build_model(split_df, paths):
    out = _search(split_df, paths, n_trials=1)
    params = dict(out["best_params"])
    params.pop("lr"), params.pop("weight_decay")
    architectures.build_model("CNN", **params)


# --------------------------------------------------------------------------- #
# apply_best_params / resumable storage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", ["CNN+bio", "BiLSTM", "Transformer"])
def test_apply_best_params_roundtrip_builds_a_model(tmp_path, model_name):
    p = tmp_path / "best_params.json"
    tuned = tuning.suggest_config(_fresh_trial(1), model_name)
    tuning.save_best_params(p, model_name, tuned)
    overrides, train_kwargs = tuning.apply_best_params(tuning.load_best_params(p, model_name))
    assert set(train_kwargs) == {"lr", "weight_decay"}
    assert not {"lr", "weight_decay"} & set(overrides)
    model = architectures.build_model(model_name, **overrides)
    assert model.cfg.rnn_layers == tuned.get("rnn_layers", model.cfg.rnn_layers)
    assert model.cfg.dropout == tuned["dropout"]  # tuned values reach the config


def test_tuned_rnn_layers_beats_the_registry_pin(tmp_path):
    p = tmp_path / "best_params.json"
    tuning.save_best_params(p, "BiLSTM", {"rnn_layers": 2, "lr": 0.001})
    overrides, _ = tuning.apply_best_params(tuning.load_best_params(p, "BiLSTM"))
    assert architectures.build_model("BiLSTM").cfg.rnn_layers == 1
    assert architectures.build_model("BiLSTM", **overrides).cfg.rnn_layers == 2


def test_tuple_fields_are_derived_from_model_config():
    default = architectures.ModelConfig()
    expected = {
        f.name for f in dataclasses.fields(default) if isinstance(getattr(default, f.name), tuple)
    }
    assert expected and set(tuning._TUPLE_FIELDS) == expected


def test_search_deep_reports_when_every_trial_fails(split_df, paths, monkeypatch):
    real = tuning.train_model

    def nan_rho(*a, **k):
        result = real(*a, **k)
        result.best_val_rho = float("nan")
        return result

    monkeypatch.setattr(tuning, "train_model", nan_rho)
    with pytest.raises(RuntimeError, match="all 2 trials failed"):
        _search(split_df, paths)


def test_search_deep_zero_trials_has_its_own_message(split_df, paths):
    with pytest.raises(RuntimeError, match="no trials"):
        _search(split_df, paths, n_trials=0)


def test_search_deep_sqlite_storage_resumes(split_df, paths, tmp_path):
    url = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    assert len(_search(split_df, paths, n_trials=1, storage=url)["study"].trials) == 1
    assert len(_search(split_df, paths, n_trials=1, storage=url)["study"].trials) == 2
