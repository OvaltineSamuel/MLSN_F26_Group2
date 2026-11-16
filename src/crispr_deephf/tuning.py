"""Optional Optuna hyper-parameter search for the deep models (off by default).

The notebook ships with tuned defaults, so this module is only imported when the
team deliberately switches the tuning cell on. Optuna is a guarded optional
import: without it :data:`OPTUNA_AVAILABLE` is ``False`` and calling
:func:`search_deep` or :func:`suggest_config` raises a clear ``ImportError``.
Importing this module requires torch; the package ``__init__`` does not import
it.

The search maximises **validation** Spearman rho, never test rho
-----------------------------------------------------------------
Every trial is scored on the validation split, and :func:`search_deep` never
reads, loads or even builds a loader for the test split. Tuning against the
test set would leak it into the model-selection loop and invalidate every
number in the final comparison. The test split is only ever touched afterwards,
once, by :func:`runner.run_deep_model` using the tuned values.

Provenance
----------
Adapted from ``cmd_search`` in the team's ``models/Experiments.py`` (not
modified): the same TPE sampler seed, the same "maximize validation score"
objective, one throw-away checkpoint per trial that is deleted afterwards, and
the same rule that a broken trial stops the search loudly instead of being
swallowed.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from crispr_deephf import config, data
from crispr_deephf.architectures import (
    MODEL_REGISTRY,
    GuideBiLSTM,
    GuideCNN,
    GuideCNNBiLSTM,
    GuideLSTM,
    GuideTransformer,
    ModelConfig,
    build_model,
)
from crispr_deephf.torch_data import build_splits, check_integrity, make_loader
from crispr_deephf.train_torch import set_seed, train_model

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    optuna = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False

__all__ = [
    "OPTUNA_AVAILABLE",
    "load_best_params",
    "save_best_params",
    "search_deep",
    "suggest_config",
]

#: Fixed training settings the search does not tune (the team's ``Experiments.py`` defaults).
_BATCH_SIZE = 256
_PATIENCE = 10

#: ``ModelConfig`` fields that are tuples; JSON turns them into lists, so they are restored on load.
_TUPLE_FIELDS = ("kernel_widths", "hidden_dims")

# Categorical choices are named strings (Optuna stores primitives only) mapped to tuples here.
_KERNELS = {"3-5-7": (3, 5, 7), "3-5": (3, 5), "5": (5,), "7": (7,)}
_HIDDEN = {"256-128": (256, 128), "512-128": (512, 128), "128-64": (128, 64), "256": (256,)}

# Every candidate head count divides every candidate embed_dim, so a Transformer trial can
# never hand ``nn.MultiheadAttention`` a width its heads do not divide.
_EMBED_DIMS = (32, 64, 128)
_N_HEADS = (1, 2, 4, 8)


def _require_optuna() -> None:
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is not installed; `pip install optuna` to run a search")


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def suggest_config(trial: object, model_name: str) -> dict:
    """Sample ``ModelConfig`` overrides plus ``lr`` and ``weight_decay`` for *model_name*.

    The ranges depend on the architecture family, read off the class registered in
    :data:`architectures.MODEL_REGISTRY` rather than a parallel list of names: convolutional
    models get ``kernel_widths``/``channels_per_width``/``bottleneck_channels``, recurrent
    ones ``rnn_hidden``/``rnn_layers`` (and ``rnn_merge`` when bidirectional), the
    Transformer ``embed_dim``/``n_heads``/``n_transformer_layers``. ``bio_embed_dim`` is
    only suggested for models that actually build the bio branch. The batch size is not
    searched.

    Parameters
    ----------
    trial : an ``optuna.Trial`` (or anything with its ``suggest_*`` methods).
    model_name : key of :data:`architectures.MODEL_REGISTRY`.

    Returns
    -------
    dict
        Keys are ``ModelConfig`` fields, plus ``lr`` and ``weight_decay`` which the training
        loop takes instead. Split off those two before calling ``build_model(**overrides)``.

    Raises
    ------
    KeyError
        If *model_name* is not registered.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {model_name!r}; available: {sorted(MODEL_REGISTRY)}")
    cls, registry = MODEL_REGISTRY[model_name]
    use_bio = registry.get("use_bio", ModelConfig().use_bio)

    params: dict = {
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }
    if cls in (GuideCNN, GuideCNNBiLSTM):
        params["kernel_widths"] = _KERNELS[
            trial.suggest_categorical("kernel_widths", list(_KERNELS))
        ]
        params["channels_per_width"] = trial.suggest_categorical("channels_per_width", [32, 64, 96])
        params["bottleneck_channels"] = trial.suggest_categorical(
            "bottleneck_channels", [16, 32, 64]
        )
    if cls in (GuideLSTM, GuideBiLSTM, GuideCNNBiLSTM):
        params["rnn_hidden"] = trial.suggest_categorical("rnn_hidden", [64, 128, 256])
        params["rnn_layers"] = trial.suggest_categorical("rnn_layers", [1, 2])
        if cls is not GuideLSTM:  # a one-directional LSTM has no two states to merge
            params["rnn_merge"] = trial.suggest_categorical("rnn_merge", ["sum", "concat"])
    if cls is GuideTransformer:
        params["embed_dim"] = trial.suggest_categorical("embed_dim", list(_EMBED_DIMS))
        params["n_heads"] = trial.suggest_categorical("n_heads", list(_N_HEADS))
        params["n_transformer_layers"] = trial.suggest_categorical(
            "n_transformer_layers", [1, 2, 3]
        )
    if cls is not GuideBiLSTM:  # GuideBiLSTM keeps the notebook's own fixed 16-16 head
        params["hidden_dims"] = _HIDDEN[trial.suggest_categorical("hidden_dims", list(_HIDDEN))]
    if use_bio:
        params["bio_embed_dim"] = trial.suggest_categorical("bio_embed_dim", [16, 32, 48])
    return params


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def search_deep(
    df: pd.DataFrame,
    *,
    model_name: str,
    target_col: str,
    split_col: str,
    paths: config.Paths,
    n_trials: int = 40,
    epochs: int = 20,
    device: str,
) -> dict:
    """Search hyper-parameters for one single-target deep model, maximising **validation** rho.

    Each trial trains a fresh model on the train split and is scored by the best mean
    **validation** Spearman rho :func:`train_model` reached. The test split is never read
    here (see the module docstring): tuning on it would make the final comparison
    optimistic. The study is in memory and uses a seeded TPE sampler, so the same inputs
    reproduce the same search.

    Parameters
    ----------
    df : cleaned dataframe carrying the frozen split column(s).
    model_name : key of :data:`architectures.MODEL_REGISTRY` with a single output.
    target_col : the one target column to tune against.
    split_col : frozen split-label column (the per-enzyme one for *target_col*).
    paths : project directories; per-trial checkpoints go to ``paths.models`` and are
        deleted, the bio-feature cache to ``paths.cache``.
    n_trials : number of Optuna trials.
    epochs : epochs per trial (early stopping still applies).
    device : torch device string.

    Returns
    -------
    dict
        ``{"best_params", "best_value", "study"}``: the winning :func:`suggest_config` dict,
        its validation rho, and the ``optuna.Study``. Persist the params with
        :func:`save_best_params`.

    Raises
    ------
    ValueError
        If the model has more than one output: a multi-output head is only safe on the
        ``shared_split`` column and is not tuned here.
    ImportError
        If optuna is not installed.
    """
    _require_optuna()
    probe = build_model(model_name)
    if probe.cfg.n_targets != 1:
        raise ValueError(
            f"{model_name!r} has {probe.cfg.n_targets} outputs; search_deep only tunes "
            "single-target models"
        )
    use_bio = probe.cfg.use_bio

    frame = data.filter_for_target(df, target_col)
    data.assert_unique_sequences(frame)
    check_integrity(frame, "21mer", split_col, [target_col], allow_missing=False)
    target_key = config.TARGET_KEYS[target_col]
    paths.models.mkdir(parents=True, exist_ok=True)
    splits, _ = build_splits(
        frame,
        target_cols=[target_col],
        split_col=split_col,
        bio=use_bio,
        cache_path=paths.cache / f"bio_{target_key}" if use_bio else None,
    )
    # Only train and val are used below. The test split is deliberately never touched.
    train_loader = make_loader(
        splits["train"], batch_size=_BATCH_SIZE, shuffle=True, seed=config.RANDOM_SEED
    )
    val_loader = make_loader(
        splits["val"], batch_size=_BATCH_SIZE, shuffle=False, seed=config.RANDOM_SEED
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_config(trial, model_name)
        lr, weight_decay = params.pop("lr"), params.pop("weight_decay")
        set_seed(config.RANDOM_SEED)
        model = build_model(model_name, **params)
        ckpt = paths.models / f"tune_{model_name}_{target_key}_trial{trial.number}.pt"
        try:
            result = train_model(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                patience=_PATIENCE,
                device=device,
                checkpoint_path=ckpt,
                target_names=[target_col],
                verbose=False,
            )
        finally:
            ckpt.unlink(missing_ok=True)
        return result.best_val_rho  # validation Spearman rho, maximised

    study = optuna.create_study(
        direction="maximize",
        study_name=f"deephf-{model_name}-{target_key}",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(objective, n_trials=n_trials)  # no catch: a broken trial must stop it loudly

    # best_trial.params holds the raw sampled values (named choices, not tuples), so rebuild
    # the suggest_config dict by replaying the winning trial through a fixed one.
    best = optuna.trial.FixedTrial(study.best_trial.params)
    return {
        "best_params": suggest_config(best, model_name),
        "best_value": study.best_value,
        "study": study,
    }


# --------------------------------------------------------------------------- #
# Persisted best params
# --------------------------------------------------------------------------- #
def _read_all(path: Path) -> dict[str, dict]:
    """Return the whole ``best_params.json`` mapping, or ``{}`` when the file is absent."""
    if not path.exists():
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc}); fix or delete best_params") from exc
    if not isinstance(content, dict):
        raise ValueError(  # noqa: TRY004 - same "bad best_params file" failure as bad JSON
            f"{path} must hold a JSON object keyed by model name (best_params)"
        )
    return content


def save_best_params(path: Path, model_name: str, params: Mapping) -> None:
    """Record *params* under *model_name* in the shared ``best_params.json``.

    Load-modify-write, so other models' entries survive; the write is atomic (a temp file
    in the same directory, fsynced, then :func:`os.replace`), so a process killed mid-write
    leaves the previous complete file. An unreadable existing file raises ``ValueError``
    rather than being silently overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _read_all(path)
    content[model_name] = {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(content, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_best_params(path: Path, model_name: str) -> dict:
    """Return the tuned params saved for *model_name*, or ``{}`` if there are none.

    Tuple-valued ``ModelConfig`` fields (``kernel_widths``, ``hidden_dims``) come back as
    tuples, matching what :func:`suggest_config` produced.
    """
    params = dict(_read_all(Path(path)).get(model_name, {}))
    for key in _TUPLE_FIELDS:
        if key in params:
            params[key] = tuple(params[key])
    return params
