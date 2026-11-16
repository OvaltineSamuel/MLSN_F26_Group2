"""Resumable experiment runner: train every model, write every result as it lands.

The full run is 8 deep models x 3 nucleases x 3 seeds plus the Tier-1 trees, and free
Colab sessions disconnect. Three rules keep a disconnect cheap:

* every finished run is saved to its own JSON file **atomically** (temp file in the same
  directory, then :func:`os.replace`), so a session killed mid-write can never leave a
  truncated file that resume would trust;
* a rerun skips any run whose saved result is readable, was produced with the same
  settings and on the same data, and retrains only what is missing;
* an unreadable result file is skipped with a warning, never fatal and never trusted.

Data-leak guard
---------------
The per-enzyme split columns (``WT_split``, ``eSp_split``, ``HF1_split``) are drawn
independently, so one guide can be WT-**train** and HF1-**test**. A multi-output head
trained on the WT-train row has then seen a guide it is later scored on for HF1-test.
:func:`run_deep_model` therefore *refuses* any multi-target run that is not on the
``shared_split`` column (see :func:`_guard_multi_target`). The warning originates from the
team's ``models/CNN.py`` ("head must have one output, because split isn't constant across
cas9 variants, so multi-output will cause a data leak").

Importing this module requires torch; the package ``__init__`` does not import it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from crispr_deephf import config, data, models
from crispr_deephf.architectures import build_model
from crispr_deephf.torch_data import build_splits, check_integrity, make_loader
from crispr_deephf.train_torch import evaluate_split, predict, set_seed, train_model

__all__ = [
    "aggregate",
    "is_result_payload",
    "load_results",
    "result_key",
    "results_frame",
    "run_deep_model",
    "run_stack",
    "run_tier1",
    "save_result",
]

#: Target key used in file names and payloads for a multi-target (multi-task) run.
MULTI_KEY = "MULTI"

#: Seed recorded for the (deterministic) Tier-1 tree models and the stack.
TIER1_SEED = 0

#: Metric columns carried by results and aggregated across seeds.
METRIC_COLS = ("Pearson_r", "Spearman_rho", "RMSE", "MAE", "R2")

_DEFAULT_LR = 1e-3

#: Per-model overrides of the generic run mode, applied in ``full`` mode only.
#: ``BiLSTM``: lr, batch size and epochs come from the teammate's completed grid search
#: in ``notebooks/BiLSTM.ipynb`` (lr {0.0005, 0.001} x batch {64, 128, 256} x epochs
#: {10, 15, 20}); every other model uses the mode's own values.
_FULL_MODE_OVERRIDES: dict[str, dict[str, float | int]] = {
    "BiLSTM": {"lr": 0.0005, "batch_size": 128, "epochs": 20},
}


# --------------------------------------------------------------------------- #
# Keys and atomic persistence
# --------------------------------------------------------------------------- #
def result_key(model_name: str, target_key: str, seed: int) -> str:
    """Return the filesystem-safe key ``<model>__<target>__seed<N>`` of one run.

    Characters outside ``[A-Za-z0-9._+-]`` (spaces, slashes, ...) become ``_``.
    """
    raw = f"{model_name}__{target_key}__seed{seed}"
    return re.sub(r"[^A-Za-z0-9._+-]", "_", raw)


def _json_default(obj):
    """Serialise numpy scalars and arrays that ``json`` cannot handle itself."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def save_result(results_dir: Path, key: str, payload: dict) -> Path:
    """Write *payload* to ``<results_dir>/<key>.json`` atomically.

    The JSON goes to a temp file in the **same directory** (so :func:`os.replace` stays on
    one filesystem and is atomic), is flushed and fsynced, and only then replaces the
    target. A process killed at any point leaves either the previous complete file or no
    file -- never a truncated ``.json``. Stray temp files do not end in ``.json`` and are
    ignored by :func:`load_results`.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{key}.json"
    fd, tmp_name = tempfile.mkstemp(dir=results_dir, prefix=f"{key}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def _read_result(path: Path) -> dict | None:
    """Return the payload at *path*, or ``None`` (with a warning) if it is unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:  # JSONDecodeError/UnicodeDecodeError are ValueErrors
        warnings.warn(f"skipping unreadable result file {path.name}: {exc}", stacklevel=3)
        return None
    if not isinstance(payload, dict):
        warnings.warn(f"skipping result file {path.name}: not a JSON object", stacklevel=3)
        return None
    return payload


def is_result_payload(payload: object) -> bool:
    """True if *payload* looks like a run result written by this module.

    The results directory is not private: :func:`tuning.save_best_params` writes
    ``best_params.json`` there, and anyone may drop a JSON note beside the runs.
    ``model_name`` is the one field every result carries and every consumer indexes by, so
    its absence is the test. Without this check the notebook's diagnostics cell raised
    ``KeyError: 'model_name'`` on ``best_params.json`` as soon as the Optuna section had
    run once -- found by executing the notebook, not by reading it.
    """
    return isinstance(payload, dict) and "model_name" in payload


def load_results(results_dir: Path) -> list[dict]:
    """Read every valid ``*.json`` result in *results_dir*, in file-name order.

    Unreadable or truncated files are skipped with a warning rather than raising, so one
    corrupt file cannot take down a whole report. JSON files that are *not* run results
    (``best_params.json`` and anything else sharing the directory) are skipped silently --
    they are not corrupt, they are simply not results. See :func:`is_result_payload`.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    loaded = (_read_result(p) for p in sorted(results_dir.glob("*.json")))
    return [r for r in loaded if is_result_payload(r)]


def _reusable(payload: dict | None, *, mode: dict, digest: str, targets: list[str]) -> bool:
    """True if a saved payload was made with these settings, data and targets."""
    return (
        payload is not None
        and isinstance(payload.get("test"), dict)
        and payload.get("mode") == mode
        and payload.get("data_digest") == digest
        and payload.get("targets") == targets
    )


def _by_target(rows: list[dict]) -> dict[str, dict]:
    """Index metric rows by target, dropping the label fields that the index replaces."""
    drop = {"Target", "Model", "Split"}
    return {r["Target"]: {k: v for k, v in r.items() if k not in drop} for r in rows}


# --------------------------------------------------------------------------- #
# Deep models
# --------------------------------------------------------------------------- #
def _run_settings(model_name: str, mode: str | dict) -> dict:
    """Return the recorded training settings: mode name, epochs, patience, batch size, lr."""
    if isinstance(mode, str):
        name, base = mode, config.get_mode(mode)
    else:
        base = dict(mode)
        name = next((n for n, m in config.RUN_MODES.items() if m == base), "custom")
    settings = {k: base[k] for k in ("epochs", "patience", "batch_size")}
    settings["lr"] = _DEFAULT_LR
    if name == "full":
        settings.update(_FULL_MODE_OVERRIDES.get(model_name, {}))
    return {"name": name, **settings}


def _guard_multi_target(target_cols: Sequence[str], split_col: str, n_model_targets: int) -> None:
    """Refuse a multi-target run unless it uses the shared split column.

    Per-enzyme splits are drawn independently, so a guide can be WT-train and HF1-test; a
    multi-output head trained on its WT row has then seen it when scored on HF1-test.
    ``shared_split`` gives one split per guide across all enzymes and is the only safe
    choice.

    Raises
    ------
    ValueError
        If more than one target (or a multi-output model) is requested on any split column
        other than :data:`config.SHARED_SPLIT_COL`.
    """
    if max(len(target_cols), n_model_targets) > 1 and split_col != config.SHARED_SPLIT_COL:
        raise ValueError(
            f"refusing a multi-target run on split column {split_col!r}: per-enzyme splits "
            "(WT_split, eSp_split, HF1_split) are drawn independently, so one guide can be "
            "train for one enzyme and test for another, and a multi-output head would be "
            "scored on guides it was trained on (data leak). Use "
            f"split_col={config.SHARED_SPLIT_COL!r}, which assigns one split per guide."
        )


def run_deep_model(
    df: pd.DataFrame,
    *,
    model_name: str,
    target_cols: Sequence[str],
    split_col: str,
    paths: config.Paths,
    mode: str | dict,
    seeds: Sequence[int],
    device: str,
    force: bool = False,
) -> list[dict]:
    """Train one registered architecture on *target_cols*, once per seed, resumably.

    Each seed's result is saved to ``paths.results`` the moment it finishes. Resume rule: a
    seed is skipped only if its saved result is readable AND its recorded mode settings
    (name, epochs, patience, batch size, lr), data digest and targets ALL equal the current
    run's -- so a ``quick`` result can never stand in for a ``full`` one. ``force`` retrains
    regardless. An unreadable or stale result file is warned about and retrained.

    Single-target runs use :func:`data.filter_for_target` with the per-enzyme split column
    and the strict (no-NaN) integrity check. A multi-target run is only allowed on
    ``shared_split`` and uses the mask for unmeasured entries.

    Parameters
    ----------
    df : cleaned dataframe with the frozen split column(s).
    model_name : key of :data:`architectures.MODEL_REGISTRY`.
    target_cols : target column(s); more than one requires ``split_col="shared_split"``.
    split_col : frozen split-label column to read.
    paths : project directories; results go to ``paths.results``, weights to ``paths.models``.
    mode : ``"quick"`` / ``"full"`` or a settings dict. In ``full`` mode ``BiLSTM`` uses its
        grid-searched lr, batch size and epochs (see ``_FULL_MODE_OVERRIDES``).
    seeds : model-initialisation seeds; the split is never re-derived.
    device : torch device string.
    force : retrain even if a reusable result exists.

    Returns
    -------
    list of dict
        One payload per seed (loaded or new): ``model_name``, ``target``, ``targets``,
        ``seed``, ``mode``, ``split_col``, ``val`` and ``test`` (``{target_col: metrics}``),
        ``best_epoch``, ``history``, ``duration_s``, ``data_digest`` and, for single-target
        runs, ``predictions`` (``val``/``test`` lists in split-row order, for stacking).

    Raises
    ------
    ValueError
        On a multi-target run over a per-enzyme split column (the data-leak guard), or when
        the model's number of outputs does not match ``len(target_cols)``.
    """
    target_cols = list(target_cols)
    probe = build_model(model_name)
    n_outputs, use_bio = probe.cfg.n_targets, probe.cfg.use_bio
    _guard_multi_target(target_cols, split_col, n_outputs)  # before any data is touched
    if n_outputs != len(target_cols):
        raise ValueError(
            f"{model_name!r} has {n_outputs} outputs but {len(target_cols)} target columns "
            "were given"
        )

    multi = len(target_cols) > 1
    if multi:
        if split_col not in df.columns:
            raise ValueError(f"{split_col!r} not in the dataframe; run data.add_shared_split")
        keep = df[split_col].notna() & df[target_cols].notna().any(axis=1)
        frame = df.loc[keep].reset_index(drop=True)
        target_key = MULTI_KEY
    else:
        frame = data.filter_for_target(df, target_cols[0])
        target_key = config.TARGET_KEYS[target_cols[0]]
    data.assert_unique_sequences(frame)
    digest = check_integrity(frame, "21mer", split_col, target_cols, allow_missing=multi)

    settings = _run_settings(model_name, mode)
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.models.mkdir(parents=True, exist_ok=True)
    splits = None  # built lazily: a fully resumed run never touches the data
    results = []
    for seed in seeds:
        key = result_key(model_name, target_key, seed)
        out_path = paths.results / f"{key}.json"
        if not force and out_path.exists():
            cached = _read_result(out_path)
            if _reusable(cached, mode=settings, digest=digest, targets=target_cols):
                results.append(cached)
                continue
            if cached is not None:
                warnings.warn(f"{key}: saved result used other settings or data; retraining")
        if splits is None:
            splits, _ = build_splits(
                frame,
                target_cols=target_cols,
                split_col=split_col,
                bio=use_bio,
                cache_path=paths.cache / f"bio_{target_key}" if use_bio else None,
                allow_missing=multi,
            )
        results.append(
            _train_one(
                model_name=model_name,
                key=key,
                seed=seed,
                splits=splits,
                target_cols=target_cols,
                target_key=target_key,
                split_col=split_col,
                settings=settings,
                digest=digest,
                paths=paths,
                device=device,
            )
        )
    return results


def _train_one(
    *,
    model_name: str,
    key: str,
    seed: int,
    splits: dict,
    target_cols: list[str],
    target_key: str,
    split_col: str,
    settings: dict,
    digest: str,
    paths: config.Paths,
    device: str,
) -> dict:
    """Train and score one seed, save its payload, and return it."""
    set_seed(seed)
    model = build_model(model_name)
    bs = settings["batch_size"]
    train_loader = make_loader(splits["train"], batch_size=bs, shuffle=True, seed=seed)
    val_loader = make_loader(splits["val"], batch_size=bs, shuffle=False, seed=seed)
    test_loader = make_loader(splits["test"], batch_size=bs, shuffle=False, seed=seed)

    start = time.perf_counter()
    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=settings["epochs"],
        lr=settings["lr"],
        patience=settings["patience"],
        device=device,
        checkpoint_path=paths.models / f"{key}.pt",
        target_names=target_cols,  # real names, so history keys are not t0/t1/t2
        verbose=False,
    )
    duration = time.perf_counter() - start

    val_rows = evaluate_split(model, val_loader, device, target_cols, model_name, "Val")
    test_rows = evaluate_split(model, test_loader, device, target_cols, model_name, "Test")
    payload = {
        "model_name": model_name,
        "family": "deep",
        "target": target_key,
        "targets": target_cols,
        "seed": seed,
        "mode": settings,
        "split_col": split_col,
        "val": _by_target(val_rows),
        "test": _by_target(test_rows),
        "best_epoch": result.best_epoch,
        "history": result.history,
        "duration_s": duration,
        "data_digest": digest,
    }
    if len(target_cols) == 1:  # per-guide predictions, so run_stack can combine models
        payload["predictions"] = {
            "val": predict(model, val_loader, device)[0][:, 0].tolist(),
            "test": predict(model, test_loader, device)[0][:, 0].tolist(),
            # guide ids in prediction order, so run_stack can prove the alignment
            "ids_val": [str(i) for i in splits["val"].ids],
            "ids_test": [str(i) for i in splits["test"].ids],
        }
    save_result(paths.results, key, payload)
    return payload


# --------------------------------------------------------------------------- #
# Tier-1 trees
# --------------------------------------------------------------------------- #
def _tier1_payload(
    name: str,
    target: str,
    digest: str,
    mode: dict,
    ids: dict[str, list[str]],
    y_val: np.ndarray,
    y_test: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
    duration: float | None,
) -> dict:
    """Build one Tier-1 payload from validation and test predictions."""
    split_col = f"{config.TARGET_KEYS[target]}_split"
    return {
        "model_name": name,
        "family": "tree",
        "target": config.TARGET_KEYS[target],
        "targets": [target],
        "seed": TIER1_SEED,
        "mode": mode,
        "split_col": split_col,
        "val": _by_target([models.compute_metrics(y_val, p_val, name, "Val", target)]),
        "test": _by_target([models.compute_metrics(y_test, p_test, name, "Test", target)]),
        "best_epoch": None,
        "history": [],
        "duration_s": duration,
        "data_digest": digest,
        "predictions": {
            "val": np.asarray(p_val).tolist(),
            "test": np.asarray(p_test).tolist(),
            "ids_val": ids["val"],
            "ids_test": ids["test"],
        },
    }


def run_tier1(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    paths: config.Paths,
    force: bool = False,
) -> list[dict]:
    """Train every installed Tier-1 model per target, plus the booster mean-ensemble.

    Moved out of the legacy Tier-1 notebook. For each target the splits are read from the
    frozen ``<KEY>_split`` column; each model's result is
    saved as it finishes and skipped on rerun unless *force*. The ``Ensemble`` is the mean
    of the available gradient boosters (needs at least two) and is rebuilt from their
    saved predictions, so it works across a resumed session. Trained models are dumped to
    ``paths.models`` as joblib files.

    Parameters
    ----------
    X : feature matrix aligned row-for-row with *df*.
    feature_names : column names of *X* (only their count is recorded).

    Returns
    -------
    list of dict
        One payload per (model, target), same shape as :func:`run_deep_model`'s.
    """
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.models.mkdir(parents=True, exist_ok=True)
    trainers = models.available_trainers()
    boosters = [m for m in trainers if m != "Ridge"]
    names_blob = chr(10).join(map(str, feature_names)).encode()
    fingerprint = hashlib.sha256(names_blob).hexdigest()[:12]
    results: list[dict] = []
    for target in config.TARGET_COLS:
        tkey = config.TARGET_KEYS[target]
        # filter_for_target is the single place rows are dropped; "_row" keeps X aligned.
        frame = data.filter_for_target(df.assign(_row=np.arange(len(df))), target)
        digest = check_integrity(frame, "21mer", f"{tkey}_split", [target])
        sp = data.get_split_arrays(frame, X[frame["_row"].to_numpy()], target)
        y_val, y_test = sp["val"]["y"], sp["test"]["y"]
        ids = {s: [str(i) for i in sp[s]["id"]] for s in ("val", "test")}
        # Resume key beyond the frame digest: the feature set (count + names) must match.
        tier1_mode = {"name": "tier1", "n_features": len(feature_names), "features": fingerprint}
        ens_mode = {**tier1_mode, "components": list(boosters)}  # booster set changed -> redo
        per_model: dict[str, dict] = {}

        def cached_payload(
            name: str,
            mode: dict,
            tkey: str = tkey,
            digest: str = digest,
            target: str = target,
        ) -> dict | None:
            path = paths.results / f"{result_key(name, tkey, TIER1_SEED)}.json"
            if force or not path.exists():
                return None
            payload = _read_result(path)
            ok = _reusable(payload, mode=mode, digest=digest, targets=[target])
            return payload if ok else None

        for name, trainer in trainers.items():
            payload = cached_payload(name, tier1_mode)
            if payload is None:
                t0 = time.perf_counter()
                fitted = trainer(sp["train"]["X"], sp["train"]["y"], sp["val"]["X"], y_val)
                dt = time.perf_counter() - t0
                payload = _tier1_payload(
                    name,
                    target,
                    digest,
                    tier1_mode,
                    ids,
                    y_val,
                    y_test,
                    fitted.predict(sp["val"]["X"]),
                    fitted.predict(sp["test"]["X"]),
                    dt,
                )
                joblib.dump(fitted, paths.models / f"{name.lower()}_{tkey}.joblib")
                save_result(paths.results, result_key(name, tkey, TIER1_SEED), payload)
            per_model[name] = payload

        if len(boosters) >= 2:
            payload = cached_payload("Ensemble", ens_mode)
            if payload is None:
                mean = {
                    s: np.mean([per_model[b]["predictions"][s] for b in boosters], axis=0)
                    for s in ("val", "test")
                }
                payload = _tier1_payload(
                    "Ensemble",
                    target,
                    digest,
                    ens_mode,
                    ids,
                    y_val,
                    y_test,
                    mean["val"],
                    mean["test"],
                    None,  # an ensemble has no training time of its own
                )
                save_result(paths.results, result_key("Ensemble", tkey, TIER1_SEED), payload)
            per_model["Ensemble"] = payload
        results.extend(per_model.values())
    return results


# --------------------------------------------------------------------------- #
# Stacking
# --------------------------------------------------------------------------- #
def _best_by_val(
    results: list[dict], family: str, target_col: str, mode_name: str | None = None
) -> list[dict]:
    """Return the seed-payloads of the *family* model with the best mean **validation** rho.

    Seeds are pooled only within one ``(model, mode)`` pair, so runs of the same model under
    different settings (e.g. quick and full) are never averaged together.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        if (
            r.get("family") == family
            and r.get("targets") == [target_col]
            and r.get("predictions") is not None
            and (mode_name is None or r.get("mode", {}).get("name") == mode_name)
        ):
            key = (r["model_name"], json.dumps(r.get("mode"), sort_keys=True))
            groups.setdefault(key, []).append(r)
    if not groups:
        raise ValueError(f"no {family!r} results with predictions for {target_col!r}")

    def mean_val_rho(group: list[dict]) -> float:
        return float(np.nanmean([p["val"][target_col]["Spearman_rho"] for p in group]))

    return max(groups.values(), key=mean_val_rho)


def run_stack(
    results: list[dict],
    df: pd.DataFrame,
    *,
    target_col: str,
    paths: config.Paths,
    mode_name: str | None = None,
) -> dict:
    """Stack the best tree model and the best deep model with a Ridge meta-learner.

    "Best" is chosen on mean **validation** Spearman rho. The deep model's seeds are
    averaged into one prediction vector. The Ridge meta-learner is then fitted on the
    **validation**-split predictions of the two models against the validation targets and
    scored on the test split. Fitting it on test predictions would be test-set leakage and
    is never done; the returned ``val`` metrics are therefore in-sample and optimistic, so
    report ``test`` only.

    Parameters
    ----------
    results : payloads from :func:`run_tier1` and :func:`run_deep_model`.
    df : the cleaned dataframe (used to recover the validation and test targets).
    target_col : the single target to stack.
    mode_name : if given (``"quick"``/``"full"``), only deep results of that run mode are
        considered; seeds are in any case never pooled across different mode settings.

    Returns
    -------
    dict
        A result payload (also saved to ``paths.results``) with ``meta_fit_split == "val"``.

    Raises
    ------
    ValueError
        If either family has no results for *target_col*, or the saved predictions do not
        line up with *df*: different data digest, different row count, or guide ids that
        differ from the frame's ``gRNA_ID`` in order (so a reordered loader cannot silently
        pair predictions with the wrong guides).
    """
    tree_runs = _best_by_val(results, "tree", target_col)
    deep_runs = _best_by_val(results, "deep", target_col, mode_name)
    digests = {r["data_digest"] for r in tree_runs + deep_runs}
    split_cols = {r["split_col"] for r in tree_runs + deep_runs}
    if len(digests) != 1 or len(split_cols) != 1:
        raise ValueError(
            f"tree and deep results come from different data: {digests}, {split_cols}"
        )
    split_col = split_cols.pop()
    frame = data.filter_for_target(df, target_col)
    labels = frame[split_col].to_numpy()

    feats, truth = {}, {}
    for split in ("val", "test"):
        y = frame.loc[labels == split, target_col].to_numpy(dtype=np.float64)
        want = frame.loc[labels == split, "gRNA_ID"].astype(str).tolist()
        for r in tree_runs + deep_runs:
            got = r["predictions"].get(f"ids_{split}")
            if got != want:
                raise ValueError(
                    f"{r['model_name']!r} {split} predictions are not aligned with the "
                    f"frame's guides (saved ids "
                    f"{'missing -- rerun with force' if got is None else 'differ in order'}); "
                    "refusing to stack predictions that may belong to different guides"
                )
        tree = np.mean([r["predictions"][split] for r in tree_runs], axis=0)
        deep = np.mean([r["predictions"][split] for r in deep_runs], axis=0)
        if not (len(tree) == len(deep) == len(y)):
            raise ValueError(
                f"{split} predictions ({len(tree)}, {len(deep)}) do not match the "
                f"{len(y)} rows of that split in df"
            )
        feats[split], truth[split] = np.column_stack([tree, deep]), y

    meta = Ridge(alpha=1.0, random_state=config.RANDOM_SEED)
    meta.fit(feats["val"], truth["val"])  # validation only; test is scored, never fitted
    components = [tree_runs[0]["model_name"], deep_runs[0]["model_name"]]
    name = f"Stack({components[0]}+{components[1]})"
    p_val, p_test = meta.predict(feats["val"]), meta.predict(feats["test"])
    tkey = config.TARGET_KEYS[target_col]
    payload = {
        "model_name": name,
        "family": "stack",
        "target": tkey,
        "targets": [target_col],
        "seed": TIER1_SEED,
        "mode": {"name": "stack"},
        "split_col": split_col,
        "meta_fit_split": "val",
        "meta_coef": meta.coef_.tolist(),
        "components": components,
        "val": _by_target([models.compute_metrics(truth["val"], p_val, name, "Val", target_col)]),
        "test": _by_target(
            [models.compute_metrics(truth["test"], p_test, name, "Test", target_col)]
        ),
        "best_epoch": None,
        "history": [],
        "duration_s": 0.0,
        "data_digest": digests.pop(),
    }
    save_result(paths.results, result_key(name, tkey, TIER1_SEED), payload)
    return payload


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def results_frame(results: list[dict]) -> pd.DataFrame:
    """Flatten payloads into long-form rows: ``Model, Target, Seed, Split`` + metrics.

    One row per (payload, split, target); ``Split`` is ``"Val"`` or ``"Test"`` and
    ``Target`` the efficiency column (a multi-task run yields one row per target). Payloads
    without a ``test`` mapping are ignored.
    """
    rows = []
    for r in results:
        if not isinstance(r.get("test"), dict):
            continue
        for split, label in (("val", "Val"), ("test", "Test")):
            for target, metrics in (r.get(split) or {}).items():
                rows.append(
                    {
                        "Model": r["model_name"],
                        "Target": target,
                        "Seed": r.get("seed"),
                        "Split": label,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise seeds: mean and s.d. of every metric per ``(Model, Target)``.

    Groups additionally by ``Split`` when the frame has one. Output columns are
    ``Model, Target, [Split], n_seeds`` followed by ``<Metric>_mean`` and ``<Metric>_std``
    for each metric present (e.g. ``Spearman_rho_mean``). The s.d. is the sample s.d.
    (ddof=1), so it is NaN when ``n_seeds == 1`` (e.g. the deterministic Tier-1 trees).
    """
    keys = ["Model", "Target"] + (["Split"] if "Split" in frame.columns else [])
    metrics = [m for m in METRIC_COLS if m in frame.columns]
    grouped = frame.groupby(keys, sort=True)
    out = grouped["Seed"].nunique().rename("n_seeds").to_frame()
    for m in metrics:
        out[f"{m}_mean"] = grouped[m].mean()
        out[f"{m}_std"] = grouped[m].std()
    return out.reset_index()
