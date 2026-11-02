"""Tier-1 baseline model trainers and evaluation metrics.

Each trainer takes the same ``(X_train, y_train, X_val, y_val)`` signature and
returns a fitted estimator, so the notebook can loop over them uniformly.

The gradient-boosting libraries (LightGBM, XGBoost, CatBoost) are imported
*lazily and optionally*: if a library is missing from the environment (common
on brand-new Python versions that lack pre-built wheels), the corresponding
trainer is simply left out of :func:`available_trainers` instead of crashing the
whole notebook. Ridge always works (scikit-learn is a hard dependency).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config

__all__ = [
    "LIGHTGBM_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "CATBOOST_AVAILABLE",
    "compute_metrics",
    "train_ridge",
    "train_lightgbm",
    "train_xgboost",
    "train_catboost",
    "available_trainers",
]

# --------------------------------------------------------------------------- #
# Optional dependency detection
# --------------------------------------------------------------------------- #
try:
    import lightgbm as lgb  # noqa: F401

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor  # noqa: F401

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor  # noqa: F401

    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    CATBOOST_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    split_name: str = "Test",
    target_name: str = "Target",
) -> dict:
    """Return the benchmark metric dict used throughout the project.

    Pearson *r* and Spearman *rho* are the headline metrics in the DeepHF paper;
    RMSE / MAE / R2 describe calibration and explained variance.
    """
    return {
        "Target": target_name,
        "Model": model_name,
        "Split": split_name,
        "Pearson_r": pearsonr(y_true, y_pred)[0],
        "Spearman_rho": spearmanr(y_true, y_pred)[0],
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Trainers (uniform signature: X_train, y_train, X_val, y_val -> fitted model)
# --------------------------------------------------------------------------- #
def train_ridge(X_train, y_train, X_val=None, y_val=None, *, alpha: float = 1.0):
    """L2-regularised linear regression -- the linear performance floor."""
    model = Ridge(alpha=alpha, random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, *, early_stopping_rounds: int = 40):
    """LightGBM with leaf-wise boosting and early stopping on the val set."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=45,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=25,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, *, early_stopping_rounds: int = 40):
    """XGBoost histogram-based boosting with early stopping."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_catboost(X_train, y_train, X_val, y_val, *, early_stopping_rounds: int = 40):
    """CatBoost oblivious-tree boosting with early stopping."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        subsample=0.8,
        random_seed=config.RANDOM_SEED,
        early_stopping_rounds=early_stopping_rounds,
        verbose=0,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def available_trainers() -> dict[str, Callable]:
    """Return ``{name: trainer}`` for every model installed in this environment.

    Ridge is always present; the three boosters are included only if their
    library imported successfully.
    """
    trainers: dict[str, Callable] = {"Ridge": train_ridge}
    if LIGHTGBM_AVAILABLE:
        trainers["LightGBM"] = train_lightgbm
    if XGBOOST_AVAILABLE:
        trainers["XGBoost"] = train_xgboost
    if CATBOOST_AVAILABLE:
        trainers["CatBoost"] = train_catboost
    return trainers
