"""The single training loop every deep model uses, plus prediction and scoring.

Ported from the team's ``models/Train_CNN.py`` (``set_seed``, ``train``,
``evaluate_model``) and extended for masked multi-target loss:

* the loss averages over **measured entries only** (``mask == 1``); unmeasured
  targets are zero-filled in ``y`` and must never pull predictions toward zero;
* "trained on MSE, selected on Spearman": early stopping, the learning-rate
  schedule and best-weights checkpointing all follow the mean validation
  Spearman rho, the paper's headline metric, not the loss.

Importing this module requires torch; the package ``__init__`` does not import
it, so callers import it explicitly.
"""

from __future__ import annotations

import copy
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from crispr_deephf.models import compute_metrics, safe_spearman

__all__ = [
    "TrainResult",
    "evaluate_split",
    "masked_mse",
    "predict",
    "set_seed",
    "train_model",
]


# --------------------------------------------------------------------------- #
# Reproducibility and loss
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every RNG we use and force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over measured entries only.

    Divides by ``mask.sum()``, *not* ``y.numel()``: dividing by the full size
    would count every unmeasured (zero-filled) entry as a perfect prediction and
    silently shrink the loss and the gradient of multi-task models.

    Parameters
    ----------
    pred, y, mask : torch.Tensor
        Same shape ``(N, T)``; ``mask`` is 1.0 where ``y`` is a real measurement.

    Returns
    -------
    torch.Tensor
        Scalar loss. When nothing is measured it is a zero that still carries
        grad (``pred.sum() * 0``), so ``.backward()`` works and nothing is NaN.
    """
    n_measured = mask.sum()
    if n_measured.item() == 0:
        return pred.sum() * 0.0
    return (((pred - y) ** 2) * mask).sum() / n_measured


# --------------------------------------------------------------------------- #
# Prediction and scoring
# --------------------------------------------------------------------------- #
def _forward(model: nn.Module, x: torch.Tensor, bio: torch.Tensor) -> torch.Tensor:
    """Run *model*, passing ``bio`` only to models that use it."""
    return model(x, bio if model.cfg.use_bio else None)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_pred, y_true, mask)``, each ``(N, T)``, in loader order."""
    model.to(device)
    model.eval()
    preds, trues, masks = [], [], []
    for x, bio, y, mask in loader:
        out = _forward(model, x.to(device), bio.to(device))
        preds.append(out.cpu().numpy())
        trues.append(y.numpy())
        masks.append(mask.numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)


def _rho_per_target(y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray) -> dict[int, float]:
    """Spearman rho per target index over measured entries; targets with none are omitted."""
    rhos = {}
    for t in range(y_true.shape[1]):
        keep = mask[:, t] == 1.0
        if keep.any():
            rhos[t] = safe_spearman(y_true[keep, t], y_pred[keep, t])
    return rhos


_METRIC_KEYS = ("Pearson_r", "Spearman_rho", "RMSE", "MAE", "R2")


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_names: Sequence[str],
    model_name: str = "Model",
    split_name: str = "Test",
) -> list[dict]:
    """Score *model* on a split: one :func:`compute_metrics` dict per target.

    Only measured entries (``mask == 1``) of each target are scored. Every row
    carries ``n_measured``. A target with fewer than 2 measured entries cannot be
    scored: it still gets its row, with NaN metrics and a warning, so a tiny split
    cannot crash a run at final evaluation.
    """
    y_pred, y_true, mask = predict(model, loader, device)
    rows = []
    for t, name in enumerate(target_names):
        keep = mask[:, t] == 1.0
        n = int(keep.sum())
        if n < 2:
            warnings.warn(
                f"target {name!r} has {n} measured entries in {split_name}; metrics are NaN",
                stacklevel=2,
            )
            row = {"Target": name, "Model": model_name, "Split": split_name}
            row.update({k: float("nan") for k in _METRIC_KEYS})
        else:
            row = compute_metrics(y_true[keep, t], y_pred[keep, t], model_name, split_name, name)
        row["n_measured"] = n
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    """Outcome of :func:`train_model`; ``model`` holds the best weights."""

    model: nn.Module
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_rho: float = float("-inf")


def _validate(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[float, float, dict[int, float]]:
    """Return ``(masked val loss, mean val rho, per-target rho by index)``.

    The mean is over targets with measured entries only.
    """
    y_pred, y_true, mask = predict(model, loader, device)
    n = mask.sum()
    loss = float((((y_pred - y_true) ** 2) * mask).sum() / n) if n else 0.0
    rhos = _rho_per_target(y_pred, y_true, mask)
    mean = float(np.mean(list(rhos.values()))) if rhos else 0.0
    return loss, mean, rhos


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: str | Path = "best_model.pt",
    target_names: Sequence[str] | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Train with Adam on masked MSE, selecting on mean validation Spearman rho.

    ``ReduceLROnPlateau`` and early stopping both watch the mean val rho; the
    best weights are written to *checkpoint_path* and restored into *model*
    before returning.

    Parameters
    ----------
    model : nn.Module
        Any model from :func:`crispr_deephf.architectures.build_model`.
    train_loader, val_loader : DataLoader
        Yield ``(x, bio, y, mask)``.
    patience : int
        Epochs without improvement before stopping; the LR scheduler uses half.
    target_names : sequence of str, optional
        Keys for ``history[i]["val_rho_per_target"]``; defaults to ``t0, t1, ...``.
    verbose : bool
        Print per-epoch progress (turn off for multi-model sweeps).
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=patience // 2
    )

    result = TrainResult(model=model)
    best_state, stale = None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0.0
        for x, bio, y, mask in train_loader:
            x, bio, y, mask = x.to(device), bio.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            # Trained on masked MSE, but selected on Spearman
            loss = masked_mse(_forward(model, x, bio), y, mask)
            loss.backward()
            opt.step()
            n_measured = float(mask.sum())
            running += loss.item() * n_measured
            seen += n_measured

        train_loss = running / seen if seen else 0.0
        val_loss, val_rho, per_target = _validate(model, val_loader, device)
        names = target_names or [f"t{i}" for i in range(max(per_target, default=-1) + 1)]
        result.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rho": val_rho,
                "val_rho_per_target": {names[t]: r for t, r in per_target.items()},
            }
        )
        sched.step(val_rho)

        if verbose:
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Spearman: {val_rho:.4f}"
            )

        if val_rho > result.best_val_rho:
            result.best_val_rho, result.best_epoch, stale = val_rho, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            if verbose:
                print(f"New best model saved with mean Spearman: {val_rho:.4f}")
        else:
            stale += 1
            if verbose:
                print(f"No improvement. Stale count: {stale}/{patience}")
            if stale >= patience:
                if verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best mean Spearman: {result.best_val_rho:.4f}"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return result
