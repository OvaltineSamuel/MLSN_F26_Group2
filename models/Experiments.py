from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

import Train_CNN as TrC
import CNN as cnn
import Bio_Features as bf

DEFAULTS = dict(
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    epochs=60,
    batch_size=256,
)

#each arm is a dict of ModelConfig overrides
ARMS = {
    "full": {},
    "no_bio": dict(use_bio=False),
    "no_pam_skip": dict(pam_proximal_skip=0),
    "single_kernel": dict(kernel_widths=(5,)),
    "no_dropout": dict(dropout=0.0),
}

def run_one(
    df: pd.DataFrame, 
    seq_col: str,
    cfg: cnn.ModelConfig,
    init_seed: int,
    cache: Path,
    ckpt: Path,
    device: torch.device,
    lr: float,
    weight_decay: float,
    patience: int,
    epochs: int,
    batch_size: int,
    with_test: bool,
    quiet: bool = True):
    """Train one model, return the validation loss"""
    
    TrC.set_seed(init_seed)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        tr, va, te, _ = TrC.build_loaders(df, seq_col, cfg, batch_size, init_seed, cache_path=cache)
        model = cnn.GuideCNN(cfg).to(device)
        model = TrC.train(
            model,
            tr,
            va,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            checkpoint_path=str(ckpt),
        )
        #this is the best validation checkpoint
        _, val_rho = TrC.evaluate_model(model, va, device=device)
        out = {"val": val_rho, "val_mean": float(np.mean(val_rho))}
        if with_test:
            _, test_rho = TrC.evaluate_model(model, te, device=device)
            out |= {"test": test_rho, "test_mean": float(np.mean(test_rho))}
    out["n_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return out

def warm_cache(df: pd.DataFrame, seq_col: str, cache: Path):
    """Compute bio features for all sequences and cache them"""
    if Path(cache).exists():
        print(f"Cache {cache} already exists, skipping warmup")
        return
    print(f"Warming up bio feature cache at {cache}")
    bf.bio_matrix(df[seq_col].astype(str).tolist(), cache_path=str(cache))
    
#-------------------------------------------------------------------------------
#-------------------ablate------------------------------------------------------
#-------------------------------------------------------------------------------

def cmd_ablate(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = TrC.load_data(a.data, a.seq_col)

    cache = out / "bio_features.npz"
    warm_cache(df, a.seq_col, cache)

    hp = dict(DEFAULTS)
    base_cfg = {}
    if a.params:
        params = json.loads(Path(a.params).read_text())
        hp.update(params.get("train", {}))
        base_cfg = params.get("model", {})

    arms = a.arms or list(ARMS)
    val_mean: dict[str, list[float]] = {k: [] for k in arms}
    test_mean: dict[str, list[float]] = {k: [] for k in arms}

    for seed in range(a.seeds):
        for arm in arms:
            t0 = time.time()
            cfg = cnn.ModelConfig(**{**base_cfg, **ARMS[arm]})
            r = run_one(
                df, a.seq_col, cfg, seed, cache, out / f"ckpt_{arm}_s{seed}.pt",
                a.device, with_test=True, **hp
            )
            val_mean[arm].append(r["val_mean"])
            test_mean[arm].append(r["test_mean"])
            print(f"seed: {seed}, {arm:14s} val: {r['val_mean']:.4f} "
                  f"test {r['test_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nSpearman over {a.seeds} init seeds, one fixed split (test = held-out, val = used for checkpoint selection)")
    print(f"{'arm':14s} {'val mean':>9s} {'test mean':>10s} {'test std':>9s} {'test delta vs full':>22s} wins")

    full = np.array(test_mean.get("full", []))
    for arm in arms:
        v, x = np.array(val_mean[arm]), np.array(test_mean[arm])
        if arm != "full" and len(full) == len(x):
            d = x - full
            dsd = d.std(ddof=1) if len(d) > 1 else float("nan")
            delta = f"{d.mean():+.4f} +/- {dsd:.4f}"
            wins = f"{int((d > 0).sum())}/{len(d)} beat full"
        else:
            delta, wins = "", ""
        sd = x.std(ddof=1) if len(x) > 1 else float("nan")
        print(f"{arm:14s} {v.mean():9.4f} {x.mean():10.4f} {sd:9.4f} {delta:>22s} {wins}")

    (out / "ablation.json").write_text(json.dumps(
        {"val_mean": val_mean, "test_mean": test_mean}, indent=2))
    print(f"\nwrote {out/'ablation.json'}")
    print("note: error bars cover init variance only; the data split is fixed")

#--------------------------------------------------------------------------------------------------
#-----------------------------Search---------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

HIDDEN = {"256-128": (256, 128), "512-128": (512, 128), "128-64": (128, 64), "256": (256,)}
import optuna
def cmd_search(a):
    
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    
    df = TrC.load_data(a.data, a.seq_col)
    cache = out / "bio_features.npz"
    
    warm_cache(df, a.seq_col, cache)
    
    def objective(trial: optuna.Trial) -> float:
        model_kw = dict(
            channels_per_width=trial.suggest_categorical("channels_per_width", [32, 64, 96]),
            bottleneck_channels=trial.suggest_categorical("bottleneck_channels", [16, 32, 64]),
            bio_embed_dim=trial.suggest_categorical("bio_embed_dim", [16, 32, 48]),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            hidden_dims=HIDDEN[trial.suggest_categorical("hidden", list(HIDDEN))],)
        
        train_kw = dict(
            lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            patience=a.patience,
            epochs=a.epochs
        )
        
        r = run_one(
            df, a.seq_col, cfg=cnn.ModelConfig(**model_kw),
            init_seed=a.init_seed, cache=cache, ckpt=out/f"ckpt_trial{trial.number}.pt",
            device=a.device, with_test=False, **train_kw,)
        
        trial.set_user_attr("val_per_target", r["val"])
        trial.set_user_attr("n_params", r["n_params"])
        
        (out/f"ckpt_trial{trial.number}.pt").unlink(missing_ok=True)
        return r["val_mean"]
    
    study = optuna.create_study(
        direction="maximize", study_name="deephf",
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///{out/'study.db'}",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=a.trials)  # no catch: a broken trial must stop the search loudly
    
    done = [t for t in study.trials if t.value is not None]
    
    print(f"\n{len(done)}/{len(study.trials)} trials completed")

    top = sorted(done, key=lambda t: t.value, reverse=True)[:5]
    
    for t in top:
        print(f"  val {t.value:.4f}  params {t.params}")
        
    best = study.best_trial
    p = best.params
    best_cfg = {
        "model": dict(
            channels_per_width=p["channels_per_width"],
            bottleneck_channels=p["bottleneck_channels"],
            bio_embed_dim=p["bio_embed_dim"],
            dropout=p["dropout"],
            hidden_dims=list(HIDDEN[p["hidden"]])
        ),
        "train": dict(lr=p["lr"],
                      weight_decay=p["weight_decay"],
                      batch_size=p["batch_size"],
                      patience=a.patience,
                      epochs=a.epochs)
    }
    
    (out / "best_params.json").write_text(json.dumps(best_cfg, indent=2))
    
    print(f"\nwrote {out/'best_params.json'} -> feed to: ablate --params {out/'best_params.json'}")
    
    try:
        imp = optuna.importance.get_param_importances(study)
        
        print("importance:", {k: round(v, 3) for k, v in imp.items()})
        
    except Exception as e:
        # too few trials
        print("importance unavailable:", type(e).__name__)
        
def main() -> None:
    
        p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        
        sub = p.add_subparsers(dest="cmd", required=True)
        for name, fn in (("ablate", cmd_ablate), ("search", cmd_search)):
            s = sub.add_parser(name)
            s.set_defaults(fn=fn)
            s.add_argument("data")
            s.add_argument("--seq-col", default="21mer")
            s.add_argument("--out", default="runs")
            s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            
        ab = sub.choices["ablate"]
        
        ab.add_argument("--seeds", type=int, default=5)
        ab.add_argument("--arms", nargs="*", choices=list(ARMS))
        
        ab.add_argument("--params", help="best_params.json from `search`")
        
        se = sub.choices["search"]
        se.add_argument("--trials", type=int, default=40)
        se.add_argument("--init-seed", type=int, default=0)
        se.add_argument("--epochs", type=int, default=40)
        se.add_argument("--patience", type=int, default=8)
        
        a = p.parse_args()
        a.fn(a)

if __name__ == "__main__":
    main()