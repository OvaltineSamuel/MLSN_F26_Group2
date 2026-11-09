import Bio_Features as bf
import CNN as cnn
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import argparse
import hashlib
import random
from torch.utils.data import TensorDataset, DataLoader


def spearman_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
  return[
      float(spearmanr(y_true[:,i], y_pred[:,i]).statistic)
      for i in range(y_true.shape[1])
  ]
  
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> tuple[float, list[float]]:
    model.eval()
    
    preds, trues, total_loss, n = [], [], 0.0, 0
    
    for xb, bb, yb in loader:
        xb, bb, yb = xb.to(device), bb.to(device), yb.to(device)
    
        out = model(xb, bb if model.cfg.use_bio else None)
        
        total_loss += nn.functional.mse_loss(out, yb, reduction="sum").item()
        
        n += yb.numel()
        
        preds.append(out.cpu().numpy())
        trues.append(yb.cpu().numpy())
        
    y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
    
    return total_loss / n, spearman_per_target(y_true, y_pred)

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: str = "best_model.pt",
) -> nn.Module:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 
        mode="max", 
        factor=0.5, 
        patience=patience//2)
    
    best, stale = -np.inf, 0
    
    for epoch in range(1, epochs + 1): 
        model.train()
        running, seen = 0.0, 0
        for xb, bb, yb in train_loader:
            xb, bb, yb = xb.to(device), bb.to(device), yb.to(device)
            
            opt.zero_grad()
            # Trained on MSE, but selected on Spearman
            
            loss = nn.functional.mse_loss(model(xb, bb if model.cfg.use_bio else None), yb)
            
            loss.backward()
            opt.step()
            
            running += loss.item() * len(yb)
            seen += len(yb)
            
        val_loss, val_spearman = evaluate_model(model, val_loader, device)
        
        mean_rho = float(np.mean(val_spearman))
        
        sched.step(mean_rho)
        
        rho_str = ", ".join(f"{r:.4f}" for r in val_spearman)
        
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {running/seen:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Spearman: {mean_rho:.4f} ({rho_str})"
        )
        
        if mean_rho > best:
            best, stale = mean_rho, 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with mean Spearman: {best:.4f}")
        else:
            stale += 1
            print(f"No improvement. Stale count: {stale}/{patience}")
            if stale >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best mean Spearman: {best:.4f}")
                break

    model.load_state_dict(torch.load(checkpoint_path))
    return model

#------------------------------------------------------------------------------------------------------
#------------------------Full Data Pipeline------------------------------------------------
#------------------------------------------------------------------------------------------------------

VARIANTS = {"WT": 0, "eSp": 1, "HF1": 2}  # variant -> index into bf.TARGET_COLUMNS


def set_seed(seed: int) -> None:
    """Seed every RNG we use and force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path: str, seq_col: str, Cas9: str = "WT") -> pd.DataFrame:
    """Load the CSV and keep only rows that belong to this variant.

    A row belongs if it has a split label and a target for the variant.
    Rows without them (e.g. never measured for this variant) are dropped.
    """
    df = pd.read_csv(path)
    split_col, target = f"{Cas9}_split", bf.TARGET_COLUMNS[VARIANTS[Cas9]]
    keep = df[split_col].notna() & df[target].notna()
    print(f"{Cas9}: dropped {(~keep).sum()} of {len(df)} rows with no split/target")
    df = df[keep].reset_index(drop=True)
    check_integrity(df, seq_col, Cas9)
    return df


def check_integrity(df: pd.DataFrame, seq_col: str, Cas9: str = "WT") -> str:
    """Fail loudly on anything that could leak or corrupt. Returns a dataset hash."""
    split_col, target = f"{Cas9}_split", bf.TARGET_COLUMNS[VARIANTS[Cas9]]
    seqs = df[seq_col].astype(str)

    bad = set(df[split_col].unique()) - {"train", "val", "test"}
    assert not bad, f"unexpected split labels: {bad}"
    assert df[split_col].nunique() == 3, "need all of train/val/test"

    assert (seqs.str.len() == bf.SEQ_LEN).all(), f"sequences must be {bf.SEQ_LEN} long"
    assert seqs.str.fullmatch("[ACGT]+").all(), "sequences must be only A/C/G/T"

    y = df[target].to_numpy(dtype=np.float64)
    assert np.isfinite(y).all(), f"non-finite values in {target}"
    assert y.min() >= 0.0 and y.max() <= 1.0, f"{target} outside [0, 1]: {y.min()}..{y.max()}"

    n_splits = df.groupby(seqs)[split_col].nunique()
    leaked = n_splits[n_splits > 1]
    assert leaked.empty, f"{len(leaked)} sequences appear in more than one split, e.g. {list(leaked.index[:3])}"

    digest = hashlib.sha256(
        "\n".join(f"{s},{p}" for s, p in zip(seqs, df[split_col])).encode()
    ).hexdigest()[:12]
    print(f"integrity ok: {len(df)} rows, data hash {digest}")
    return digest

def one_hot_encode_sequences(seqs: list[str]) -> np.ndarray:
    """Convert a list of DNA sequences into one-hot encoded format."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((len(seqs), 4, bf.SEQ_LEN), dtype=np.float32)
    
    for i, seq in enumerate(seqs):
        assert len(seq) == bf.SEQ_LEN, f"Sequence length must be {bf.SEQ_LEN}, but got {len(seq)}"
        for j, nucleotide in enumerate(seq):
            if nucleotide in mapping:
                one_hot[i, mapping[nucleotide], j] = 1.0
    return one_hot

def build_loaders(
    df: pd.DataFrame,
    seq_col: str,
    cfg: cnn.ModelConfig,
    batch_size: int,
    seed: int,
    cache_path: str | None,
    Cas9: str = "WT",  # different CNNs must be made for each Cas9 variant
):
    """df must come from load_data (filtered + checked). seed only controls batch shuffling."""
    check_integrity(df, seq_col, Cas9)

    seqs = df[seq_col].astype(str).tolist()
    X = one_hot_encode_sequences(seqs)
    Y = df[[bf.TARGET_COLUMNS[VARIANTS[Cas9]]]].to_numpy(dtype=np.float32)  # N x 1

    glob, chans = bf.bio_matrix(seqs, cache_path=cache_path)
    B = np.concatenate([glob, chans.reshape(len(seqs), -1)], axis=1)

    split = df[f"{Cas9}_split"].to_numpy()
    idx = {name: np.where(split == name)[0] for name in ("train", "val", "test")}

    # scaler sees training rows only
    scaler = bf.Standardizer().fit(B[idx["train"]])
    Bs = scaler.transform(B)

    def make(name: str, shuffle: bool):
        i = idx[name]
        ds = TensorDataset(torch.from_numpy(X[i]), torch.from_numpy(Bs[i]), torch.from_numpy(Y[i]))
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=g)

    print(f"Train size: {len(idx['train'])}, Val size: {len(idx['val'])}, Test size: {len(idx['test'])}")
    return make("train", True), make("val", False), make("test", False), scaler

def main() -> None:
    
    p = argparse.ArgumentParser(description=__doc__)
    
    p.add_argument("data", help="CSV/XLSX with a 21-mer column and three efficiency columns")
    p.add_argument("--seq_col", default="21mer", help="Column name for the 21-mer sequences")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--bio-cache", default="bio_features.npz")
    p.add_argument("--no-bio", action="store_true", help="Disable bio features, ablation sequence only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--init-seed", type=int, default=42, help="model init + batch order; the data split is fixed in the CSV")
    args = p.parse_args()
    
    set_seed(args.init_seed)
    
    df = load_data(args.data, args.seq_col)
    print(f"Loaded {len(df)} sequences from {args.data}")
    
    cfg = cnn.ModelConfig(use_bio=not args.no_bio)
    train_loader, val_loader, test_loader, scaler = build_loaders(
        df, args.seq_col, cfg, args.batch_size, args.init_seed, cache_path=args.bio_cache
    )
    
    model = cnn.GuideCNN(cfg)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params} trainable parameters. Bio branch: {'off' if args.no_bio else 'on'}")
    
    model = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        checkpoint_path=f"best_model{'_nobio' if args.no_bio else ''}.pt"
    )
    
    _, test_spearman = evaluate_model(model, test_loader, args.device)
    
    print("held-out test Spearman correlations:", ", ".join(f"{r:.4f}" for r in test_spearman))
    
    print(f"{'mean':20s}{np.mean(test_spearman):.4f}")
if __name__ == "__main__":
    main()
    

    
    
