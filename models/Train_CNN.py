import Bio_Features as bf
import CNN as cnn
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
import pandas as pd


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
        patience=patience//2, 
        verbose=True)
    
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

def load_data(path: str, seq_col: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df

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
    cache_path: str | None
):
    seqs = df[seq_col].astype(str).tolist()
    
    X = one_hot_encode_sequences(seqs)
    
    Y = df[bf.TARGET_COLUMNS].to_numpy(dtype=np.float32)
    
    if Y.max() > 1.5:
        Y = Y/100.0
    B = bf.bio_matrix(seqs, cache_path=cache_path)
    splits = {}
    for key, target_col in zip(["WT", "HF1", "eSp"], bf.TARGET_COLUMNS):
        s = df[f"{key}_split"].to_numpy()
        
        
