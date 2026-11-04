import Bio_Features as bf
import CNN as cnn
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr


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
        
        
