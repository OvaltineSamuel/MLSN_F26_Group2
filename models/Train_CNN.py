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
