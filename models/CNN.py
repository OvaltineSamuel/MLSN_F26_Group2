import Bio_Features as bf
import numpy as np
import torch
from dataclasses import dataclass
import torch.nn as nn
import scipy

@dataclass
class ModelConfig:
  kernel_widths: tuple[int, ...] = (3,5,7)
  channels_per_width: int=64
  bottleneck_channels: int = 32
  bio_embed_dim: int = 32
  hidden_dims: tuple[int, ...] = (256, 128)
  dropout: float = 0.3
  n_bio_features: int = len(bf.GLOBAL_FEATURE_NAMES) + len(bf.POSITIONAL_CHANNEL_NAMES) * bf.SEQ_LEN
  n_targets: int = len(bf.TARGET_COLUMNS)
  use_bio: bool = True
  pam_proximal_skip: int = 4  # raw one-hot of the last N positions, 0 to disable
  
class GuideCNN(nn.Module):
  def __init__(self, cfg: ModelConfig) -> None:
    super().__init__()
    self.cfg = cfg

    self.branches = nn.ModuleList(
        [
            nn.Conv1d(4, cfg.channels_per_width, k, padding=k//2)
            for k in cfg.kernel_widths
        ]
    )
    trunk_channels = cfg.channels_per_width * len(cfg.kernel_widths)

    self.bn1 = nn.BatchNorm1d(trunk_channels)
# normalize after 3 paralel convolutions
    self.conv2 = nn.Conv1d(trunk_channels, trunk_channels, kernel_size=3, padding=1)
# convolve again with kernel width 3, searching for deeper patterns
    self.bn2 = nn.BatchNorm1d(trunk_channels)
# same normalize again, needs to be a seperate object because
# because each normalization error keeps its own stats

    self.bottleneck = nn.Conv1d(trunk_channels, cfg.bottleneck_channels, kernel_size=1)

    seq_dim = cfg.bottleneck_channels * bf.SEQ_LEN

    skip_dim = 4*cfg.pam_proximal_skip
#===================BIO FEATURES===============================

# Bio has it's own MLP so 11 features aren't drowned out by the conv features

    if cfg.use_bio:
      self.bio_mlp = nn.Sequential(
          nn.Linear(cfg.n_bio_features, cfg.bio_embed_dim),
          nn.ReLU(),
          nn.Linear(cfg.bio_embed_dim, cfg.bio_embed_dim),
          nn.ReLU(),
      )
      bio_dim = cfg.bio_embed_dim
    else:
      self.bio_mlp = None
      bio_dim = 0

# reduce the risk of drowning out features: we should see how much difference the bio
# features make, if they need more influence

    layers: list[nn.Module] = []
    in_dim = seq_dim + skip_dim + bio_dim
    for h in cfg.hidden_dims:
      layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
# consider adding batch normalization after linear for stability
      in_dim = h

    layers.append(nn.Linear(in_dim, cfg.n_targets))
#final output

    self.head = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor, bio: torch.Tensor | None = None) -> torch.Tensor:
    # x: (N,4,21), b(x): (N,64,21)
    h = torch.cat([nn.functional.relu(b(x)) for b in self.branches], dim=1)
    # h: (N,192,21), the 3 outputs stack along the channel axis

    h = self.bn1(h)
    # per channel normalization

    h = nn.functional.relu(self.bn2(self.conv2(h)))
    # another width-3 convolve, then normalize then zero negatives

    h = nn.functional.relu(self.bottleneck(h))

    parts = [h.flatten(1)]

    if self.cfg.pam_proximal_skip:
      #bypass convolutions for the pam proximal end
      parts.append(x[:,:,-self.cfg.pam_proximal_skip :].flatten(1))

    if self.bio_mlp is not None:
      if bio is None:
        raise ValueError("model was built with use_bio=True but bio is None")
      parts.append(self.bio_mlp(bio))

    # since efficiency is a fraction in [0,1]

    return torch.sigmoid(self.head(torch.cat(parts, dim=1)))




