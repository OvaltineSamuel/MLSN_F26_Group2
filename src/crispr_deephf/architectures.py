"""PyTorch architectures for tiers 2 and 3 plus the beyond-the-paper extras.

Every model here shares one contract so the training loop (Task 7) and the
notebook's run switches (Task 11) never special-case a model:

* ``forward(x, bio=None)`` where ``x`` is ``(N, 4, 21)`` float32 one-hot
  (channel-first A, C, G, T, exactly what :func:`torch_data.one_hot_encode`
  produces) and ``bio`` is ``(N, n_bio_features)`` float32 in the
  :func:`bio_features.flatten_bio` layout;
* the return is ``(N, n_targets)`` squashed through a sigmoid, because indel
  efficiency is a fraction in ``[0, 1]``;
* a model built with ``use_bio=True`` raises ``ValueError`` when ``bio`` is
  ``None`` instead of failing deep inside a layer;
* every parameter a model declares lies on the forward path, so one backward
  pass leaves no gradient-free weights.

Provenance
----------
``GuideCNN`` is the team's ``models/CNN.py`` ported structurally unchanged
(multi-kernel branches, batch norm, a width-3 convolution, a 1x1 bottleneck,
the PAM-proximal raw one-hot skip, and a separate bio MLP concatenated before
the head). Their explanatory comments are carried over where the code they
annotate still lives in one piece; the two that moved with the code they
describe are the batch-norm suggestion on the shared head (:func:`_make_head`)
and their multi-output data-leak warning, which is now on
``ModelConfig.n_targets``. ``GuideBiLSTM`` reproduces the tuned Keras model
from ``notebooks/BiLSTM.ipynb``. Neither teammate source is modified.

Importing this module requires torch; the package ``__init__`` does not import
it, so callers import it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from crispr_deephf.bio_features import N_BIO_FEATURES, SEQ_LEN

__all__ = [
    "EXTRA_MODELS",
    "MODEL_REGISTRY",
    "TIER2_MODELS",
    "TIER3_MODELS",
    "GuideBiLSTM",
    "GuideCNN",
    "GuideCNNBiLSTM",
    "GuideLSTM",
    "GuideTransformer",
    "ModelConfig",
    "build_model",
]

_MERGE_MODES = ("sum", "concat")

# ``Dense(16, elu) -> Dense(16, elu) -> Dense(1)``: the head width of the tuned
# Keras BiLSTM in ``notebooks/BiLSTM.ipynb``, kept verbatim (see GuideBiLSTM).
_KERAS_DENSE_UNITS = 16


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Hyper-parameters shared by every architecture in this module.

    A superset of the team's ``models/CNN.py:ModelConfig`` using the same field
    names where they overlap, so the two stay recognisably the same model. Not
    every field applies to every architecture (``kernel_widths`` means nothing
    to the RNNs); unused fields are simply ignored by that model.

    Attributes
    ----------
    kernel_widths, channels_per_width, bottleneck_channels
        Convolutional trunk: one parallel branch per kernel width, then a 1x1
        bottleneck down to ``bottleneck_channels``.
    rnn_hidden, rnn_layers, rnn_merge
        Recurrent trunk. ``rnn_merge`` ("sum" or "concat") decides how the final
        forward and backward states of a bidirectional model are combined. The
        default ``rnn_layers=2`` is here so depth can be experimented with;
        every recurrent entry in :data:`MODEL_REGISTRY` pins it back to 1 (see
        the comment above the registry).
    n_heads, n_transformer_layers
        Transformer encoder shape.
    embed_dim
        Width of the per-position ``Linear(4, embed_dim)`` projection used by
        the recurrent and transformer models.
    bio_embed_dim
        Output width of the bio-feature MLP.
    hidden_dims, dropout
        The shared prediction head (``GuideBiLSTM`` keeps the notebook's own
        head instead; see its docstring).
    n_bio_features
        Width of ``bio``; defaults to :data:`bio_features.N_BIO_FEATURES`.
    n_targets
        Output width of the head. 1 for a single Cas9 variant, 3 for the
        multi-task model. Nothing else changes between them.

        The team's warning on this field (``models/CNN.py``) still stands and
        is the reason it defaults to 1: *"head must have one output, because
        split isn't constant across cas9 variants, so multi-output will cause a
        data leak"*. With a per-enzyme split column (``WT_split``,
        ``eSp_split``, ``HF1_split``) each split is drawn independently, so one
        guide can be WT-**train** and HF1-**test**; a multi-output head trained
        on that guide's WT-train row has already seen it, and scoring it on the
        HF1-test row silently inflates the result. Resolution: ``n_targets > 1``
        is safe **only** on the project's ``shared_split`` column, which assigns
        one split per guide across all three enzymes (plus the Task 5 mask, so
        unmeasured targets contribute no loss). Never run a multi-target head
        against a per-enzyme split column.
    use_bio
        Build the bio branch. ``False`` makes the model sequence-only, and it
        then declares no bio parameters at all.
    pam_proximal_skip
        ``GuideCNN`` only: raw one-hot of the last N positions bypasses the
        convolutions; 0 disables the skip.
    """

    kernel_widths: tuple[int, ...] = (3, 5, 7)
    channels_per_width: int = 64
    bottleneck_channels: int = 32
    rnn_hidden: int = 128
    rnn_layers: int = 2
    rnn_merge: str = "sum"
    n_heads: int = 4
    n_transformer_layers: int = 2
    embed_dim: int = 64
    bio_embed_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.3
    n_bio_features: int = N_BIO_FEATURES
    n_targets: int = 1
    use_bio: bool = True
    pam_proximal_skip: int = 4

    def __post_init__(self) -> None:
        if self.rnn_merge not in _MERGE_MODES:
            raise ValueError(f"rnn_merge must be one of {_MERGE_MODES}, got {self.rnn_merge!r}")


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
class _BioMLP(nn.Module):
    """Two ``Linear``+``ReLU`` layers mapping ``bio`` to ``bio_embed_dim``.

    The team's design (``models/CNN.py``): the biophysical features get their
    own small encoder so a handful of scalars are not drowned out by the
    hundreds of sequence features they are concatenated with.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_bio_features, cfg.bio_embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.bio_embed_dim, cfg.bio_embed_dim),
            nn.ReLU(),
        )

    def forward(self, bio: torch.Tensor | None) -> torch.Tensor:
        """Encode ``bio``, raising a clear error when it was not supplied."""
        if bio is None:
            raise ValueError("model was built with use_bio=True but bio is None")
        return self.net(bio)


def _make_bio_branch(cfg: ModelConfig) -> tuple[_BioMLP | None, int]:
    """Return ``(bio_mlp, bio_dim)``; ``(None, 0)`` when ``use_bio`` is False."""
    if not cfg.use_bio:
        return None, 0
    return _BioMLP(cfg), cfg.bio_embed_dim


def _make_head(in_dim: int, cfg: ModelConfig) -> nn.Sequential:
    """Build the shared prediction head: the ``hidden_dims`` MLP then ``n_targets``.

    Parameters
    ----------
    in_dim
        Width of the concatenated trunk + skip + bio features. It must match
        what ``forward`` actually concatenates, or the head would carry
        parameters no gradient ever reaches.
    cfg
        Supplies ``hidden_dims``, ``dropout`` and ``n_targets``.
    """
    layers: list[nn.Module] = []
    for h in cfg.hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
        # consider adding batch normalization after linear for stability
        in_dim = h
    layers.append(nn.Linear(in_dim, cfg.n_targets))
    return nn.Sequential(*layers)


def _rnn_out_dim(cfg: ModelConfig, *, bidirectional: bool) -> int:
    """Width of the pooled recurrent state after applying ``rnn_merge``."""
    if bidirectional and cfg.rnn_merge == "concat":
        return 2 * cfg.rnn_hidden
    return cfg.rnn_hidden


def _final_state(h_n: torch.Tensor, cfg: ModelConfig, *, bidirectional: bool) -> torch.Tensor:
    """Pool an LSTM final hidden state ``(layers*dirs, N, H)`` to ``(N, out_dim)``.

    For a bidirectional model the last two rows are the top layer's forward and
    backward states; ``rnn_merge`` decides whether they are summed (the tuned
    Keras ``merge_mode='sum'``) or concatenated.
    """
    if not bidirectional:
        return h_n[-1]
    forward, backward = h_n[-2], h_n[-1]
    if cfg.rnn_merge == "sum":
        return forward + backward
    return torch.cat([forward, backward], dim=1)


# --------------------------------------------------------------------------- #
# Tiers 2 and 3: recurrent models
# --------------------------------------------------------------------------- #
class _GuideRNN(nn.Module):
    """Shared trunk for :class:`GuideLSTM` and :class:`GuideBiLSTM`.

    ``x`` arrives channel-first as ``(N, 4, 21)`` and is transposed to
    ``(N, 21, 4)`` here, since ``nn.LSTM(batch_first=True)`` wants positions on
    dim 1. The Keras original fed integer tokens to ``Embedding(output_dim=64)``;
    we are handed a **one-hot** ``x`` instead, so the equivalent step is a
    per-position ``Linear(4, embed_dim)`` projection. Mechanically different
    (a dense projection instead of a lookup table), equivalent in capacity: for
    one-hot input a linear layer is an embedding lookup plus a bias.
    """

    bidirectional: bool = False

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(4, cfg.embed_dim)
        self.rnn = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
        )
        self.bio_mlp, bio_dim = _make_bio_branch(cfg)
        self.head = self._build_head(
            _rnn_out_dim(cfg, bidirectional=self.bidirectional) + bio_dim, cfg
        )

    @staticmethod
    def _build_head(in_dim: int, cfg: ModelConfig) -> nn.Module:
        """Build the prediction head; subclasses may substitute their own."""
        return _make_head(in_dim, cfg)

    def forward(self, x: torch.Tensor, bio: torch.Tensor | None = None) -> torch.Tensor:
        """Run the guide through the recurrent trunk and the head."""
        _, (h_n, _) = self.rnn(self.embed(x.transpose(1, 2)))
        parts = [_final_state(h_n, self.cfg, bidirectional=self.bidirectional)]
        if self.bio_mlp is not None:
            parts.append(self.bio_mlp(bio))
        return torch.sigmoid(self.head(torch.cat(parts, dim=1)))


class GuideLSTM(_GuideRNN):
    """Unidirectional LSTM over the 21 positions, then the shared head.

    The plainest sequence-only tier-2 model, and the reference point the
    bidirectional variant has to beat.
    """

    bidirectional = False


class GuideBiLSTM(_GuideRNN):
    """Bidirectional LSTM reproducing the teammate's tuned Keras model.

    Source: ``notebooks/BiLSTM.ipynb``, whose completed grid search over
    ``learning_rate x batch_size x epochs`` settled on ``lr=0.0005``,
    ``batch_size=128``, ``epochs=20`` with ``embedding_dim=64`` and
    ``lstm_units=128``, reaching test R2 = 0.7311 (its randomized search did
    worse, 0.7100). Task 8 uses those three hyper-parameters as this model's
    ``full``-mode defaults. The architecture carried over here is

    ``Embedding(output_dim=64) -> Bidirectional(LSTM(128), merge_mode='sum')
    -> Dense(16, elu) -> Dense(16, elu) -> Dense(1)``

    hence ``embed_dim=64``, ``rnn_hidden=128`` and ``rnn_merge="sum"`` (the
    forward and backward states are *summed*, not concatenated) as defaults, and
    hence this model keeps the notebook's own two-by-16 ELU head rather than the
    shared ``hidden_dims`` one -- ``cfg.hidden_dims`` and ``cfg.dropout`` do not
    apply to it. The final ``Dense`` is linear in Keras; here it is followed by
    the sigmoid every model in this module ends with. Keras stacked a single
    ``Bidirectional(LSTM(128))``, so every recurrent entry in
    :data:`MODEL_REGISTRY` pins ``rnn_layers=1``: as registered, this is the
    tuned single-layer architecture, not a deeper variant of it.

    Also recorded from that notebook, so nobody repeats it: the teammate built
    sinusoidal positional encodings and adding them to the tokenised vectors
    made the model *worse*. :class:`GuideTransformer` therefore learns its
    positional embedding instead of using a fixed sinusoidal one.

    Two deliberate differences from the notebook: it tokenised the 20-mer
    ``gRNA_Seq`` and reported only R2 on its own re-derived split, whereas we
    train on the frozen split of the **21-mer** and score with Spearman rho,
    which is what finally makes this model comparable to the paper's RNN row.
    Do not "fix" that back.
    """

    bidirectional = True

    @staticmethod
    def _build_head(in_dim: int, cfg: ModelConfig) -> nn.Module:
        """Build the notebook's ``Dense(16, elu) -> Dense(16, elu) -> Dense(n)`` head."""
        units = _KERAS_DENSE_UNITS
        return nn.Sequential(
            nn.Linear(in_dim, units),
            nn.ELU(),
            nn.Linear(units, units),
            nn.ELU(),
            nn.Linear(units, cfg.n_targets),
        )


# --------------------------------------------------------------------------- #
# Tiers 2 and 3: the team's CNN (ported from models/CNN.py)
# --------------------------------------------------------------------------- #
class GuideCNN(nn.Module):
    """Multi-kernel 1-D CNN, ported from the team's ``models/CNN.py``.

    Structure and comments are theirs: parallel convolutions of several kernel
    widths, batch norm, a second width-3 convolution, a 1x1 bottleneck, the
    PAM-proximal raw one-hot skip, and the bio MLP concatenated before the head.
    The only changes are mechanical -- the shared :class:`_BioMLP` and
    :func:`_make_head` replace their inlined copies, and ``SEQ_LEN`` comes from
    :mod:`crispr_deephf.bio_features` rather than the teammate's module.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.branches = nn.ModuleList(
            [nn.Conv1d(4, cfg.channels_per_width, k, padding=k // 2) for k in cfg.kernel_widths]
        )
        trunk_channels = cfg.channels_per_width * len(cfg.kernel_widths)

        # normalize after the parallel convolutions
        self.bn1 = nn.BatchNorm1d(trunk_channels)
        # convolve again with kernel width 3, searching for deeper patterns
        self.conv2 = nn.Conv1d(trunk_channels, trunk_channels, kernel_size=3, padding=1)
        # same normalize again, needs to be a separate object because each
        # normalization layer keeps its own stats
        self.bn2 = nn.BatchNorm1d(trunk_channels)

        self.bottleneck = nn.Conv1d(trunk_channels, cfg.bottleneck_channels, kernel_size=1)

        seq_dim = cfg.bottleneck_channels * SEQ_LEN
        skip_dim = 4 * cfg.pam_proximal_skip

        # Bio has its own MLP so a handful of features are not drowned out by
        # the conv features; we should see how much difference they make before
        # giving them more influence.
        self.bio_mlp, bio_dim = _make_bio_branch(cfg)

        self.head = _make_head(seq_dim + skip_dim + bio_dim, cfg)

    def forward(self, x: torch.Tensor, bio: torch.Tensor | None = None) -> torch.Tensor:
        """Convolve the guide, append the skip and bio branches, predict."""
        # x: (N,4,21), b(x): (N,channels_per_width,21)
        h = torch.cat([nn.functional.relu(b(x)) for b in self.branches], dim=1)
        # h: (N,192,21), the branch outputs stack along the channel axis

        # per channel normalization
        h = self.bn1(h)
        # another width-3 convolve, then normalize, then zero negatives
        h = nn.functional.relu(self.bn2(self.conv2(h)))
        h = nn.functional.relu(self.bottleneck(h))

        parts = [h.flatten(1)]

        if self.cfg.pam_proximal_skip:
            # bypass the convolutions for the pam proximal end
            parts.append(x[:, :, -self.cfg.pam_proximal_skip :].flatten(1))

        if self.bio_mlp is not None:
            parts.append(self.bio_mlp(bio))

        # since efficiency is a fraction in [0,1]
        return torch.sigmoid(self.head(torch.cat(parts, dim=1)))


# --------------------------------------------------------------------------- #
# Extras: beyond the paper
# --------------------------------------------------------------------------- #
class GuideCNNBiLSTM(nn.Module):
    """Convolutional front-end feeding a bidirectional LSTM.

    The convolutions pick up local motifs, the 1x1 bottleneck keeps the
    per-position vector small, and the BiLSTM then reads those motif
    activations in order. Deliberately *not* the full :class:`GuideCNN` trunk:
    no second width-3 convolution and no PAM-proximal skip, because the
    recurrence already supplies long-range context and every declared parameter
    has to earn a gradient.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.branches = nn.ModuleList(
            [nn.Conv1d(4, cfg.channels_per_width, k, padding=k // 2) for k in cfg.kernel_widths]
        )
        trunk_channels = cfg.channels_per_width * len(cfg.kernel_widths)
        self.bn1 = nn.BatchNorm1d(trunk_channels)
        self.bottleneck = nn.Conv1d(trunk_channels, cfg.bottleneck_channels, kernel_size=1)

        self.rnn = nn.LSTM(
            input_size=cfg.bottleneck_channels,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
        )

        self.bio_mlp, bio_dim = _make_bio_branch(cfg)
        self.head = _make_head(_rnn_out_dim(cfg, bidirectional=True) + bio_dim, cfg)

    def forward(self, x: torch.Tensor, bio: torch.Tensor | None = None) -> torch.Tensor:
        """Convolve, read the motif activations with a BiLSTM, predict."""
        h = torch.cat([nn.functional.relu(b(x)) for b in self.branches], dim=1)
        h = nn.functional.relu(self.bn1(h))
        h = nn.functional.relu(self.bottleneck(h))
        # (N, bottleneck, 21) -> (N, 21, bottleneck) for batch_first recurrence
        _, (h_n, _) = self.rnn(h.transpose(1, 2))
        parts = [_final_state(h_n, self.cfg, bidirectional=True)]
        if self.bio_mlp is not None:
            parts.append(self.bio_mlp(bio))
        return torch.sigmoid(self.head(torch.cat(parts, dim=1)))


class GuideTransformer(nn.Module):
    """Transformer encoder over the 21 positions with a learned position embedding.

    Like the recurrent models this transposes ``(N, 4, 21)`` to ``(N, 21, 4)``
    and projects with ``Linear(4, embed_dim)`` -- the one-hot equivalent of an
    embedding lookup. The positional embedding is *learned* rather than
    sinusoidal, following the teammate's finding (``notebooks/BiLSTM.ipynb``)
    that sinusoidal encodings hurt on this data. The encoder output is
    mean-pooled over positions before the head.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Linear(4, cfg.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, cfg.embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.embed_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_transformer_layers, enable_nested_tensor=False
        )

        self.bio_mlp, bio_dim = _make_bio_branch(cfg)
        self.head = _make_head(cfg.embed_dim + bio_dim, cfg)

    def forward(self, x: torch.Tensor, bio: torch.Tensor | None = None) -> torch.Tensor:
        """Attend over the 21 positions, mean-pool, predict."""
        h = self.embed(x.transpose(1, 2)) + self.pos_embed
        parts = [self.encoder(h).mean(dim=1)]
        if self.bio_mlp is not None:
            parts.append(self.bio_mlp(bio))
        return torch.sigmoid(self.head(torch.cat(parts, dim=1)))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#: Run name -> (class, ``ModelConfig`` overrides). The run name is what the
#: notebook, the results table and the saved checkpoints all key on.
#:
#: Every recurrent entry pins ``rnn_layers=1`` rather than inheriting the
#: ``ModelConfig`` default of 2 (which stays, so depth remains easy to
#: experiment with). Three reasons:
#:
#: 1. Fidelity: one layer is the teammate's grid-searched, empirically
#:    validated architecture. Publishing two layers while citing their tuning
#:    would misrepresent their work.
#: 2. Clean comparisons: it holds recurrent depth constant across the tier-2
#:    comparison and the tier-2-vs-tier-3 ablation. If ``BiLSTM`` were one
#:    layer and ``BiLSTM+bio`` two, "bio features help" would be confounded
#:    with "more depth helps" and the ablation would prove nothing.
#: 3. The paper: the DeepHF RNN row is a single BiLSTM layer, so this is the
#:    closer reproduction of what we measure ourselves against.
MODEL_REGISTRY: dict[str, tuple[type[nn.Module], dict[str, object]]] = {
    # Tier 2 -- sequence only.
    "LSTM": (GuideLSTM, {"use_bio": False, "rnn_layers": 1}),
    "BiLSTM": (GuideBiLSTM, {"use_bio": False, "rnn_layers": 1}),
    "CNN": (GuideCNN, {"use_bio": False}),
    # Tier 3 -- sequence plus biophysical features.
    "BiLSTM+bio": (GuideBiLSTM, {"use_bio": True, "rnn_layers": 1}),
    "CNN+bio": (GuideCNN, {"use_bio": True}),
    # Extras -- beyond the paper.
    "CNN-BiLSTM": (GuideCNNBiLSTM, {"use_bio": True, "rnn_layers": 1}),
    "Transformer": (GuideTransformer, {"use_bio": True}),
    # The one multi-target entry. Safe *only* on the ``shared_split`` column,
    # which gives a guide the same split across all three enzymes; on a
    # per-enzyme split column a guide can be WT-train and HF1-test, and a
    # 3-output head would train on it and then be scored on it -- the data leak
    # the team warned about on ``ModelConfig.n_targets`` (see its docstring).
    "MultiTask-BiLSTM+bio": (
        GuideBiLSTM,
        {"use_bio": True, "n_targets": 3, "rnn_layers": 1},
    ),
}

#: Registry names grouped for the notebook's run switches. Together these three
#: lists partition :data:`MODEL_REGISTRY` -- a name missing from all of them
#: would silently never run.
TIER2_MODELS: list[str] = ["LSTM", "BiLSTM", "CNN"]
TIER3_MODELS: list[str] = ["BiLSTM+bio", "CNN+bio"]
EXTRA_MODELS: list[str] = ["CNN-BiLSTM", "Transformer", "MultiTask-BiLSTM+bio"]


def build_model(name: str, **overrides: object) -> nn.Module:
    """Instantiate the registered model ``name``.

    Parameters
    ----------
    name
        A key of :data:`MODEL_REGISTRY`.
    **overrides
        :class:`ModelConfig` fields overriding the registry's own overrides
        (which in turn override the dataclass defaults).

    Returns
    -------
    nn.Module
        The model, with its resolved :class:`ModelConfig` on ``.cfg``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    TypeError
        If an override is not a :class:`ModelConfig` field.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    cls, registry_overrides = MODEL_REGISTRY[name]
    cfg = replace(ModelConfig(), **{**registry_overrides, **overrides})
    return cls(cfg)
