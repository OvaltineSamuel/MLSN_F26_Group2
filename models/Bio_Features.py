import Bio.SeqUtils
from Bio.SeqUtils import MeltingTemp as mt
import RNA
import numpy as np




SEQ_LEN = 21 #20-nt spacer plus first nt in PAM
SPACER_LEN = 20

SCAFFOLD = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTT"

TARGET_COLUMNS = ["WT_Efficiency", "eSpCas9_Efficiency", "SpCas9_HF1_Efficiency"]

TARGET_BY_CAS9 = {
    "WT":  "WT_Efficiency",
    "eSp": "eSpCas9_Efficiency",
    "HF1": "SpCas9_HF1_Efficiency",
}

SCAFFOLD_REF = RNA.fold(SCAFFOLD.replace("T", "U"))[0]

BASES = "ACGT"
DINUCS = [a + b for a in BASES for b in BASES]

GLOBAL_FEATURE_NAMES = (
    ["gc_content", "gc_count"]
    + [f"count_{b}" for b in BASES]
    + [f"count_{d}" for d in DINUCS]
    + ["tm_full", "tm_1_4", "tm_16_20", "tm_5_13",
       "mfe_spacer", "mfe_full", "mfe_difference",
       "ensemble_energy", "ensemble_diversity",
       "scaffold_integrity", "scaffold_bp_lost",
       "accessibility_mean", "accessibility_pam_prox",
       "intra_pairing_mean", "scaffold_pairing_mean"]
)
POSITIONAL_CHANNEL_NAMES = ["p_unpaired", "p_intra", "p_scaffold"]
# p_intra is hairpin within spacer, scaffold is within scaffold

def melting_temp(seq:str)-> float:
  return float(mt.Tm_NN(seq))

def fold(rna:str) -> tuple[str,float]:
  structure, mfe = RNA.fold(rna)
  return structure, mfe

def unpaired_probabilities(rna:str, spacer_len: int=SPACER_LEN) -> np.ndarray:
  """Per-position probability that this NT
  is unpaired. Used to encode the position
  accessibility: ie, what are the odds a spacer
  NT is free to bind with target DNA

  Returns (p_unpaired, p_intra, p_scaffold), each length spacer_len."""

  fc = RNA.fold_compound(rna)
  # ViennaRNA object holding sequence and
  # energy rules

  _, mfe = fc.mfe()
  # finds most stable fold and saves the free energy of it

  fc.exp_params_rescale(mfe)
  #to avoid floating point overflow when we partition

  fc.pf()
  #partition function, weighs all folds by stability

  bpp = fc.bpp()
  #base pair probability matrix:
  #bpp[i][j] is probability position i is
  # paired with position j

  n = len(rna)
  intra = np.zeros(spacer_len, dtype=np.float64)
  scaffold = np.zeros(spacer_len, dtype=np.float64)
  for i in range(1, n+1):
    for j in range(i+1, n+1):
      p = bpp[i][j]
      for a, b in ((i, j), (j, i)):
                if a <= spacer_len:
                    if b <= spacer_len:
                        intra[a - 1] += p
                    else:
                        scaffold[a - 1] += p
  unpaired = np.clip(1.0 - intra - scaffold, 0.0, 1.0)
  # loops over every (i,j), if probability is nonzero
  # adds the probability of those 2 being paired such
  # that paired[i] = P(i being paired with anything)

  return unpaired, np.clip(intra, 0, 1), np.clip(scaffold, 0, 1)

def _scaffold_integrity(full_structure: str, spacer_len: int = SPACER_LEN) -> tuple[float, float]:
    ref = SCAFFOLD_REF
    tail = full_structure[spacer_len:]
    kept = sum(1 for a, b in zip(tail, ref) if a == b)
    return kept / len(ref), float(len(ref) - kept)

def compute_bio_features(seq21:str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (global_features (F,), positional_channels (3, 21))."""
    seq21 = seq21.strip().upper().replace("U","T")
    spacer = seq21[:20]

    gc_count = sum(1 for b in spacer if b in "GC")
    mono = [float(spacer.count(b)) for b in BASES]
    di = [float(sum(1 for i in range(len(spacer)-1) if spacer[i:i+2]==d)) for d in DINUCS]


    tm_full = melting_temp(spacer)
    tm_pam_distal_5 = melting_temp(spacer[0:5])
    tm_mid_8 = melting_temp(spacer[5:13])
    tm_pam_proximal_5 = melting_temp(spacer[15:20])

    spacer_rna = spacer.replace("T", "U")
    full_rna = (spacer + SCAFFOLD).replace("T", "U")

    _, mfe_spacer = fold(spacer_rna)
    struct_full, mfe_full = fold(full_rna)

    fc = RNA.fold_compound(full_rna)
    _, m = fc.mfe()
    fc.exp_params_rescale(m)
    ens_energy = float(fc.pf()[1])
    ens_div = float(fc.mean_bp_distance())
    # fold diversity: high means many folds with reasonable likelihood

    unpaired, intra, scaf = unpaired_probabilities(full_rna)

    integrity, bp_lost = _scaffold_integrity(struct_full)

    glob = np.array([
        gc_count/SPACER_LEN,
        float(gc_count),
    ] + mono + di + [
        tm_full, tm_pam_distal_5, tm_mid_8, tm_pam_proximal_5,
        float(mfe_spacer), float(mfe_full), float(mfe_full - mfe_spacer),
        ens_energy, ens_div,
        integrity, bp_lost,
        float(unpaired.mean()),
        float(unpaired[15:20].mean()),
        float(intra.mean()),
        float(scaf.mean())
    ], dtype = np.float32)

    chans = np.zeros((3, SEQ_LEN), dtype=np.float32)
    chans[0,:SPACER_LEN] = unpaired
    chans[1,:SPACER_LEN] = intra
    chans[2,:SPACER_LEN] = scaf

    return glob, chans


def bio_matrix(seqs: list[str], cache_path: str| None = None) -> tuple[np.ndarray, np.ndarray]:
  """ (N, len(GLOBAL_FEATURE_NAMES)), cache the biofeature results"""
  if cache_path:
    # np.savez_compressed silently appends ".npz"; normalise so save and load agree
    cache_path = str(cache_path)
    if not cache_path.endswith(".npz"):
      cache_path += ".npz"

    try:
      cached = np.load(cache_path)
      if len(cached["glob"]) == len(seqs):
        print(f"loaded cached bio features from {cache_path}")
        return cached["glob"], cached["chans"]
    except FileNotFoundError:
      pass
  glob = np.zeros((len(seqs), len(GLOBAL_FEATURE_NAMES)), dtype=np.float32)
  chans = np.zeros((len(seqs), len(POSITIONAL_CHANNEL_NAMES), SEQ_LEN), dtype=np.float32)
  for i, seq in enumerate(seqs):
      
    glob[i], chans[i] = compute_bio_features(seq)
    if (i+1) % 2000 == 0:
        print(f"bio features: {i+1}/{len(seqs)}")
  if cache_path:
    np.savez_compressed(cache_path, glob=glob, chans=chans)
    print(f"saved bio features to {cache_path}")
  return glob, chans

class Standardizer:
  """biofeatures are on different scales"""

  def __init__(self):
    self.mean = None
    self.std = None

  def fit(self, x: np.ndarray) -> "Standardizer":
    self.mean = x.mean(axis=0)
    std = x.std(axis=0)
    self.std = np.where(std == 0, 1.0, std)
    return self

#when training we will fit the standardizer on glob
#we will call transform on train, validate and test data

  def transform(self, x: np.ndarray) -> np.ndarray:
    return ((x - self.mean) / self.std).astype(np.float32)
