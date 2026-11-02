# 🧬 CRISPR-Cas9 Guide-RNA Efficiency Prediction — MLSN F26 Group 2

Predicting the indel (cleavage) efficiency of CRISPR-Cas9 guide RNAs directly
from sequence, reproducing and extending the **DeepHF** study:

> Wang *et al.*, *"Optimized CRISPR guide RNA design for two high-fidelity Cas9
> variants by deep learning"*, **Nature Communications** 10, 5286 (2019).
> <https://www.nature.com/articles/s41467-019-12281-8>

We predict efficiency for three nucleases — **WT-SpCas9**, **eSpCas9(1.1)**, and
**SpCas9-HF1** — from the 59,852-guide DeepHF dataset.

## 🎯 Three-tier modelling plan

| Tier | Approach | Status |
|------|----------|--------|
| **1 — Baselines** | Gradient-boosted trees (LightGBM / XGBoost / CatBoost) + Ridge, on hand-crafted sequence features | ✅ implemented |
| **2 — Deep learning** | LSTM, BiLSTM, and CNN on sequence encodings | 🔜 planned |
| **3 — Bio-features + multi-task** | BiLSTM & CNN augmented with biological features; shared multi-task backbone | 🔜 planned |

## 📁 Repository layout

```
mlsn-group-2/
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Tier 0: raw .xlsx  ->  clean, split-labelled CSV
│   └── 02_tier1_baselines.ipynb    # Tier 1: feature engineering + model benchmark
├── src/crispr_deephf/              # reusable, importable pipeline code
│   ├── config.py                   # paths, schema, split ratios, constants
│   ├── data.py                     # load / clean / validate / split
│   ├── features.py                 # sequence encodings + 439 biological features
│   ├── models.py                   # baseline trainers + metrics (boosters optional)
│   └── plots.py                    # figure helpers
├── data/
│   ├── raw/                        # raw .xlsx (git-ignored)
│   └── processed/                  # cleaned CSVs from notebook 1 (git-ignored)
├── models/                         # serialized models (git-ignored)
├── results/figures/                # metrics + figures (git-ignored)
├── docs/paper/                     # the Nature paper PDF (git-ignored)
├── environment.yml  ·  requirements.txt  ·  pyproject.toml
└── .gitignore
```

> **Design note.** The notebooks are the presentation surface (narrative +
> results + figures); all reusable logic lives in `src/crispr_deephf/`, so both
> notebooks share one tested implementation and can never drift apart.

## 🚀 Getting started (local)

1. **Create the environment** (recreates a clean `mlsn-group-2` env):
   ```bash
   conda env remove -n mlsn-group-2      # only if a broken/empty one exists
   conda env create -f environment.yml
   conda activate mlsn-group-2
   ```
   Or with pip: `pip install -r requirements.txt`.

2. **Place the raw dataset** at:
   ```
   data/raw/41467_2019_12281_MOESM3_ESM.xlsx
   ```
   (DeepHF Supplementary Data — download from the paper if it's missing.)

3. **Run the notebooks in order:**
   - `notebooks/01_data_cleaning.ipynb` → writes `data/processed/deephf_clean_all.csv`
   - `notebooks/02_tier1_baselines.ipynb` → trains models, writes figures &
     predictions to `results/`

## ☁️ Running on Google Colab

Each notebook's **first cell** has two setup blocks: **[A] Local** (active) and
**[B] Google Colab** (commented out). To use Colab's faster hardware, comment out
block [A] and uncomment block [B], which installs dependencies, mounts Drive (or
clones this repo), and points `ROOT` at the project folder. Everything after the
setup cell is identical on both platforms.

## 🔬 Method notes

- **Data split** follows the paper: **76.5 % train / 8.5 % val / 15 % test**
  (hold out 15 % for test, then 10 % of the rest for validation), `seed=42`.
  The split is **frozen into `*_split` columns** by notebook 1 so notebook 2
  reproduces the exact same partition.
- **Tier-1 features (439):** positional one-hot (84) + positional dinucleotides
  (320) + global nucleotide counts/frequencies (24) + regional GC content (7) +
  biological motifs (4, incl. the Pol III `TTTT` terminator).
- **Metrics:** Pearson *r* and Spearman *ρ* (the paper's headline metrics), plus
  RMSE, MAE, and R².

## ⚠️ Python 3.14 note

`lightgbm` / `xgboost` / `catboost` may not yet ship wheels for the newest Python
releases. `environment.yml` pins **Python 3.11** for reliable installs. If you
run on 3.14, `models.py` imports each booster optionally and the notebook simply
**skips** any that aren't installed (Ridge always runs).

## 👥 Team

MLSN F26 — Group 2 (6 members).
