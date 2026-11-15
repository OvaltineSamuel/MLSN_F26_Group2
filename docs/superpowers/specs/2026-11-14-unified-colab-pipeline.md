# Spec — Unified CRISPR DeepHF Colab Pipeline

**Date:** 2026-11-14
**Owner:** Samuel Huang (team lead, MLSN F26 Group 2, 6 members)
**Agreed in:** grilling session, Q1–Q18

## Problem

The project currently has three notebooks that cannot run end to end as one
piece of work:

- `notebooks/01_data_cleaning.ipynb` — local-first, writes the cleaned CSV.
- `notebooks/02_tier1_baselines.ipynb` — local-first, Tier-1 trees.
- `notebooks/CNN_Train.ipynb` — Colab-only, reads a CSV name that no longer
  exists (`deephf_cleaned_all_59852.csv`), duplicates code that also lives in
  `models/*.py`.

Teammates' Tier-3 work (`models/Bio_Features.py`, `models/CNN.py`,
`models/Train_CNN.py`, `models/Experiments.py`) sits in a directory that
`.gitignore` treats as a model-artifact dump, and imports by bare module name
(`import Bio_Features as bf`), so it only works when run from inside `models/`.

## Goal

**One notebook**, `notebooks/crispr_deephf_pipeline.ipynb`, that runs the whole
project — cleaning, Tier 1, Tier 2, Tier 3 and extra models — on Google Colab
by default and locally by uncommenting one block. It must reproduce the DeepHF
paper's results and give the team a slide-ready comparison showing where we
match the paper, where we beat it, and why.

This is a **reproduction and extension exercise for a machine-learning course**,
not biological research. No wet-lab claims are made.

## Reference

Wang et al., *"Optimized CRISPR guide RNA design for two high-fidelity Cas9
variants by deep learning"*, Nature Communications 10, 4284 (2019).
PDF at `docs/paper/`.

### Paper's reported test Spearman ρ (from the article text, p.4–5)

| Model | WT-SpCas9 | eSpCas9(1.1) | SpCas9-HF1 |
|---|---|---|---|
| XGBoost | 0.8454 | 0.8310 | 0.8184 |
| MLP | 0.8416 | 0.8457 | 0.8440 |
| CNN | 0.8455 | 0.8313 | 0.8343 |
| RNN (BiLSTM) | 0.8555 | 0.8491 | 0.8512 |
| RNN + biofeature | **0.8670** | **0.8624** | **0.8603** |

Linear and Ridge appear only as bars in Fig. 5c–e (≈0.75–0.79); they are
recorded as figure-read approximations and labelled as such, never as exact.

Paper dataset sizes: WT-SpCas9 55,604 · eSpCas9(1.1) 58,167 · SpCas9-HF1 56,888
guides, from a 59,852-row supplementary table.

**Comparability caveat that must be stated wherever we compare:** the paper
reports the mean ± s.d. of **tenfold shuffled validation** (n = 10). We report a
**single fixed split** (seed 42) averaged over **3 model-init seeds**. Our
numbers are therefore noisier and not a like-for-like significance test.

## Requirements

### R1 — Single notebook, two runtimes
- `notebooks/crispr_deephf_pipeline.ipynb` is the only notebook the team runs.
- Setup cell: **Colab block active**, **LOCAL block commented out**, each clearly
  delimited. A runtime check prints a loud warning if the active block does not
  match the detected environment.
- Colab path: clone the public repo `https://github.com/OvaltineSamuel/MLSN_F26_Group2`,
  `pip install -e .`, mount Drive, read/write artifacts under the Drive project
  folder so results survive a disconnect.
- Every main section carries a markdown note explaining what it does and why,
  written for teammates who did not write that code.

### R2 — Data
- Raw file resolution order: (1) Drive project folder `data/raw/`, (2) automatic
  download from the Nature supplementary-data URL, (3) manual upload prompt.
- Cleaning as today: schema standardisation, PAM typo repair, `[0,1]` range and
  `N20 NGG` contract assertions.
- Splits stay **per enzyme** at **76.5 / 8.5 / 15**, seed 42, frozen into
  `WT_split`, `eSp_split`, `HF1_split` — unchanged, because the paper uses these
  ratios and teammates' code already reads these columns.
- **New:** a `shared_split` column assigning each guide one split across all
  three enzymes, for the multi-task model only.

### R3 — Models
All deep models in **PyTorch**.

- **Tier 1 (tabular):** Ridge, LightGBM, XGBoost, CatBoost, and a mean-ensemble
  of the available boosters, on the existing 439 engineered features.
- **Tier 2 (sequence only):** LSTM, BiLSTM, CNN.
- **Tier 3 (sequence + bio features):** BiLSTM+bio (the paper's winning design)
  and the team's multi-kernel GuideCNN+bio.
- **Extras, beyond the paper:** CNN→BiLSTM hybrid, Transformer encoder,
  multi-task model predicting all three enzymes at once (uses `shared_split`),
  and a stacking ensemble over the best tree and deep models.

### R4 — One shared bio-feature module
Both Tier-3 models and the extras consume **one** bio-feature implementation:
the team's ViennaRNA set (MFE, ensemble energy and diversity, base-pairing
probability channels, scaffold integrity, regional melting temperatures, GC,
mono/di-nucleotide counts) plus any paper feature it lacks. Features are
computed once and cached to Drive; the standardizer is fit on **train rows
only**.

### R5 — Run controls
A settings cell near the top exposes:
- `RUN_TIER1`, `RUN_TIER2`, `RUN_TIER3`, `RUN_EXTRAS` on/off switches.
- `MODE = "quick" | "full"`. `quick` = a few epochs, 1 seed, for verifying the
  notebook runs end to end. `full` = paper-level training, 3 seeds per deep
  model, sized for an A100/L4 Colab Pro GPU.
- Results are written to disk as each model finishes, and a completed model is
  skipped on re-run, so a timed-out session resumes instead of restarting.
- Optuna hyperparameter search exists but is **off by default**.

### R6 — Results and presentation
- Headline metric: **test Spearman ρ**; Pearson *r*, RMSE and R² alongside.
- Deep models report **mean ± s.d. over 3 seeds**.
- A scoreboard table and figure comparing **ours vs the paper**, per enzyme,
  with the tenfold caveat printed alongside.
- Slide-ready figures saved to `results/figures/`.
- Explanation cells for *why* our numbers differ are written **after** the real
  full run, from the real numbers — never predicted in advance.

### R7 — Nothing of the team's work is deleted
- `models/*.py` and `notebooks/CNN_Train.ipynb` stay where they are; a
  `models/README.md` explains that the notebook now uses the package copy.
- `01_data_cleaning.ipynb` and `02_tier1_baselines.ipynb` move to
  `notebooks/legacy/`.
- Work lands on branch `refactor/single-colab-notebook` as a PR, never a direct
  push to `main`.

### R8 — Engineering hygiene
- Reusable logic lives in `src/crispr_deephf/`; the notebook holds narrative,
  calls and results.
- `pytest` suite running on CPU against a **synthetic dataset** (no Excel file
  needed), covering: the data contract, split ratios, no sequence leaking across
  splits, feature shapes, and one forward pass per model.
- GitHub Actions on every PR: lint (`ruff`) + tests.
- `requirements.txt`, `environment.yml`, `pyproject.toml`, `.gitignore` and
  `README.md` all updated to match.
- The notebook is committed **without outputs** in the PR; the executed version
  with results is a follow-up commit after the Colab full run.

## Out of scope
- Fine-tuning a pretrained DNA language model (too heavy for the course budget).
- Off-target prediction, endogenous-site validation, the paper's transfer
  learning to U6/T7 datasets.
- Any wet-lab or clinical interpretation.

## Execution note
The agent cannot reach a Colab GPU. It builds everything and verifies **quick**
mode locally on CPU. The user then runs **full** mode on Colab Pro via the VS
Code Colab kernel and returns the result files, after which the comparison
commentary is written from the real numbers.
