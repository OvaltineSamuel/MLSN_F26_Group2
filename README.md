# 🧬 CRISPR-Cas9 Guide-RNA Efficiency Prediction — MLSN F26 Group 2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OvaltineSamuel/MLSN_F26_Group2/blob/refactor/single-colab-notebook/notebooks/crispr_deephf_pipeline.ipynb)

> **Badge note:** the link above points at the `refactor/single-colab-notebook`
> branch, because that is where the notebook currently lives. **Once this branch
> is merged, change `refactor/single-colab-notebook` to `main` in two places:**
> that URL, and `REPO_BRANCH` in the notebook's setup cell. A badge pointing at
> `main` would 404 until then, and the setup cell clones `REPO_BRANCH`
> explicitly so a pre-merge click does not land on a `main` that has no
> `crispr_deephf` package.

Predicting the indel (cleavage) efficiency of CRISPR-Cas9 guide RNAs directly
from sequence, reproducing and extending the **DeepHF** study:

> Wang *et al.*, *"Optimized CRISPR guide RNA design for two high-fidelity Cas9
> variants by deep learning"*, **Nature Communications** 10, 4284 (2019).
> <https://www.nature.com/articles/s41467-019-12281-8>

We predict efficiency for three nucleases — **WT-SpCas9**, **eSpCas9(1.1)** and
**SpCas9-HF1** — from the 59,852-guide DeepHF supplementary dataset.

This is a **reproduction-and-extension exercise for a machine-learning course**,
not biological research. No wet-lab or clinical claims are made anywhere in it.

---

## 🚀 One notebook, start to finish

Everything the team runs lives in **one** notebook:

```
notebooks/crispr_deephf_pipeline.ipynb
```

It goes raw `.xlsx` → cleaning → frozen splits → EDA → features → Tier 1 →
Tier 2 → Tier 3 → extras → scoreboard vs the paper → slide-ready figures, in
that order, with a markdown note before each section explaining what it does
and why (written for teammates who did not write that code).

### Colab (the default, and the way the reported run happens)

1. Click the **Open in Colab** badge above.
2. **Runtime → Change runtime type → GPU** (A100 or L4 on Colab Pro for the
   full run; quick mode is fine on CPU).
3. **Run all.** Cell 1 ships with the **[A] GOOGLE COLAB** block active: it
   mounts Drive, clones/pulls this repo, `pip install -e .[deep,boosters]`, and
   points the project root at the shared Drive folder so every artifact
   survives a disconnect.
4. Nothing else needs editing. The notebook ships as `MODE = "quick"`; see the
   switches below before starting a full run.

### Locally

Cell 1's second block, **[B] LOCAL**, is shipped commented out. To go local:
comment out block [A], select every line below the `# --- uncomment from here ---`
sentinel and press **Ctrl-/**. The cell prints a loud `RUNTIME MISMATCH` warning
if the active block does not match the kernel it is actually running on, so a
wrong choice announces itself immediately instead of failing three cells later.

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate mlsn-group-2
pip install -e ".[deep,boosters]"
```

Then put the supplementary table at `data/raw/41467_2019_12281_MOESM3_ESM.xlsx`.
You do not have to: the notebook resolves the raw file in order — (1) the
project `data/raw/`, (2) automatic download from the Nature supplementary-data
URL, (3) a manual upload prompt.

---

## 🎛️ Run switches

A single settings cell near the top of the notebook (section 2b) controls the
whole run:

| Switch | Ships as | What it does |
|---|---|---|
| `MODE` | `"quick"` | `"quick"` = 3 epochs, 1 seed, patience 2 — an end-to-end smoke test, **never a reported number**. `"full"` = 60 epochs, 3 seeds, patience 10 — the configuration we report. Within a mode **every deep model trains on the same schedule**; see *One schedule for every deep model* below. |
| `RUN_TIER1` | `True` | Ridge / LightGBM / XGBoost / CatBoost / mean-ensemble on the 439 tabular features |
| `RUN_TIER2` | `True` | LSTM, BiLSTM, CNN (sequence only) |
| `RUN_TIER3` | **`False`** | BiLSTM+bio, CNN+bio (sequence + bio features). Off by default: these need the bio features, and `MODE` does not make them cheap — a *quick* Tier-3 run still folds the enzyme's full row set. |
| `RUN_EXTRAS` | **`False`** | CNN→BiLSTM, Transformer, multi-task, stacking. Off for the same reason. |
| `RUN_TUNING` | `False` | Optuna search (section 11). Off on purpose — it is hours of GPU time and its results are **not** applied automatically. |
| `FORCE_RERUN` | `False` | `True` ignores every saved result and retrains from scratch |

**A reported run needs three edits: `MODE = "full"`, `RUN_TIER3 = True` and
`RUN_EXTRAS = True`.** The shipped state is the cheap end-to-end check: quick mode with
the two bio tiers off folds nothing, so *Run all* costs minutes. Turning the bio tiers on
costs **ViennaRNA folding time before the first reported epoch**: about **65 min on Colab for
WT-SpCas9 alone** (the shipped `ENZYMES = ["WT"]`: 55,604 folds at 70 ms/guide measured on a
Colab A100 instance; 38.7 ms/guide on a desktop core), or **about 4.5 h for all three enzymes plus
multi-task** (230,961 folds). The GPU does not help: folding is single-threaded CPU code. Section 6
of the notebook has the arithmetic. This is paid whatever `MODE` says, because `run_deep_model` folds the enzyme's whole row set regardless.

### One schedule for every deep model

Every deep model in every tier trains on its mode's own epochs, batch size and learning
rate. No model is special-cased, and that is deliberate: together with the pinned
`rnn_layers = 1`, it is what makes the tier deltas measurements rather than coincidences —
`LSTM` → `BiLSTM` is bidirectionality alone, `BiLSTM` → `BiLSTM+bio` is the bio features
alone. A teammate's grid search (`notebooks/BiLSTM.ipynb`: lr 0.0005, batch 128, 20
epochs, test R² 0.7311) found a better schedule for `BiLSTM` specifically; it is kept as
`runner.TUNED_OVERRIDES` and **no run mode applies it**. Passing it
(`run_deep_model(..., overrides=runner.TUNED_OVERRIDES)`) makes that one model stronger
but breaks schedule-matching against `LSTM` and `BiLSTM+bio`, so a result produced with it
must be reported separately and labelled as tuned.

### The executed notebook lives in `notebooks/executed/`

`notebooks/crispr_deephf_pipeline.ipynb` is the clean, unexecuted copy the team edits, and
the test suite holds it to two contracts: no cell outputs, and the safe shipped switch
defaults (no `MODE = "full"` in any code cell). The executed copy of the reported
`MODE = "full"` run — outputs, figures and all — is committed to **`notebooks/executed/`**
instead, where neither contract applies, so the results commit cannot turn CI red.

### Resume, not restart

Every finished run is written to `results/*.json` the moment it lands, atomically
(temp file in the same directory, then `os.replace`), so a session killed
mid-write can never leave a half-JSON that a later run would trust. On a re-run a
model is skipped only when its saved result is readable **and** its recorded
settings (mode name, epochs, patience, batch size, lr), data digest and target
list all match the current run — so a `quick` result can never be mistaken for a
`full` one. A free-Colab disconnect therefore costs you one model, not the
afternoon.

---

## 🤖 Model roster

All deep models are **PyTorch**. The three tiers plus extras partition
`architectures.MODEL_REGISTRY`, so a model missing from all of them could not
silently fail to run.

| Tier | Models | Input |
|---|---|---|
| **Tier 1 — tabular** | Ridge, LightGBM, XGBoost, CatBoost, **Ensemble** (mean of the available boosters) | 439 engineered features |
| **Tier 2 — sequence only** | LSTM, BiLSTM, CNN | 21-mer tokens / one-hot |
| **Tier 3 — sequence + bio** | **BiLSTM+bio** (the paper's winning design), CNN+bio (the team's multi-kernel GuideCNN) | sequence + 100 biophysical features |
| **Extras — beyond the paper** | CNN→BiLSTM hybrid, Transformer encoder, **multi-task** BiLSTM+bio predicting all three nucleases at once, and a **Stack** of the best tree with the best deep model (Ridge meta-learner) | as above |

The boosters are optional imports: on a Python version without wheels for them,
`models.available_trainers()` simply leaves them out and Ridge still runs. The
`Ensemble` needs at least two boosters and is absent otherwise — every
downstream table and figure handles that.

**Two safety rules are enforced in code, not by convention:**

- The per-nuclease split columns (`WT_split`, `eSp_split`, `HF1_split`) are drawn
  independently, so one guide can be WT-**train** and HF1-**test**. A
  multi-output head trained on its WT row would then be scored on it for HF1.
  `runner.run_deep_model` therefore **refuses** any multi-target run that is not
  on the `shared_split` column. (The warning comes from the team's own
  `models/CNN.py`.)
- The stack's Ridge meta-learner is fitted on **validation** predictions only and
  scored on test. Its validation metrics are in-sample and are labelled as such.

---

## 📊 The target to beat

The paper's reported test Spearman ρ (article text, p.4–5):

| Model | WT-SpCas9 | eSpCas9(1.1) | SpCas9-HF1 |
|---|---|---|---|
| XGBoost | 0.8454 | 0.8310 | 0.8184 |
| MLP | 0.8416 | 0.8457 | 0.8440 |
| CNN | 0.8455 | 0.8313 | 0.8343 |
| RNN (BiLSTM) | 0.8555 | 0.8491 | 0.8512 |
| **RNN + biofeature** | **0.8670** | **0.8624** | **0.8603** |

Linear and Ridge appear only as bars in Fig. 5c–e (≈0.75–0.79). Those are stored
separately as **figure-read approximations** (±0.01) and are labelled
`figure (approx)` wherever they appear; a "win" over one has to clear the reading
tolerance before the scoreboard calls it a win. We build no MLP, so that row
deliberately has no counterpart on our side.

> ### ⚠️ The comparability caveat
> The paper reports the **mean ± s.d. of tenfold shuffled validation (n = 10)**.
> We report a **single fixed split (seed 42)** averaged over **3 model-init
> seeds**. Our numbers are noisier and **this is not a like-for-like
> significance test**. The notebook prints this caveat under every table and as
> the caption of every comparison figure, and `benchmark.format_caveat()` is the
> single place its wording lives.

Two further honesty notes the scoreboard prints for itself: our best-per-nuclease
model is selected *post hoc* from all models scored on the test set (optimistically
biased in our favour), and the multi-task model is the only one on `shared_split`,
so its test set is a **different set of guides** and its ρ is not row-comparable
with the models beside it.

**Our own numbers are not in this README yet.** They land as a follow-up commit
after the `MODE = "full"` Colab run, together with the executed notebook, the
slide figures, and the section-13 write-up explaining *why* our numbers differ —
written from the real numbers, never predicted in advance.

---

## 📁 Repository layout

```
MLSN_F26_Group2/
├── notebooks/
│   ├── crispr_deephf_pipeline.ipynb   # ← THE notebook: the whole project (kept output-free)
│   ├── executed/                      # the executed MODE="full" copy, with outputs
│   ├── CNN_Train.ipynb                # teammate's original Colab CNN notebook (kept)
│   ├── BiLSTM.ipynb                   # teammate's original Keras BiLSTM + grid search (kept)
│   └── legacy/
│       ├── 01_data_cleaning.ipynb     # superseded by section 3 of the pipeline
│       └── 02_tier1_baselines.ipynb   # superseded by section 7
├── src/crispr_deephf/                 # all reusable logic, tested
│   ├── config.py                      # paths, schema, split ratios, paper reference numbers
│   ├── data.py                        # raw-file resolution, cleaning, validation, frozen splits
│   ├── features.py                    # sequence encodings + the 439 engineered features
│   ├── bio_features.py                # ViennaRNA/Biopython biophysical features, cached
│   ├── models.py                      # Tier-1 trainers (boosters optional) + metrics
│   ├── architectures.py               # every PyTorch model + the run-name registry
│   ├── torch_data.py                  # one-hot encoding, split tensors, integrity digests
│   ├── train_torch.py                 # training loop, early stopping, masked loss, evaluation
│   ├── runner.py                      # the resumable experiment runner and the stack
│   ├── tuning.py                      # Optuna search (resumable SQLite storage), off by default
│   ├── benchmark.py                   # ours-vs-paper scoreboard + the caveats
│   └── plots.py                       # every figure helper
├── models/
│   ├── README.md                      # ← why the notebook does not import these
│   ├── Bio_Features.py · CNN.py · Train_CNN.py · Experiments.py   # teammates' originals, untouched
│   └── *.joblib · *.pt                # trained weights (git-ignored)
├── tests/                             # pytest suite, CPU-only, synthetic data (no Excel needed)
├── data/
│   ├── raw/                           # the supplementary .xlsx (git-ignored)
│   ├── processed/                     # cleaned, split-labelled CSVs (git-ignored)
│   └── cache/                         # bio-feature .npz caches (git-ignored)
├── results/ · results/figures/        # per-run JSON + slide figures (git-ignored)
├── docs/paper/                        # the Nature paper PDF (git-ignored)
├── docs/superpowers/                  # the spec and plan this branch delivers
├── .github/workflows/ci.yml           # ruff + pytest on every PR
└── environment.yml · requirements.txt · pyproject.toml · .gitignore
```

> **Nothing of the team's work was deleted.** `models/*.py`,
> `notebooks/CNN_Train.ipynb` and `notebooks/BiLSTM.ipynb` are exactly as their
> authors wrote them. `models/README.md` explains that the pipeline imports the
> ported copies in `src/crispr_deephf/` instead, and that changes belong in the
> package so the two stay in step. The two old local-first notebooks moved to
> `notebooks/legacy/` rather than being removed.

> **Design note.** The notebook is the presentation surface — narrative, calls,
> results, figures. Every piece of reusable logic lives in `src/crispr_deephf/`
> and is tested, so the notebook and the package can never drift apart.

---

## 🔬 Method notes

- **Splits** follow the paper: **76.5 % train / 8.5 % val / 15 % test** (hold out
  15 % for test, then 10 % of the rest for validation), `seed=42`, **per
  nuclease**, frozen into `WT_split` / `eSp_split` / `HF1_split` so the notebook
  *reads* the partition rather than re-deriving it. A fourth column,
  `shared_split`, gives each guide one split across all three nucleases and
  exists **only** for the multi-task model.
- **Tier-1 features (439):** positional one-hot (84) + positional adjacent
  dinucleotides (320) + global nucleotide counts/frequencies (24) + regional GC
  content (7) + biological motifs (4, incl. the Pol III `TTTT` terminator).
- **Bio features (100 per guide):** GC, mono/dinucleotide counts, regional
  melting temperatures, MFE of the spacer and of the spacer+scaffold construct,
  ensemble energy and diversity, scaffold integrity, accessibility, plus three
  per-position channels over the 21 positions. Computed **once** and cached;
  the standardizer is fit on **train rows only**.
  **They are expensive:** 70 ms per guide measured on a Colab A100 instance (38.7 ms on a
  desktop core; the GPU is idle), so WT-SpCas9 alone is about **65 min** and the four cache
  files (230,961 folds for 59,852 distinct guides) about **4.5 hours**. The notebook's
  `ENZYMES` setting (shipped `["WT"]`) picks which enzymes are modelled; the multi-task model
  is skipped automatically unless all three are. Section 6 of the notebook shows that arithmetic and
  skips the pre-warm in quick mode.
- **Metrics:** test **Spearman ρ** is the headline; Pearson *r*, RMSE, MAE and R²
  alongside. Deep models report **mean ± s.d. over 3 seeds**; a single-seed run
  reports `n/a`, never `0`.
- **A finding about the paper.** Counting the supplementary data gives
  WT 55,604 · eSpCas9(1.1) **58,617** · HF1 56,888. WT and HF1 match the article
  text exactly; the text says 58,167 for eSpCas9(1.1). Two exact matches and one
  transposed pair point at a digit transposition in the published text.
  `config.PAPER_N_GUIDES` deliberately keeps the paper's own number, and nothing
  in the pipeline uses the count, so no result depends on it. Section 4b of the
  notebook surfaces it.

---

## ✅ Tests and lint

The suite runs on **CPU** against a **synthetic** 400-guide dataset, so it needs
neither the Excel file nor a network nor a GPU:

```bash
pip install -e ".[dev]"      # add [deep,boosters] to run the torch/booster tests too
ruff check src tests
pytest -v
```

What it covers: the data contract and PAM repair, split ratios and the guarantee
that no sequence leaks across splits, feature shapes, one forward pass per
registered architecture, the resume/atomic-write behaviour, the scoreboard's
honesty guards, the Optuna wrapper, and a static contract check on the notebook
itself (valid `nbformat`, no committed outputs, every call matching the real
package signature). It also pins the invariants the reported numbers rest on:
every deep model sharing one training schedule per mode, a warning when results
from two run modes are pooled into one mean, recovery from a truncated or empty
bio-feature cache, and the diagnostics cell's refusal to pair predictions with
the dataframe by anything other than guide id.
`tests/test_smoke_pipeline.py` is the end-to-end one:
split → features → Tier 1 → a deep model → stack → scoreboard → figures, in
quick mode, in one test.

Tests that need torch or a booster `importorskip` cleanly, so CI installs only
the `dev` extra. GitHub Actions runs `ruff` + `pytest` on every PR.

---

## 👥 Team

MLSN F26 — Group 2 (6 members). Team lead: Samuel Huang.
