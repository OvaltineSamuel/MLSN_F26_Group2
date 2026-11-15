# Unified CRISPR DeepHF Colab Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one notebook that runs the whole DeepHF reproduction — cleaning, Tier 1 trees, Tier 2 sequence models, Tier 3 bio-feature models and four extra architectures — on Colab by default and locally by uncommenting a block, with a scoreboard comparing our results to the paper's.

**Architecture:** All reusable logic lives in `src/crispr_deephf/`; the notebook is narrative plus calls. Teammates' `models/*.py` is ported into the package (originals untouched) so one bio-feature implementation, one loader and one training loop serve every deep model. A run registry writes each model's results to disk as it finishes so a timed-out Colab session resumes.

**Tech Stack:** Python 3.11+, PyTorch, scikit-learn, LightGBM/XGBoost/CatBoost, ViennaRNA + Biopython (bio features), Optuna (optional), pytest + ruff + GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-14-unified-colab-pipeline.md`

## Global Constraints

- Splits: **76.5 / 8.5 / 15**, `RANDOM_SEED = 42`, per enzyme, frozen in `WT_split` / `eSp_split` / `HF1_split`. Never re-derive a split at train time; always read the frozen column.
- The bio-feature standardizer and every scaler are fit on **train rows only**.
- Headline metric is **test Spearman ρ**; Pearson *r*, RMSE, MAE, R² reported alongside. Deep models report mean ± s.d. over **3 seeds** in `full` mode, 1 seed in `quick`.
- All deep models are PyTorch and expose `forward(x, bio=None)` where `x` is `(N, 4, 21)` float32 and `bio` is `(N, n_bio)` float32.
- `models/*.py`, `notebooks/CNN_Train.ipynb` and `notebooks/BiLSTM.ipynb` are **never modified or deleted**. Legacy notebooks are `git mv`d, not deleted. `BiLSTM.ipynb` is currently untracked and must be `git add`ed in Task 1 so the teammate's work enters the repo.
- Paper comparison numbers are the article-text values in the spec table; Linear/Ridge are figure-read approximations and must be labelled `approx` wherever shown.
- Every comparison to the paper prints the tenfold-vs-single-split caveat.
- Work happens on branch `refactor/single-colab-notebook`; no direct commits to `main`.
- Tests run on CPU against synthetic data, with no raw Excel file and no network.

## Review Focus

1. **A guide measured for one enzyme but not another** — rows with `NaN` targets must be dropped per enzyme, never fed to a model as 0. Covered in Task 3 and Task 5 tests.
2. **`shared_split` on a guide with partial measurements** — the multi-task model must mask the missing enzymes in its loss rather than learn a filled-in value. Covered in Task 6 and Task 7 tests.
3. **A cached bio-feature file that does not match the current data** — a cache keyed only by row count silently returns the wrong features for a different dataset. Cache must be keyed by a content hash. Covered in Task 4 tests.
4. **A resumed run finding a half-written result file** — a session killed mid-write must not leave a corrupt result that the resume logic trusts. Writes must be atomic. Covered in Task 8 tests.
5. **A constant or near-constant prediction** — Spearman ρ is `NaN` when a model outputs one value, which silently poisons early stopping and the scoreboard. Must be handled explicitly. Covered in Task 7 tests.

---

### Task 1: Repo hygiene, test scaffolding and CI

**Files:**
- Create: `tests/conftest.py`, `tests/test_repo_layout.py`, `.github/workflows/ci.yml`, `models/README.md`
- Modify: `.gitignore`, `requirements.txt`, `environment.yml`, `pyproject.toml`
- Move: `notebooks/01_data_cleaning.ipynb` → `notebooks/legacy/01_data_cleaning.ipynb`, `notebooks/02_tier1_baselines.ipynb` → `notebooks/legacy/02_tier1_baselines.ipynb`
- Untrack: `notebooks/catboost_info/`

**Interfaces:**
- Produces: `synthetic_df` pytest fixture — a `pandas.DataFrame` with the full cleaned schema (`gRNA_ID`, `gRNA_Seq`, `PAM`, `21mer`, the three `*_Efficiency` columns), `n=400` rows of random valid `N20`+`NGG` sequences, seeded with `numpy.random.default_rng(0)`, where ~10% of each efficiency column is `NaN` and efficiencies are drawn from `rng.uniform(0, 1)`. Every later task's tests consume it.

- [ ] **Step 1: Create the branch and fill the local environment gaps**

The conda env `mlsn-f26-group2` (Python 3.12) already has numpy, pandas, sklearn, scipy, lightgbm, xgboost, catboost, seaborn, openpyxl and nbformat. It is **missing** `torch`, `ViennaRNA`, `biopython`, `optuna`, `pytest` and `ruff`, all of which later tasks need to actually verify their tests rather than skip them.

```bash
git checkout -b refactor/single-colab-notebook
conda run -n mlsn-f26-group2 pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run -n mlsn-f26-group2 pip install ViennaRNA biopython optuna pytest ruff
```

If the ViennaRNA wheel does not install on Windows, record that and run Task 4's tests on Colab instead — do not mark the task done on a skipped test.

- [ ] **Step 2: Write `tests/conftest.py` with the `synthetic_df` fixture**

Signature: `@pytest.fixture def synthetic_df() -> pd.DataFrame`. Build `21mer` as `gRNA_Seq + PAM[0]` so the data contract holds. Also provide `synthetic_seqs() -> list[str]` returning `synthetic_df["21mer"].tolist()`.

- [ ] **Step 3: Write the failing layout test**

```python
def test_package_imports_without_torch_installed():
    import crispr_deephf
    assert crispr_deephf.__version__

def test_legacy_notebooks_preserved():
    assert Path("notebooks/legacy/01_data_cleaning.ipynb").exists()
    assert Path("notebooks/legacy/02_tier1_baselines.ipynb").exists()

def test_teammate_sources_untouched():
    for name in ("Bio_Features.py", "CNN.py", "Train_CNN.py", "Experiments.py"):
        assert Path("models") .joinpath(name).exists()
```

- [ ] **Step 4: Run it and watch the legacy test fail**

Run: `pytest tests/test_repo_layout.py -v`
Expected: FAIL on `test_legacy_notebooks_preserved`.

- [ ] **Step 5: Move the legacy notebooks and untrack the CatBoost logs**

```bash
mkdir -p notebooks/legacy
git mv notebooks/01_data_cleaning.ipynb notebooks/legacy/
git mv notebooks/02_tier1_baselines.ipynb notebooks/legacy/
git rm -r --cached notebooks/catboost_info
```

- [ ] **Step 6: Update `.gitignore`**

Add `catboost_info/`, `**/catboost_info/`, `*.npz`, `runs/`, `*.pt`. Keep the existing `!models/*.py` re-include so teammates' sources stay tracked. Keep the `instagram_post/` line already in the working tree and give the file a trailing newline.

- [ ] **Step 7: Track the BiLSTM notebook and write `models/README.md`**

```bash
git add notebooks/BiLSTM.ipynb
```

`models/README.md` explains in a short paragraph that these files are the team's original Tier-3 sources, kept as written; that the pipeline notebook imports the ported copies in `src/crispr_deephf/`; and that changes should go to the package so both stay in step. Name `notebooks/BiLSTM.ipynb` and `notebooks/CNN_Train.ipynb` there too, as the Keras and PyTorch originals the package ports.

- [ ] **Step 8: Update the dependency files**

`requirements.txt` and `environment.yml` gain `pytest>=8.0` and `ruff>=0.6`. `pyproject.toml` gains a `[project.optional-dependencies] dev = ["pytest>=8.0", "ruff>=0.6"]` group and a `deep = ["torch>=2.1", "biopython>=1.82", "ViennaRNA>=2.6", "optuna>=3.5"]` group, plus `[tool.ruff] line-length = 100`.

- [ ] **Step 9: Write `.github/workflows/ci.yml`**

On `pull_request` and `push`: Ubuntu, Python 3.11, `pip install -e ".[dev]"`, then `ruff check src tests`, then `pytest -v`. Do **not** install the `deep` or `boosters` extras — tests must skip cleanly without them.

- [ ] **Step 10: Run the tests**

Run: `pytest -v`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "chore: test scaffolding, CI, repo hygiene; archive legacy notebooks"
```

---

### Task 2: Config — run modes, paper reference numbers, bio-feature constants

**Files:**
- Modify: `src/crispr_deephf/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `config.TARGET_COLS`, `TARGET_KEYS`, `RANDOM_SEED`, `Paths`, `get_paths`.
- Produces:
  - `PAPER_SPEARMAN: dict[str, dict[str, float]]` — `{model_name: {"WT": r, "eSp": r, "HF1": r}}` with the spec's exact text values for `"XGBoost"`, `"MLP"`, `"CNN"`, `"RNN"`, `"RNN+biofeature"`.
  - `PAPER_SPEARMAN_APPROX: dict[str, dict[str, float]]` — `{"Linear": ..., "Ridge": ...}`, figure-read.
  - `PAPER_CAVEAT: str` — the tenfold-vs-single-split sentence.
  - `PAPER_N_GUIDES: dict[str, int]` — `{"WT": 55604, "eSp": 58167, "HF1": 56888}`.
  - `SHARED_SPLIT_COL: str = "shared_split"`.
  - `RUN_MODES: dict[str, dict]` — `{"quick": {"epochs": 3, "n_seeds": 1, "patience": 2, "batch_size": 256}, "full": {"epochs": 60, "n_seeds": 3, "patience": 10, "batch_size": 256}}`.
  - `get_mode(name: str) -> dict` — returns a copy, raises `KeyError` with the valid names listed.
  - `RAW_XLSX_URL: str` — the Nature supplementary-data URL for `RAW_XLSX_NAME`.
  - `Paths` gains a `cache` field at `root/data/cache`, created by `mkdirs`.

- [ ] **Step 1: Write the failing test**

```python
def test_paper_numbers_are_exact_text_values():
    assert config.PAPER_SPEARMAN["RNN+biofeature"] == {"WT": 0.8670, "eSp": 0.8624, "HF1": 0.8603}
    assert config.PAPER_SPEARMAN["XGBoost"]["HF1"] == 0.8184
    assert set(config.PAPER_SPEARMAN) == {"XGBoost", "MLP", "CNN", "RNN", "RNN+biofeature"}

def test_approx_numbers_kept_separate():
    assert set(config.PAPER_SPEARMAN_APPROX) == {"Linear", "Ridge"}
    assert not set(config.PAPER_SPEARMAN) & set(config.PAPER_SPEARMAN_APPROX)

def test_run_modes():
    assert config.get_mode("quick")["n_seeds"] == 1
    assert config.get_mode("full")["n_seeds"] == 3
    config.get_mode("quick")["epochs"] = 999
    assert config.get_mode("quick")["epochs"] == 3   # returns a copy
    with pytest.raises(KeyError):
        config.get_mode("medium")

def test_paths_include_cache(tmp_path):
    p = config.get_paths(tmp_path)
    assert p.cache.is_dir()
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: module 'crispr_deephf.config' has no attribute 'PAPER_SPEARMAN'`.

- [ ] **Step 3: Add the constants to `config.py`**

Each paper block gets a comment citing where the number came from (article text p.4–5 vs Fig. 5c–e). `PAPER_CAVEAT` reads: *"The paper reports mean ± s.d. over tenfold shuffled validation (n=10); we report a single fixed split (seed 42) averaged over 3 model-init seeds, so the two are not a like-for-like significance test."*

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/config.py tests/test_config.py
git commit -m "feat(config): run modes, paper reference metrics, cache dir"
```

---

### Task 3: Data — shared split, raw-file acquisition

**Files:**
- Modify: `src/crispr_deephf/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `config.SHARED_SPLIT_COL`, `config.RAW_XLSX_URL`, existing `add_split_labels`, `variant_valid_mask`.
- Produces:
  - `add_shared_split(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame` — adds `shared_split` with `"train"`/`"val"`/`"test"` for **every** row, using the same 76.5/8.5/15 ratios, so a guide has one split across all enzymes.
  - `ensure_raw_excel(paths: config.Paths, *, allow_download: bool = True) -> Path` — returns the raw `.xlsx`, searching `paths.data_raw` and `paths.root` first, then downloading `config.RAW_XLSX_URL` into `paths.data_raw` when `allow_download`, then raising `FileNotFoundError` with upload instructions.
  - `split_counts(df: pd.DataFrame, split_col: str) -> dict[str, int]`.
  - `filter_for_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame` — rows where the target is non-null **and** the enzyme's split label is set; index reset. This is the one place rows get dropped per enzyme.

- [ ] **Step 1: Write the failing tests**

```python
def test_shared_split_covers_every_row_once(synthetic_df):
    out = data.add_shared_split(synthetic_df)
    assert out["shared_split"].isin(["train", "val", "test"]).all()

def test_shared_split_ratios(synthetic_df):
    c = data.split_counts(data.add_shared_split(synthetic_df), "shared_split")
    n = len(synthetic_df)
    assert c["test"] / n == pytest.approx(0.15, abs=0.02)
    assert c["val"] / n == pytest.approx(0.085, abs=0.02)

def test_shared_split_is_deterministic(synthetic_df):
    a = data.add_shared_split(synthetic_df)["shared_split"].tolist()
    b = data.add_shared_split(synthetic_df)["shared_split"].tolist()
    assert a == b

def test_filter_for_target_drops_unmeasured_rows(synthetic_df):
    df = data.add_split_labels(synthetic_df)
    out = data.filter_for_target(df, "WT_Efficiency")
    assert out["WT_Efficiency"].notna().all()
    assert len(out) == int(df["WT_Efficiency"].notna().sum())

def test_no_sequence_leaks_across_splits(synthetic_df):
    df = data.add_split_labels(synthetic_df)
    for key in ("WT", "eSp", "HF1"):
        g = df.dropna(subset=[f"{key}_split"]).groupby("21mer")[f"{key}_split"].nunique()
        assert (g <= 1).all()

def test_ensure_raw_excel_raises_without_file_or_download(tmp_path):
    paths = config.get_paths(tmp_path)
    with pytest.raises(FileNotFoundError):
        data.ensure_raw_excel(paths, allow_download=False)
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_data.py -v`
Expected: FAIL — `add_shared_split` undefined.

- [ ] **Step 3: Implement the four functions in `data.py`**

`add_shared_split` reuses `make_variant_split_indices(len(df), seed)` over all rows. `ensure_raw_excel` downloads with `urllib.request` into a `.part` file and renames on success, so a killed download never leaves a truncated `.xlsx`. Add every new name to `__all__`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/data.py tests/test_data.py
git commit -m "feat(data): shared multi-task split and raw-file acquisition"
```

---

### Task 4: Bio-feature module ported from the team's code

**Files:**
- Create: `src/crispr_deephf/bio_features.py`
- Test: `tests/test_bio_features.py`

**Interfaces:**
- Consumes: nothing from earlier tasks except `config`.
- Produces:
  - `SEQ_LEN = 21`, `SPACER_LEN = 20`, `SCAFFOLD: str`, `GLOBAL_FEATURE_NAMES: list[str]`, `POSITIONAL_CHANNEL_NAMES: list[str]`, `N_BIO_FEATURES: int` = `len(GLOBAL_FEATURE_NAMES) + len(POSITIONAL_CHANNEL_NAMES) * SEQ_LEN`.
  - `VIENNA_AVAILABLE: bool`, `BIOPYTHON_AVAILABLE: bool` — set by guarded imports, matching the optional-dependency pattern already in `models.py`.
  - `compute_bio_features(seq21: str) -> tuple[np.ndarray, np.ndarray]` — `(globals (F,), channels (3, 21))`, both float32.
  - `bio_matrix(seqs: Sequence[str], cache_path: Path | str | None = None, *, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]` — shapes `(N, F)` and `(N, 3, 21)`.
  - `flatten_bio(glob: np.ndarray, chans: np.ndarray) -> np.ndarray` — `(N, N_BIO_FEATURES)`, the single layout every model consumes.
  - `class Standardizer` with `fit(x) -> Standardizer`, `transform(x) -> np.ndarray`, `mean`/`std` attributes.
  - `sequence_digest(seqs: Sequence[str]) -> str` — sha256 of the joined sequences, 16 hex chars.

Port the logic from `models/Bio_Features.py` unchanged in substance. Two defects to fix while porting: the cache is validated by **`sequence_digest`** stored inside the `.npz`, not by row count; and `melting_temp` on a degenerate slice must not raise.

- [ ] **Step 1: Write the failing tests**

```python
pytestmark = pytest.mark.skipif(not bio_features.VIENNA_AVAILABLE, reason="ViennaRNA not installed")

def test_feature_shapes():
    g, c = bio_features.compute_bio_features("ACGT" * 5 + "A")
    assert g.shape == (len(bio_features.GLOBAL_FEATURE_NAMES),)
    assert c.shape == (3, 21)
    assert np.isfinite(g).all() and np.isfinite(c).all()

def test_flatten_matches_declared_width(synthetic_seqs):
    g, c = bio_features.bio_matrix(synthetic_seqs[:8], verbose=False)
    assert bio_features.flatten_bio(g, c).shape == (8, bio_features.N_BIO_FEATURES)

def test_cache_roundtrip(tmp_path, synthetic_seqs):
    p = tmp_path / "bio.npz"
    g1, c1 = bio_features.bio_matrix(synthetic_seqs[:8], cache_path=p, verbose=False)
    g2, c2 = bio_features.bio_matrix(synthetic_seqs[:8], cache_path=p, verbose=False)
    assert np.allclose(g1, g2) and np.allclose(c1, c2)

def test_cache_rejected_for_different_sequences(tmp_path, synthetic_seqs):
    p = tmp_path / "bio.npz"
    bio_features.bio_matrix(synthetic_seqs[:8], cache_path=p, verbose=False)
    g, _ = bio_features.bio_matrix(synthetic_seqs[8:16], cache_path=p, verbose=False)
    assert not np.allclose(g, bio_features.bio_matrix(synthetic_seqs[:8], verbose=False)[0])

def test_standardizer_zero_variance_column():
    x = np.array([[1.0, 5.0], [1.0, 7.0]], dtype=np.float32)
    out = bio_features.Standardizer().fit(x).transform(x)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_bio_features.py -v`
Expected: FAIL (module missing) or SKIP if ViennaRNA is absent — if it skips, install `ViennaRNA` locally first so this task is actually verified.

- [ ] **Step 3: Implement `bio_features.py`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_bio_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/bio_features.py tests/test_bio_features.py
git commit -m "feat(bio): shared ViennaRNA bio-feature module with content-keyed cache"
```

---

### Task 5: Torch data layer — encoding, integrity, loaders

**Files:**
- Create: `src/crispr_deephf/torch_data.py`
- Test: `tests/test_torch_data.py`

**Interfaces:**
- Consumes: `data.filter_for_target`, `bio_features.bio_matrix`, `bio_features.flatten_bio`, `bio_features.Standardizer`, `config.TARGET_KEYS`.
- Produces:
  - `one_hot_encode(seqs: Sequence[str]) -> np.ndarray` — `(N, 4, 21)` float32, channel-first.
  - `check_integrity(df: pd.DataFrame, seq_col: str, split_col: str, target_cols: list[str]) -> str` — asserts valid split labels, `[ACGT]{21}` sequences, finite targets in `[0,1]`, and no sequence in more than one split; returns a 12-char dataset digest. Ported from `models/Train_CNN.py:check_integrity`, generalised to several targets.
  - `@dataclass SplitTensors` with `x`, `bio`, `y`, `mask`, `ids` as numpy arrays.
  - `build_splits(df, *, target_cols, split_col, seq_col="21mer", bio=True, cache_path=None) -> tuple[dict[str, SplitTensors], Standardizer]` — returns `{"train"/"val"/"test": SplitTensors}`. `y` is `(N, len(target_cols))`, `mask` is `(N, len(target_cols))` float32 marking measured entries. The standardizer is fit on the **train** rows only and applied to all three.
  - `make_loader(split: SplitTensors, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader` — yields `(x, bio, y, mask)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_one_hot_shape_and_sum(synthetic_seqs):
    x = torch_data.one_hot_encode(synthetic_seqs[:5])
    assert x.shape == (5, 4, 21)
    assert np.allclose(x.sum(axis=1), 1.0)

def test_check_integrity_rejects_leaked_sequence(synthetic_df):
    df = data.add_split_labels(synthetic_df)
    df = data.filter_for_target(df, "WT_Efficiency")
    df.loc[0, "21mer"] = df.loc[len(df) - 1, "21mer"]
    df.loc[0, "WT_split"], df.loc[len(df) - 1, "WT_split"] = "train", "test"
    with pytest.raises(AssertionError, match="more than one split"):
        torch_data.check_integrity(df, "21mer", "WT_split", ["WT_Efficiency"])

def test_mask_marks_missing_targets(synthetic_df):
    df = data.add_shared_split(synthetic_df)
    splits, _ = torch_data.build_splits(df, target_cols=config.TARGET_COLS,
                                        split_col="shared_split", bio=False)
    tr = splits["train"]
    assert tr.mask.shape == tr.y.shape
    assert set(np.unique(tr.mask)) <= {0.0, 1.0}
    assert np.isfinite(tr.y).all()          # NaN targets zero-filled, not NaN
    assert tr.mask.sum() < tr.mask.size     # synthetic data has missing values

def test_standardizer_fit_on_train_only(synthetic_df):
    df = data.add_shared_split(synthetic_df)
    splits, scaler = torch_data.build_splits(df, target_cols=["WT_Efficiency"],
                                             split_col="shared_split", bio=False)
    assert splits["train"].x.shape[0] > splits["test"].x.shape[0]
    assert scaler is None or scaler.mean is not None
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_torch_data.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `torch_data.py`**

`torch` is imported at module top; the test module is marked `pytest.importorskip("torch")`. With `bio=False`, `bio` arrays are zero-width `(N, 0)` and the returned standardizer is `None`, so a sequence-only model needs no branch at the call site.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_torch_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/torch_data.py tests/test_torch_data.py
git commit -m "feat(torch): shared encoding, integrity checks and masked split loaders"
```

---

### Task 6: Architectures — Tiers 2 and 3 plus the extras

**Files:**
- Create: `src/crispr_deephf/architectures.py`
- Test: `tests/test_architectures.py`

**Interfaces:**
- Consumes: `bio_features.N_BIO_FEATURES`, `bio_features.SEQ_LEN`.
- Produces:
  - `@dataclass ModelConfig` — fields: `kernel_widths=(3,5,7)`, `channels_per_width=64`, `bottleneck_channels=32`, `rnn_hidden=128`, `rnn_layers=2`, `rnn_merge="sum"` (`"sum"` or `"concat"`), `n_heads=4`, `n_transformer_layers=2`, `embed_dim=64`, `bio_embed_dim=32`, `hidden_dims=(256,128)`, `dropout=0.3`, `n_bio_features=bio_features.N_BIO_FEATURES`, `n_targets=1`, `use_bio=True`, `pam_proximal_skip=4`. Superset of the team's `models/CNN.py:ModelConfig`, same field names where they overlap.
  - Modules, each `forward(x: Tensor(N,4,21), bio: Tensor(N,F) | None) -> Tensor(N, n_targets)` with a sigmoid output: `GuideLSTM`, `GuideBiLSTM`, `GuideCNN` (ported from `models/CNN.py`), `GuideCNNBiLSTM`, `GuideTransformer`.
  - `MODEL_REGISTRY: dict[str, tuple[type, dict]]` mapping the run name to its class and `ModelConfig` overrides:
    `"LSTM"`, `"BiLSTM"`, `"CNN"` → `use_bio=False`; `"BiLSTM+bio"`, `"CNN+bio"` → `use_bio=True`; `"CNN-BiLSTM"`, `"Transformer"`, `"MultiTask-BiLSTM+bio"` → `use_bio=True`, and the multi-task entry `n_targets=3`.
  - `build_model(name: str, **overrides) -> nn.Module`.
  - `TIER2_MODELS`, `TIER3_MODELS`, `EXTRA_MODELS: list[str]` — the registry names grouped for the notebook's run switches.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("name", list(architectures.MODEL_REGISTRY))
def test_forward_shape_and_range(name):
    model = architectures.build_model(name)
    x = torch.rand(4, 4, 21)
    bio = torch.rand(4, model.cfg.n_bio_features) if model.cfg.use_bio else None
    out = model(x, bio)
    assert out.shape == (4, model.cfg.n_targets)
    assert ((out >= 0) & (out <= 1)).all()

@pytest.mark.parametrize("name", list(architectures.MODEL_REGISTRY))
def test_backward_reaches_every_parameter(name):
    model = architectures.build_model(name)
    x = torch.rand(4, 4, 21)
    bio = torch.rand(4, model.cfg.n_bio_features) if model.cfg.use_bio else None
    model(x, bio).sum().backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused

def test_multitask_has_three_outputs():
    assert architectures.build_model("MultiTask-BiLSTM+bio").cfg.n_targets == 3

def test_bio_model_rejects_missing_bio():
    with pytest.raises(ValueError):
        architectures.build_model("CNN+bio")(torch.rand(2, 4, 21), None)

def test_groups_cover_registry():
    grouped = set(architectures.TIER2_MODELS + architectures.TIER3_MODELS + architectures.EXTRA_MODELS)
    assert grouped == set(architectures.MODEL_REGISTRY)
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_architectures.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `architectures.py`**

**Carrying over the team's Keras BiLSTM (`notebooks/BiLSTM.ipynb`).** That notebook is TensorFlow/Keras, tokenises the **20-mer** `gRNA_Seq` rather than the 21-mer, derives its own split instead of reading the frozen columns, and reports only R² (best: grid search, R² = 0.7311), so its numbers cannot be placed beside the paper's Spearman ρ. The PyTorch `GuideBiLSTM` therefore keeps that notebook's **tuned architecture** as its defaults — `embed_dim=64`, `rnn_hidden=128`, bidirectional outputs combined by **sum** (not concatenation), then `Dense(16, ELU) → Dense(16, ELU) → Dense(1)` — with a comment citing the notebook and its grid-search result (`lr=0.0005`, `batch_size=128`, `epochs=20`), which Task 8 uses as that model's `full`-mode defaults. The teammate's finding that sinusoidal positional encodings *hurt* is recorded in the notebook's Tier-2 markdown so nobody repeats it. Run under our frozen split on the 21-mer and scored with Spearman, this model becomes comparable to the paper's RNN row for the first time.

Shared pieces: a `_BioMLP` (two `Linear`+`ReLU` layers to `bio_embed_dim`, matching the team's design) and a `_make_head(in_dim, cfg)` building the `hidden_dims` MLP with dropout and a final `Linear(.., n_targets)`. `GuideCNN` is the team's module ported verbatim in structure — multi-kernel branches, `BatchNorm`, a width-3 conv, a 1×1 bottleneck, the PAM-proximal one-hot skip, and the bio MLP concatenated before the head. The RNN models transpose `(N,4,21)` to `(N,21,4)` and take the concatenation of the final forward and backward hidden states. `GuideTransformer` projects to `embed_dim`, adds a learned positional embedding, and mean-pools the encoder output.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_architectures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/architectures.py tests/test_architectures.py
git commit -m "feat(models): tier-2, tier-3 and extra PyTorch architectures"
```

---

### Task 7: Training loop and metrics

**Files:**
- Create: `src/crispr_deephf/train_torch.py`
- Modify: `src/crispr_deephf/models.py` (add `safe_spearman`)
- Test: `tests/test_train_torch.py`

**Interfaces:**
- Consumes: `torch_data.SplitTensors`, `torch_data.make_loader`, `architectures.build_model`, `models.compute_metrics`.
- Produces:
  - `models.safe_spearman(y_true, y_pred) -> float` — the Spearman ρ, returning `0.0` instead of `NaN` when either input is constant.
  - `set_seed(seed: int) -> None` — ported from `models/Train_CNN.py`, seeds `random`, `numpy`, `torch`, CUDA, and sets deterministic cuDNN.
  - `masked_mse(pred, y, mask) -> Tensor` — MSE over measured entries only; returns a zero that still carries grad when the batch has no measured entry.
  - `@dataclass TrainResult` with `model: nn.Module`, `history: list[dict]`, `best_epoch: int`, `best_val_rho: float`.
  - `train_model(model, train_loader, val_loader, *, epochs, lr=1e-3, weight_decay=1e-4, patience, device, checkpoint_path) -> TrainResult` — Adam, `ReduceLROnPlateau` on mean val ρ, early stopping on mean val ρ, best weights restored. Ported from `models/Train_CNN.py:train`, extended for masked multi-target loss.
  - `predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(y_pred, y_true, mask)`.
  - `evaluate_split(model, loader, device, target_names) -> list[dict]` — one `compute_metrics` dict per target, over measured entries only.

- [ ] **Step 1: Write the failing tests**

```python
def test_safe_spearman_on_constant_prediction():
    assert models.safe_spearman(np.array([1.0, 2, 3]), np.array([0.5, 0.5, 0.5])) == 0.0

def test_masked_mse_ignores_unmeasured_entries():
    pred = torch.tensor([[0.0, 9.9]]); y = torch.tensor([[0.0, 0.0]]); mask = torch.tensor([[1.0, 0.0]])
    assert train_torch.masked_mse(pred, y, mask).item() == pytest.approx(0.0)

def test_set_seed_makes_init_reproducible():
    train_torch.set_seed(0); a = architectures.build_model("CNN")
    train_torch.set_seed(0); b = architectures.build_model("CNN")
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)

def test_training_runs_and_improves_or_holds(synthetic_df):
    # two epochs on synthetic data: assert it completes, records history, restores best
    res = ...  # build loaders with bio=False, train "CNN" for epochs=2, patience=1, device="cpu"
    assert len(res.history) == 2
    assert {"epoch", "train_loss", "val_loss", "val_rho"} <= set(res.history[0])
    assert np.isfinite(res.best_val_rho)

def test_evaluate_split_returns_one_row_per_target(synthetic_df):
    rows = ...  # evaluate a 3-target multitask model on the synthetic test split
    assert len(rows) == 3
    assert all("Spearman_rho" in r for r in rows)
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_train_torch.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `safe_spearman` in `models.py` and `train_torch.py`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_torch.py tests/test_architectures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/train_torch.py src/crispr_deephf/models.py tests/test_train_torch.py
git commit -m "feat(train): masked multi-target training loop with Spearman early stopping"
```

---

### Task 8: Experiment runner with resume

**Files:**
- Create: `src/crispr_deephf/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: everything from Tasks 3–7, `config.get_mode`, `config.Paths`.
- Produces:
  - `result_key(model_name: str, target_key: str, seed: int) -> str` — `f"{model_name}__{target_key}__seed{seed}"`, filesystem-safe.
  - `save_result(results_dir: Path, key: str, payload: dict) -> Path` — writes JSON **atomically** (temp file in the same directory, then `os.replace`).
  - `load_results(results_dir: Path) -> list[dict]` — reads every valid result JSON, skipping and warning about unreadable ones.
  - `run_deep_model(df, *, model_name, target_cols, split_col, paths, mode, seeds, device, force=False) -> list[dict]` — trains one architecture for the given targets, once per seed; skips any seed whose result JSON already exists unless `force`; saves each result as it finishes. Each payload holds `model_name`, `target`, `seed`, `mode`, `test` and `val` metric dicts, `best_epoch`, `history`, `duration_s`, and the `check_integrity` data digest.
  - `run_tier1(df, X, feature_names, *, paths, force=False) -> list[dict]` — the existing Tier-1 loop moved out of notebook 2, plus the booster mean-ensemble, saving results the same way.
  - `run_stack(results: list[dict], df, *, target_col, paths) -> dict` — the extras' stacking ensemble: fits a `Ridge` meta-learner on the **validation**-split predictions of the best tree model and the best deep model, then scores it on the test split. Validation-fitted, never test-fitted.
  - `results_frame(results: list[dict]) -> pd.DataFrame` — long-form rows: `Model`, `Target`, `Seed`, `Split`, metric columns.
  - `aggregate(frame: pd.DataFrame) -> pd.DataFrame` — mean and s.d. per `(Model, Target)` across seeds, with an `n_seeds` column.

- [ ] **Step 1: Write the failing tests**

```python
def test_save_is_atomic_and_roundtrips(tmp_path):
    p = runner.save_result(tmp_path, "m__WT__seed0", {"model_name": "m", "Spearman_rho": 0.5})
    assert p.exists() and not list(tmp_path.glob("*.tmp*"))
    assert runner.load_results(tmp_path)[0]["Spearman_rho"] == 0.5

def test_corrupt_result_is_skipped_not_fatal(tmp_path):
    runner.save_result(tmp_path, "good__WT__seed0", {"model_name": "good"})
    (tmp_path / "broken__WT__seed0.json").write_text('{"model_name": "bro')
    loaded = runner.load_results(tmp_path)
    assert [r["model_name"] for r in loaded] == ["good"]

def test_completed_seed_is_skipped_on_rerun(tmp_path, synthetic_df):
    # run "CNN" in quick mode once, then again; the second call retrains nothing
    ...
    assert second_call_duration_total == 0.0

def test_aggregate_reports_mean_and_sd():
    frame = pd.DataFrame({"Model": ["CNN"] * 3, "Target": ["WT_Efficiency"] * 3,
                          "Seed": [0, 1, 2], "Spearman_rho": [0.80, 0.82, 0.84]})
    agg = runner.aggregate(frame)
    assert agg.loc[0, "Spearman_rho_mean"] == pytest.approx(0.82)
    assert agg.loc[0, "n_seeds"] == 3
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `runner.py`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/runner.py tests/test_runner.py
git commit -m "feat(runner): resumable multi-seed experiment runner with atomic results"
```

---

### Task 9: Benchmark against the paper, plus figures

**Files:**
- Create: `src/crispr_deephf/benchmark.py`
- Modify: `src/crispr_deephf/plots.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `runner.aggregate`, `config.PAPER_SPEARMAN`, `config.PAPER_SPEARMAN_APPROX`, `config.PAPER_CAVEAT`, `config.TARGET_KEYS`.
- Produces:
  - `PAPER_EQUIVALENT: dict[str, str]` — maps our run names to the paper's rows: `{"XGBoost": "XGBoost", "CNN": "CNN", "BiLSTM": "RNN", "BiLSTM+bio": "RNN+biofeature", "Ridge": "Ridge", "Linear": "Linear"}`. Models with no paper counterpart (`CNN+bio`, `CNN-BiLSTM`, `Transformer`, `MultiTask-BiLSTM+bio`, `LightGBM`, `CatBoost`, ensembles) map to nothing and show a blank paper column.
  - `scoreboard(agg: pd.DataFrame) -> pd.DataFrame` — columns `Model`, `Target`, `Ours_rho`, `Ours_sd`, `Paper_rho`, `Paper_source` (`"text"` / `"figure (approx)"` / `""`), `Delta`, `Beats_paper`.
  - `best_per_target(agg: pd.DataFrame) -> pd.DataFrame` — our best model per enzyme against the paper's best (`RNN+biofeature`).
  - `format_caveat() -> str` — returns `config.PAPER_CAVEAT` for printing under every table and figure.
  - In `plots.py`: `plot_vs_paper(scoreboard_df) -> plt.Figure` — grouped bars, ours vs paper, per enzyme, with error bars from `Ours_sd`, approximations hatched, and the caveat as the figure caption; `plot_training_curves(history_frame) -> plt.Figure`; `plot_tier_progression(agg) -> plt.Figure` — Tier 1 → 2 → 3 → extras, showing whether each tier buys anything.

- [ ] **Step 1: Write the failing tests**

```python
def test_scoreboard_marks_source_of_paper_number():
    agg = pd.DataFrame({"Model": ["BiLSTM+bio", "Ridge", "Transformer"],
                        "Target": ["WT_Efficiency"] * 3,
                        "Spearman_rho_mean": [0.87, 0.78, 0.86],
                        "Spearman_rho_std": [0.001, 0.0, 0.002], "n_seeds": [3, 1, 3]})
    sb = benchmark.scoreboard(agg).set_index("Model")
    assert sb.loc["BiLSTM+bio", "Paper_rho"] == 0.8670
    assert sb.loc["BiLSTM+bio", "Paper_source"] == "text"
    assert sb.loc["Ridge", "Paper_source"] == "figure (approx)"
    assert pd.isna(sb.loc["Transformer", "Paper_rho"])

def test_beats_paper_flag():
    ...  # 0.87 vs 0.8670 -> True; 0.86 with no paper row -> False/NA, never True
    assert sb.loc["BiLSTM+bio", "Beats_paper"] is True or sb.loc["BiLSTM+bio", "Beats_paper"] == True

def test_caveat_is_non_empty():
    assert "tenfold" in benchmark.format_caveat()

def test_plot_vs_paper_returns_figure():
    assert isinstance(plots.plot_vs_paper(sb), plt.Figure)
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `benchmark.py` and the three plot helpers**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_benchmark.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/benchmark.py src/crispr_deephf/plots.py tests/test_benchmark.py
git commit -m "feat(benchmark): ours-vs-paper scoreboard and presentation figures"
```

---

### Task 10: Optional Optuna tuning

**Files:**
- Create: `src/crispr_deephf/tuning.py`
- Test: `tests/test_tuning.py`

**Interfaces:**
- Consumes: `runner.run_deep_model`, `architectures.build_model`, `models.available_trainers`.
- Produces:
  - `OPTUNA_AVAILABLE: bool` (guarded import, same pattern as `models.py`).
  - `suggest_config(trial, model_name: str) -> dict` — `ModelConfig` overrides plus `lr` and `weight_decay`, with ranges per architecture family.
  - `search_deep(df, *, model_name, target_col, split_col, paths, n_trials=40, epochs=20, device) -> dict` — returns `{"best_params": ..., "best_value": ..., "study": ...}`, maximising val Spearman ρ. Adapted from `models/Experiments.py:cmd_search`.
  - `save_best_params(path: Path, model_name: str, params: dict) -> None` / `load_best_params(path: Path, model_name: str) -> dict` — a single `best_params.json` keyed by model name, so the notebook can apply tuned values without re-running the search.

- [ ] **Step 1: Write the failing tests**

```python
pytestmark = pytest.mark.skipif(not tuning.OPTUNA_AVAILABLE, reason="optuna not installed")

def test_suggest_config_keys_are_valid_model_config_fields():
    import optuna
    params = tuning.suggest_config(optuna.trial.FixedTrial({}), "CNN")
    fields = {f.name for f in dataclasses.fields(architectures.ModelConfig)} | {"lr", "weight_decay"}
    assert set(params) <= fields

def test_best_params_roundtrip(tmp_path):
    p = tmp_path / "best_params.json"
    tuning.save_best_params(p, "CNN", {"lr": 0.001})
    tuning.save_best_params(p, "BiLSTM", {"lr": 0.002})
    assert tuning.load_best_params(p, "CNN") == {"lr": 0.001}
    assert tuning.load_best_params(p, "BiLSTM") == {"lr": 0.002}   # first entry not clobbered
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_tuning.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `tuning.py`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_tuning.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crispr_deephf/tuning.py tests/test_tuning.py
git commit -m "feat(tuning): optional Optuna search with persisted best params"
```

---

### Task 11: Assemble the unified notebook

**Files:**
- Create: `notebooks/crispr_deephf_pipeline.ipynb`
- Modify: `src/crispr_deephf/__init__.py` (export the new modules)
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: the whole package.
- Produces: the notebook. Section order:
  0. Title, paper reference, three-tier plan, how to run (markdown).
  1. **Setup cell** — Colab block active, LOCAL block commented out, runtime mismatch warning.
  2. **Settings cell** — `MODE`, `RUN_TIER1`/`2`/`3`/`EXTRAS`, `RUN_TUNING = False`, `DEVICE`, `FORCE_RERUN = False`.
  3. Data acquisition and cleaning → writes the cleaned CSV with frozen splits and `shared_split`.
  4. EDA: summary table, efficiency distributions (paper Fig. 2d).
  5. Features: 439 tabular features and the sequence encodings, explained.
  6. Bio features: computed once, cached, explained.
  7. Tier 1 — trees and ensemble.
  8. Tier 2 — LSTM, BiLSTM, CNN.
  9. Tier 3 — BiLSTM+bio, CNN+bio.
  10. Extras — CNN-BiLSTM, Transformer, multi-task, stacking.
  11. Optional Optuna cell, off by default.
  12. **Results** — aggregate table, scoreboard vs paper, all figures saved to `results/figures/`, caveat printed, and a markdown cell to be filled in after the full run.

- [ ] **Step 1: Write the failing notebook contract test**

```python
NB = Path("notebooks/crispr_deephf_pipeline.ipynb")

def test_notebook_is_valid_and_has_no_outputs():
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")

def test_setup_cell_has_colab_active_and_local_commented():
    src = nbformat.read(NB, as_version=4).cells[1].source
    assert "google.colab" in src
    assert "[A] GOOGLE COLAB" in src and "[B] LOCAL" in src
    local = src.split("[B] LOCAL")[1]
    assert all(l.strip().startswith("#") or not l.strip() for l in local.splitlines())

def test_every_code_cell_compiles():
    for c in nbformat.read(NB, as_version=4).cells:
        if c.cell_type == "code" and not c.source.lstrip().startswith(("!", "%")):
            compile(c.source, "<cell>", "exec")

def test_switches_and_sections_present():
    nb = nbformat.read(NB, as_version=4)
    all_src = "\n".join(c.source for c in nb.cells)
    for token in ("MODE", "RUN_TIER1", "RUN_TIER2", "RUN_TIER3", "RUN_EXTRAS", "RUN_TUNING"):
        assert token in all_src
    assert sum(1 for c in nb.cells if c.cell_type == "markdown") >= 12
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_notebook.py -v`
Expected: FAIL — notebook missing.

- [ ] **Step 3: Build the notebook**

Write it with `nbformat` from a Python script so the JSON is well-formed, then delete the script. Every code cell is preceded by a markdown note aimed at a teammate who did not write that section: what it does, why the paper does it that way, and what to look at in the output.

- [ ] **Step 4: Run tests**

Run: `pytest -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/crispr_deephf_pipeline.ipynb src/crispr_deephf/__init__.py tests/test_notebook.py
git commit -m "feat(notebook): unified Colab/local pipeline notebook"
```

---

### Task 12: Quick-mode end-to-end verification and docs

**Files:**
- Modify: `README.md`
- Create: `tests/test_smoke_pipeline.py`

**Interfaces:**
- Consumes: everything.
- Produces: a `quick`-mode smoke test proving the pipeline runs end to end on CPU without the raw Excel file.

- [ ] **Step 1: Write the smoke test**

```python
def test_quick_pipeline_end_to_end(tmp_path, synthetic_df):
    """Clean -> split -> tier1 -> one deep model -> scoreboard, on synthetic data."""
    paths = config.get_paths(tmp_path)
    df = data.add_shared_split(data.add_split_labels(synthetic_df))
    X, names = features.extract_crispr_features(df["21mer"].tolist(), verbose=False)
    r1 = runner.run_tier1(df, X, names, paths=paths)
    r2 = runner.run_deep_model(df, model_name="CNN", target_cols=["WT_Efficiency"],
                               split_col="WT_split", paths=paths,
                               mode=config.get_mode("quick"), seeds=[0], device="cpu")
    sb = benchmark.scoreboard(runner.aggregate(runner.results_frame(r1 + r2)))
    assert len(sb) >= 2
    assert sb["Ours_rho"].notna().all()
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_smoke_pipeline.py -v`
Expected: PASS (fix anything it exposes — this is the first time the pieces meet).

- [ ] **Step 3: Execute the real notebook in quick mode locally**

Run: `jupyter nbconvert --to notebook --execute notebooks/crispr_deephf_pipeline.ipynb --output /tmp/quick_check.ipynb` with the LOCAL block active and `MODE="quick"`, against the real `data/raw/` Excel file.
Expected: every cell runs with no exception. Record the wall-clock time. **Do not commit the executed copy.**

- [ ] **Step 4: Rewrite `README.md`**

Update: the repo layout (new modules, `notebooks/legacy/`, `models/README.md`), the one-notebook workflow, the Colab-first instructions, the run switches and `MODE`, the model roster across all tiers plus extras, the paper's reference numbers as the target to beat with the tenfold caveat, how to run the tests, and a note that the full-run results commit follows.

- [ ] **Step 5: Run the full suite and lint**

Run: `ruff check src tests && pytest -v`
Expected: PASS.

- [ ] **Step 6: Commit and open the PR**

```bash
git add -A
git commit -m "docs: README for the unified pipeline; add end-to-end smoke test"
git push -u origin refactor/single-colab-notebook
```

Open the PR describing: one notebook replacing three, teammates' code ported with originals preserved, the new models, the resume mechanism, and that the full Colab run lands as a follow-up commit.

---

## After the full Colab run (not part of this plan's tasks)

The user runs `MODE="full"` on Colab Pro. With the real result files in hand:
fill in the "why our numbers differ from the paper" markdown cell from the real
numbers, commit the executed notebook with outputs, and add the slide-ready
figures.
