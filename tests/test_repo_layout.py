"""Repository layout guards: package imports cleanly, legacy and teammate files survive."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_without_torch_installed():
    import crispr_deephf

    assert crispr_deephf.__version__


def test_legacy_notebooks_preserved():
    assert (ROOT / "notebooks/legacy/01_data_cleaning.ipynb").exists()
    assert (ROOT / "notebooks/legacy/02_tier1_baselines.ipynb").exists()


def test_teammate_sources_untouched():
    for name in ("Bio_Features.py", "CNN.py", "Train_CNN.py", "Experiments.py"):
        assert (ROOT / "models" / name).exists()
