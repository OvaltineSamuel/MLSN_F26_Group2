"""Config constants: paper reference metrics, run modes and the cache directory."""

from __future__ import annotations

import pytest

from crispr_deephf import config


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
