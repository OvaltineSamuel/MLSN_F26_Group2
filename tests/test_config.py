"""Config constants: paper reference metrics, run modes and the cache directory."""

from __future__ import annotations

import pytest

from crispr_deephf import config


def test_paper_spearman_full_table():
    # Every cell, literal, from the article text (p.4-5). Order: WT / eSp / HF1.
    assert config.PAPER_SPEARMAN == {
        "XGBoost": {"WT": 0.8454, "eSp": 0.8310, "HF1": 0.8184},
        "MLP": {"WT": 0.8416, "eSp": 0.8457, "HF1": 0.8440},
        "CNN": {"WT": 0.8455, "eSp": 0.8313, "HF1": 0.8343},
        "RNN": {"WT": 0.8555, "eSp": 0.8491, "HF1": 0.8512},
        "RNN+biofeature": {"WT": 0.8670, "eSp": 0.8624, "HF1": 0.8603},
    }


def test_paper_n_guides_full():
    assert config.PAPER_N_GUIDES == {"WT": 55604, "eSp": 58167, "HF1": 56888}


def test_approx_numbers_kept_separate():
    assert set(config.PAPER_SPEARMAN_APPROX) == {"Linear", "Ridge"}
    assert not set(config.PAPER_SPEARMAN) & set(config.PAPER_SPEARMAN_APPROX)


def test_paper_caveat_text():
    assert config.PAPER_CAVEAT
    assert "tenfold" in config.PAPER_CAVEAT


def test_raw_xlsx_url_exact():
    assert config.RAW_XLSX_URL == (
        "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-12281-8"
        "/MediaObjects/41467_2019_12281_MOESM3_ESM.xlsx"
    )


def test_shared_split_col():
    assert config.SHARED_SPLIT_COL == "shared_split"


def test_run_modes():
    assert config.get_mode("quick")["n_seeds"] == 1
    assert config.get_mode("full")["n_seeds"] == 3
    config.get_mode("quick")["epochs"] = 999
    assert config.get_mode("quick")["epochs"] == 3   # returns a copy
    with pytest.raises(KeyError) as excinfo:
        config.get_mode("medium")
    assert "quick" in str(excinfo.value)
    assert "full" in str(excinfo.value)


def test_paths_include_cache(tmp_path):
    p = config.get_paths(tmp_path)
    assert p.cache.is_dir()
    assert p.cache == p.root / "data" / "cache"
    assert p.root == tmp_path.resolve()
