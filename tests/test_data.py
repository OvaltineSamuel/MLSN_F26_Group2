"""Tests for the data module: splits, per-target filtering, raw-file acquisition."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from crispr_deephf import config, data


def test_shared_split_covers_every_row_once(synthetic_df):
    out = data.add_shared_split(synthetic_df)
    assert out["shared_split"].isin(["train", "val", "test"]).all()
    assert len(out) == len(synthetic_df)


def test_shared_split_ratios(synthetic_df):
    c = data.split_counts(data.add_shared_split(synthetic_df), "shared_split")
    n = len(synthetic_df)
    assert sum(c.values()) == n
    assert c["test"] / n == pytest.approx(0.15, abs=0.02)
    assert c["val"] / n == pytest.approx(0.085, abs=0.02)
    assert c["train"] / n == pytest.approx(0.765, abs=0.02)


def test_shared_split_is_deterministic(synthetic_df):
    a = data.add_shared_split(synthetic_df)["shared_split"].tolist()
    b = data.add_shared_split(synthetic_df)["shared_split"].tolist()
    assert a == b


def test_shared_split_does_not_mutate_input_or_per_enzyme_columns(synthetic_df):
    labelled = data.add_split_labels(synthetic_df)
    before = labelled.copy()
    out = data.add_shared_split(labelled)
    assert "shared_split" not in labelled.columns
    for key in ("WT", "eSp", "HF1"):
        col = f"{key}_split"
        assert out[col].equals(before[col])


def test_filter_for_target_drops_unmeasured_rows(synthetic_df):
    df = data.add_split_labels(synthetic_df)
    out = data.filter_for_target(df, "WT_Efficiency")
    assert out["WT_Efficiency"].notna().all()
    assert len(out) == int(df["WT_Efficiency"].notna().sum())
    assert out.index.tolist() == list(range(len(out)))
    assert out["WT_split"].isin(["train", "val", "test"]).all()


def test_filter_for_target_drops_rows_with_unset_split(synthetic_df):
    df = data.add_split_labels(synthetic_df)
    row = df.index[df["WT_Efficiency"].notna()][0]
    df.loc[row, "WT_split"] = None
    out = data.filter_for_target(df, "WT_Efficiency")
    assert len(out) == int(df["WT_Efficiency"].notna().sum()) - 1


def test_no_sequence_leaks_across_splits(synthetic_df):
    df = data.add_shared_split(data.add_split_labels(synthetic_df))
    for key in ("WT", "eSp", "HF1", "shared"):
        g = df.dropna(subset=[f"{key}_split"]).groupby("21mer")[f"{key}_split"].nunique()
        assert (g <= 1).all()


def test_ensure_raw_excel_raises_without_file_or_download(tmp_path):
    paths = config.get_paths(tmp_path)
    with pytest.raises(FileNotFoundError):
        data.ensure_raw_excel(paths, allow_download=False)


def test_ensure_raw_excel_finds_existing_file(tmp_path):
    paths = config.get_paths(tmp_path)
    f = paths.data_raw / config.RAW_XLSX_NAME
    f.write_bytes(b"x")
    assert data.ensure_raw_excel(paths, allow_download=False) == f


def test_ensure_raw_excel_download_is_atomic(tmp_path, monkeypatch):
    paths = config.get_paths(tmp_path)
    payload = b"PK-fake-xlsx"

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            out, self._data = self._data[:n], self._data[n:]
            return out

        _data = payload

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: _Resp())
    out = data.ensure_raw_excel(paths)
    assert out == paths.data_raw / config.RAW_XLSX_NAME
    assert out.read_bytes() == payload
    assert not list(paths.data_raw.glob("*.part"))


def test_ensure_raw_excel_failed_download_leaves_no_file(tmp_path, monkeypatch):
    paths = config.get_paths(tmp_path)

    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(data.urllib.request, "urlopen", boom)
    with pytest.raises(FileNotFoundError):
        data.ensure_raw_excel(paths)
    assert not list(Path(paths.data_raw).glob("*.xlsx"))
    assert not list(Path(paths.data_raw).glob("*.part"))
