"""Tests for the ViennaRNA bio-feature module (kept small: folding is slow)."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest

from crispr_deephf import bio_features

#: Applied to the tests that actually fold a sequence. It is deliberately **not** a
#: module-level ``pytestmark``: the cache tests at the bottom of this file stub the folding
#: out, so they can (and must) run in CI, which installs the dev extra only.
requires_folding = pytest.mark.skipif(
    not (bio_features.VIENNA_AVAILABLE and bio_features.BIOPYTHON_AVAILABLE),
    reason="ViennaRNA / Biopython not installed",
)

_ORIGINAL = Path(__file__).resolve().parents[1] / "models" / "Bio_Features.py"


@requires_folding
def test_feature_shapes():
    g, c = bio_features.compute_bio_features("ACGT" * 5 + "A")
    assert g.shape == (len(bio_features.GLOBAL_FEATURE_NAMES),)
    assert c.shape == (3, 21)
    assert g.dtype == np.float32 and c.dtype == np.float32
    assert np.isfinite(g).all() and np.isfinite(c).all()


def test_declared_width():
    expected = len(bio_features.GLOBAL_FEATURE_NAMES) + 3 * bio_features.SEQ_LEN
    assert bio_features.N_BIO_FEATURES == expected


@requires_folding
def test_flatten_matches_declared_width(synthetic_seqs):
    g, c = bio_features.bio_matrix(synthetic_seqs[:8], verbose=False)
    flat = bio_features.flatten_bio(g, c)
    assert flat.shape == (8, bio_features.N_BIO_FEATURES)
    # Layout contract: globals first, then channels reshaped to (N, 3*21).
    n_glob = g.shape[1]
    assert np.array_equal(flat[:, :n_glob], g)
    assert np.array_equal(flat[:, n_glob:], c.reshape(8, -1))


@requires_folding
def test_cache_roundtrip(tmp_path, synthetic_seqs):
    p = tmp_path / "bio.npz"
    g1, c1 = bio_features.bio_matrix(synthetic_seqs[:8], cache_path=p, verbose=False)
    g2, c2 = bio_features.bio_matrix(synthetic_seqs[:8], cache_path=p, verbose=False)
    assert np.allclose(g1, g2) and np.allclose(c1, c2)


@requires_folding
def test_cache_rejected_for_different_sequences(tmp_path, synthetic_seqs):
    p = tmp_path / "bio.npz"
    bio_features.bio_matrix(synthetic_seqs[:8], cache_path=p, verbose=False)
    g, _ = bio_features.bio_matrix(synthetic_seqs[8:16], cache_path=p, verbose=False)
    assert not np.allclose(g, bio_features.bio_matrix(synthetic_seqs[:8], verbose=False)[0])
    # and the cache was refreshed with the new content key
    with np.load(p) as cached:
        assert str(cached["digest"]) == bio_features.sequence_digest(synthetic_seqs[8:16])


def test_sequence_digest():
    d = bio_features.sequence_digest(["ACGT", "TTTT"])
    assert len(d) == 16 and int(d, 16) >= 0
    assert d == bio_features.sequence_digest(["ACGT", "TTTT"])
    assert d != bio_features.sequence_digest(["TTTT", "ACGT"])
    assert d != bio_features.sequence_digest(["ACGTT", "TTT"])  # joiner prevents collisions


@requires_folding
@pytest.mark.parametrize("seq", ["", "A", "N" * 5, "ACGTX"])
def test_melting_temp_degenerate_does_not_raise(seq):
    assert isinstance(bio_features.melting_temp(seq), float)


def test_standardizer_zero_variance_column():
    x = np.array([[1.0, 5.0], [1.0, 7.0]], dtype=np.float32)
    out = bio_features.Standardizer().fit(x).transform(x)
    assert np.isfinite(out).all()
    assert out.dtype == np.float32


@requires_folding
def test_matches_original_teammate_module(synthetic_seqs):
    """The port must reproduce the original feature values exactly."""
    spec = importlib.util.spec_from_file_location("_orig_bio_features", _ORIGINAL)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    for seq in synthetic_seqs[:4]:
        g_new, c_new = bio_features.compute_bio_features(seq)
        g_old, c_old = orig.compute_bio_features(seq)
        assert np.array_equal(g_new, g_old)
        assert np.array_equal(c_new, c_old)


# --------------------------------------------------------------------------- #
# Cache durability: atomic write, and a damaged file must recompute, not raise
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub_folding(monkeypatch):
    """Replace ``compute_bio_features`` with a cheap deterministic stand-in.

    The cache is the thing under test, not ViennaRNA, and folding 8 guides costs a third of
    a second. Stubbing it also lets these tests run in CI, which installs no ViennaRNA.
    Returns a counter of how many sequences were "folded", so a test can prove the cache was
    read rather than recomputed -- and recomputed rather than trusted.
    """
    calls: list[str] = []

    def fake(seq: str):
        calls.append(seq)
        value = float(len(calls))
        glob = np.full(len(bio_features.GLOBAL_FEATURE_NAMES), value, dtype=np.float32)
        chans = np.full((3, bio_features.SEQ_LEN), value, dtype=np.float32)
        return glob, chans

    monkeypatch.setattr(bio_features, "compute_bio_features", fake)
    return calls


def test_cache_is_written_atomically_and_leaves_no_temp_file(tmp_path, stub_folding):
    seqs = ["ACGT" * 5 + "A", "TTTT" * 5 + "G"]
    bio_features.bio_matrix(seqs, cache_path=tmp_path / "bio_WT", verbose=False)
    assert (tmp_path / "bio_WT.npz").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bio_WT.npz"], "temp file left behind"


def test_a_write_killed_midway_leaves_the_previous_cache_intact(
    tmp_path, stub_folding, monkeypatch
):
    """The guarantee the temp file buys: either the previous complete cache, or no cache.

    Writing with ``np.savez_compressed(path, ...)`` published bytes straight into the target,
    so a session killed mid-write (a disconnected Colab runtime, a full Drive) replaced 37
    minutes of folding with a half zip that the next run then tried to read. The stub below
    writes a partial zip to whatever it is handed and *then* fails, which is the only shape
    of failure that can tell an atomic write from a direct one -- a stub that raises before
    writing anything passes either way.
    """
    first = ["ACGT" * 5 + "A", "TTTT" * 5 + "G"]
    cache = tmp_path / "bio_WT.npz"
    bio_features.bio_matrix(first, cache_path=cache, verbose=False)
    good_bytes = cache.read_bytes()

    def half_write(target, **arrays):
        payload = b"PK\x03\x04 half a zip and then the session died"
        if hasattr(target, "write"):  # the temp file handle of an atomic write
            target.write(payload)
        else:  # a path: this is a direct write, and it has just clobbered the cache
            Path(str(target)).write_bytes(payload)
        raise OSError("killed mid-write")

    monkeypatch.setattr(bio_features.np, "savez_compressed", half_write)
    with pytest.raises(OSError, match="killed mid-write"):
        bio_features.bio_matrix(["GGGG" * 5 + "C"], cache_path=cache, verbose=False)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["bio_WT.npz"], (
        "a failed write must leave no temp file and publish no new file"
    )
    assert cache.read_bytes() == good_bytes, "the previous complete cache was overwritten"
    stub_folding.clear()
    bio_features.bio_matrix(first, cache_path=cache, verbose=False)
    assert stub_folding == [], "the surviving cache must still be reusable"


@pytest.mark.parametrize(
    ("damage", "expected_exc"),
    [("truncate", zipfile.BadZipFile), ("empty", EOFError)],
)
def test_damaged_cache_is_recomputed_rather_than_raising(
    tmp_path, stub_folding, damage, expected_exc
):
    """A cache damaged by a killed session must recompute cleanly.

    ``np.savez_compressed`` wrote straight to the target, so a session killed mid-write left
    a half zip -- and the read guard caught only ``(OSError, ValueError, KeyError)``. Neither
    of these two failures is any of those: a truncated zip raises ``zipfile.BadZipFile``
    (which subclasses ``Exception`` directly) and a zero-byte file raises ``EOFError``. The
    expected exception is asserted first, so this test cannot pass just because the cache
    happened to be readable.
    """
    seqs = ["ACGT" * 5 + "A", "TTTT" * 5 + "G"]
    cache = tmp_path / "bio_WT.npz"
    good, _ = bio_features.bio_matrix(seqs, cache_path=cache, verbose=False)
    intact = cache.read_bytes()

    if damage == "truncate":
        cache.write_bytes(intact[: len(intact) // 2])
    else:
        cache.write_bytes(b"")
    with pytest.raises(expected_exc), np.load(cache) as f:  # the raise the guard absorbs
        f["glob"]

    stub_folding.clear()
    again, _ = bio_features.bio_matrix(seqs, cache_path=cache, verbose=False)
    assert len(stub_folding) == len(seqs), "the damaged cache should have been recomputed"
    assert again.shape == good.shape
    # and the overwritten cache is readable again, with the right content key
    with np.load(cache) as cached:
        assert str(cached["digest"]) == bio_features.sequence_digest(seqs)
    stub_folding.clear()
    bio_features.bio_matrix(seqs, cache_path=cache, verbose=False)
    assert stub_folding == [], "the repaired cache should be reused, not folded again"
