"""Tests for the ViennaRNA bio-feature module (kept small: folding is slow)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from crispr_deephf import bio_features

pytestmark = pytest.mark.skipif(
    not (bio_features.VIENNA_AVAILABLE and bio_features.BIOPYTHON_AVAILABLE),
    reason="ViennaRNA / Biopython not installed",
)

_ORIGINAL = Path(__file__).resolve().parents[1] / "models" / "Bio_Features.py"


def test_feature_shapes():
    g, c = bio_features.compute_bio_features("ACGT" * 5 + "A")
    assert g.shape == (len(bio_features.GLOBAL_FEATURE_NAMES),)
    assert c.shape == (3, 21)
    assert g.dtype == np.float32 and c.dtype == np.float32
    assert np.isfinite(g).all() and np.isfinite(c).all()


def test_declared_width():
    expected = len(bio_features.GLOBAL_FEATURE_NAMES) + 3 * bio_features.SEQ_LEN
    assert bio_features.N_BIO_FEATURES == expected


def test_flatten_matches_declared_width(synthetic_seqs):
    g, c = bio_features.bio_matrix(synthetic_seqs[:8], verbose=False)
    flat = bio_features.flatten_bio(g, c)
    assert flat.shape == (8, bio_features.N_BIO_FEATURES)
    # Layout contract: globals first, then channels reshaped to (N, 3*21).
    n_glob = g.shape[1]
    assert np.array_equal(flat[:, :n_glob], g)
    assert np.array_equal(flat[:, n_glob:], c.reshape(8, -1))


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
    # and the cache was refreshed with the new content key
    with np.load(p) as cached:
        assert str(cached["digest"]) == bio_features.sequence_digest(synthetic_seqs[8:16])


def test_sequence_digest():
    d = bio_features.sequence_digest(["ACGT", "TTTT"])
    assert len(d) == 16 and int(d, 16) >= 0
    assert d == bio_features.sequence_digest(["ACGT", "TTTT"])
    assert d != bio_features.sequence_digest(["TTTT", "ACGT"])
    assert d != bio_features.sequence_digest(["ACGTT", "TTT"])  # joiner prevents collisions


@pytest.mark.parametrize("seq", ["", "A", "N" * 5, "ACGTX"])
def test_melting_temp_degenerate_does_not_raise(seq):
    assert isinstance(bio_features.melting_temp(seq), float)


def test_standardizer_zero_variance_column():
    x = np.array([[1.0, 5.0], [1.0, 7.0]], dtype=np.float32)
    out = bio_features.Standardizer().fit(x).transform(x)
    assert np.isfinite(out).all()
    assert out.dtype == np.float32


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
