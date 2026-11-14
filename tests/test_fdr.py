"""Tests for stats/fdr.py (Benjamini-Hochberg false-discovery control)."""

import numpy as np
import pytest

from stats.fdr import benjamini_hochberg, bh_significant


def test_known_q_values():
    # Worked example: p = (0.01, 0.04, 0.03, 0.20) at m = 4.
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q = benjamini_hochberg(p)
    # sorted p: .01, .03, .04, .20 -> q_raw: .04, .06, .0533, .20
    # monotonic from the top: .04, .0533, .0533, .20
    assert q[0] == pytest.approx(0.04, abs=1e-12)
    assert q[2] == pytest.approx(0.16 / 3, abs=1e-12)
    assert q[1] == pytest.approx(0.16 / 3, abs=1e-12)
    assert q[3] == pytest.approx(0.20, abs=1e-12)


def test_many_nulls_rarely_significant():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 10000)  # all null: uniform p-values
    sig = bh_significant(p, alpha=0.05)
    assert sig.sum() <= 600  # ~5% of 10,000, with slack


def test_strong_signals_survive():
    p = np.concatenate([np.full(50, 1e-6), np.random.default_rng(1).uniform(0, 1, 9950)])
    sig = bh_significant(p, alpha=0.05)
    assert sig[:50].all()
    assert sig[50:].sum() <= 550


def test_nan_p_values_pass_through():
    p = np.array([0.001, np.nan, 0.5, np.nan])
    q = benjamini_hochberg(p)
    assert np.isnan(q[1]) and np.isnan(q[3])
    assert np.isfinite(q[0]) and np.isfinite(q[2])
    assert not bh_significant(p, alpha=0.05)[1]


def test_q_values_are_monotone_in_p():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 500)
    q = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert np.all((q >= 0) & (q <= 1))


def test_empty_and_all_nan():
    assert benjamini_hochberg([]).shape == (0,)
    q = benjamini_hochberg([np.nan, np.nan])
    assert np.isnan(q).all()
    assert not bh_significant([np.nan]).any()
