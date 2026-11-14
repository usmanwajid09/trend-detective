"""Tests for stats/mannkendall.py, verified against pymannkendall 1.4.3."""

import numpy as np
import pytest

from stats.mannkendall import (
    mann_kendall,
    mann_kendall_grid,
    sen_slope_ci,
    sens_slope,
)

pmk = pytest.importorskip("pymannkendall")


def _series(kind, seed=0):
    rng = np.random.default_rng(seed)
    n = 50
    t = np.arange(n, dtype=float)
    if kind == "trend":
        return 0.05 * t + rng.normal(0, 1, n)
    if kind == "noise":
        return rng.normal(10, 1, n)
    if kind == "ties":
        return np.round(0.03 * t + rng.normal(0, 0.5, n))
    if kind == "autocorr":
        # trend-free AR(1) with rho = 0.7: the classic false-positive trap
        e = rng.normal(0, 1, n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + e[i]
        return x
    if kind == "decreasing":
        return -0.04 * t + rng.normal(0, 1, n)
    raise ValueError(kind)


def test_sens_slope_matches_pymannkendall():
    for kind in ("trend", "noise", "ties", "decreasing"):
        x = _series(kind)
        slope, intercept = sens_slope(x)
        ref_slope, ref_intercept = pmk.sens_slope(x)
        assert slope == pytest.approx(ref_slope, rel=1e-12, abs=1e-12)
        assert intercept == pytest.approx(ref_intercept, rel=1e-9, abs=1e-9)


def test_original_matches_pymannkendall():
    for kind in ("trend", "noise", "ties", "decreasing", "autocorr"):
        x = _series(kind)
        r = mann_kendall(x)
        ref = pmk.original_test(x, 0.05)
        assert r["s"] == ref.s
        assert r["var_s"] == pytest.approx(ref.var_s, rel=1e-12)
        assert r["z"] == pytest.approx(ref.z, rel=1e-9, abs=1e-9)
        assert r["p_value"] == pytest.approx(ref.p, rel=1e-6, abs=1e-9)
        assert r["tau"] == pytest.approx(ref.Tau, rel=1e-12)
        assert r["slope"] == pytest.approx(ref.slope, rel=1e-12, abs=1e-12)
        assert r["significant"] == bool(ref.h)
        assert r["trend"] == ref.trend


def test_hamed_rao_matches_pymannkendall():
    for kind in ("trend", "noise", "autocorr", "ties"):
        x = _series(kind)
        r = mann_kendall(x, correction="hamed-rao")
        ref = pmk.hamed_rao_modification_test(x, 0.05)
        assert r["var_s"] == pytest.approx(ref.var_s, rel=1e-9)
        assert r["z"] == pytest.approx(ref.z, rel=1e-8, abs=1e-8)
        assert r["p_value"] == pytest.approx(ref.p, rel=1e-6, abs=1e-9)
        assert r["significant"] == bool(ref.h)
    # ... and with an explicit lag (Hamed & Rao's recommended 3).
    x = _series("autocorr", seed=9)
    r = mann_kendall(x, correction="hamed-rao", lag=3)
    ref = pmk.hamed_rao_modification_test(x, 0.05, lag=3)
    assert r["var_s"] == pytest.approx(ref.var_s, rel=1e-9)
    assert r["p_value"] == pytest.approx(ref.p, rel=1e-6, abs=1e-9)


def test_yue_wang_matches_pymannkendall():
    for kind in ("trend", "noise", "autocorr"):
        x = _series(kind)
        r = mann_kendall(x, correction="yue-wang")
        ref = pmk.yue_wang_modification_test(x, 0.05)
        assert r["var_s"] == pytest.approx(ref.var_s, rel=1e-9)
        assert r["p_value"] == pytest.approx(ref.p, rel=1e-6, abs=1e-9)


def test_exact_slope_recovered():
    t = np.arange(2000, 2020, dtype=float)
    x = 2.0 * (t - 2000) + 5.0
    slope, intercept = sens_slope(x, t)
    assert slope == pytest.approx(2.0, abs=1e-12)
    r = mann_kendall(x, t)
    assert r["slope_per_decade"] == pytest.approx(20.0, abs=1e-9)
    assert r["significant"] is True


def test_flat_series_is_not_significant():
    x = np.full(30, 7.5)
    r = mann_kendall(x)
    assert r["s"] == 0
    assert r["p_value"] == 1.0
    assert r["significant"] is False
    assert r["slope"] == 0.0


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        mann_kendall([1.0, 2.0])
    with pytest.raises(ValueError):
        sens_slope([1.0])


def test_ci_contains_slope_and_is_symmetric_on_noise():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 60)
    lo, hi = sen_slope_ci(x)
    slope, _ = sens_slope(x)
    assert lo <= slope <= hi
    assert lo < hi
    # On pure noise the interval is wide relative to the slope.
    assert hi - lo > abs(slope)


def test_ci_widens_with_autocorrelation_correction():
    x = _series("autocorr", seed=3)
    r0 = mann_kendall(x)
    r1 = mann_kendall(x, correction="hamed-rao", lag=3)
    assert r1["var_s"] >= r0["var_s"]
    assert (r1["ci_hi"] - r1["ci_lo"]) >= (r0["ci_hi"] - r0["ci_lo"])


def test_autocorr_correction_cuts_false_positives():
    """On trend-free AR(1) data the uncorrected test fires far too often;
    the Hamed-Rao correction (first 3 lags, as Hamed & Rao propose) must
    bring the rate down towards the nominal 5%.

    Measured with pymannkendall's own implementation on 300 repeats:
    n=100, rho=0.7 -> raw 39%, Hamed-Rao lag=3 9%.
    """
    rng = np.random.default_rng(123)
    n_rep, n = 60, 100
    raw_hits = hr_hits = 0
    for _ in range(n_rep):
        e = rng.normal(0, 1, n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + e[i]
        raw_hits += mann_kendall(x)["significant"]
        hr_hits += mann_kendall(x, correction="hamed-rao", lag=3)["significant"]
    assert raw_hits > hr_hits
    assert hr_hits <= 12  # ~9% expected (about 5 of 60); 12 is generous slack


def test_nan_values_are_skipped():
    x = _series("trend", seed=5).astype(float)
    x[10] = np.nan
    x[25] = np.nan
    r = mann_kendall(x)
    assert r["n"] == 48
    assert r["significant"] is True


def test_grid_matches_single_series():
    rng = np.random.default_rng(11)
    times = np.arange(1980, 2025, dtype=float)
    cells = np.column_stack([
        _series("trend", seed=1)[:45],
        _series("noise", seed=2)[:45],
        _series("autocorr", seed=3)[:45],
        _series("decreasing", seed=4)[:45],
    ])
    grid = mann_kendall_grid(cells, times, correction="hamed-rao", lag=3)
    for j in range(4):
        ref = mann_kendall(cells[:, j], times, correction="hamed-rao", lag=3)
        assert grid["slope"][j] == pytest.approx(ref["slope"], rel=1e-9)
        assert grid["p_value"][j] == pytest.approx(ref["p_value"], rel=1e-6)
        assert grid["var_s"][j] == pytest.approx(ref["var_s"], rel=1e-9)
        assert grid["ci_lo"][j] == pytest.approx(ref["ci_lo"], rel=1e-6)
        assert grid["ci_hi"][j] == pytest.approx(ref["ci_hi"], rel=1e-6)
        assert bool(grid["significant"][j]) == ref["significant"]


def test_grid_handles_missing_cells():
    rng = np.random.default_rng(21)
    times = np.arange(2000, 2030, dtype=float)
    cells = rng.normal(0, 1, (30, 5))
    cells[:, 3] = np.nan                      # no data at all
    cells[:25, 4] = np.nan                    # only 5 years: below min_n
    cells[::7, 0] = np.nan                    # scattered gaps
    grid = mann_kendall_grid(cells, times, min_n=10)
    assert np.isnan(grid["slope"][3])
    assert np.isnan(grid["slope"][4])
    assert np.isnan(grid["p_value"][3])
    assert not grid["significant"][3]
    assert np.isfinite(grid["slope"][0])
    assert grid["n"][0] == 25  # 30 years minus rows 0, 7, 14, 21, 28


def test_grid_shape_and_keys():
    cells = np.random.default_rng(0).normal(0, 1, (20, 7))
    grid = mann_kendall_grid(cells, np.arange(20, dtype=float), correction=None)
    for key in ("n", "s", "tau", "var_s", "z", "p_value", "slope",
                "slope_per_decade", "intercept", "ci_lo", "ci_hi",
                "significant", "rho1"):
        assert key in grid
        assert np.asarray(grid[key]).shape == (7,), key
