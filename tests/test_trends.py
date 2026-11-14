"""Tests for the trend/significance maths in src/trends.py."""

import numpy as np
import pytest

from src.trends import analyse_trend, record_years, verdict


def test_exact_slope_recovered():
    years = np.arange(2000, 2020)
    values = 2.0 * (years - 2000) + 5.0  # exactly +2 per year
    t = analyse_trend(years, values)
    assert t["slope_per_decade"] == pytest.approx(20.0, abs=1e-9)
    assert t["significant"] is True
    assert t["p_value"] < 0.05
    assert t["n"] == 20


def test_no_trend_is_not_significant():
    rng = np.random.default_rng(42)
    years = np.arange(1980, 2025)
    values = 10.0 + rng.normal(0, 1, size=len(years))  # pure noise, no trend
    t = analyse_trend(years, values)
    assert t["significant"] is False
    assert t["p_value"] > 0.05


def test_verdict_words_match_numbers():
    sig = {"slope_per_decade": 0.21, "p_value": 0.0001, "significant": True, "n": 44}
    assert "Significant warming" in verdict(sig, "°C", "temperature")
    assert "+0.210" in verdict(sig, "°C", "temperature")
    flat = {"slope_per_decade": -0.06, "p_value": 0.52, "significant": False, "n": 44}
    assert "No significant trend" in verdict(flat, "°C", "temperature")
    wet = {"slope_per_decade": 0.15, "p_value": 0.016, "significant": True, "n": 44}
    assert "Significant wetter" in verdict(wet, "mm/day", "precipitation")


def test_verdict_p_value_never_bare_zero_or_raw_exponent():
    # p underflows to exactly 0 for the full-record Lahore cell: never "p = 0".
    underflow = {"slope_per_decade": 0.122, "p_value": 0.0, "significant": True, "n": 147}
    assert "p < 0.001" in verdict(underflow, "°C", "temperature")
    assert "p = 0," not in verdict(underflow, "°C", "temperature")
    tiny = {"slope_per_decade": 0.246, "p_value": 5.68e-05, "significant": True, "n": 44}
    assert "5.7×10⁻⁵" in verdict(tiny, "°C", "temperature")
    assert "e-05" not in verdict(tiny, "°C", "temperature")
    ordinary = {"slope_per_decade": -0.06, "p_value": 0.53, "significant": False, "n": 44}
    assert "p = 0.53" in verdict(ordinary, "°C", "temperature")


def test_record_years_basic():
    r = record_years([2000, 2001, 2002, 2003, 2004], [1.0, 5.0, 3.0, 5.0, 2.0])
    assert r == {"max": {"year": 2001, "value": 5.0},   # tie: earliest wins
                "min": {"year": 2000, "value": 1.0}}


def test_record_years_skips_nan_gaps():
    # None and NaN years are missing data, never records.
    r = record_years([2000, 2001, 2002, 2003, 2004],
                     [1.0, float("nan"), 3.0, None, 2.0])
    assert r == {"max": {"year": 2002, "value": 3.0},
                "min": {"year": 2000, "value": 1.0}}


def test_record_years_no_data_is_null():
    r = record_years([2000, 2001], [float("nan"), None])
    assert r == {"max": None, "min": None}


def test_record_years_real_global_series():
    """Warmest 2024 (+1.28 °C), coldest 1909 (−0.48 °C) — the README
    numbers, from the cached GISTEMP global series."""
    from src.data import load_series

    years, values, _ = load_series("global", "temperature")
    r = record_years(years, values)
    assert r["max"] == {"year": 2024, "value": 1.28}
    assert r["min"] == {"year": 1909, "value": -0.48}


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        analyse_trend([2000, 2001], [1.0, 2.0])


def test_real_cached_data_trends():
    """Regression test on the real bundled data: the numbers in the README."""
    from src.data import load_series

    years, values, _ = load_series("reykjavik", "temperature")
    t = analyse_trend(years, values)
    assert t["slope_per_decade"] == pytest.approx(0.194, abs=0.01)
    assert t["significant"] is True

    years, values, _ = load_series("lahore", "temperature")
    t = analyse_trend(years, values)
    assert t["significant"] is False

    years, values, _ = load_series("lahore", "precipitation")
    t = analyse_trend(years, values)
    assert t["significant"] is True
    assert t["slope_per_decade"] > 0

    years, values, _ = load_series("global", "temperature")
    t = analyse_trend(years, values)
    assert t["slope_per_decade"] == pytest.approx(0.081, abs=0.01)
    assert t["significant"] is True
