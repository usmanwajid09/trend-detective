"""Tests for the trend/significance maths in src/trends.py."""

import numpy as np
import pytest

from src.trends import analyse_trend, verdict


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
