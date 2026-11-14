"""Tests for the change-point detector (src/changepoint.py) and the
/api/changepoint endpoint."""

import numpy as np
import pytest

from app import app
from src.changepoint import _cusum_peak, detect_changepoint


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _acceleration_series(seed=7, break_year=1960):
    """Flat, then warming at 0.05/yr after the break (a known kink)."""
    rng = np.random.default_rng(seed)
    years = np.arange(1900, 2000, dtype=float)
    values = np.where(years < break_year, 0.0,
                      0.05 * (years - break_year))
    return years, values + rng.normal(0, 0.05, len(years))


def test_cusum_peak_recovers_a_level_shift():
    """The classic CUSUM case: a step in the mean at 1960.  The peak
    lands on the last pre-break year (1959)."""
    years = np.arange(1900, 2000, dtype=float)
    values = np.where(years < 1960, 0.0, 1.0)
    slope, intercept = np.polyfit(years, values, 1)
    residuals = values - (slope * years + intercept)
    _, k = _cusum_peak(residuals)
    assert abs(int(years[k]) - 1960) <= 1


def test_detect_recovers_known_acceleration():
    """A clear acceleration (flat, then fast warming) is supported, and
    the recovered break is near the true kink.

    Honest caveat (also in the module docstring): for a slope change the
    CUSUM peak marks where the series crosses its single-line fit, which
    lags the kink — here by ~17 years — so the test asserts recovery
    within 20 years, plus the before/after slopes that actually matter.
    """
    years, values = _acceleration_series()
    result = detect_changepoint(years, values)
    assert result["supported"] is True
    assert abs(result["break_year"] - 1960) <= 20
    assert result["p_value"] < 0.05
    assert result["after"]["slope_per_decade"] > result["before"]["slope_per_decade"]
    assert result["before"]["slope_per_decade"] == pytest.approx(0.0, abs=0.05)
    assert result["after"]["slope_per_decade"] == pytest.approx(0.5, abs=0.1)


def test_detect_flat_series_unsupported():
    """White noise around a flat mean: no break is supported."""
    years = np.arange(1900, 2000, dtype=float)
    for seed in (1, 11):
        rng = np.random.default_rng(seed)
        values = rng.normal(0, 0.1, len(years))
        result = detect_changepoint(years, values)
        assert result["supported"] is False
        assert result["break_year"] is None
        assert result["p_value"] >= 0.05


def test_detect_level_shift_is_not_an_acceleration():
    """A genuine level shift is detected by the CUSUM (p < 0.05) but is
    NOT reported as an acceleration: after the step the series cools
    slightly, so the after-slope is not steeper."""
    years = np.arange(1900, 2000, dtype=float)
    values = np.where(years < 1960, 0.0,
                      1.0 - 0.01 * (years - 1960))
    values = values + np.random.default_rng(3).normal(0, 0.02, len(years))
    result = detect_changepoint(years, values)
    assert result["p_value"] < 0.05
    assert result["supported"] is False
    assert result["break_year"] is None


def test_endpoint_shape_and_values(client):
    resp = client.get("/api/changepoint?variable=temperature")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["variable"] == "temperature"
    assert d["label"] == "Global mean"
    assert isinstance(d["method"], str) and "CUSUM" in d["method"]
    assert d["supported"] is True
    assert d["break_year"] == 2000
    assert d["statistic"] == pytest.approx(2.28, abs=0.05)
    assert d["p_value"] < 0.05
    assert d["before"]["start_year"] == 1880
    assert d["before"]["end_year"] == 1999
    assert d["before"]["slope_per_decade"] == pytest.approx(0.0547, abs=0.005)
    assert d["after"]["start_year"] == 2000
    assert d["after"]["end_year"] == 2025
    assert d["after"]["slope_per_decade"] == pytest.approx(0.26, abs=0.01)
    assert d["after"]["slope_per_decade"] > d["before"]["slope_per_decade"]


def test_endpoint_precipitation(client):
    """Global rainfall has no clear break (p = 0.51): the honest result
    is 'unsupported', and the endpoint must answer 200, not 400."""
    resp = client.get("/api/changepoint?variable=precipitation")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["variable"] == "precipitation"
    assert d["label"] == "Global mean"
    assert d["supported"] is False
    assert d["break_year"] is None
    assert d["p_value"] == pytest.approx(0.512, abs=0.05)


def test_endpoint_bad_variable(client):
    assert client.get("/api/changepoint?variable=ice").status_code == 400
