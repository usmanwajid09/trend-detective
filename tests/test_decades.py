"""Tests for the /api/decades endpoint (the "more trends" depth package)."""

import numpy as np
import pytest

from app import app, decadal_trends


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_decades_global_shape(client):
    resp = client.get("/api/decades?variable=temperature")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["variable"] == "temperature"
    assert d["unit"] == "°C"
    assert d["label"] == "Global mean"
    assert d["source"] == "NASA GISTEMP v4"
    assert [x["decade"] for x in d["decades"]] == [
        "1980s", "1990s", "2000s", "2010s", "2020s"]
    assert [(x["start_year"], x["end_year"]) for x in d["decades"]] == [
        (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019),
        (2020, 2029)]
    assert [x["n"] for x in d["decades"]] == [10, 10, 10, 10, 6]
    for x in d["decades"]:
        assert set(x) == {"decade", "start_year", "end_year",
                          "slope_per_decade", "p_value", "significant", "n"}


def test_decades_global_values(client):
    d = client.get("/api/decades?variable=temperature").get_json()
    by_decade = {x["decade"]: x for x in d["decades"]}
    # Warming accelerates: the 2010s are the first decade with a
    # significant trend, and the 2020s slope is the steepest.
    assert by_decade["2010s"]["significant"] is True
    assert by_decade["2010s"]["slope_per_decade"] == pytest.approx(0.36, abs=0.01)
    assert by_decade["2020s"]["slope_per_decade"] == pytest.approx(0.675, abs=0.05)
    assert by_decade["2020s"]["slope_per_decade"] > by_decade["2010s"]["slope_per_decade"]
    assert by_decade["1980s"]["significant"] is False
    # Full-period anchor: 1880-2025 Mann-Kendall + Sen's slope.
    full = d["full_period"]
    assert (full["start_year"], full["end_year"], full["n"]) == (1880, 2025, 146)
    assert full["slope_per_decade"] == pytest.approx(0.08, abs=0.005)
    assert full["significant"] is True


def test_decades_1981_2025_anchor():
    """The 1981-2025 window (the decades' span) must reproduce the known
    ~+0.212 C/decade global warming rate."""
    from src.data import load_series

    years, values, _ = load_series("global", "temperature")
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)
    mask = (years >= 1981) & (years <= 2025)
    _, full = decadal_trends(years[mask], values[mask])
    assert full["slope_per_decade"] == pytest.approx(0.212, abs=0.005)
    assert (full["start_year"], full["end_year"], full["n"]) == (1981, 2025, 45)


def test_decades_cell_lahore(client):
    resp = client.get("/api/decades?variable=temperature&lat=31.52&lon=74.36")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["label"] == "Cell 31°, 75°"
    assert d["source"] == "NASA GISTEMP v4 gridded"
    assert len(d["decades"]) == 5
    # Same full-period value as /api/cell reports for this cell.
    assert d["full_period"]["slope_per_decade"] == pytest.approx(0.122, abs=0.005)
    assert d["full_period"]["significant"] is True


def test_decades_cell_precipitation_label(client):
    resp = client.get("/api/decades?variable=precipitation&lat=31.52&lon=74.36")
    d = resp.get_json()
    assert d["label"] == "Cell 31.25°, 73.75°"
    assert d["unit"] == "mm/day"


def test_decades_sparse_decades_are_null():
    """A decade with < 5 finite years reports null slopes, real n."""
    years = np.arange(1980, 2030, dtype=float)
    values = np.full(len(years), np.nan)
    # 1980s: only 3 finite years; 1990s: complete; 2020s: 2 finite years.
    values[0:3] = [0.1, 0.2, 0.15]
    values[10:20] = np.linspace(0.2, 0.4, 10)
    values[40:42] = [0.5, 0.6]
    decades, full = decadal_trends(years, values)
    by_decade = {x["decade"]: x for x in decades}
    assert by_decade["1980s"]["slope_per_decade"] is None
    assert by_decade["1980s"]["p_value"] is None
    assert by_decade["1980s"]["significant"] is False
    assert by_decade["1980s"]["n"] == 3
    assert by_decade["1990s"]["slope_per_decade"] is not None
    assert by_decade["1990s"]["n"] == 10
    assert by_decade["2020s"]["slope_per_decade"] is None
    assert by_decade["2020s"]["n"] == 2
    # Too few years overall: the full period is null too.
    assert full["slope_per_decade"] is not None  # 15 finite years here
    decades2, full2 = decadal_trends(years[:4], values[:4])
    assert all(x["slope_per_decade"] is None for x in decades2)
    assert full2["slope_per_decade"] is None
    assert full2["n"] == 3


def test_decades_bad_params(client):
    assert client.get("/api/decades?variable=ice").status_code == 400
    # global has no precipitation series
    assert client.get("/api/decades?variable=precipitation").status_code == 400
    # half a lat/lon pair
    assert client.get("/api/decades?lat=31").status_code == 400
    assert client.get("/api/decades?lon=75").status_code == 400
    assert client.get("/api/decades?lat=abc&lon=75").status_code == 400
    assert client.get("/api/decades?lat=100&lon=75").status_code == 400
