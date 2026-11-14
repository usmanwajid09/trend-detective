"""Tests for the M2 map endpoints: /api/map and /api/cell."""

import numpy as np
import pytest

from app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_map_temperature(client):
    resp = client.get("/api/map?variable=temperature")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["variable"] == "temperature"
    assert d["unit"] == "°C"
    assert d["years"] == [1880, 2026]
    assert len(d["lat"]) == 90 and len(d["lon"]) == 180
    slope = np.array([[np.nan if v is None else v for v in row]
                      for row in d["slope_per_decade"]])
    assert slope.shape == (90, 180)
    sig = np.array(d["significant"])
    assert sig.shape == (90, 180)
    # BH discoveries precomputed in data/grid/map_temperature.npz
    assert int(sig.sum()) == 6170
    assert int((~np.isnan(slope) & sig).sum()) == 6170
    # no JSON NaN leaked through (nulls only)
    assert "NaN" not in resp.get_data(as_text=True)


def test_map_precipitation(client):
    resp = client.get("/api/map?variable=precipitation")
    d = resp.get_json()
    assert len(d["lat"]) == 72 and len(d["lon"]) == 144
    assert np.array(d["significant"]).sum() == 2918


def test_map_bad_variable(client):
    assert client.get("/api/map?variable=ice").status_code == 400


def test_map_window_on_demand(client):
    # A 25-year window is computed live from the annual cache.
    resp = client.get("/api/map?variable=temperature&start_year=2000&end_year=2024")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["years"] == [2000, 2024]
    assert len(d["lat"]) == 90
    # a second identical request is served from the in-memory cache
    resp2 = client.get("/api/map?variable=temperature&start_year=2000&end_year=2024")
    assert resp2.get_json() == d


def test_cell_lahore_temperature(client):
    resp = client.get("/api/cell?variable=temperature&lat=31.52&lon=74.36")
    assert resp.status_code == 200
    d = resp.get_json()
    assert (d["cell_lat"], d["cell_lon"]) == (31.0, 75.0)
    assert len(d["years"]) == len(d["values"]) == 147
    assert d["trend"]["slope_per_decade"] == pytest.approx(0.122, abs=0.005)
    assert d["trend"]["significant"] is True
    assert d["trend"]["significant_bh"] is True
    assert d["trend"]["q_value"] == pytest.approx(0.0, abs=1e-12)
    assert d["trend"]["ci_lo_per_decade"] > 0  # whole CI above zero
    assert "Significant warming" in d["verdict"]
    assert d["source"] == "NASA GISTEMP v4 gridded"


def test_cell_lahore_precipitation(client):
    resp = client.get("/api/cell?variable=precipitation&lat=31.52&lon=74.36")
    d = resp.get_json()
    assert (d["cell_lat"], d["cell_lon"]) == (31.25, 73.75)
    assert len(d["years"]) == 48
    assert d["trend"]["significant_bh"] is True
    assert d["trend"]["slope_per_decade"] == pytest.approx(0.085, abs=0.01)
    assert "Significant wetter" in d["verdict"]


def test_cell_reykjavik_warming_matches_point_series(client):
    # The map cell and the POWER point series should tell the same story
    # for Reykjavik (both warm significantly).
    resp = client.get("/api/cell?variable=temperature&lat=64.15&lon=-21.94")
    d = resp.get_json()
    assert d["trend"]["significant"] is True
    assert d["trend"]["slope_per_decade"] > 0
    s = client.get("/api/series?region=reykjavik&variable=temperature").get_json()
    assert s["trend"]["significant"] is True


def test_cell_bad_params(client):
    assert client.get("/api/cell?variable=temperature").status_code == 400
    assert client.get("/api/cell?variable=temperature&lat=abc&lon=0").status_code == 400
    assert client.get("/api/cell?variable=ice&lat=0&lon=0").status_code == 400
    assert client.get("/api/cell?variable=temperature&lat=100&lon=0").status_code == 400


def test_index_has_map_section(client):
    html = client.get("/").get_data(as_text=True)
    assert "Global trend map" in html
    assert "map-canvas" in html
    assert "cell-panel" in html
