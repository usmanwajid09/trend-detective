"""Tests for seasonal trend maps: aggregation, precomputed map files and
the /api/map?season= endpoint."""

import json

import numpy as np
import pytest

import app as app_mod
from app import app
from src import grid
from src.grid import load_trend_map


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


SEASONAL_MAPS = [
    ("temperature", "DJF", (90, 180)),
    ("temperature", "MAM", (90, 180)),
    ("temperature", "JJA", (90, 180)),
    ("temperature", "SON", (90, 180)),
    ("precipitation", "DJF", (72, 144)),
    ("precipitation", "MAM", (72, 144)),
    ("precipitation", "JJA", (72, 144)),
    ("precipitation", "SON", (72, 144)),
]


def test_aggregate_djf_winter_mean():
    """December 2000 + Jan/Feb 2001 form winter 2001 (mean 20, n = 3)."""
    years = np.array([2000] * 12 + [2001] * 12)
    months = np.tile(np.arange(1, 13), 2)
    vals = np.zeros((24, 1, 1))
    vals[11, 0, 0] = 30.0   # Dec 2000
    vals[12, 0, 0] = 10.0   # Jan 2001
    vals[13, 0, 0] = 20.0   # Feb 2001
    out_years, means, counts = grid.aggregate_monthly(years, months, vals,
                                                      "DJF")
    assert list(out_years) == [2000, 2001, 2002]
    assert means[1, 0, 0] == pytest.approx(20.0)
    assert counts[1, 0, 0] == 3
    # Dec 2001 is alone in winter 2002 -> too few months, NaN.
    assert np.isnan(means[2, 0, 0])


def test_seasonal_map_is_not_the_annual_map():
    """The DJF map must be a genuinely different aggregation, not a copy
    of the annual map, and must carry all 147 winters (1880-2026)."""
    djf = load_trend_map("data/grid/map_temperature_DJF.npz")
    annual = load_trend_map("data/grid/map_temperature.npz")
    assert int(djf["n"].max()) == 147
    assert not np.allclose(djf["slope_per_decade"], annual["slope_per_decade"],
                           equal_nan=True)
    # Winter warms faster than the annual mean in the Arctic: the DJF
    # maximum significant slope beats the annual one.
    assert float(np.nanmax(np.where(
        djf["significant"], djf["slope_per_decade"], -np.inf))) > float(
        np.nanmax(np.where(
            annual["significant"], annual["slope_per_decade"], -np.inf)))


@pytest.mark.parametrize("variable,season,shape", SEASONAL_MAPS)
def test_seasonal_map_files_exist(variable, season, shape):
    path = f"data/grid/map_{variable}_{season}.npz"
    result = load_trend_map(path)
    assert result["meta"]["season"] == season
    assert result["meta"]["dataset"] == variable
    assert result["slope_per_decade"].shape == shape
    assert result["significant"].shape == shape
    assert result["meta"]["years"] == (
        [1880, 2026] if variable == "temperature" else [1979, 2026])
    # every seasonal map has real discoveries (not an empty file)
    assert int(result["significant"].sum()) > 1000


def test_seasonal_map_annual_files_untouched():
    """The seasonal build must not have clobbered the annual maps."""
    for variable in ("temperature", "precipitation"):
        result = load_trend_map(f"data/grid/map_{variable}.npz")
        assert result["meta"]["season"] == "annual"


def test_map_season_djf_payload(client):
    resp = client.get("/api/map?variable=temperature&season=DJF")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["season"] == "DJF"
    assert d["variable"] == "temperature"
    assert d["years"] == [1880, 2026]
    slope = np.array([[np.nan if v is None else v for v in row]
                      for row in d["slope_per_decade"]])
    assert slope.shape == (90, 180)
    # DJF Arctic amplification: the strongest significant winter warming
    # beats the annual maximum (see the report for the numbers).
    assert bool(np.nanmax(np.where(np.array(d["significant"]), slope, -np.inf)) > 1.5)


def test_map_season_precipitation(client):
    resp = client.get("/api/map?variable=precipitation&season=JJA")
    d = resp.get_json()
    assert d["season"] == "JJA"
    assert len(d["lat"]) == 72 and len(d["lon"]) == 144
    assert int(np.array(d["significant"]).sum()) == 1603


def test_map_season_annual_default(client):
    resp = client.get("/api/map?variable=temperature")
    d = resp.get_json()
    assert d["season"] == "annual"
    assert d["years"] == [1880, 2026]
    assert int(np.array(d["significant"]).sum()) == 6170


def test_map_season_bad_season(client):
    assert client.get("/api/map?season=winter").status_code == 400
    assert client.get("/api/map?season=djf").status_code == 400  # case-sensitive


def test_map_season_cannot_combine_with_window(client):
    resp = client.get("/api/map?season=DJF&start_year=2000&end_year=2024")
    assert resp.status_code == 400


def test_map_season_missing_file_is_503(client, tmp_path, monkeypatch):
    """A seasonal map with no precomputed file is a 503, never computed
    on demand."""
    monkeypatch.setattr(app_mod, "GRID_DIR", str(tmp_path))
    app_mod._MAP_CACHE.clear()
    resp = client.get("/api/map?variable=temperature&season=DJF")
    assert resp.status_code == 503
    assert "precomputed map missing" in resp.get_json()["error"]
    app_mod._MAP_CACHE.clear()
