"""Tests for the NSIDC sea ice extent loader and /api/seaice."""

import csv
import os

import numpy as np
import pytest

from app import app
from src.seaice import build_cache, load_cache, load_monthly_csvs


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _write_month(tmp_path, month, rows):
    path = tmp_path / f"N_{month:02d}_extent_v4.0.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", " mo", "source_dataset", " region", " extent",
                    "   area"])
        for year, extent in rows:
            w.writerow([year, month, "NSIDC-0051", "N", extent, extent - 2.0])
    return path


def test_load_monthly_csvs_strips_header_spaces(tmp_path):
    for m in range(1, 13):
        _write_month(tmp_path, m, [(2000, 10.0), (2001, 9.5)])
    months = load_monthly_csvs(str(tmp_path))
    assert months[9][2000] == 10.0
    assert months[9][2001] == 9.5
    assert months[1][2000] == 10.0


def test_load_monthly_csvs_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_monthly_csvs(str(tmp_path))


def test_build_cache_series(tmp_path):
    for m in range(1, 13):
        _write_month(tmp_path, m,
                     [(2000 + y, 10.0 - 0.1 * y) for y in range(12)])
    meta = build_cache(str(tmp_path), str(tmp_path))
    assert meta["n_years"] == 12
    assert meta["years"] == [2000, 2011]
    assert meta["september_min_year"] == 2011
    cache = load_cache(str(tmp_path / "seaice.npz"))
    assert list(cache["years"]) == list(range(2000, 2012))
    assert cache["september"][0] == pytest.approx(10.0)
    assert cache["september"][-1] == pytest.approx(8.9)
    assert cache["march"][0] == pytest.approx(10.0)
    assert np.isfinite(cache["annual_mean"]).all()


def test_build_cache_ignores_minus_9999_sentinel(tmp_path):
    # NSIDC writes -9999 for missing months (Dec 1987 / Jan 1988 gap).
    # A sentinel must never leak into the annual mean.
    rows = [(2000 + y, 10.0 - 0.1 * y) for y in range(12)]
    for m in range(1, 13):
        _write_month(tmp_path, m, rows)
    # January 2001 is missing in the source file.
    jan_path = tmp_path / "N_01_extent_v4.0.csv"
    with open(jan_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", " mo", "source_dataset", " region", " extent",
                    "   area"])
        for year, extent in rows:
            w.writerow([year, 1, "NSIDC-0051", "N",
                        -9999 if year == 2001 else extent, extent - 2.0])
    build_cache(str(tmp_path), str(tmp_path))
    cache = load_cache(str(tmp_path / "seaice.npz"))
    # 2001 annual mean = mean of the other 11 months (all 9.9), never -822.
    assert cache["annual_mean"][1] == pytest.approx(9.9, abs=0.05)
    assert np.isfinite(cache["annual_mean"]).all()
    assert (cache["annual_mean"] > 0).all()


def test_real_cache_september_is_declining():
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "grid", "seaice.npz")
    if not os.path.exists(path):
        pytest.skip("sea ice cache not built")
    cache = load_cache(path)
    assert cache["meta"]["years"] == [1979, 2025]
    assert cache["september"][0] > cache["september"][-1]
    # September 1979 ~7.05, 2025 ~4.75 million km2 (NSIDC Sea Ice Index v4).
    assert cache["september"][0] == pytest.approx(7.05, abs=0.05)
    assert cache["september"][-1] == pytest.approx(4.75, abs=0.05)
    assert cache["meta"]["september_min_year"] == 2012


def test_seaice_endpoint(client):
    resp = client.get("/api/seaice")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["variable"] == "sea_ice_extent"
    assert d["unit"] == "million km²"
    assert d["years"][0] == 1979 and d["years"][-1] == 2025
    assert len(d["september"]) == len(d["years"]) == 47
    assert len(d["march"]) == 47
    sept = d["trends"]["september"]
    assert sept["significant"] is True
    assert sept["slope_per_decade"] < 0
    assert sept["slope_per_decade"] == pytest.approx(-0.741, abs=0.01)
    assert "shrinking" in d["verdict"]
    assert "NSIDC" in d["source"]
