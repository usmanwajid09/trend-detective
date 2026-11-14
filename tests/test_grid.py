"""Tests for src/grid.py: aggregation, caches, cell lookup, trend maps."""

import numpy as np
import pytest

from src import grid
from src.data import load_series
from src.trends import analyse_trend
from stats.mannkendall import mann_kendall


def _synthetic_monthly(n_years=6, n_lat=3, n_lon=4, seed=0):
    """Monthly (T, lat, lon) values with a known trend in one cell."""
    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(2000, 2000 + n_years), 12)
    months = np.tile(np.arange(1, 13), n_years)
    vals = rng.normal(0, 1, (len(years), n_lat, n_lon))
    # cell (1, 2) warms 0.1/yr
    vals[:, 1, 2] += 0.1 * (years - 2000)
    return years, months, vals


def test_aggregate_annual_means():
    years, months, vals = _synthetic_monthly(n_years=20)
    out_years, means, counts = grid.aggregate_monthly(years, months, vals, "annual")
    assert list(out_years) == list(range(2000, 2020))
    assert means.shape == (20, 3, 4)
    assert (counts == 12).all()
    # the warming cell: annual means rise ~0.1/yr
    t = analyse_trend(out_years, means[:, 1, 2])
    assert t["slope_per_decade"] == pytest.approx(1.0, abs=0.2)
    # a noise cell has no strong trend
    t0 = analyse_trend(out_years, means[:, 0, 0])
    assert abs(t0["slope_per_decade"]) < 2.0


def test_aggregate_djf_spans_new_year():
    # December 2000 must join Jan/Feb 2001, not 2000.
    years = np.array([2000] * 12 + [2001] * 12)
    months = np.tile(np.arange(1, 13), 2)
    vals = np.zeros((24, 1, 1))
    vals[11, 0, 0] = 30.0   # Dec 2000
    vals[12, 0, 0] = 10.0   # Jan 2001
    vals[13, 0, 0] = 20.0   # Feb 2001
    out_years, means, counts = grid.aggregate_monthly(years, months, vals, "DJF")
    # Jan/Feb 2000 belong to winter 2000; Dec 2000 joins Jan/Feb 2001;
    # Dec 2001 alone is too few months for winter 2002.
    assert list(out_years) == [2000, 2001, 2002]
    assert means[0, 0, 0] == pytest.approx(0.0)   # Jan+Feb 2000, both 0
    assert counts[0, 0, 0] == 2
    assert means[1, 0, 0] == pytest.approx(20.0)  # (30+10+20)/3
    assert counts[1, 0, 0] == 3
    assert np.isnan(means[2, 0, 0])
    assert counts[2, 0, 0] == 1


def test_aggregate_requires_enough_months():
    years, months, vals = _synthetic_monthly()
    vals[0:9, 0, 0] = np.nan  # only 3 months of 2000 for cell (0,0)
    _, means, counts = grid.aggregate_monthly(years, months, vals, "annual")
    assert np.isnan(means[0, 0, 0])
    assert counts[0, 0, 0] == 3
    assert np.isfinite(means[0, 1, 1])


def test_real_cache_loads_and_meta():
    cache = grid.load_dataset("temperature")
    assert cache["values"].shape[1:] == (90, 180)
    assert cache["meta"]["source"] == "NASA GISTEMP v4 gridded"
    assert cache["meta"]["season"] == "annual"
    assert cache["years"][0] == 1880
    assert cache["years"][-1] >= 2025
    pcache = grid.load_dataset("precipitation")
    assert pcache["values"].shape[1:] == (72, 144)
    assert pcache["years"][0] == 1979


def test_cell_index_lahore():
    cache = grid.load_dataset("temperature")
    ilat, ilon, clat, clon = grid.cell_index(31.52, 74.36, cache["lat"], cache["lon"])
    assert (clat, clon) == (31.0, 75.0)
    cs = grid.cell_series(cache, 31.52, 74.36)
    assert cs["cell_lat"] == 31.0 and cs["cell_lon"] == 75.0
    assert cs["n"] == len(cs["years"])
    assert cs["unit"] == "°C"


def test_cell_index_wraps_longitude_for_gpcp():
    cache = grid.load_dataset("precipitation")
    # GPCP lons are 0..360; -21.94 (Reykjavik) must wrap to 338.06.
    _, ilon1, _, clon1 = grid.cell_index(64.15, -21.94, cache["lat"], cache["lon"])
    _, ilon2, _, clon2 = grid.cell_index(64.15, 338.06, cache["lat"], cache["lon"])
    assert ilon1 == ilon2 and clon1 == clon2


def test_trend_map_crop_matches_single_series():
    cache = grid.load_dataset("temperature")
    # crop to a small region so the test stays fast
    crop = dict(cache)
    crop["lat"] = cache["lat"][55:60]
    crop["lon"] = cache["lon"][125:130]
    crop["values"] = cache["values"][:, 55:60, 125:130]
    crop["counts"] = cache["counts"][:, 55:60, 125:130]
    result = grid.trend_map(crop, start_year=1981, end_year=2024)
    assert result["slope_per_decade"].shape == (5, 5)
    years = np.array(crop["years"])
    sel = (years >= 1981) & (years <= 2024)
    for a in range(5):
        for b in range(5):
            v = crop["values"][sel, a, b]
            ref = mann_kendall(v, years[sel], correction="hamed-rao", lag=3)
            assert result["slope_per_decade"][a, b] == pytest.approx(
                ref["slope_per_decade"], rel=1e-5, abs=1e-6)
            assert result["p_value"][a, b] == pytest.approx(
                ref["p_value"], rel=1e-4, abs=1e-9)


def test_trend_map_bh_removes_false_discoveries():
    rng = np.random.default_rng(4)
    years = np.arange(1980, 2025)
    # 200 null cells + 5 strongly warming cells
    vals = rng.normal(0, 1, (len(years), 205))
    vals[:, :5] += 0.08 * (years - 1980)[:, None]
    cache = {"years": years, "lat": np.arange(205, dtype=float),
             "lon": np.array([0.0]),
             "values": vals[:, :, None], "counts": np.full((len(years), 205, 1), 12, dtype=np.uint8),
             "meta": {"source": "synthetic", "unit": "x", "season": "annual"}}
    result = grid.trend_map(cache)
    raw = (result["p_value"] < 0.05).sum()
    disc = result["significant"].sum()
    assert disc <= raw
    assert result["significant"][:5, 0].all()  # the real trends survive
    assert disc < 20  # BH must not crown many null cells


def test_lahore_power_vs_gistemp():
    """Open item 3: POWER says no Lahore warming; GISTEMP disagrees.

    The README claim ("no significant warming in Lahore") holds only for
    the NASA POWER point series.  The GISTEMP 2-degree cell containing
    Lahore warms significantly over the same 1981-2024 window, and so do
    all eight neighbouring cells, so the finding must be presented as a
    dataset disagreement, not as "Lahore is not warming".
    """
    years, values, _ = load_series("lahore", "temperature")
    assert analyse_trend(years, values)["significant"] is False

    cache = grid.load_dataset("temperature")
    cs = grid.cell_series(cache, 31.52, 74.36)
    y = np.array(cs["years"])
    v = np.array([np.nan if x is None else x for x in cs["values"]])
    sel = (y >= 1981) & (y <= 2024)
    trend = analyse_trend(y[sel], v[sel])
    assert trend["significant"] is True
    assert trend["slope_per_decade"] == pytest.approx(0.26, abs=0.05)
    mk = mann_kendall(v[sel], y[sel], correction="hamed-rao", lag=3)
    assert mk["significant"] is True
    assert mk["ci_lo"] > 0  # the whole 95% CI sits above zero


def test_fetch_grid_jobs():
    from src.fetch_data import GRID_JOBS

    names = [name for name, _, _ in GRID_JOBS]
    assert "GISTEMP gridded" in names
    assert "GPCP v2.3" in names
    urls = [url for _, url, _ in GRID_JOBS]
    assert any("gistemp250_GHCNv4.nc.gz" in u for u in urls)
    assert any("precip.mon.mean.nc" in u for u in urls)
