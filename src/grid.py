"""Gridded data loaders, caches and per-cell trend maps.

Data (real, downloaded by ``python -m src.fetch_data --grid``):
- GISTEMP v4 gridded temperature anomaly, 2-degree grid, monthly,
  Jan 1880 - present (NetCDF, int16 scaled by 0.01).
- GPCP v2.3 precipitation, 2.5-degree grid, monthly, Jan 1979 - present
  (NetCDF, mm/day).

Raw NetCDF lives in data/raw/ (never committed).  The app never reads
NetCDF at runtime: the build step aggregates monthly values into annual
(or seasonal) means and writes a small compressed cache
(data/grid/<dataset>.npz) that the app loads with numpy alone.  A year
is kept only when enough months are present (9 of 12 for annual,
2 of 3 for a season); a season year for DJF is the year of its
January/February, so December belongs to the following year's winter.

Trend maps are computed with stats.mannkendall.mann_kendall_grid
(Hamed-Rao autocorrelation correction, lag 3) and false-discovery
control via stats.fdr.benjamini_hochberg.
"""

import gzip
import json
import os
import shutil

import numpy as np

from stats.fdr import benjamini_hochberg
from stats.mannkendall import mann_kendall_grid

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
GRID_DIR = os.path.join(BASE, "data", "grid")

SEASONS = {
    "annual": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
MIN_MONTHS = {"annual": 9, "DJF": 2, "MAM": 2, "JJA": 2, "SON": 2}

DATASETS = {
    "temperature": {
        "raw_file": "gistemp250_GHCNv4.nc",
        "variable": "tempanomaly",
        "source": "NASA GISTEMP v4 gridded",
        "source_url": "https://data.giss.nasa.gov/gistemp/",
        "unit": "°C",
        "unit_label": "temperature anomaly",
        "cache_file": "temperature.npz",
        "map_file": "map_temperature.npz",
    },
    "precipitation": {
        "raw_file": "precip.mon.mean.nc",
        "variable": "precip",
        "source": "GPCP Version 2.3",
        "source_url": "https://psl.noaa.gov/data/gridded/data.gpcp.html",
        "unit": "mm/day",
        "unit_label": "precipitation",
        "cache_file": "precipitation.npz",
        "map_file": "map_precipitation.npz",
    },
}


def _season_bucket(year, month, season):
    """The aggregation year a month belongs to (DJF spans New Year)."""
    if season == "DJF" and month == 12:
        return year + 1
    return year


def aggregate_monthly(month_years, month_numbers, values, season):
    """Aggregate a monthly (T, lat, lon) array into seasonal means.

    month_years/month_numbers: (T,) int arrays.  values: (T, lat, lon)
    float array, NaN where missing.  Returns (years, means, counts) with
    means (Y, lat, lon) float32 (NaN where too few months) and counts
    (Y, lat, lon) uint8.
    """
    months = SEASONS[season]
    min_months = MIN_MONTHS[season]
    buckets = [_season_bucket(int(y), int(mo), season)
               for y, mo in zip(month_years, month_numbers)
               if int(mo) in months]
    years = np.array(sorted(set(buckets)), dtype=np.int32)
    y_index = {int(y): i for i, y in enumerate(years)}
    sums = np.zeros((len(years),) + values.shape[1:], dtype=np.float64)
    counts = np.zeros((len(years),) + values.shape[1:], dtype=np.int32)
    for j, (y, mo) in enumerate(zip(month_years, month_numbers)):
        if int(mo) not in months:
            continue
        i = y_index[_season_bucket(int(y), int(mo), season)]
        vals = values[j]
        finite = np.isfinite(vals)
        sums[i][finite] += vals[finite]
        counts[i] += finite
    means = np.where(counts >= min_months,
                     sums / np.where(counts == 0, 1, counts), np.nan)
    return years, means.astype(np.float32), counts.astype(np.uint8)


def _time_parts(ds):
    """(years, months) int arrays for a NetCDF file's time variable."""
    import netCDF4 as nc

    tvar = ds.variables["time"]
    dates = nc.num2date(tvar[:], tvar.units, only_use_cftime_datetimes=False)
    years = np.array([d.year for d in dates], dtype=np.int32)
    months = np.array([d.month for d in dates], dtype=np.int32)
    return years, months


def aggregate_netcdf(path, variable, season="annual", chunk=120):
    """Stream a monthly NetCDF variable into seasonal means.

    Returns (years, lat, lon, means, counts).  Reads in chunks so the
    full monthly grid (tens of millions of values) is never in memory.
    """
    import netCDF4 as nc

    months = SEASONS[season]
    with nc.Dataset(path) as ds:
        var = ds.variables[variable]
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)
        years_all, months_all = _time_parts(ds)
        n_time = var.shape[0]
        shape = var.shape[1:]
        buckets = sorted(set(
            _season_bucket(int(y), int(mo), season)
            for y, mo in zip(years_all, months_all) if int(mo) in months))
        years = np.array(buckets, dtype=np.int32)
        y_index = {int(y): i for i, y in enumerate(years)}
        sums = np.zeros((len(years),) + shape, dtype=np.float64)
        counts = np.zeros((len(years),) + shape, dtype=np.int32)
        for start in range(0, n_time, chunk):
            end = min(start + chunk, n_time)
            block = np.ma.filled(var[start:end], np.nan).astype(np.float32)
            for j in range(end - start):
                y, mo = int(years_all[start + j]), int(months_all[start + j])
                if mo not in months:
                    continue
                i = y_index[_season_bucket(y, mo, season)]
                vals = block[j]
                finite = np.isfinite(vals)
                sums[i][finite] += vals[finite]
                counts[i] += finite
    means = np.where(counts >= MIN_MONTHS[season],
                     sums / np.where(counts == 0, 1, counts), np.nan)
    return years, lat, lon, means.astype(np.float32), counts.astype(np.uint8)


def gunzip(path, dest):
    with gzip.open(path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def _cache_filename(dataset, season):
    """Cache filename for a dataset/season.  Annual keeps the historic
    name; seasonal caches get a distinct name so a seasonal build can
    never overwrite the annual cache the app serves."""
    if season == "annual":
        return DATASETS[dataset]["cache_file"]
    return f"{dataset}_{season}.npz"


def _map_filename(dataset, season):
    if season == "annual":
        return DATASETS[dataset]["map_file"]
    return f"map_{dataset}_{season}.npz"


def build_cache(dataset, season="annual", raw_dir=RAW_DIR, out_dir=GRID_DIR,
                out_file=None):
    """Build a grid cache from the raw NetCDF. Returns metadata.

    The cache is written to ``out_dir/<out_file>``; when ``out_file`` is
    omitted it defaults to the annual name for season="annual" and to
    ``<dataset>_<season>.npz`` otherwise, so seasonal builds never
    clobber the annual cache.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    info = DATASETS[dataset]
    raw = os.path.join(raw_dir, info["raw_file"])
    if not os.path.exists(raw):
        raise FileNotFoundError(
            f"raw data missing: {raw} (run python -m src.fetch_data --grid)")
    years, lat, lon, values, counts = aggregate_netcdf(
        raw, info["variable"], season)
    meta = {
        "dataset": dataset,
        "variable": info["variable"],
        "source": info["source"],
        "source_url": info["source_url"],
        "unit": info["unit"],
        "season": season,
        "min_months": MIN_MONTHS[season],
        "years": [int(years[0]), int(years[-1])],
        "n_years": int(len(years)),
        "n_lat": int(len(lat)),
        "n_lon": int(len(lon)),
        "grid": f"{len(lat)}x{len(lon)}",
    }
    os.makedirs(out_dir, exist_ok=True)
    out_name = out_file if out_file is not None else _cache_filename(dataset, season)
    out = os.path.join(out_dir, out_name)
    np.savez_compressed(out, years=years.astype(np.int32), lat=lat, lon=lon,
                        values=values, counts=counts,
                        meta=np.array([json.dumps(meta)]))
    meta["cache_file"] = out_name
    meta["cache_bytes"] = os.path.getsize(out)
    return meta


def load_cache(path):
    """Load a grid cache npz into a dict (meta parsed from JSON)."""
    with np.load(path, allow_pickle=False) as z:
        cache = {
            "years": z["years"],
            "lat": z["lat"],
            "lon": z["lon"],
            "values": z["values"],
            "counts": z["counts"],
            "meta": json.loads(str(z["meta"][0])),
        }
    return cache


def load_dataset(dataset, grid_dir=GRID_DIR):
    """Load a built cache by dataset name."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    return load_cache(os.path.join(grid_dir, DATASETS[dataset]["cache_file"]))


def _normalise_lon(lon, lons):
    """Wrap a longitude into the grid's convention (0..360 or -180..180)."""
    lo, hi = float(lons.min()), float(lons.max())
    if lo >= 0:  # 0..360 grid (GPCP)
        return lon % 360.0
    return ((lon + 180.0) % 360.0) - 180.0


def cell_index(lat, lon, lats, lons):
    """Nearest grid cell to (lat, lon); returns (ilat, ilon, clat, clon)."""
    lon = _normalise_lon(lon, lons)
    ilat = int(np.argmin(np.abs(lats - lat)))
    ilon = int(np.argmin(np.abs(lons - lon)))
    return ilat, ilon, float(lats[ilat]), float(lons[ilon])


def cell_series(cache, lat, lon):
    """Annual series for the grid cell containing (lat, lon).

    Returns years (list), values (list, None where missing), the cell
    centre, and the source metadata.
    """
    ilat, ilon, clat, clon = cell_index(lat, lon, cache["lat"], cache["lon"])
    series = cache["values"][:, ilat, ilon]
    return {
        "years": [int(y) for y in cache["years"]],
        "values": [None if not np.isfinite(v) else round(float(v), 4)
                   for v in series],
        "cell_lat": clat,
        "cell_lon": clon,
        "lat": float(lat),
        "lon": float(lon),
        "n": int(np.isfinite(series).sum()),
        "source": cache["meta"]["source"],
        "source_url": cache["meta"]["source_url"],
        "unit": cache["meta"]["unit"],
    }


def global_mean_series(cache):
    """Area-weighted global mean series from a gridded cache.

    Cosine-of-latitude weights (area on a lat/lon grid); each year
    averages only the cells that have data, so a year is NaN only when
    no cell has data at all.  Returns (years, values) as arrays.
    """
    years = cache["years"]
    values = cache["values"]
    weights = np.cos(np.deg2rad(cache["lat"].astype(np.float64)))
    out = np.full(len(years), np.nan)
    for t in range(len(years)):
        v = values[t].astype(np.float64)
        finite = np.isfinite(v)
        if finite.any():
            w = np.broadcast_to(weights[:, None], v.shape)[finite]
            out[t] = float(np.average(v[finite], weights=w))
    return years.astype(np.int32), out


def trend_map(cache, alpha=0.05, min_n=10, correction="hamed-rao", lag=3,
              start_year=None, end_year=None):
    """Per-cell Mann-Kendall trends for a cache, with BH FDR control.

    Optionally restrict to a year window first.  Returns a dict of
    2-D (lat, lon) arrays plus the axes and the window used.
    """
    years = cache["years"]
    values = cache["values"]
    lo = 0 if start_year is None else int(np.searchsorted(years, start_year))
    hi = len(years) if end_year is None else int(
        np.searchsorted(years, end_year, side="right"))
    if hi - lo < 3:
        raise ValueError("year window needs at least 3 years")
    window_years = years[lo:hi]
    flat = values[lo:hi].reshape(len(window_years), -1)
    res = mann_kendall_grid(flat, window_years.astype(float),
                            alpha=alpha, correction=correction,
                            min_n=min_n, lag=lag)
    n_lat, n_lon = values.shape[1], values.shape[2]
    shape = (n_lat, n_lon)
    q_flat = benjamini_hochberg(res["p_value"])
    significant_flat = np.isfinite(q_flat) & (q_flat <= alpha)
    q = q_flat.reshape(shape)
    out = {
        "lat": cache["lat"],
        "lon": cache["lon"],
        "years": [int(window_years[0]), int(window_years[-1])],
        "slope_per_decade": res["slope_per_decade"].reshape(shape).astype(np.float32),
        "p_value": res["p_value"].reshape(shape).astype(np.float32),
        "q_value": q.astype(np.float32),
        "significant": significant_flat.reshape(shape),
        "n": res["n"].reshape(shape).astype(np.int16),
        "ci_lo_per_decade": (res["ci_lo"] * 10).reshape(shape).astype(np.float32),
        "ci_hi_per_decade": (res["ci_hi"] * 10).reshape(shape).astype(np.float32),
        "rho1": res["rho1"].reshape(shape).astype(np.float32),
    }
    return out


def save_trend_map(path, result, meta):
    arrays = {k: v for k, v in result.items()
              if isinstance(v, np.ndarray) and v.dtype != object}
    np.savez_compressed(path, meta=np.array([json.dumps(meta)]), **arrays)
    return os.path.getsize(path)


def load_trend_map(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files if k != "meta"}
        result["meta"] = json.loads(str(z["meta"][0]))
    return result


def build_map(dataset, grid_dir=GRID_DIR, cache_path=None, map_file=None,
              **kwargs):
    """Compute the default (full-period) trend map and save it.

    ``cache_path`` loads a cache from an explicit path instead of the
    annual cache (used for seasonal maps built from scratch caches);
    ``map_file`` overrides the output filename, which otherwise defaults
    to the annual name (season="annual") or ``map_<dataset>_<season>.npz``.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    cache = load_cache(cache_path) if cache_path else load_dataset(dataset,
                                                                  grid_dir)
    result = trend_map(cache, **kwargs)
    meta = {
        "dataset": dataset,
        "source": cache["meta"]["source"],
        "unit": cache["meta"]["unit"],
        "season": cache["meta"]["season"],
        "years": result["years"],
        "alpha": kwargs.get("alpha", 0.05),
        "correction": kwargs.get("correction", "hamed-rao"),
        "lag": kwargs.get("lag", 3),
        "min_n": kwargs.get("min_n", 10),
        "fdr": "Benjamini-Hochberg",
    }
    out_name = map_file if map_file is not None else _map_filename(
        dataset, cache["meta"]["season"])
    out = os.path.join(grid_dir, out_name)
    size = save_trend_map(out, result, meta)
    meta["map_file"] = out_name
    meta["map_bytes"] = size
    return meta


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Build Trend Detective grid caches")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="build data/grid caches from raw NetCDF")
    p_build.add_argument("--dataset", choices=list(DATASETS) + ["all"],
                         default="all")
    p_build.add_argument("--season", choices=list(SEASONS), default="annual")
    p_build.add_argument("--out-dir", default=GRID_DIR,
                         help="directory for the cache file(s)")
    p_build.add_argument("--out-file", default=None,
                         help="override the cache filename")
    p_map = sub.add_parser("map", help="compute and save default trend maps")
    p_map.add_argument("--dataset", choices=list(DATASETS) + ["all"],
                       default="all")
    p_map.add_argument("--season", choices=list(SEASONS) + [None],
                       default=None,
                       help="season of the cache to map (default: annual)")
    p_map.add_argument("--cache-dir", default=GRID_DIR,
                       help="directory holding the cache file(s)")
    p_map.add_argument("--map-file", default=None,
                       help="override the output map filename")
    args = parser.parse_args(argv)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    if args.command == "build":
        for name in names:
            meta = build_cache(name, season=args.season,
                               out_dir=args.out_dir, out_file=args.out_file)
            print(f"OK   {name}: {meta['n_years']} years, {meta['grid']} grid, "
                  f"{meta['cache_bytes']} bytes -> {meta['cache_file']}")
    else:
        for name in names:
            season = args.season or "annual"
            if args.cache_dir == GRID_DIR and season == "annual" \
                    and args.map_file is None:
                meta = build_map(name)
            else:
                cache_path = os.path.join(
                    args.cache_dir, _cache_filename(name, season))
                meta = build_map(name, cache_path=cache_path,
                                 map_file=args.map_file)
            print(f"OK   {name} map: {meta['years'][0]}-{meta['years'][1]}, "
                  f"{meta['map_bytes']} bytes -> {meta['map_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
