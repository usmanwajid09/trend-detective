"""Arctic sea ice extent (NSIDC Sea Ice Index, Version 4).

Source files: one CSV per month, downloaded to data/raw/seaice/ from
https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/
(N_01_extent_v4.0.csv ... N_12_extent_v4.0.csv, Northern Hemisphere).
Each file holds one month's extent (million sq km) per year, 1979
onwards.  The CSV headers carry stray spaces, so keys are stripped
before reading.

The build step (``python -m src.seaice build``) combines the 12 files
into a small committed cache, data/grid/seaice.npz: the September
(minimum) extent, the March (maximum) extent and the annual mean, all
1979-2025.  The app never reads the CSVs at runtime; /api/seaice
serves the cache plus Mann-Kendall trends computed on the fly.
"""

import csv
import datetime
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw", "seaice")
GRID_DIR = os.path.join(BASE, "data", "grid")
CACHE_FILE = "seaice.npz"
SOURCE = "NSIDC Sea Ice Index, Version 4 (NOAA@NSIDC, G02135)"
SOURCE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135/"
UNIT = "million km²"


def load_monthly_csvs(raw_dir=RAW_DIR):
    """Read the 12 monthly CSVs -> {month: {year: extent}}."""
    months = {}
    for m in range(1, 13):
        path = os.path.join(raw_dir, f"N_{m:02d}_extent_v4.0.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"sea ice CSV missing: {path}")
        rows = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                rows[int(row["year"])] = float(row["extent"])
        months[m] = rows
    return months


def _clean(v):
    """NSIDC writes -9999 for a missing month (e.g. Dec 1987 / Jan 1988,
    the SSM/I data gap). A sentinel must never reach a mean or a trend."""
    v = float(v)
    return np.nan if v < 0 else v


def build_cache(raw_dir=RAW_DIR, out_dir=GRID_DIR):
    """Combine the monthly CSVs into data/grid/seaice.npz. Returns metadata."""
    months = load_monthly_csvs(raw_dir)
    years = sorted(set.intersection(*[set(v) for v in months.values()]))
    if len(years) < 10:
        raise ValueError("need at least 10 common years across months")
    monthly = np.array(
        [[_clean(months[m][y]) for m in range(1, 13)] for y in years],
        dtype=np.float32)
    september = monthly[:, 8].copy()
    march = monthly[:, 2].copy()
    counts = np.isfinite(monthly).sum(axis=1)
    with np.errstate(invalid="ignore"):
        annual_mean = np.where(
            counts >= 9,
            np.nanmean(np.where(np.isfinite(monthly), monthly, np.nan),
                        axis=1),
            np.nan).astype(np.float32)
    meta = {
        "dataset": "sea_ice_extent",
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "unit": UNIT,
        "years": [int(years[0]), int(years[-1])],
        "n_years": int(len(years)),
        "september_min_year": int(years[int(np.nanargmin(september))]),
        "september_min_value": round(float(np.nanmin(september)), 3),
        "built": datetime.date.today().isoformat(),
    }
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, CACHE_FILE)
    np.savez_compressed(out, years=np.array(years, dtype=np.int32),
                        september=september, march=march,
                        annual_mean=annual_mean,
                        meta=np.array([json.dumps(meta)]))
    meta["cache_file"] = CACHE_FILE
    meta["cache_bytes"] = os.path.getsize(out)
    return meta


def load_cache(path=None):
    """Load data/grid/seaice.npz into a dict (meta parsed from JSON)."""
    if path is None:
        path = os.path.join(GRID_DIR, CACHE_FILE)
    with np.load(path, allow_pickle=False) as z:
        cache = {
            "years": z["years"],
            "september": z["september"],
            "march": z["march"],
            "annual_mean": z["annual_mean"],
            "meta": json.loads(str(z["meta"][0])),
        }
    return cache


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Build the sea ice cache")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build data/grid/seaice.npz from CSVs")
    args = parser.parse_args(argv)
    if args.command == "build":
        meta = build_cache()
        print(f"OK   sea ice: {meta['n_years']} years "
              f"({meta['years'][0]}-{meta['years'][1]}), "
              f"{meta['cache_bytes']} bytes -> {meta['cache_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
