"""Load the cached real NASA datasets bundled under data/.

Sources (fetched at build time, 2026-11-13):
- NASA POWER (power.larc.nasa.gov), monthly point data, community RE:
  T2M (mean air temperature at 2 m, deg C) and PRECTOTCORR
  (precipitation, corrected, mm/day). 1981-2024.
- NASA GISTEMP v4 (data.giss.nasa.gov), GLB.Ts+dSST.csv:
  global land-ocean annual temperature anomaly, J-D column, deg C
  relative to the 1951-1980 baseline. 1880-2025.
"""

import csv
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")

REGIONS = {
    "lahore": {
        "name": "Lahore, Pakistan",
        "lat": 31.52,
        "lon": 74.36,
        "file": "power_lahore.json",
        "source": "NASA POWER",
        "variables": ["temperature", "precipitation"],
    },
    "reykjavik": {
        "name": "Reykjavik, Iceland",
        "lat": 64.15,
        "lon": -21.94,
        "file": "power_reykjavik.json",
        "source": "NASA POWER",
        "variables": ["temperature", "precipitation"],
    },
    "global": {
        "name": "Global (land + ocean)",
        "lat": None,
        "lon": None,
        "file": "gistemp_global.csv",
        "source": "NASA GISTEMP v4",
        "variables": ["temperature"],
    },
}

VARIABLES = {
    "temperature": {
        "label": "Temperature",
        "unit": "°C",
        "power_param": "T2M",
        "note": "POWER: annual mean 2 m air temperature. GISTEMP: annual "
                "temperature anomaly vs the 1951–1980 baseline.",
    },
    "precipitation": {
        "label": "Precipitation",
        "unit": "mm/day",
        "power_param": "PRECTOTCORR",
        "note": "POWER: annual mean precipitation (bias-corrected), mm/day.",
    },
}


def _annual_means(monthly):
    """monthly: {YYYYMM: value} -> (years, annual means), skipping fill values."""
    by_year = {}
    for key, value in monthly.items():
        if value is None or value < -900:
            continue
        by_year.setdefault(int(str(key)[:4]), []).append(float(value))
    years = sorted(by_year)
    means = [sum(by_year[y]) / len(by_year[y]) for y in years]
    return years, means


def load_series(region_id, variable_id):
    """Return (years, values, meta) for a region/variable from the cache."""
    if region_id not in REGIONS:
        raise ValueError(f"unknown region: {region_id}")
    region = REGIONS[region_id]
    if variable_id not in region["variables"]:
        raise ValueError(f"{variable_id} not available for {region['name']}")
    path = os.path.join(DATA_DIR, region["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"cached data missing: {path}")

    if region_id == "global":
        lines = open(path).read().splitlines()
        hdr = next(i for i, line in enumerate(lines) if line.startswith("Year,"))
        years, values = [], []
        for row in csv.DictReader(lines[hdr:]):
            try:
                year = int(row["Year"])
                jd = float(row["J-D"])  # "***" for the incomplete current year
            except (ValueError, TypeError):
                continue
            years.append(year)
            values.append(jd)
        meta = {"source": "NASA GISTEMP v4",
                "url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"}
    else:
        param = VARIABLES[variable_id]["power_param"]
        doc = json.load(open(path))
        monthly = doc["properties"]["parameter"][param]
        years, values = _annual_means(monthly)
        meta = {"source": "NASA POWER",
                "url": "https://power.larc.nasa.gov/api/temporal/monthly/point"}

    meta.update({
        "region": region["name"],
        "variable": VARIABLES[variable_id]["label"],
        "unit": VARIABLES[variable_id]["unit"],
        "note": VARIABLES[variable_id]["note"],
    })
    return years, values, meta
