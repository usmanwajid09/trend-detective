"""Trend Detective — NASA Space Apps 2026, "Be An Earth System Trend Detective!"

Flask app serving real NASA data (cached under data/):
- NASA POWER monthly T2M / PRECTOTCORR for Lahore and Reykjavik, 1981-2024
- NASA GISTEMP v4 global annual temperature anomaly, 1880-2025
- Gridded trend maps (data/grid/): GISTEMP v4 2-degree and GPCP v2.3
  2.5-degree annual means, tested per cell with Mann-Kendall + Sen's
  slope (Hamed-Rao autocorrelation correction) and Benjamini-Hochberg
  false-discovery control.

Run:  python app.py   (then open http://127.0.0.1:5000)
"""

import json
import os
import sys

import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import REGIONS, VARIABLES, load_series  # noqa: E402
from src.grid import DATASETS, cell_series, load_dataset, load_trend_map, trend_map  # noqa: E402
from src.trends import analyse_trend, verdict  # noqa: E402
from stats.mannkendall import mann_kendall  # noqa: E402

app = Flask(__name__)

GRID_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grid")

# In-memory cache of loaded trend maps: (variable, start, end) -> payload.
_MAP_CACHE = {}


def _jsonable(arr, round_to=None):
    """2-D numeric array -> nested lists with NaN as None (JSON-safe)."""
    out = []
    for row in arr:
        line = []
        for v in row:
            fv = float(v)
            if np.isnan(fv):
                line.append(None)
            elif round_to is not None:
                line.append(round(fv, round_to))
            else:
                line.append(fv)
        out.append(line)
    return out


def _load_map_payload(variable, start_year=None, end_year=None):
    """Trend-map payload, from the precomputed file when the window is the
    default full period, else computed on demand (slow, cached in memory)."""
    key = (variable, start_year, end_year)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    info = DATASETS[variable]
    if start_year is None and end_year is None:
        path = os.path.join(GRID_DIR, info["map_file"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"precomputed map missing: {path}")
        result = load_trend_map(path)
        meta = result["meta"]
        payload = {
            "variable": variable,
            "unit": meta["unit"],
            "source": meta["source"],
            "years": meta["years"],
            "lat": [round(float(v), 2) for v in result["lat"]],
            "lon": [round(float(v), 2) for v in result["lon"]],
            "slope_per_decade": _jsonable(result["slope_per_decade"], 3),
            "p_value": _jsonable(result["p_value"]),
            "q_value": _jsonable(result["q_value"]),
            "significant": [[bool(v) for v in row] for row in result["significant"]],
            "n": [[int(v) for v in row] for row in result["n"]],
            "method": ("Mann-Kendall + Sen's slope, Hamed-Rao autocorrelation "
                       "correction (lag 3); Benjamini-Hochberg FDR, alpha 0.05"),
        }
    else:
        cache = load_dataset(variable, GRID_DIR)
        result = trend_map(cache, start_year=start_year, end_year=end_year)
        payload = {
            "variable": variable,
            "unit": cache["meta"]["unit"],
            "source": cache["meta"]["source"],
            "years": result["years"],
            "lat": [round(float(v), 2) for v in result["lat"]],
            "lon": [round(float(v), 2) for v in result["lon"]],
            "slope_per_decade": _jsonable(result["slope_per_decade"], 3),
            "p_value": _jsonable(result["p_value"]),
            "q_value": _jsonable(result["q_value"]),
            "significant": [[bool(v) for v in row] for row in result["significant"]],
            "n": [[int(v) for v in row] for row in result["n"]],
            "method": ("Mann-Kendall + Sen's slope, Hamed-Rao autocorrelation "
                       "correction (lag 3); Benjamini-Hochberg FDR, alpha 0.05"),
        }
    _MAP_CACHE[key] = payload
    return payload


def _series_payload(region_id, variable_id):
    years, values, meta = load_series(region_id, variable_id)
    trend = analyse_trend(years, values)
    label = "precipitation" if variable_id == "precipitation" else "temperature"
    return {
        "region": meta["region"],
        "variable": meta["variable"],
        "unit": meta["unit"],
        "source": meta["source"],
        "source_url": meta["url"],
        "note": meta["note"],
        "years": years,
        "values": [round(v, 3) for v in values],
        "trend": trend,
        "verdict": verdict(trend, meta["unit"], label),
    }


@app.route("/")
def index():
    return render_template("index.html", regions=REGIONS, variables=VARIABLES)


@app.route("/api/series")
def series():
    region_id = request.args.get("region", "lahore")
    variable_id = request.args.get("variable", "temperature")
    try:
        return jsonify(_series_payload(region_id, variable_id))
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/summary")
def summary():
    """One-line finding per region/variable — the "what changed, where" panel."""
    out = []
    for region_id in REGIONS:
        for variable_id in REGIONS[region_id]["variables"]:
            payload = _series_payload(region_id, variable_id)
            out.append({
                "region": payload["region"],
                "variable": payload["variable"],
                "unit": payload["unit"],
                "slope_per_decade": payload["trend"]["slope_per_decade"],
                "p_value": payload["trend"]["p_value"],
                "significant": payload["trend"]["significant"],
                "years": f"{payload['trend']['start_year']}-{payload['trend']['end_year']}",
                "verdict": payload["verdict"],
            })
    return jsonify(out)


@app.route("/api/map")
def map_data():
    """Per-cell trend map for a variable (default: full-period, precomputed).

    Query params: variable=temperature|precipitation (default temperature),
    optional start_year/end_year for an on-demand window (slower).
    """
    variable = request.args.get("variable", "temperature")
    if variable not in DATASETS:
        return jsonify({"error": f"unknown variable: {variable}"}), 400
    start_year = request.args.get("start_year", type=int)
    end_year = request.args.get("end_year", type=int)
    try:
        return jsonify(_load_map_payload(variable, start_year, end_year))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503


@app.route("/api/cell")
def cell_data():
    """Time series + Mann-Kendall trend for the grid cell at (lat, lon).

    Query params: variable=temperature|precipitation, lat, lon (degrees).
    The trend line and its 95% confidence band are given by the median
    point (median_year, median_value) and the Sen's-slope CI: the band is
    the region between the lines through the median point with slopes
    ci_lo and ci_hi.
    """
    variable = request.args.get("variable", "temperature")
    if variable not in DATASETS:
        return jsonify({"error": f"unknown variable: {variable}"}), 400
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "lat and lon query params are required"}), 400
    if not (-90 <= lat <= 90 and -540 <= lon <= 540):
        return jsonify({"error": "lat must be -90..90 and lon -540..540"}), 400
    try:
        cache = load_dataset(variable, GRID_DIR)
    except (FileNotFoundError, OSError) as exc:
        return jsonify({"error": str(exc)}), 503
    series = cell_series(cache, lat, lon)
    years = np.array(series["years"], dtype=float)
    values = np.array([np.nan if v is None else v for v in series["values"]],
                      dtype=float)
    finite = np.isfinite(values)
    label = "precipitation" if variable == "precipitation" else "temperature"
    payload = dict(series)
    payload["variable"] = label
    if int(finite.sum()) >= 10:
        mk = mann_kendall(values, years, correction="hamed-rao", lag=3)
        payload["trend"] = {
            "slope_per_decade": round(mk["slope_per_decade"], 4),
            "ci_lo_per_decade": round(mk["ci_lo"] * 10, 4),
            "ci_hi_per_decade": round(mk["ci_hi"] * 10, 4),
            "p_value": mk["p_value"],
            "significant": mk["significant"],
            "n": mk["n"],
            "trend": mk["trend"],
            "median_year": float(np.median(years[finite])),
            "median_value": float(np.median(values[finite])),
        }
        payload["verdict"] = verdict(
            {"slope_per_decade": mk["slope_per_decade"],
             "p_value": mk["p_value"], "significant": mk["significant"],
             "n": mk["n"]},
            series["unit"], label)
        try:
            default_map = _load_map_payload(variable)
            ilat = int(np.argmin(np.abs(np.array(default_map["lat"]) - series["cell_lat"])))
            ilon = int(np.argmin(np.abs(np.array(default_map["lon"]) - series["cell_lon"])))
            payload["trend"]["q_value"] = default_map["q_value"][ilat][ilon]
            payload["trend"]["significant_bh"] = bool(
                default_map["significant"][ilat][ilon])
        except (FileNotFoundError, KeyError, IndexError):
            payload["trend"]["q_value"] = None
            payload["trend"]["significant_bh"] = None
    else:
        payload["trend"] = None
        payload["verdict"] = (
            f"Too few years with data (n = {int(finite.sum())}) to test a trend.")
    return jsonify(payload)


if __name__ == "__main__":
    # Local dev: python app.py  -> http://127.0.0.1:5000
    # Hosting (Render etc.) sets $PORT; gunicorn runs `app:app` instead.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
