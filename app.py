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

from src.changepoint import detect_changepoint  # noqa: E402
from src.data import REGIONS, VARIABLES, load_series  # noqa: E402
from src.grid import (DATASETS, SEASONS, cell_series, global_mean_series,  # noqa: E402
                      load_dataset, load_trend_map, trend_map)
from src.seaice import (SOURCE as SEAICE_SOURCE,  # noqa: E402
                        SOURCE_URL as SEAICE_URL, UNIT as SEAICE_UNIT,
                        load_cache as load_seaice_cache)
from src.trends import analyse_trend, fmt_p, verdict  # noqa: E402
from stats.mannkendall import mann_kendall  # noqa: E402

app = Flask(__name__)

GRID_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grid")

# In-memory cache of loaded trend maps: (variable, start, end, season) -> payload.
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


def _load_map_payload(variable, start_year=None, end_year=None,
                      season="annual"):
    """Trend-map payload, from the precomputed file when the window is the
    default full period, else computed on demand (slow, cached in memory).

    Seasonal maps are precomputed files too (map_<variable>_<season>.npz);
    they are never computed on demand — a missing seasonal file raises
    FileNotFoundError (the route turns it into a 503).
    """
    if season != "annual" and (start_year is not None or end_year is not None):
        raise ValueError("seasonal maps are precomputed for the full period "
                         "only; start_year/end_year cannot be combined "
                         "with season")
    key = (variable, start_year, end_year, season)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    info = DATASETS[variable]
    if start_year is None and end_year is None:
        if season == "annual":
            path = os.path.join(GRID_DIR, info["map_file"])
        else:
            path = os.path.join(GRID_DIR, f"map_{variable}_{season}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"precomputed map missing: {path}")
        result = load_trend_map(path)
        meta = result["meta"]
        payload = {
            "variable": variable,
            "unit": meta["unit"],
            "source": meta["source"],
            "season": season,
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
            "season": "annual",
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
    optional season=annual|DJF|MAM|JJA|SON (default annual; seasonal maps
    are precomputed files, missing ones return 503), optional
    start_year/end_year for an on-demand annual window (slower, and not
    combinable with a season).
    """
    variable = request.args.get("variable", "temperature")
    if variable not in DATASETS:
        return jsonify({"error": f"unknown variable: {variable}"}), 400
    season = request.args.get("season", "annual")
    if season not in SEASONS:
        return jsonify({"error": f"unknown season: {season}"}), 400
    start_year = request.args.get("start_year", type=int)
    end_year = request.args.get("end_year", type=int)
    try:
        return jsonify(_load_map_payload(variable, start_year, end_year,
                                         season=season))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


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


# ---------------------------------------------------------------------------
# Decadal trends and change point (the "more trends" depth package)
# ---------------------------------------------------------------------------


def _decade_windows():
    """The five calendar decades shown by /api/decades."""
    return [("1980s", 1980, 1989), ("1990s", 1990, 1999),
            ("2000s", 2000, 2009), ("2010s", 2010, 2019),
            ("2020s", 2020, 2029)]


def _mk_or_null(years, values):
    """Mann-Kendall + Sen's slope for one (sub)series, or null slopes
    when fewer than 5 finite years are available."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n < 5:
        return {"slope_per_decade": None, "p_value": None,
                "significant": False, "n": n}
    mk = mann_kendall(values, years, correction="hamed-rao", lag=3)
    return {"slope_per_decade": round(mk["slope_per_decade"], 4),
            "p_value": mk["p_value"],
            "significant": bool(mk["significant"]), "n": mk["n"]}


def decadal_trends(years, values):
    """Decade-by-decade Mann-Kendall trends plus a full-period anchor.

    Returns (decades, full_period): decades is one entry per calendar
    decade (1980s-2020s) with slope_per_decade/p_value null when the
    decade holds fewer than 5 finite years; full_period is the same
    test over the whole series.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    decades = []
    for name, start, end in _decade_windows():
        in_window = (years >= start) & (years <= end)
        res = _mk_or_null(years[in_window], values[in_window])
        decades.append({"decade": name, "start_year": start,
                        "end_year": end, **res})
    full = _mk_or_null(years, values)
    finite_years = years[np.isfinite(values)]
    full_period = {
        "slope_per_decade": full["slope_per_decade"],
        "p_value": full["p_value"],
        "significant": full["significant"],
        "start_year": int(finite_years.min()) if len(finite_years) else None,
        "end_year": int(finite_years.max()) if len(finite_years) else None,
        "n": full["n"],
    }
    return decades, full_period


def _parse_lat_lon_pair():
    """(lat, lon) from query params, or None for the global series.

    Raises ValueError on a half-given pair or non-numeric/out-of-range
    values — the route turns those into 400s.
    """
    lat_raw = request.args.get("lat")
    lon_raw = request.args.get("lon")
    if (lat_raw is None) != (lon_raw is None):
        raise ValueError("lat and lon must be given together")
    if lat_raw is None:
        return None
    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except (ValueError, TypeError):
        raise ValueError("lat and lon must be numbers")
    if not (-90 <= lat <= 90 and -540 <= lon <= 540):
        raise ValueError("lat must be -90..90 and lon -540..540")
    return lat, lon


def _global_series(variable):
    """Global mean series for /api/decades and /api/changepoint.

    Temperature keeps the published GISTEMP global (J-D) series.
    Precipitation has no published global series, so its global mean is
    the area-weighted mean of the GPCP gridded cache
    (data/grid/precipitation.npz) — same Mann-Kendall method as
    temperature, just a different source.  Returns (years, values, meta).
    """
    if variable == "temperature":
        return load_series("global", "temperature")
    cache = load_dataset(variable, GRID_DIR)
    years, values = global_mean_series(cache)
    meta = {"source": cache["meta"]["source"],
            "url": cache["meta"]["source_url"],
            "region": "Global (land + ocean)",
            "variable": VARIABLES[variable]["label"],
            "unit": cache["meta"]["unit"],
            "note": "GPCP: area-weighted global mean of the gridded "
                    "annual means."}
    return [int(y) for y in years], [float(v) for v in values], meta


@app.route("/api/decades")
def decades():
    """Mann-Kendall trend per calendar decade (1980s-2020s).

    Query params: variable=temperature|precipitation (default
    temperature); without lat/lon the global mean series is used, with
    lat/lon the grid cell at that point (same source as /api/cell).
    """
    variable = request.args.get("variable", "temperature")
    if variable not in DATASETS:
        return jsonify({"error": f"unknown variable: {variable}"}), 400
    try:
        pair = _parse_lat_lon_pair()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        if pair is None:
            years, values, meta = _global_series(variable)
            label_text = "Global mean"
            unit = meta["unit"]
            source = meta["source"]
        else:
            lat, lon = pair
            cache = load_dataset(variable, GRID_DIR)
            series = cell_series(cache, lat, lon)
            years = np.array(series["years"], dtype=float)
            values = np.array([np.nan if v is None else v
                               for v in series["values"]], dtype=float)
            label_text = (f"Cell {series['cell_lat']:g}°, "
                          f"{series['cell_lon']:g}°")
            unit = series["unit"]
            source = series["source"]
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"error": str(exc)}), 503
    decade_list, full_period = decadal_trends(years, values)
    return jsonify({
        "variable": variable,
        "unit": unit,
        "label": label_text,
        "source": source,
        "decades": decade_list,
        "full_period": full_period,
    })


@app.route("/api/changepoint")
def changepoint():
    """Single change point in the global mean series (CUSUM scan; see
    src/changepoint.py for the method in plain words).

    Query param: variable=temperature|precipitation (default temperature).
    Only the global series is analysed: temperature uses the published
    GISTEMP global series, precipitation the area-weighted mean of the
    GPCP gridded cache (see _global_series).
    """
    variable = request.args.get("variable", "temperature")
    if variable not in DATASETS:
        return jsonify({"error": f"unknown variable: {variable}"}), 400
    try:
        years, values, meta = _global_series(variable)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    result = detect_changepoint(years, values)
    before = result["before"]
    after = result["after"]
    return jsonify({
        "variable": variable,
        "label": "Global mean",
        "method": result["method"],
        "supported": result["supported"],
        "break_year": result["break_year"],
        "statistic": result["statistic"],
        "p_value": result["p_value"],
        "before": before,
        "after": after,
    })


@app.route("/api/seaice")
def seaice_data():
    """Arctic sea ice extent: September (minimum), March (maximum) and
    annual-mean series, 1979-2025, with Mann-Kendall trends (same method
    as the map cells).  Not a map — one line chart per series."""
    try:
        cache = load_seaice_cache()
    except (FileNotFoundError, OSError) as exc:
        return jsonify({"error": str(exc)}), 503
    years = cache["years"].astype(float)
    trends = {}
    for key in ("september", "march", "annual_mean"):
        values = cache[key].astype(float)
        mk = mann_kendall(values, years, correction="hamed-rao", lag=3)
        trends[key] = {
            "slope_per_decade": round(mk["slope_per_decade"], 4),
            "p_value": mk["p_value"],
            "significant": mk["significant"],
            "n": mk["n"],
        }
    sept = trends["september"]
    direction = "shrinking" if sept["slope_per_decade"] < 0 else "growing"
    verdict_text = (
        f"September Arctic sea ice is {direction}: "
        f"{sept['slope_per_decade']:+.3f} million km² per decade "
        f"({fmt_p(sept['p_value'])}, n = {sept['n']}), "
        f"from {float(cache['september'][0]):.2f} million km² in "
        f"{int(cache['years'][0])} to {float(cache['september'][-1]):.2f} "
        f"in {int(cache['years'][-1])}.")
    return jsonify({
        "variable": "sea_ice_extent",
        "unit": SEAICE_UNIT,
        "source": SEAICE_SOURCE,
        "source_url": SEAICE_URL,
        "years": [int(y) for y in cache["years"]],
        "september": [round(float(v), 3) for v in cache["september"]],
        "march": [round(float(v), 3) for v in cache["march"]],
        "annual_mean": [round(float(v), 3) for v in cache["annual_mean"]],
        "trends": trends,
        "verdict": verdict_text,
    })


if __name__ == "__main__":
    # Local dev: python app.py  -> http://127.0.0.1:5000
    # Hosting (Render etc.) sets $PORT; gunicorn runs `app:app` instead.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
