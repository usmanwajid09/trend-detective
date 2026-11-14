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
from src.grid import (DATASETS, SEASONS, cell_series, find_hotspots,  # noqa: E402
                      global_mean_series, load_dataset, load_trend_map,
                      trend_map)
from src.seaice import (SOURCE as SEAICE_SOURCE,  # noqa: E402
                        SOURCE_URL as SEAICE_URL, UNIT as SEAICE_UNIT,
                        load_cache as load_seaice_cache)
from src.trends import (analyse_trend, fmt_p, record_years,  # noqa: E402
                        verdict)
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
    rounded = [round(v, 3) for v in values]
    return {
        "region": meta["region"],
        "variable": meta["variable"],
        "unit": meta["unit"],
        "source": meta["source"],
        "source_url": meta["url"],
        "note": meta["note"],
        "years": years,
        "values": rounded,
        "records": record_years(years, rounded),
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


def _cell_trend(series, label):
    """Mann-Kendall + Sen's slope for a cell_series dict.

    Returns (trend, verdict_text), or (None, too-few-data text) when
    fewer than 10 finite years are available.  The BH q-value is not
    part of this: the caller adds it from the precomputed map
    (see _map_q).  Shared by /api/cell and /api/report so both quote
    the same numbers.
    """
    years = np.array(series["years"], dtype=float)
    values = np.array([np.nan if v is None else v for v in series["values"]],
                      dtype=float)
    finite = np.isfinite(values)
    if int(finite.sum()) < 10:
        return None, (f"Too few years with data (n = {int(finite.sum())}) "
                      "to test a trend.")
    mk = mann_kendall(values, years, correction="hamed-rao", lag=3)
    trend = {
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
    verdict_text = verdict(
        {"slope_per_decade": mk["slope_per_decade"],
         "p_value": mk["p_value"], "significant": mk["significant"],
         "n": mk["n"]},
        series["unit"], label)
    return trend, verdict_text


def _map_q(variable, cell_lat, cell_lon):
    """(q_value, significant_bh) for a cell from the precomputed annual
    map, or (None, None) when the map has no value there."""
    default_map = _load_map_payload(variable)
    ilat = int(np.argmin(np.abs(np.array(default_map["lat"]) - cell_lat)))
    ilon = int(np.argmin(np.abs(np.array(default_map["lon"]) - cell_lon)))
    q = default_map["q_value"][ilat][ilon]
    return q, (None if q is None
               else bool(default_map["significant"][ilat][ilon]))


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
    label = "precipitation" if variable == "precipitation" else "temperature"
    payload = dict(series)
    payload["variable"] = label
    payload["records"] = record_years(series["years"], series["values"])
    trend, verdict_text = _cell_trend(series, label)
    payload["trend"] = trend
    payload["verdict"] = verdict_text
    if trend is not None:
        try:
            q_value, significant_bh = _map_q(variable, series["cell_lat"],
                                             series["cell_lon"])
        except (FileNotFoundError, KeyError, IndexError):
            q_value, significant_bh = None, None
        payload["trend"]["q_value"] = q_value
        payload["trend"]["significant_bh"] = significant_bh
    return jsonify(payload)


# In-memory cache of the hotspots (the precomputed maps never change
# while the app runs).
_HOTSPOTS_CACHE = None


@app.route("/api/hotspots")
def hotspots():
    """Top cells where significant warming meets significant drying
    (drought risk) or significant wetting (flood risk).

    Both signals must be significant after Benjamini-Hochberg FDR
    (q <= 0.05) on the precomputed annual maps; see
    src/grid.py:find_hotspots for the alignment and the ordering.
    """
    global _HOTSPOTS_CACHE
    if _HOTSPOTS_CACHE is None:
        try:
            _HOTSPOTS_CACHE = find_hotspots(GRID_DIR)
        except (FileNotFoundError, OSError) as exc:
            return jsonify({"error": str(exc)}), 503
    return jsonify(_HOTSPOTS_CACHE)


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


# ---------------------------------------------------------------------------
# Plain-text findings report (/api/report)
# ---------------------------------------------------------------------------


def _coord_slug(lat, lon):
    """'31n-75e' / '31.25n-73.75w' — a filename-safe cell label."""
    def part(v, pos, neg):
        return f"{abs(v):g}{pos if v >= 0 else neg}"
    return f"{part(lat, 'n', 's')}-{part(lon, 'e', 'w')}"


def _window_trend(years, values, start, end):
    """Mann-Kendall + Sen's slope (+95% CI) for a year window, or None
    when fewer than 5 finite years fall inside it."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (years >= start) & (years <= end) & np.isfinite(values)
    if int(mask.sum()) < 5:
        return None
    mk = mann_kendall(values[mask], years[mask], correction="hamed-rao",
                      lag=3)
    return {"slope_per_decade": mk["slope_per_decade"],
            "ci_lo_per_decade": mk["ci_lo"] * 10,
            "ci_hi_per_decade": mk["ci_hi"] * 10,
            "p_value": mk["p_value"],
            "significant": bool(mk["significant"]),
            "n": mk["n"],
            "start_year": int(years[mask].min()),
            "end_year": int(years[mask].max())}


def _slope_text(slope, unit, p_value, n, ci_lo=None, ci_hi=None):
    """'+0.246 °C/decade (95% CI +0.137 to +0.359, p ≈ 5.7×10⁻⁵, n = 44)'."""
    text = f"{slope:+.3f} {unit}/decade"
    if ci_lo is not None:
        text += f" (95% CI {ci_lo:+.3f} to {ci_hi:+.3f})"
    return f"{text} ({fmt_p(p_value)}, n = {n})"


def _decade_table_text(decades, unit):
    lines = []
    for d in decades:
        if d["slope_per_decade"] is None:
            lines.append(f"  {d['decade']} ({d['start_year']}-{d['end_year']}): "
                         f"too few years (n = {d['n']})")
        else:
            sig = "significant" if d["significant"] else "not significant"
            lines.append(f"  {d['decade']}: "
                         f"{_slope_text(d['slope_per_decade'], unit, d['p_value'], d['n'])}"
                         f" — {sig}")
    return "\n".join(lines)


def _records_text(records, unit, high_word, low_word):
    if records["max"] is None:
        return "  No year has data."
    return (f"  {high_word}: {records['max']['year']} "
            f"({records['max']['value']:+.3f} {unit}); "
            f"{low_word}: {records['min']['year']} "
            f"({records['min']['value']:+.3f} {unit})")


def _changepoint_text(result, unit):
    """One honest change-point line (or the honest 'none' line)."""
    if not result["supported"]:
        p = result["p_value"]
        return ("No supported change point in this series "
                f"({fmt_p(p) if p is not None else 'permutation test unavailable'}).")
    before, after = result["before"], result["after"]
    ratio = ""
    if (before["slope_per_decade"] and after["slope_per_decade"]
            and before["slope_per_decade"] > 0):
        ratio = (f" — about "
                 f"{after['slope_per_decade'] / before['slope_per_decade']:.1f}x "
                 "faster")
    return (f"Change detected around {result['break_year']} "
            f"({fmt_p(result['p_value'])}): "
            f"{_slope_text(before['slope_per_decade'], unit, before['p_value'], before['n'])} "
            f"before ({before['start_year']}-{before['end_year']}), "
            f"{_slope_text(after['slope_per_decade'], unit, after['p_value'], after['n'])} "
            f"after ({after['start_year']}-{after['end_year']}){ratio}.\n"
            "  (Caveat: the CUSUM peak marks where the series crosses its "
            "single-line fit, so \"around\" is the claim, not the exact year.)")


def _lahore_neighbour_window_slopes(cache):
    """1981-2024 Mann-Kendall slopes of the 8 cells around Lahore's
    (31N, 75E) cell, computed from the gridded cache."""
    slopes = []
    for dlat in (-2, 0, 2):
        for dlon in (-2, 0, 2):
            if dlat == 0 and dlon == 0:
                continue
            s = cell_series(cache, 31.0 + dlat, 75.0 + dlon)
            yy = np.array(s["years"], dtype=float)
            vv = np.array([np.nan if v is None else v for v in s["values"]],
                          dtype=float)
            w = _window_trend(yy, vv, 1981, 2024)
            if w is not None:
                slopes.append(w["slope_per_decade"])
    return slopes


def _cell_report(variable, lat, lon):
    """(filename, text) findings report for one grid cell.

    Every number is computed here from the cached real data: the
    full-period trend, the 1981-2024 window (same Mann-Kendall test),
    the decade table, the cell's own change-point scan, and — only for
    the Lahore temperature cell — the POWER-vs-GISTEMP disagreement
    note, with the POWER numbers computed live from the Lahore series.
    """
    cache = load_dataset(variable, GRID_DIR)
    series = cell_series(cache, lat, lon)
    label = "precipitation" if variable == "precipitation" else "temperature"
    unit = series["unit"]
    years = np.array(series["years"], dtype=float)
    values = np.array([np.nan if v is None else v for v in series["values"]],
                      dtype=float)
    finite = np.isfinite(values)
    trend, verdict_text = _cell_trend(series, label)
    records = record_years(series["years"], series["values"])
    window = _window_trend(years, values, 1981, 2024)
    decades, full_period = decadal_trends(years, values)
    changepoint = detect_changepoint(years, values)
    high_word = "Wettest" if label == "precipitation" else "Warmest"
    low_word = "Driest" if label == "precipitation" else "Coldest"

    lines = [
        "TREND DETECTIVE — FINDINGS FOR ONE GRID CELL",
        "=" * 45,
        f"Location: cell centre {series['cell_lat']:g}°, {series['cell_lon']:g}° "
        f"(nearest to {lat:g}°, {lon:g}°)",
        f"Variable: {VARIABLES[variable]['label']} — "
        f"{series['source']} ({unit})",
        f"Period: {int(years[finite].min()) if finite.any() else '—'}-"
        f"{int(years[finite].max()) if finite.any() else '—'}, "
        f"n = {int(finite.sum())} years with data",
        "",
        "TREND (Mann-Kendall + Sen's slope, Hamed-Rao autocorrelation "
        "correction, lag 3)",
    ]
    if trend is None:
        lines.append(f"  {verdict_text}")
    else:
        try:
            q_value, significant_bh = _map_q(variable, series["cell_lat"],
                                             series["cell_lon"])
        except (FileNotFoundError, KeyError, IndexError):
            q_value, significant_bh = None, None
        lines.append("  " + _slope_text(trend["slope_per_decade"], unit,
                                        trend["p_value"], trend["n"],
                                        trend["ci_lo_per_decade"],
                                        trend["ci_hi_per_decade"]))
        if q_value is None:
            lines.append("  Benjamini-Hochberg q-value unavailable for this cell.")
        else:
            bh = ("significant after Benjamini-Hochberg FDR (q ≤ 0.05)"
                  if significant_bh else
                  "not significant after Benjamini-Hochberg FDR (q > 0.05)")
            q_text = "q < 0.001" if q_value <= 0 else f"q = {q_value:.3g}"
            lines.append(f"  {q_text} — {bh}")
        lines.append(f"  Verdict: {verdict_text}")
    lines += [
        "",
        "RECORD YEARS",
        _records_text(records, unit, high_word, low_word),
        "",
        "1981–2024 WINDOW (same Mann-Kendall test, same cell)",
    ]
    if window is None:
        lines.append("  Too few years with data in 1981–2024 to test.")
    else:
        lines.append("  " + _slope_text(window["slope_per_decade"], unit,
                                        window["p_value"], window["n"],
                                        window["ci_lo_per_decade"],
                                        window["ci_hi_per_decade"])
                     + (" — significant" if window["significant"]
                        else " — not significant"))
    lines += [
        "",
        "TREND BY DECADE",
        _decade_table_text(decades, unit),
        "",
        "CHANGE POINT (CUSUM scan + 2,000-shuffle permutation test)",
        "  " + _changepoint_text(changepoint, unit),
    ]
    if (variable == "temperature" and series["cell_lat"] == 31.0
            and series["cell_lon"] == 75.0):
        pyears, pvalues, _ = load_series("lahore", "temperature")
        ptrend = analyse_trend(pyears, pvalues)
        neighbours = _lahore_neighbour_window_slopes(cache)
        lines += [
            "",
            "NOTE — LAHORE: THE TWO DATASETS DISAGREE",
            "  The NASA POWER point series for Lahore (1981–2024) shows "
            f"no significant warming: {_slope_text(ptrend['slope_per_decade'], '°C', ptrend['p_value'], ptrend['n'])}.",
            "  The GISTEMP map cell above says the opposite over the same "
            "1981–2024 window" + (f": {_slope_text(window['slope_per_decade'], '°C', window['p_value'], window['n'])}." if window else "."),
        ]
        if neighbours:
            lines.append(
                f"  All {len(neighbours)} neighbouring GISTEMP cells with data agree: "
                f"over the same window they warm at "
                f"{min(neighbours):+.2f} to {max(neighbours):+.2f} °C/decade.")
        lines.append(
            "  The honest finding is a dataset disagreement, not \"Lahore is "
            "not warming\": the station-based GISTEMP record says Lahore warms "
            "like the rest of the region; the POWER reanalysis point series "
            "says it does not.")
    lines += [
        "",
        "DATA SOURCES",
        f"  {series['source']} — {series['source_url']}",
        "METHOD",
        "  Mann-Kendall test + Sen's slope (stats/mannkendall.py), Hamed-Rao "
        "autocorrelation correction (lag 3); Benjamini-Hochberg "
        "false-discovery control, alpha 0.05. A coloured cell on the map is "
        "one with q ≤ 0.05.",
        "  Generated by Trend Detective (NASA Space Apps 2026) from the "
        "cached real data — no numbers invented.",
    ]
    slug = _coord_slug(series["cell_lat"], series["cell_lon"])
    filename = f"trend-detective-findings-{slug}-{variable}.txt"
    return filename, "\n".join(lines) + "\n"


def _global_report(variable):
    """(filename, text) global findings report for one variable."""
    years, values, meta = _global_series(variable)
    label = "precipitation" if variable == "precipitation" else "temperature"
    unit = meta["unit"]
    years_a = np.array(years, dtype=float)
    values_a = np.array(values, dtype=float)
    decades, full = decadal_trends(years_a, values_a)
    records = record_years(years, values)
    changepoint = detect_changepoint(years, values)
    high_word = "Wettest" if label == "precipitation" else "Warmest"
    low_word = "Driest" if label == "precipitation" else "Coldest"

    lines = [
        "TREND DETECTIVE — GLOBAL FINDINGS",
        "=" * 40,
        f"Variable: {meta['variable']} — {meta['source']} ({unit})",
        f"Region: {meta['region']}",
        "",
        "HEADLINE",
    ]
    if variable == "temperature":
        lines.append(
            "  Full record "
            f"({full['start_year']}-{full['end_year']}, n = {full['n']}): "
            + _slope_text(full["slope_per_decade"], unit, full["p_value"],
                          full["n"])
            + (" — significant" if full["significant"] else " — not significant"))
        window = _window_trend(years_a, values_a, 1981, 2025)
        if window is not None:
            lines.append(
                "  Since 1981 "
                f"({window['start_year']}-{window['end_year']}, n = {window['n']}): "
                + _slope_text(window["slope_per_decade"], unit,
                              window["p_value"], window["n"],
                              window["ci_lo_per_decade"],
                              window["ci_hi_per_decade"])
                + (" — significant" if window["significant"] else " — not significant"))
    else:
        lines.append(
            "  Full record "
            f"({full['start_year']}-{full['end_year']}, n = {full['n']}): "
            + _slope_text(full["slope_per_decade"], unit, full["p_value"],
                          full["n"])
            + (" — significant" if full["significant"] else " — not significant")
            + " (area-weighted mean of the GPCP gridded cache)")
    lines += [
        "",
        "RECORD YEARS",
        _records_text(records, unit, high_word, low_word),
        "",
        "TREND BY DECADE",
        _decade_table_text(decades, unit),
        "",
        "CHANGE POINT (CUSUM scan + 2,000-shuffle permutation test)",
        "  " + _changepoint_text(changepoint, unit),
        "",
        "ARCTIC SEA ICE — A THIRD, INDEPENDENT WITNESS",
    ]
    try:
        cache = load_seaice_cache()
        syears = cache["years"].astype(float)
        for key, name in (("september", "September (minimum)"),
                          ("march", "March (maximum)"),
                          ("annual_mean", "Annual mean")):
            mk = mann_kendall(cache[key].astype(float), syears,
                              correction="hamed-rao", lag=3)
            lines.append(
                f"  {name}: "
                f"{_slope_text(mk['slope_per_decade'], SEAICE_UNIT, mk['p_value'], mk['n'])}"
                f", from {float(cache[key][0]):.2f} in {int(syears[0])} "
                f"to {float(cache[key][-1]):.2f} in {int(syears[-1])}.")
        sept = cache["september"]
        i_min = int(np.argmin(sept))
        lines.append(f"  Record-low September: {float(sept[i_min]):.2f} "
                     f"million km² in {int(cache['years'][i_min])}.")
    except (FileNotFoundError, OSError) as exc:
        lines.append(f"  Sea ice cache unavailable: {exc}")
    lines += [
        "",
        "DATA SOURCES",
        f"  {meta['source']} — {meta['url']}",
        f"  {SEAICE_SOURCE} — {SEAICE_URL}",
        "METHOD",
        "  Mann-Kendall test + Sen's slope (stats/mannkendall.py), Hamed-Rao "
        "autocorrelation correction (lag 3); Benjamini-Hochberg "
        "false-discovery control, alpha 0.05, on the map cells. Change "
        "point: CUSUM scan on OLS residuals, support by permutation test.",
        "  Generated by Trend Detective (NASA Space Apps 2026) from the "
        "cached real data — no numbers invented.",
    ]
    filename = f"trend-detective-findings-global-{variable}.txt"
    return filename, "\n".join(lines) + "\n"


@app.route("/api/report")
def report():
    """Plain-text findings report as JSON: {"filename": ..., "text": ...}.

    Query params: variable=temperature|precipitation (default
    temperature); without lat/lon the global report, with lat/lon the
    report for that grid cell.  Every number is computed server-side
    from the cached real data.
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
            filename, text = _global_report(variable)
        else:
            filename, text = _cell_report(variable, *pair)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"filename": filename, "text": text})


if __name__ == "__main__":
    # Local dev: python app.py  -> http://127.0.0.1:5000
    # Hosting (Render etc.) sets $PORT; gunicorn runs `app:app` instead.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
