"""Trend Detective — NASA Space Apps 2026, "Be An Earth System Trend Detective!"

Flask app serving real NASA data (cached under data/):
- NASA POWER monthly T2M / PRECTOTCORR for Lahore and Reykjavik, 1981-2024
- NASA GISTEMP v4 global annual temperature anomaly, 1880-2025

Run:  python app.py   (then open http://127.0.0.1:5000)
"""

import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import REGIONS, VARIABLES, load_series  # noqa: E402
from src.trends import analyse_trend, verdict  # noqa: E402

app = Flask(__name__)


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


if __name__ == "__main__":
    # Local dev: python app.py  -> http://127.0.0.1:5000
    # Hosting (Render etc.) sets $PORT; gunicorn runs `app:app` instead.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
