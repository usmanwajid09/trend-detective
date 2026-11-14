"""Tests for the Flask routes in app.py."""

from pathlib import Path

import pytest

from app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Trend Detective" in html
    assert "Lahore, Pakistan" in html
    assert "Reykjavik, Iceland" in html
    assert "Global" in html


def test_series_lahore_temperature(client):
    resp = client.get("/api/series?region=lahore&variable=temperature")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["region"] == "Lahore, Pakistan"
    assert d["unit"] == "°C"
    assert len(d["years"]) == len(d["values"]) == 44
    assert d["years"][0] == 1981 and d["years"][-1] == 2024
    assert d["trend"]["significant"] is False  # honest: no Lahore warming trend
    assert "No significant trend" in d["verdict"]
    assert "NASA POWER" in d["source"]


def test_series_reykjavik_warming(client):
    resp = client.get("/api/series?region=reykjavik&variable=temperature")
    d = resp.get_json()
    assert d["trend"]["significant"] is True
    assert d["trend"]["slope_per_decade"] > 0
    assert "Significant warming" in d["verdict"]


def test_series_global(client):
    resp = client.get("/api/series?region=global&variable=temperature")
    d = resp.get_json()
    assert d["source"] == "NASA GISTEMP v4"
    assert d["years"][0] == 1880
    assert d["trend"]["slope_per_decade"] > 0
    assert d["trend"]["significant"] is True


def test_summary_covers_all_series(client):
    resp = client.get("/api/summary")
    d = resp.get_json()
    assert len(d) == 5  # 2 regions x 2 variables + global temperature
    by_key = {(r["region"], r["variable"]) for r in d}
    assert ("Lahore, Pakistan", "Precipitation") in by_key
    lahore_rain = [r for r in d if r["region"] == "Lahore, Pakistan"
                   and r["variable"] == "Precipitation"][0]
    assert lahore_rain["significant"] is True


def test_series_records(client):
    """Record years ride along in /api/series (global and regional)."""
    d = client.get("/api/series?region=global&variable=temperature").get_json()
    assert d["records"] == {"max": {"year": 2024, "value": 1.28},
                            "min": {"year": 1909, "value": -0.48}}
    d = client.get("/api/series?region=lahore&variable=temperature").get_json()
    assert d["records"] == {"max": {"year": 1999, "value": 27.428},
                            "min": {"year": 2020, "value": 23.866}}
    d = client.get("/api/series?region=lahore&variable=precipitation").get_json()
    assert d["records"] == {"max": {"year": 2023, "value": 3.272},
                            "min": {"year": 1998, "value": 0.611}}


def test_bad_region_is_400(client):
    resp = client.get("/api/series?region=nowhere&variable=temperature")
    assert resp.status_code == 400


def test_global_has_no_precipitation(client):
    resp = client.get("/api/series?region=global&variable=precipitation")
    assert resp.status_code == 400


def test_index_third_fix_round(client):
    """Third fix round (2026-11-14): the slider state machine can no
    longer stick on "Recomputing…" (loadMap never throws, the flag is
    cleared when the newest map lands, the cell refetch cannot throw),
    the map uses a squared colour scale over the Esri World Street Map
    with a Boundaries and Places label layer on top, the decades table
    fits, and the leaderboard says "up to 10"."""
    html = client.get("/").get_data(as_text=True)
    assert "World_Street_Map" in html
    assert "World_Boundaries_and_Places" in html
    assert "World_Light_Gray" not in html
    assert "world.geo.json" not in html
    assert "loadBorders" not in html
    assert "Temp q" not in html
    assert "Precip q" not in html
    assert html.count("<th>Temp p</th>") == 2
    assert "animateCounters" not in html
    assert "data-count" not in html
    assert "1880&ndash;2026</output>" in html
    assert "Tested, not significant" not in html
    assert "Only cells with a significant trend are coloured" in html
    assert "refetchCellQuietly" in html
    assert "Could not load the trend map" in html
    assert "<th>Slope / dec</th>" in html
    assert "<th>Sig.</th>" in html
    assert "up to 10 in" in html
    # Fix round 4 (same evening): the leaderboard ReferenceError that
    # broke every window change, and the restore targeting the loaded
    # window rather than the requested one.
    assert "const short = picked" in html
    assert "const loadedBefore" in html
    assert "const restoreStart" in html
    assert "Esri World Street Map" in html
    assert "Esri, HERE, Garmin" in html
    # Fix round 5: the inspector is capped at the map column's height
    # (desktop only) so a tall cell panel cannot leave a blank hole
    # under the map; the slider also applies on release, and the
    # decades chart drops its crowded "Year" axis title.
    assert "addEventListener('change'" in html
    assert "opts.scales.x.title.display = false" in html
    assert "SQUARE" in html  # slopeColor comment: squared colour scale
    css = (Path(__file__).parent.parent / "static" / "style.css").read_text()
    assert "rgb(127,165,205)" in css  # legend gradient, same squared stops
    assert "@media (min-width: 1081px)" in css
    assert ".inspector { max-height: 45rem; overflow-y: auto; }" in css
