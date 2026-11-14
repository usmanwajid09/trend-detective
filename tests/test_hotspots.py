"""Tests for the /api/hotspots endpoint (drought/flood risk cells).

The lists pair the two precomputed annual trend maps: a cell qualifies
only when BOTH variables are significant after Benjamini-Hochberg FDR
(q <= 0.05) and the temperature slope is positive — drought adds a
negative precipitation slope, flood a positive one.
"""

import pytest

from app import app
from src.grid import find_hotspots


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


FIELDS = {"lat", "lon", "place", "temp_slope_per_decade",
          "precip_slope_per_decade", "temp_p_value", "precip_p_value",
          "temp_q_value", "precip_q_value"}


def test_hotspots_shape(client):
    resp = client.get("/api/hotspots")
    assert resp.status_code == 200
    d = resp.get_json()
    assert set(d) == {"drought", "flood"}
    # 421 drought and 636 flood candidates in the real maps, so both
    # lists hit the cap of 10.
    assert len(d["drought"]) == 10
    assert len(d["flood"]) == 10
    for entry in d["drought"] + d["flood"]:
        assert set(entry) == FIELDS
        assert -90 <= entry["lat"] <= 90
        assert -180 <= entry["lon"] <= 180


def test_hotspots_drought_signs_and_significance(client):
    d = client.get("/api/hotspots").get_json()
    for entry in d["drought"]:
        assert entry["temp_slope_per_decade"] > 0  # warming
        assert entry["precip_slope_per_decade"] < 0  # drying
        assert entry["temp_q_value"] <= 0.05
        assert entry["precip_q_value"] <= 0.05
        assert entry["temp_p_value"] > 0 and entry["precip_p_value"] > 0
    # Drought list is ranked by temperature slope (the driver of
    # evaporation/drought risk), steepest first.
    slopes = [e["temp_slope_per_decade"] for e in d["drought"]]
    assert slopes == sorted(slopes, reverse=True)


def test_hotspots_flood_signs_and_significance(client):
    d = client.get("/api/hotspots").get_json()
    for entry in d["flood"]:
        assert entry["temp_slope_per_decade"] > 0  # warming
        assert entry["precip_slope_per_decade"] > 0  # wetting
        assert entry["temp_q_value"] <= 0.05
        assert entry["precip_q_value"] <= 0.05
    # Flood list is ranked by precipitation slope (the driver of
    # flood risk), steepest first.
    slopes = [e["precip_slope_per_decade"] for e in d["flood"]]
    assert slopes == sorted(slopes, reverse=True)


def test_hotspots_top_cells_are_real(client):
    """The top entries are the known fastest cells in the real maps."""
    d = client.get("/api/hotspots").get_json()
    top_drought = d["drought"][0]
    assert (top_drought["lat"], top_drought["lon"]) == (23.75, 46.25)
    assert top_drought["temp_slope_per_decade"] == pytest.approx(0.57, abs=0.01)
    assert top_drought["precip_slope_per_decade"] < 0
    assert top_drought["place"] == "Saudi Arabia"
    top_flood = d["flood"][0]
    assert (top_flood["lat"], top_flood["lon"]) == (8.75, -73.75)
    assert top_flood["precip_slope_per_decade"] == pytest.approx(0.462, abs=0.01)
    assert top_flood["place"] == "Caribbean Sea"


def test_hotspots_places_match_cells(client):
    """Every top-10 row names its real place (test_placenames.py pins
    the table itself; this pins the endpoint's wiring of it)."""
    d = client.get("/api/hotspots").get_json()
    drought_places = [e["place"] for e in d["drought"]]
    flood_places = [e["place"] for e in d["flood"]]
    assert drought_places == [
        "Saudi Arabia", "South Atlantic Ocean", "South Atlantic Ocean",
        "Arctic Ocean", "Arctic Canada", "Greenland Sea", "Arctic Ocean",
        "Saudi Arabia", "Saudi Arabia", "Saudi Arabia"]
    assert flood_places == [
        "Caribbean Sea", "South Pacific Ocean", "South Pacific Ocean",
        "Indian Ocean", "South Pacific Ocean", "Philippine Sea",
        "Indian Ocean", "Philippine Sea", "South Pacific Ocean",
        "Caribbean Sea"]


def test_find_hotspots_top_parameter():
    small = find_hotspots(top=3)
    assert len(small["drought"]) == 3
    assert len(small["flood"]) == 3
    full = find_hotspots()
    assert small["drought"] == full["drought"][:3]
    assert small["flood"] == full["flood"][:3]
