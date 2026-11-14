"""Tests for /api/cell start_year/end_year: the selected-cell panel
follows the map window instead of always quoting the full record.

The windowed trend is computed from the same cell series, restricted
to the window; its q-value comes from the map payload for the same
window (in-memory cached by _load_map_payload, so this is cheap once
the map itself has been loaded).  A cell with too few years in the
window falls back to the full record and says so in window_note.
"""

import numpy as np
import pytest

from app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


LAHORE_TEMP = {"variable": "temperature", "lat": 31.52, "lon": 74.36}


def _map_q_at(client, variable, cell_lat, cell_lon, start_year):
    """q_value for a cell from the /api/map payload for a window."""
    d = client.get(
        "/api/map",
        query_string={"variable": variable, "start_year": start_year},
    ).get_json()
    ilat = int(np.argmin(np.abs(np.array(d["lat"]) - cell_lat)))
    ilon = int(np.argmin(np.abs(np.array(d["lon"]) - cell_lon)))
    return d["q_value"][ilat][ilon], d["significant"][ilat][ilon]


def test_cell_default_is_full_record(client):
    d = client.get("/api/cell", query_string=LAHORE_TEMP).get_json()
    assert d["trend_years"] == [1880, 2026]
    assert "window_note" not in d
    # Full series still returned (the chart and record years need it).
    assert len(d["years"]) == len(d["values"]) == 147
    assert d["trend"]["slope_per_decade"] == pytest.approx(0.122, abs=0.005)


def test_cell_window_2016_lahore(client):
    d = client.get(
        "/api/cell",
        query_string={**LAHORE_TEMP, "start_year": 2016},
    ).get_json()
    assert d["trend_years"] == [2016, 2026]
    assert "window_note" not in d
    assert d["trend"] is not None
    assert d["trend"]["n"] == 11  # all eleven years have data here
    # The fit is over the window, not the full record: its median
    # point sits inside 2016-2026 (the full-record median is 1953).
    assert 2016 <= d["trend"]["median_year"] <= 2026
    # The series itself is still the full record.
    assert len(d["years"]) == 147
    # q comes from the 2016 map, not the full-record map.
    q, sig = _map_q_at(client, "temperature", 31.0, 75.0, 2016)
    assert d["trend"]["q_value"] == pytest.approx(q, abs=1e-9)
    assert d["trend"]["significant_bh"] == sig


def test_cell_window_precipitation(client):
    d = client.get(
        "/api/cell",
        query_string={"variable": "precipitation", "lat": 31.52,
                      "lon": 74.36, "start_year": 2016},
    ).get_json()
    assert d["trend_years"] == [2016, 2026]
    assert d["trend"] is not None
    assert d["trend"]["n"] == 11


def test_cell_window_too_few_years_falls_back_to_full_record(client):
    # Antarctic cell (-79, -45): 25 years of data overall, none in
    # 2016-2026 — the window cannot be tested, so the full record is
    # shown and the note says so.
    d = client.get(
        "/api/cell",
        query_string={"variable": "temperature", "lat": -79.0,
                      "lon": -45.0, "start_year": 2016},
    ).get_json()
    assert d["trend"] is not None
    assert d["trend_years"] == [1880, 2026]
    assert "window_note" in d
    assert "full record" in d["window_note"]


def test_cell_window_bad_params(client):
    base = dict(LAHORE_TEMP)
    assert client.get("/api/cell",
                      query_string={**base, "start_year": 2030}).status_code == 400
    assert client.get("/api/cell",
                      query_string={**base, "start_year": 2026,
                                    "end_year": 2026}).status_code == 400
    assert client.get("/api/cell",
                      query_string={**base, "start_year": 2020,
                                    "end_year": 2010}).status_code == 400
