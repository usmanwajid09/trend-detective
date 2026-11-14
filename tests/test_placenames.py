"""Tests for src/placenames.py — coarse place names for hotspot cells.

The table is deliberately tiny: ocean/sea names and a few well-known
land regions. A cell in no box must return None (the frontend then
shows coordinates alone) rather than an invented name.
"""

from src.placenames import REGIONS, place_name


def test_top_hotspot_rows_are_named():
    """Every cell in the real top-10 lists has a place name."""
    from src.grid import find_hotspots

    h = find_hotspots()
    for entry in h["drought"] + h["flood"]:
        assert entry["place"], (entry["lat"], entry["lon"])


def test_known_places():
    assert place_name(23.75, 46.25) == "Saudi Arabia"
    assert place_name(21.25, 43.75) == "Saudi Arabia"
    assert place_name(8.75, -73.75) == "Caribbean Sea"
    assert place_name(8.75, -76.25) == "Caribbean Sea"
    assert place_name(-18.75, -28.75) == "South Atlantic Ocean"
    assert place_name(-18.75, -176.25) == "South Pacific Ocean"
    assert place_name(-18.75, -173.75) == "South Pacific Ocean"
    assert place_name(-3.75, 58.75) == "Indian Ocean"
    assert place_name(6.25, 136.25) == "Philippine Sea"
    assert place_name(3.75, 133.75) == "Philippine Sea"
    assert place_name(71.25, -168.75) == "Arctic Ocean"
    assert place_name(71.25, -18.75) == "Greenland Sea"
    assert place_name(81.25, -73.75) == "Arctic Ocean"
    assert place_name(66.25, -78.75) == "Arctic Canada"


def test_lon_wraps_to_180_convention():
    # The precipitation grid ships 0..360; 183.75E is 176.25W.
    assert place_name(-18.75, 183.75) == "South Pacific Ocean"
    assert place_name(-18.75, -176.25) == place_name(-18.75, 183.75)


def test_unknown_cell_returns_none():
    # Central Asia and the central US are in no named region — the
    # frontend must fall back to coordinates, never invent a name.
    assert place_name(45.0, 100.0) is None
    assert place_name(35.0, -100.0) is None


def test_every_region_box_is_well_formed():
    for name, lat_min, lat_max, lon_min, lon_max in REGIONS:
        assert name
        assert -90 <= lat_min < lat_max <= 90
        assert -180 <= lon_min < lon_max <= 180
