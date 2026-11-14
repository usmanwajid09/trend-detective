"""Coarse place names for hotspot cells.

A hotspot row is a pair of coordinates, which means nothing to a
judge, so /api/hotspots also returns a short place name. The table is
deliberately tiny and coarse — ocean/sea names and a few well-known
land regions, keyed to latitude/longitude boxes. Nothing is looked up
at runtime (no external geocoding API, no network call).

Honesty rule: a box is only named when the name is right for the
whole box. A cell that falls in no box gets ``None`` and the frontend
falls back to coordinates alone — never an invented name.
"""

# (name, lat_min, lat_max, lon_min, lon_max); checked in order, so
# specific regions come before the broad oceans that contain them.
REGIONS = [
    # Well-known land regions (only where the whole box is that region).
    ("Saudi Arabia", 16.0, 32.5, 34.5, 56.0),
    ("Amazon rainforest", -12.0, 5.0, -72.0, -48.0),
    ("Sahara", 15.0, 32.0, -17.0, 34.5),
    # Seas, gulfs and bays.
    ("Greenland Sea", 68.0, 82.0, -25.0, 0.0),
    ("Caribbean Sea", 8.0, 20.0, -88.0, -60.0),
    ("Gulf of Mexico", 18.0, 30.0, -97.0, -80.0),
    ("Philippine Sea", 3.0, 30.0, 128.0, 160.0),
    ("Mediterranean Sea", 30.0, 46.0, -6.0, 36.0),
    # Broad ocean boxes (checked after the named seas above).
    ("Arctic Canada", 60.0, 72.0, -100.0, -60.0),
    ("Arctic Ocean", 70.0, 90.0, -180.0, 180.0),
    ("North Atlantic Ocean", 0.0, 50.0, -60.0, 0.0),
    ("South Atlantic Ocean", -45.0, 0.0, -45.0, 15.0),
    ("North Pacific Ocean", 0.0, 45.0, -180.0, -110.0),
    ("South Pacific Ocean", -45.0, 0.0, -180.0, -70.0),
    ("North Pacific Ocean", 0.0, 45.0, 150.0, 180.0),
    ("South Pacific Ocean", -45.0, 0.0, 150.0, 180.0),
    ("Indian Ocean", -45.0, 10.0, 20.0, 110.0),
    ("Southern Ocean", -60.0, -45.0, -180.0, 180.0),
]


def place_name(lat, lon):
    """Coarse place name for a cell centre, or None when unknown.

    Longitude is normalised to -180..180 first (the precipitation grid
    ships 0..360). Boundaries are inclusive; a point on a shared edge
    takes the first matching region, which is the more specific one.
    """
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    lat = float(lat)
    for name, lat_min, lat_max, lon_min, lon_max in REGIONS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return None
