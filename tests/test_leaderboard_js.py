"""Regression test for the leaderboard sign filter, run against the
SHIPPED JavaScript in templates/index.html.

The bug (found by a live visual check, 2026-11-14): "Fastest cooling"
listed warming cells.  At the 2016 window only 3 cells genuinely
cooled, and ranks 4-10 of the cooling list were warming cells (+0.37
to +0.56 degC/decade) because the grouping sorted the whole
significant set and took the top 10 without filtering by sign.

The test extracts pickLeaderboardGroups + dedupCells from the template
(between their marker comments) and runs them in node against a
2016-window-shaped cell list, so it tests the code the browser
actually runs — not a Python re-implementation of it.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "index.html"


def _extract(html, start_marker, end_marker):
    start = html.index(start_marker)
    start = html.index("*/", start) + 2
    end = html.index(end_marker, start)
    return html[start:end]


def _run_groups(cells, is_temp):
    html = TEMPLATE.read_text()
    js = (
        _extract(html, "/* DEDUP-START */", "/* DEDUP-END */")
        + "\n"
        + _extract(html, "/* LEADERBOARD-PURE-START", "/* LEADERBOARD-PURE-END */")
        + "\nconsole.log(JSON.stringify(pickLeaderboardGroups("
        + json.dumps(cells)
        + ", "
        + ("true" if is_temp else "false")
        + ")));"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js)
        path = f.name
    try:
        out = subprocess.run(
            ["node", path], capture_output=True, text=True, timeout=30
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _cells_2016_shaped():
    """3 genuinely cooling cells plus many warming ones, spread out so
    dedup (4 degrees) never collapses them — the shape of the 2016
    window that exposed the bug."""
    cells = [
        {"lat": 47, "lon": 79, "slope": -2.594, "p": 1e-4, "q": 0.01},
        {"lat": 20, "lon": -100, "slope": -1.410, "p": 1e-3, "q": 0.02},
        {"lat": -30, "lon": 150, "slope": -1.266, "p": 1e-3, "q": 0.03},
    ]
    for i in range(20):
        cells.append(
            {"lat": -80 + i * 8, "lon": -170 + i * 17,
             "slope": 0.373 + i * 0.01, "p": 1e-4, "q": 0.01}
        )
    return cells


def test_cooling_never_lists_warming_cells():
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    groups = _run_groups(_cells_2016_shaped(), True)
    by_key = {g["key"]: g["cells"] for g in groups}
    cooling = by_key["cooling"]
    assert len(cooling) == 3  # only the 3 genuinely cooling cells
    assert all(c["slope"] < 0 for c in cooling)
    assert [c["slope"] for c in cooling] == [-2.594, -1.410, -1.266]
    warming = by_key["warming"]
    assert all(c["slope"] > 0 for c in warming)
    assert len(warming) == 10  # dedup cap, all genuinely warming


def test_full_record_lists_still_fill():
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    cells = []
    for i in range(15):
        cells.append({"lat": -80 + i * 10, "lon": -170 + i * 20,
                      "slope": 0.9 - i * 0.03, "p": 1e-4, "q": 0.01})
    for i in range(12):
        cells.append({"lat": -70 + i * 10, "lon": -160 + i * 20,
                      "slope": -0.9 + i * 0.03, "p": 1e-4, "q": 0.01})
    groups = _run_groups(cells, True)
    by_key = {g["key"]: g["cells"] for g in groups}
    assert len(by_key["warming"]) == 10
    assert all(c["slope"] > 0 for c in by_key["warming"])
    assert len(by_key["cooling"]) == 10
    assert all(c["slope"] < 0 for c in by_key["cooling"])


def test_precipitation_groups_filter_by_sign():
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    cells = [
        {"lat": 10, "lon": 20, "slope": -0.5, "p": 1e-4, "q": 0.01},
        {"lat": 30, "lon": 40, "slope": 0.4, "p": 1e-4, "q": 0.01},
        {"lat": 50, "lon": 60, "slope": 0.1, "p": 1e-4, "q": 0.01},
    ]
    groups = _run_groups(cells, False)
    by_key = {g["key"]: g["cells"] for g in groups}
    assert len(by_key["drying"]) == 1  # only the negative slope
    assert all(c["slope"] < 0 for c in by_key["drying"])
    assert len(by_key["wetting"]) == 2
    assert all(c["slope"] > 0 for c in by_key["wetting"])
