"""Tests for the /api/report endpoint (plain-text findings reports)."""

import pytest

from app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_report_cell_lahore_temperature(client):
    resp = client.get("/api/report?variable=temperature&lat=31.52&lon=74.36")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["filename"] == "trend-detective-findings-31n-75e-temperature.txt"
    text = d["text"]
    assert "cell centre 31°, 75°" in text
    # The inspected cell's own slope, from the same Mann-Kendall test
    # /api/cell reports.
    assert "+0.122 °C/decade" in text
    # The 1981-2024 window line, computed server-side from the cache.
    assert "1981–2024 WINDOW" in text
    assert "+0.246 °C/decade" in text
    assert "5.7×10⁻⁵" in text
    # Decade table and record years.
    assert "TREND BY DECADE" in text
    assert "2010s:" in text
    assert "Warmest: 2022" in text
    assert "Coldest: 1893" in text
    # The Lahore disagreement note, with both datasets' live numbers.
    assert "DATASETS DISAGREE" in text
    assert "-0.061 °C/decade" in text  # NASA POWER point series
    assert "+0.20 to +0.30 °C/decade" in text  # neighbouring cells
    assert "NASA GISTEMP v4 gridded" in text


def test_report_cell_slope_matches_api_cell(client):
    """The report must quote the same slope /api/cell does."""
    cell = client.get(
        "/api/cell?variable=temperature&lat=64.15&lon=-21.94").get_json()
    slope = cell["trend"]["slope_per_decade"]
    text = client.get(
        "/api/report?variable=temperature&lat=64.15&lon=-21.94"
    ).get_json()["text"]
    assert f"{slope:+.3f} °C/decade" in text
    assert "cell centre 65°, -21°" in text
    # No Lahore note anywhere but the Lahore cell.
    assert "DATASETS DISAGREE" not in text


def test_report_cell_precipitation(client):
    d = client.get(
        "/api/report?variable=precipitation&lat=31.52&lon=74.36").get_json()
    assert d["filename"] == "trend-detective-findings-31.25n-73.75e-precipitation.txt"
    text = d["text"]
    assert "cell centre 31.25°, 73.75°" in text
    assert "Wettest:" in text and "Driest:" in text
    assert "DATASETS DISAGREE" not in text  # temperature-only note


def test_report_global_temperature(client):
    resp = client.get("/api/report?variable=temperature")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["filename"] == "trend-detective-findings-global-temperature.txt"
    text = d["text"]
    assert "GLOBAL FINDINGS" in text
    assert "Since 1981" in text
    assert "+0.211 °C/decade" in text
    assert "Warmest: 2024" in text
    assert "Coldest: 1909" in text
    assert "2010s:" in text
    assert "Change detected around 2000" in text
    assert "4.8x faster" in text
    assert "September (minimum): -0.741 million km²/decade" in text
    assert "Record-low September: 3.57 million km² in 2012" in text
    assert "NASA GISTEMP v4" in text
    assert "NSIDC Sea Ice Index" in text


def test_report_global_precipitation(client):
    d = client.get("/api/report?variable=precipitation").get_json()
    assert d["filename"] == "trend-detective-findings-global-precipitation.txt"
    text = d["text"]
    assert "Wettest: 1998" in text
    assert "Driest: 1991" in text
    assert "not significant" in text
    assert "No supported change point" in text
    assert "September (minimum): -0.741" in text


def test_report_bad_params(client):
    assert client.get("/api/report?variable=ice").status_code == 400
    assert client.get("/api/report?lat=31").status_code == 400
    assert client.get("/api/report?lon=75").status_code == 400
    assert client.get("/api/report?lat=abc&lon=75").status_code == 400
    assert client.get("/api/report?lat=100&lon=75").status_code == 400
