"""Regression test for the year-slider window state machine, run against
the SHIPPED JavaScript in templates/index.html.

The bug (found by a live visual check, 2026-11-14): after dragging the
slider 1880 -> 2016 and back to 1880, the map heading stayed stuck on
"Global trend map · Recomputing…" (no year window) and the cell panel
stayed on the 2016 slope, even though the footer and leaderboard had
updated to 1880. Root cause, reproduced in this harness: loadMap() had
no catch around fetch, so a rejected fetch (e.g. a request failing
during the Render deploy restart that overlapped the check) propagated
out of applyYearWindow as an unhandled rejection, exiting with
windowPending still true and the slider logic dead. A second, smaller
defect fed the same symptom: loadMap() called updateSeasonLabel()
after assigning the new mapData but before applyYearWindow cleared
windowPending, so the heading briefly claimed the NEW window while
still saying "Recomputing…".

The test extracts the shipped state-machine functions (by the marker
comments in the template) and runs them in node under mocked DOM +
fetch, driving the exact live sequence plus failure injections. It
tests the code the browser actually runs — not a Python
re-implementation of it.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "index.html"

# Scenarios, in the order node runs them (one fresh vm context each).
SCENARIOS = ["clean", "midflight", "fail2016", "fail1880", "throwpaint"]


def _extract(html, start_marker, end_marker):
    start = html.index(start_marker)
    assert start >= 0, f"marker missing: {start_marker}"
    start = html.index("*/", start) + 2
    end = html.index(end_marker, start)
    assert end >= 0, f"marker missing: {end_marker}"
    return html[start:end]


# Mock DOM + fetch + the stubs the shipped code calls (paintOverlay,
# renderLeaderboard, setMapLoading, selectPrimary). The shipped
# functions under test are loadMap / applyYearWindow / the slider
# handler / fetchCell / the heading helpers; everything else is a
# stub that records what the shipped code asked it to do.
MOCKS_JS = r'''
const registry = {};
function makeEl(id, extra) {
  const handlers = {};
  const el = {
    id, textContent: '', value: '', style: {}, hidden: false,
    dataset: {}, className: '',
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(ev, fn) { (handlers[ev] = handlers[ev] || []).push(fn); },
    _handlers: handlers,
    dispatch(ev) { (handlers[ev] || []).forEach((fn) => fn()); },
    getContext: () => ({}),
    querySelectorAll: () => [],
    ...(extra || {}),
  };
  registry[id] = el;
  return el;
}
const $ = (id) => registry[id] || makeEl(id);
document = { getElementById: $ };

makeEl('map-variable', { value: 'temperature' });
makeEl('map-season', { value: 'annual' });
const mapVarSel = $('map-variable');
const seasonSel = $('map-season');
const SEASON_LABELS = { annual: 'Annual', DJF: 'Winter (DJF)', MAM: 'Spring (MAM)', JJA: 'Summer (JJA)', SON: 'Autumn (SON)' };
makeEl('year-start-slider', { value: '1880', min: '1880' });
makeEl('map-heading');
makeEl('map-season-label');
makeEl('map-stats');
makeEl('map-loading');
makeEl('map-loading-text');
makeEl('year-window-label');
makeEl('year-window-note');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const scenario = typeof __scenario !== 'undefined' ? __scenario : 'clean';
// fail1880 must only fire on the RETURN, never on the initial load.
const failOnce = { map2016: scenario === 'fail2016' };
let armed = false;

function mapPayload(years) {
  const tested = years[0] === 1880 ? 7949 : 6485;
  const sig = years[0] === 1880 ? 6170 : 80;
  return {
    years, lat: [31], lon: [75],
    slope_per_decade: [[years[0] === 1880 ? 0.12 : 0.14]],
    significant: [[true]], p_value: [[0.001]], q_value: [[0.002]],
    source: 'NASA GISTEMP v4 gridded', unit: '°C',
    variable: 'temperature', season: 'annual', _tested: tested, _sig: sig,
  };
}
function cellPayload(url) {
  const m = url.match(/start_year=(\d+)/);
  const start = m ? parseInt(m[1], 10) : 1880;
  return {
    cell_lat: 31.0, cell_lon: 75.0, variable: 'temperature', unit: '°C',
    years: [1880, 2026], values: [], n: start === 1880 ? 147 : 11,
    trend: { slope_per_decade: start === 1880 ? 0.1218 : 0.1432,
             ci_lo_per_decade: 0.1, ci_hi_per_decade: 0.14,
             p_value: start === 1880 ? 0.0 : 0.938, q_value: 0.0,
             significant_bh: start === 1880, median_year: start === 1880 ? 1953 : 2021 },
    trend_years: [start, 2026], records: null, source: 'NASA GISTEMP v4 gridded',
    verdict: 'warming', window_note: null,
  };
}
fetch = async (url) => {
  if (url.startsWith('/api/map')) {
    const m = url.match(/start_year=(\d+)/);
    const start = m ? parseInt(m[1], 10) : 1880;
    if (start === 2016 && failOnce.map2016) { failOnce.map2016 = false; throw new Error('network down (deploy restart)'); }
    if (start === 1880 && armed && scenario === 'fail1880') { armed = false; throw new Error('network down (deploy restart)'); }
    await sleep(start === 1880 ? 20 : 1200); // 2016 on-demand compute is slow
    return { ok: true, status: 200, json: async () => mapPayload([start, 2026]) };
  }
  if (url.startsWith('/api/cell')) {
    await sleep(50);
    return { ok: true, status: 200, json: async () => cellPayload(url) };
  }
  throw new Error('unexpected url ' + url);
};

let mapData = null;
let leafletMap = null, overlay = null;
let primaryMarker = null, compareMarker = null;
let primaryCell = null;
let panelWindow = '(none)';
let panelSlope = null;
function setMapLoading(on, text) {}
function paintOverlay() {
  const d = mapData;
  if (scenario === 'throwpaint' && d.years[0] === 2016) throw new Error('paintOverlay boom on 2016 payload');
  $('map-stats').textContent = d.source + ' · ' + d.years[0] + '–' + d.years[1] +
    ' · ' + d._tested.toLocaleString() + ' cells tested · ' + d._sig.toLocaleString() + ' significant';
  return 'data:image/png;base64,x';
}
function renderLeaderboard() {}
async function selectPrimary(lat, lon) {
  const d = await fetchCell(lat, lon);
  primaryCell = { lat: d.cell_lat, lon: d.cell_lon, payload: d };
  const ty = d.trend_years || [d.years[0], d.years[d.years.length - 1]];
  panelWindow = ty[0] + '–' + ty[1];
  panelSlope = d.trend.slope_per_decade;
}
'''

DRIVER_JS = r'''
function state(tag) {
  return {
    tag,
    heading: $('map-heading').textContent,
    pending: windowPending,
    mapWindow: { ...mapWindow },
    mapYears: mapData ? [...mapData.years] : null,
    stats: $('map-stats').textContent,
    panel: panelWindow,
    panelSlope,
    note: $('year-window-note').textContent,
  };
}
async function setSlider(value, pauseMs) {
  yearSlider.value = String(value);
  yearSlider.dispatch('input');
  if (pauseMs) await sleep(pauseMs);
}
async function init() {
  await loadMap();
  await selectPrimary(31, 75); // Lahore preset clicked, like the live check did
  armed = true;
  return state('init');
}
async function run() {
  const out = [];
  out.push(await init());
  if (scenario === 'clean' || scenario === 'fail2016' || scenario === 'fail1880' || scenario === 'throwpaint') {
    await setSlider(2016);
    await sleep(1000); // debounce (600ms) fired, 2016 fetch still in flight
    out.push(state('in-flight 2016'));
    await sleep(2200);
    out.push(state('after 2016 settle'));
    await setSlider(1880); await sleep(2600);
    out.push(state('after 1880 return'));
    if (scenario === 'fail2016' || scenario === 'fail1880') {
      await setSlider(2016); await sleep(2600);
      out.push(state('after 2016 retry'));
      await setSlider(1880); await sleep(2600);
      out.push(state('after 1880 retry'));
    }
  } else if (scenario === 'midflight') {
    await setSlider(2016); await sleep(900); // 2016 still computing
    await setSlider(1880); await sleep(3000);
    out.push(state('after drag back mid-flight'));
  }
  __results = out;
}
run().catch((e) => { __results = [{ tag: 'DRIVER ERROR', error: String((e && e.message) || e) }]; });
'''

NODE_RUNNER = r'''
const vm = require('vm');
const SCENARIOS = ['clean', 'midflight', 'fail2016', 'fail1880', 'throwpaint'];
const rejections = [];
process.on('unhandledRejection', (e) => { rejections.push(String((e && e.message) || e)); });
(async () => {
  const results = {};
  for (const name of SCENARIOS) {
    rejections.length = 0;
    const sandbox = { console, setTimeout, clearTimeout, process, __scenario: name, __results: null };
    sandbox.global = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(MOCKS + '\n' + REGIONS + '\n' + DRIVER, sandbox, { filename: 'slider-' + name + '.js' });
    const deadline = Date.now() + 30000;
    while (sandbox.__results === null && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 100));
    }
    if (sandbox.__results === null) throw new Error('scenario ' + name + ' did not finish');
    results[name] = { states: sandbox.__results, rejections: rejections.slice() };
  }
  console.log(JSON.stringify(results));
})().catch((e) => { console.error(e && e.stack || e); process.exit(1); });
'''


def _run_harness():
    html = TEMPLATE.read_text()
    regions = "\n".join([
        _extract(html, "/* WINDOW-LABEL-START */", "/* WINDOW-LABEL-END */"),
        _extract(html, "/* LOADMAP-START */", "/* LOADMAP-END */"),
        _extract(html, "/* FETCHCELL-START */", "/* FETCHCELL-END */"),
        _extract(html, "/* SLIDER-START */", "/* SLIDER-END */"),
    ])
    node_src = (
        "const MOCKS = " + json.dumps(MOCKS_JS) + ";\n"
        "const REGIONS = " + json.dumps(regions) + ";\n"
        "const DRIVER = " + json.dumps(DRIVER_JS) + ";\n"
        + NODE_RUNNER
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(node_src)
        path = f.name
    try:
        out = subprocess.run(
            ["node", path], capture_output=True, text=True, timeout=120
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _by_tag(states):
    return {s["tag"]: s for s in states}


def test_slider_state_machine():
    results = _run_harness()
    assert set(results) == set(SCENARIOS)

    for name, res in results.items():
        assert res["rejections"] == [], f"{name}: unhandled rejections: {res['rejections']}"
        states = res["states"]
        assert not any(s.get("error") for s in states), f"{name}: {states}"
        for s in states:
            # The heading must never be stuck claiming a recompute once
            # the map has settled, and must never show a window the
            # loaded map does not cover.
            if s["tag"] not in ("in-flight 2016",):
                assert "Recomputing" not in s["heading"], f"{name} {s['tag']}: {s['heading']}"
            assert s["pending"] is False or "in-flight" in s["tag"] or "mid-flight" in s["tag"] or "mid-drag" in s["tag"], (
                f"{name} {s['tag']}: pending stuck true: {s}"
            )

    clean = _by_tag(results["clean"]["states"])
    assert clean["in-flight 2016"]["heading"] == "Global trend map \u00b7 1880\u20132026 \u00b7 Recomputing\u2026"
    assert clean["in-flight 2016"]["pending"] is True
    assert clean["after 2016 settle"]["heading"] == "Global trend map \u00b7 2016\u20132026"
    assert clean["after 2016 settle"]["mapYears"] == [2016, 2026]
    assert clean["after 2016 settle"]["panel"] == "2016\u20132026"
    assert clean["after 1880 return"]["heading"] == "Global trend map"
    assert clean["after 1880 return"]["mapYears"] == [1880, 2026]
    assert clean["after 1880 return"]["panel"] == "1880\u20132026"

    mid = _by_tag(results["midflight"]["states"])
    assert mid["after drag back mid-flight"]["heading"] == "Global trend map"
    assert mid["after drag back mid-flight"]["mapYears"] == [1880, 2026]
    assert mid["after drag back mid-flight"]["panel"] == "1880\u20132026"

    f16 = _by_tag(results["fail2016"]["states"])
    failed = f16["after 2016 settle"]
    assert failed["heading"] == "Global trend map"
    assert failed["mapYears"] == [1880, 2026]
    assert failed["panel"] == "1880\u20132026"
    assert "could not be computed" in failed["note"]
    assert f16["after 2016 retry"]["mapYears"] == [2016, 2026]
    assert f16["after 2016 retry"]["panel"] == "2016\u20132026"
    assert f16["after 1880 retry"]["mapYears"] == [1880, 2026]

    f80 = _by_tag(results["fail1880"]["states"])
    assert f80["after 2016 settle"]["panel"] == "2016\u20132026"
    returned = f80["after 1880 return"]
    # The failed return restores the previous window honestly: map,
    # heading and panel all stay on 2016, with the note saying so.
    assert returned["heading"] == "Global trend map \u00b7 2016\u20132026"
    assert returned["mapYears"] == [2016, 2026]
    assert returned["panel"] == "2016\u20132026"
    assert "could not be computed" in returned["note"]
    assert f80["after 1880 retry"]["mapYears"] == [1880, 2026]
    assert f80["after 1880 retry"]["panel"] == "1880\u20132026"

    tp = _by_tag(results["throwpaint"]["states"])
    assert tp["after 2016 settle"]["heading"] == "Global trend map"
    assert tp["after 2016 settle"]["mapYears"] == [1880, 2026]
    assert "could not be computed" in tp["after 2016 settle"]["note"]
