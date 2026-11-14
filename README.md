# Trend Detective — NASA Space Apps 2026

**Challenge:** "Be An Earth System Trend Detective!" — build a tool that picks
variables measured by NASA missions or produced by NASA models, visualises how
they change over time, and answers: what is changing, where, how much, and
whether the change is significant.

**What it is:** a small Flask app. Pick a region (Lahore, Reykjavik, or Global)
and a variable (temperature or precipitation) and it draws the time series from
real NASA data, fits a trend, and gives a plain-language verdict with the
numbers — slope per decade and p-value — next to it. A summary table answers
"what changed, where, how much" for every region/variable at once. The
**global trend map** does the same test for every cell of the planet: pick a
variable, see where the change is real, and click any cell for its time
series, trend line and 95% confidence band.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
# open http://127.0.0.1:5000
```

Tests: `.venv/bin/python -m pytest tests/ -q` (127 tests, all passing).

To re-download the data: `.venv/bin/python -m src.fetch_data`
(it only writes real downloads — a failed download is reported, never faked).

### Gridded data (Milestone 1): build the map caches

```bash
.venv/bin/python -m src.fetch_data --grid   # GISTEMP + GPCP NetCDF -> data/raw/
.venv/bin/python -m src.grid build          # annual means -> data/grid/*.npz
.venv/bin/python -m src.grid map            # per-cell trend maps -> data/grid/map_*.npz
```

Raw NetCDF stays in `data/raw/` (git-ignored). The committed caches are
small: `temperature.npz` 1.9 MB (147 years, 90×180 grid),
`precipitation.npz` 1.8 MB (48 years, 72×144 grid), trend maps ~0.2 MB each.

## Data sources (all real, cached under `data/`)

- **NASA POWER** — https://power.larc.nasa.gov — monthly point data
  (community RE), parameters T2M (2 m air temperature, °C) and PRECTOTCORR
  (bias-corrected precipitation, mm/day), 1981–2024, for Lahore, Pakistan
  (31.52, 74.36) and Reykjavik, Iceland (64.15, −21.94).
  Files: `data/power_lahore.json`, `data/power_reykjavik.json`.
- **NASA GISTEMP v4** — https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv —
  global land–ocean annual temperature anomaly (J-D column, °C vs the
  1951–1980 baseline), 1880–2025. The incomplete 2026 row has no annual value
  and is excluded. File: `data/gistemp_global.csv`.
- **NASA GISTEMP v4 gridded** — https://data.giss.nasa.gov/pub/gistemp/gistemp250_GHCNv4.nc.gz —
  monthly temperature anomaly on a 2° grid, Jan 1880–Oct 2026. Built into
  annual means in `data/grid/temperature.npz` (a year is kept when ≥ 9 of its
  12 months are present; 2026 is a 10-month mean).
- **GPCP v2.3 precipitation** — https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc —
  monthly precipitation rate on a 2.5° grid, Jan 1979–Oct 2026. Built into
  annual means in `data/grid/precipitation.npz` (same ≥ 9-month rule).

## How the map statistics work

Point series use ordinary least-squares (above). The map uses a stricter
toolkit, because testing thousands of cells at p < 0.05 flags hundreds by
luck alone, and because year-to-year weather is correlated:

- **Mann-Kendall test + Sen's slope** (`stats/mannkendall.py`). The test
  only uses the order of values and the slope is the median of all pairwise
  slopes, so one odd year cannot fake a trend. Each slope carries a 95%
  confidence interval. Verified against the `pymannkendall` reference
  package (exact match on trend, no-trend, tied and autocorrelated series,
  for the original, Hamed-Rao and Yue-Wang variants).
- **Autocorrelation correction** (Hamed & Rao 1998, first 3 lags). When one
  year resembles the last, the data hold less independent information than
  their length suggests, so the test widens its uncertainty. On synthetic
  trend-free AR(1) data (n = 100, rho = 0.7) the false-"significant" rate
  falls from 39% uncorrected to 9% corrected, against a nominal 5%.
- **False-discovery control** (`stats/fdr.py`). The Benjamini-Hochberg
  procedure turns the map's p-values into q-values; a coloured cell is one
  with q ≤ 0.05. On the precipitation map it removes 1,128 of the 4,046
  raw p < 0.05 cells.

A full global grid (16,200 cells) is analysed in about 15 seconds on this
machine; the maps are precomputed at build time so the app serves them
instantly (`/api/map`, `/api/cell`).

## Results (computed from the cached data, 2026-11-13)

Ordinary least-squares regression of annual means on year; "significant"
means p < 0.05.

| Region | Variable | Period | Change / decade | p | Verdict |
|---|---|---|---|---|---|
| Global (land + ocean) | Temperature anomaly | 1880–2025 | +0.083 °C | ~5×10⁻⁴⁷ | Significant warming |
| Global (land + ocean) | Temperature anomaly | 1981–2025 | +0.212 °C | ~6×10⁻²¹ | Significant warming |
| Reykjavik, Iceland | Temperature | 1981–2024 | +0.194 °C | 0.0009 | Significant warming |
| Reykjavik, Iceland | Precipitation | 1981–2024 | −0.004 mm/day | 0.94 | No significant trend |
| Lahore, Pakistan | Temperature | 1981–2024 | −0.061 °C | 0.52 | No significant trend |
| Lahore, Pakistan | Precipitation | 1981–2024 | +0.153 mm/day | 0.016 | Significant wetter |

**The story the data tells:** the planet is warming (+0.21 °C/decade since
1981) and so is Reykjavik, at almost exactly the global rate, while Lahore's
rainfall has significantly increased. Same planet, different local stories:
that contrast is the whole point of the tool. The 2024 global anomaly was
+1.28 °C, the warmest year in the record.

### ⚠️ Lahore: the two datasets disagree — read this before presenting

The table above says Lahore shows *no* significant temperature trend — but
that is only the NASA POWER point series. The GISTEMP 2° cell containing
Lahore (centre 31°N, 75°E) **warms significantly over the same 1981–2024
window: +0.25 °C/decade, p ≈ 6×10⁻⁵** (Sen's slope, Mann-Kendall;
95% CI +0.14 to +0.36). An OLS fit gives +0.26, p ≈ 1×10⁻⁵ — the same
story. All eight neighbouring cells agree (+0.20 to +0.30
°C/decade), so the GISTEMP cell is not an outlier — and over the full
1880–2026 record it warms at +0.12 °C/decade, p ≈ 1×10⁻²⁷.

So the honest finding is a **dataset disagreement**, not "Lahore is not
warming": the station-based GISTEMP record says Lahore warms like the rest
of the region; the POWER reanalysis point series says it does not. The map
view (GISTEMP) and the point view (POWER) show different answers for the
same city, and that tension is itself the most interesting thing the tool
found. Do not present the POWER-only line without the GISTEMP half.

### Map results (Mann-Kendall, Hamed-Rao, Benjamini-Hochberg q ≤ 0.05)

| Map | Period | Cells tested | Significant | Warming / wetter | Cooling / drier |
|---|---|---|---|---|---|
| Temperature (GISTEMP, 2°) | 1880–2026 | 7,949 of 16,200 | 6,170 | 6,084 | 86 |
| Precipitation (GPCP, 2.5°) | 1979–2026 | 10,368 of 10,368 | 2,918 | 1,686 wetter | 1,232 drier |

Temperature cells are untested where the early record is too sparse
(< 10 years of data). The median significant warming cell warms at about
+0.17 °C/decade; the fastest cells (Arctic) approach +1 °C/decade.

### Is it speeding up? Decadal trends and the change-point

The same Mann-Kendall + Sen's-slope test, run per decade on the global
mean series (endpoint `/api/decades`):

| Decade | Change / decade | p | Verdict |
|---|---|---|---|
| 1980s | +0.10 °C | 0.53 | Not significant |
| 1990s | +0.14 °C | 0.28 | Not significant |
| 2000s | +0.16 °C | 0.07 | Not significant |
| 2010s | **+0.36 °C** | 0.012 | Significant |
| 2020s (partial, n = 6) | +0.68 °C | 0.13 | Not significant |

No single 10-year decade before the 2010s is significant on its own —
the signal only emerges over longer windows — but the slope climbs every
decade. A change-point scan (CUSUM on the residuals of a straight-line
fit, support judged by a 2,000-shuffle permutation test, p = 0.0005)
says **warming accelerated around the year 2000: +0.055 °C/decade before
(1880–1999), +0.26 °C/decade after (2000–2025) — about 4.8× faster**.
Honest caveat: the CUSUM peak marks where the series crosses its
single-line fit, not the exact year the slope changed, so "around 2000"
is the claim, not "in 2000". The acceleration itself (0.055 → 0.26) is
unambiguous. The same decades table is shown for any inspected cell:
Lahore's cell (31°N, 75°E) has no significant decade on its own
(+0.55, +0.29, +0.01, −0.03, +1.54 for 2020s, n = 7) — its warming
only shows over the full 1880–2026 record (+0.12 °C/decade).

### Seasons and the leaderboard

Season selector next to the variable selector repaints the map from
precomputed seasonal trend maps (DJF/MAM/JJA/SON, both variables,
`/api/map?season=`). Winter shows the Arctic amplification plainly: the
fastest-warming DJF cell is **+1.64 °C/decade at 81°N, 19°E, versus
+0.99 °C/decade annual at the same cell — winter warming ~1.7× the
annual rate** (1,054 significant Arctic cells in DJF). At that cell the
other seasons are MAM +0.75, JJA +0.78, SON +1.09 °C/decade.

The leaderboard (computed in the browser from the map payload,
BH-significant cells only, click a row to fly to the cell):

| Group | Top cell | Slope / decade |
|---|---|---|
| Fastest warming | 81°N, 21–25°E (Arctic) | +0.99 °C |
| Fastest cooling | 9°S, 53°W (S Atlantic) | −0.47 °C |
| Wetting most | 8.75°N, 73.75°W (Caribbean) | +0.46 mm/day |
| Drying most | 18.75°N, 156.25°W (C Pacific) | −0.79 mm/day |

New preset chips fly to Karachi, Islamabad, London, Tokyo, Sydney, the
Arctic (81°N, 19°E) and the Amazon (−3°, −60°). Sahara (23°N, 13°E)
has no chip: its GISTEMP cell has no data at all (n = 0), so it was
dropped rather than shown empty.

### The ice agrees (third variable: sea ice)

Arctic sea ice extent, NSIDC Sea Ice Index Version 4, 1979–2025
(endpoint `/api/seaice`):

| Series | 1979 | 2025 | Change / decade | p |
|---|---|---|---|---|
| September (minimum) | 7.05 M km² | 4.75 M km² | −0.74 M km² | ~6×10⁻¹² |
| March (maximum) | 16.34 M km² | 14.12 M km² | −0.38 M km² | ~2×10⁻¹² |
| Annual mean | 12.35 M km² | 10.15 M km² | −0.51 M km² | ~2×10⁻⁹ |

Record-low September: 3.57 M km² in 2012. The sea-ice loss is a third,
independent witness to the same warming — not a map, one line chart.

## 60-second demo script

1. Open the page. "This is Trend Detective — it answers what is changing,
   where, how much, and whether it is significant, using real NASA data."
2. Select **Global → Temperature**. "The planet has warmed 0.21 °C per
   decade since 1981 — highly significant, p is essentially zero."
3. Scroll to **"Is it speeding up?"** "And it is speeding up. In the
   1980s the planet warmed 0.10 °C per decade; in the 2010s, 0.36. Our
   change-point scan says warming accelerated around the year 2000 —
   from 0.05 °C a decade before, to 0.26 after. Nearly five times faster."
4. Switch region to **Lahore**. "Here's the honest complication: the POWER
   point series says Lahore shows no significant warming — minus 0.06 °C per
   decade, p = 0.52. But the GISTEMP map cell for Lahore says it *is*
   warming, +0.25 °C per decade, highly significant — and so do all eight
   neighbouring cells. Two real NASA datasets, two different answers for
   the same city."
5. Scroll to the **leaderboard**. "The fastest-warming place on Earth is
   the Arctic, 81° North — almost +1 °C per decade. Switch the season to
   Winter and it is +1.6. Click any row and you fly straight to that cell."
   Then the ice: "And the ice agrees — September Arctic sea ice has
   shrunk from 7.1 to 4.8 million square kilometres since 1979, losing
   about three-quarters of a million per decade."
6. New: **hotspots**. "The map can now pair the two variables. 421 cells
   are warming *and* drying — drought risk — the fastest is Saudi Arabia,
   24° North, warming 0.57 °C per decade while its rain falls. And 636
   cells are warming and getting wetter — flood risk — the wettest is the
   Caribbean, 9° North, at plus 0.46 millimetres a day per decade. Both
   signals are significant after false-discovery correction."
7. End on the **global trend map**. "The whole planet, one honest test per
   cell — 6,170 cells warming, only 86 cooling, after correcting for false
   discoveries. Click anywhere and you get its full series with the trend
   line and confidence band."

## Project layout

```
app.py            Flask app (routes + API: /api/series, /api/summary,
                  /api/map, /api/cell, /api/decades, /api/changepoint,
                  /api/seaice, /api/hotspots, /api/report)
src/
  data.py         loads the cached NASA datasets
  trends.py       OLS trend + p-value + plain-language verdict; record_years
                  (warmest/coldest or wettest/driest year of a series)
  fetch_data.py   re-downloads the datasets from NASA (--grid for NetCDF)
  grid.py         gridded loaders, seasons, cache build, trend maps;
                  find_hotspots (warming+drying / warming+wetting cells)
  placenames.py   coarse place names for hotspot cells (tiny lat/lon
                  box table; None when the cell is in no named region)
  changepoint.py  CUSUM change-point scan + permutation test
  seaice.py       NSIDC Sea Ice Index monthly CSVs -> extent cache
stats/
  mannkendall.py  Mann-Kendall + Sen's slope, Hamed-Rao/Yue-Wang
                  corrections, vectorised grid version
  fdr.py          Benjamini-Hochberg false-discovery control
templates/        index.html (Chart.js chart, selectors, summary table, map)
static/           style.css
data/             cached real downloads (POWER JSON x2, GISTEMP CSV)
data/grid/        committed caches: annual means + precomputed trend maps
                  (annual + seasonal DJF/MAM/JJA/SON, both variables)
data/raw/         raw NetCDF (git-ignored, re-downloadable)
tests/            pytest: trend maths + Flask routes (127 tests)
.github/workflows/ci.yml   pytest on every push
```

## AI disclosure

This starter was built with AI assistance (Muse). All data is real NASA data
from the sources above; no numbers were invented. NASA Space Apps requires
disclosure of AI tools used — this line is that disclosure.

## Still to do for the hackathon

- Record the demo video and submit the project for judging (submissions
  close Sunday 23:59 local time). The demo script above is written for the
  current build.
- Milestones 3–4 from the implementation plan are done (time controls:
  decades + season selector + change-point; more variables: leaderboard
  and sea ice; Milestone 3's year-window slider already existed via
  /api/map).
- Milestone 5 (contradictions/report export) is done: `/api/report`
  returns a plain-text findings report for a cell or the globe (trend +
  CI, 1981–2024 window, decade table, change point, record years, and the
  Lahore dataset-disagreement note only for the Lahore cell), and
  `/api/hotspots` lists the top warming+drying (drought risk) and
  warming+wetting (flood risk) cells, each with a coarse place name
  from `src/placenames.py` (Saudi Arabia, Caribbean Sea, South Atlantic
  Ocean, …; coordinates alone when the cell is in no named region).
- Fix round after a live visual check (2026-11-14): basemap switched to
  Esri World Light Gray Base — English labels, no API key (OpenStreetMap
  labels render in local scripts; CARTO light_all returns an "API KEY
  REQUIRED" watermark tile, verified). Non-significant cells paint as a
  translucent grey and no-data cells as transparent, so the basemap shows
  through instead of splotchy grey blocks and a white wash; the legend
  matches. The year slider caps at 2016 (a window needs ≥ 10 years of
  data), rapid slider changes are serialised (only the latest load may
  touch the UI) and a failed window restores slider, heading, map,
  leaderboard and stats together; the heading always describes the map
  that is actually loaded. The key-findings counters only animate after
  the first live fetch resolves, so a recording never captures
  intermediate numbers. Hotspot tables scroll inside their own card
  (grid items get `min-width: 0`) and never cause page-level horizontal
  scrolling. The "6,170 of 7,949" figure in About is labelled as the
  full record (1880–2026) so it can never contradict the active window.
  The overlay painting was verified gapless against the real grids
  (Python reproduction): the edge strip seen when zoomed was the solid
  grey blocks plus unloaded tile placeholders, not a painting seam.
- Second fix round after a second live visual check (2026-11-14): the
  translucent grey still read as solid blocks over the very light
  basemap, so non-significant cells now paint nothing at all — only
  significant cells are coloured, and the legend/intro say so ("Only
  cells with a significant trend are coloured; everywhere else you see
  the basemap"). Country labels and borders now come from Esri World
  Light Gray Reference tiles in a pane above the trend overlay (the
  Natural Earth GeoJSON borders layer is removed), and the overlay
  opacity is 1.0. Hotspot tables dropped the two q columns (p stays) so
  both tables fit at desktop width without an inner scrollbar. The
  leaderboard filters by sign — "Fastest cooling" can no longer list
  warming cells when a short window has fewer than 10 cooling cells
  (tested in node against the shipped JS, see tests/test_leaderboard_js.py).
  `/api/cell` accepts `start_year`/`end_year`: the selected-cell panel
  follows the active map window (trend, q and trend line over the window;
  too few years in the window falls back to the full record with a note).
  The key-findings count-up animation is deleted entirely — the numbers
  are static final values under the "Warming up the data…" cover, so a
  recording can never capture an intermediate value. The year-window
  label's initial text is "1880–2026" (it used to read "1880–…" until
  the script ran).
- Third fix round after a third live visual check (2026-11-14): the
  slider could stick in a broken state — after 1880 → 2016 → 1880 the
  heading stayed "Global trend map · Recomputing…" with no year
  window and the cell panel stayed on the 2016 slope, although the
  footer and leaderboard had updated. Root cause, reproduced in a
  node harness before fixing: `loadMap()` had no catch around fetch,
  so a rejected fetch (the check overlapped a Render deploy restart)
  escaped `applyYearWindow` as an unhandled rejection with
  `windowPending` stuck true and the slider logic dead; the same path
  also let the heading claim the new window while still recomputing.
  Fixed: `loadMap()` catches fetch/JSON errors and returns `'error'`
  (renders a "That window could not be computed" note and restores the
  previous window); the pending flag is cleared in one place that
  always runs, for the newest load, before the heading is updated;
  the heading derives only from the loaded map data; the cell refetch
  runs after every successful window change through
  `refetchCellQuietly()`, whose failure cannot break heading state.
  Regression test: `tests/test_slider_state.py` loads the shipped
  template JavaScript under mocked DOM/fetch and drives the exact
  live sequence — clean, mid-flight drag back, failing 2016 fetch,
  failing 1880 return, throwing paint — asserting heading, pending
  flag and cell-panel window after each step, and that no unhandled
  rejections occur.
- Colour and basemap (same round): the trend colours now follow a
  squared (power) scale — painted intensity is (|slope| / maxAbs)²,
  so weak warming (+0.1 °C/decade) reads near-white instead of the
  pale pink wash of the linear scale, and strong warming still
  carries deep red; the legend gradient uses the identical computed
  stops, so the legend stays honest, and exact numbers are always in
  the cell panel. The basemap is now Esri World Street Map (verified
  tile at z=2 returns HTTP 200, readable country labels at world
  zoom), replacing the Light Gray Base; the labels-on-top layer is
  Esri World Boundaries and Places (verified HTTP 200), replacing
  the Light Gray Reference, which was nearly empty at world zoom.
  One base, one label source, no double labels. Non-significant
  cells still paint nothing, so the basemap shows through.
- "Is it speeding up?" decades tables (same round): the slope
  cells no longer repeat the unit in every row and the headers are
  shortened ("Slope / dec", "p", "Sig."), so all five columns are
  visible at desktop width without an inner scrollbar. The
  leaderboard intro now says "up to 10 in each direction" — a short
  window can have fewer than ten qualifying cooling cells.
- Fix round 4 (2026-11-14, same evening): the live gate caught a
  regression the round-3 tests missed — every load showed "Could not
  draw the trend map (groups is not defined)" and every year-window
  change failed. Cause: the leaderboard extraction moved the group
  definitions into `pickLeaderboardGroups` but left the short-list
  computation in `renderLeaderboard` still referencing the old
  `groups` variable; the throw inside `loadMap` turned every window
  change into an 'error' + restore. Fixed by capturing the picked
  groups once and reusing them, and the slider harness now runs the
  REAL `renderLeaderboard` (it used to stub it out — that is why the
  bug slipped through) and asserts the leaderboard renders on load.
  The restore path is also fixed in two ways: it restores to the
  window that was actually loaded (tracked separately from the
  previously REQUESTED window — the old code once restored to an
  intermediate 1954 that never loaded, leaving the heading claiming
  1954–2026 while everything else showed the full record), and it
  distinguishes a fetch failure (map data untouched — just reset
  slider/labels/cell) from a paint failure after the new payload was
  assigned (refetch the previous window, then sync the cell panel).
  The slider label now shows "· loading…" while a window is
  pending, and a failure note left over from an earlier attempt
  clears when the slider returns to the current window. The
  "Is it speeding up?" section moved out of the side column to a
  full-width row (it left a blank left column while scrolling), and
  the About/footer basemap credits now name Esri World Street Map
  + World Boundaries and Places instead of the stale OpenStreetMap
  text. No data changed.
- Fix round 5 (2026-11-14, same evening): the gate passed every
  functional check but found one big visual defect — a full viewport
  of blank white under the map, because the "Inspect a cell" panel
  is much taller than the map column and the grid row stretches to
  the taller column. Fixed in CSS, not by moving sections around:
  on desktop (two-column layout only, above 1080px) the inspector
  is capped at the map column's height (45rem) and scrolls inside
  itself; below 1080px the layout is one column and the panel flows
  normally. The slider also applies on release ('change' event), so
  a drag that never produced 'input' events still moves the window.
  The decades chart drops its crowded "Year" axis title (the axis is
  decade categories, not a year scale). No data changed.
  Still open: Milestone 6 (final polish).
