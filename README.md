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

Tests: `.venv/bin/python -m pytest tests/ -q` (51 tests, all passing).

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

## 60-second demo script

1. Open the page. "This is Trend Detective — it answers what is changing,
   where, how much, and whether it is significant, using real NASA data."
2. Select **Global → Temperature**. "The planet has warmed 0.21 °C per
   decade since 1981 — highly significant, p is essentially zero."
3. Switch region to **Reykjavik**. "Iceland warmed at almost exactly the
   global rate, +0.19 °C per decade."
4. Switch region to **Lahore**. "Here's the honest complication: the POWER
   point series says Lahore shows no significant warming — minus 0.06 °C per
   decade, p = 0.52. But the GISTEMP map cell for Lahore says it *is*
   warming, +0.25 °C per decade, highly significant — and so do all eight
   neighbouring cells. Two real NASA datasets, two different answers for the
   same city."
5. Switch variable to **Precipitation** (still Lahore). "What *is* changing
   in Lahore is the rain: significantly wetter, +0.15 mm per day per decade."
6. Scroll to the summary table, then the **global trend map**. "This is the
   whole planet, one honest test per cell — 6,170 cells warming, only 86
   cooling, after correcting for false discoveries. Click Lahore on the map
   and you get its full series with the trend line and confidence band."

## Project layout

```
app.py            Flask app (routes + API: /api/series, /api/summary,
                  /api/map, /api/cell)
src/
  data.py         loads the cached NASA datasets
  trends.py       OLS trend + p-value + plain-language verdict
  fetch_data.py   re-downloads the datasets from NASA (--grid for NetCDF)
  grid.py         gridded loaders, seasons, cache build, trend maps
stats/
  mannkendall.py  Mann-Kendall + Sen's slope, Hamed-Rao/Yue-Wang
                  corrections, vectorised grid version
  fdr.py          Benjamini-Hochberg false-discovery control
templates/        index.html (Chart.js chart, selectors, summary table, map)
static/           style.css
data/             cached real downloads (POWER JSON x2, GISTEMP CSV)
data/grid/        committed caches: annual means + precomputed trend maps
data/raw/         raw NetCDF (git-ignored, re-downloadable)
tests/            pytest: trend maths + Flask routes (51 tests)
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
- Milestones 3–6 from the implementation plan are not started (time
  controls, sea ice / sea level, contradictions, final polish).
