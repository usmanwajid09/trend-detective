# Trend Detective — NASA Space Apps 2026

**Challenge:** "Be An Earth System Trend Detective!" — build a tool that picks
variables measured by NASA missions or produced by NASA models, visualises how
they change over time, and answers: what is changing, where, how much, and
whether the change is significant.

**What it is:** a small Flask app. Pick a region (Lahore, Reykjavik, or Global)
and a variable (temperature or precipitation) and it draws the time series from
real NASA data, fits a trend, and gives a plain-language verdict with the
numbers — slope per decade and p-value — next to it. A summary table answers
"what changed, where, how much" for every region/variable at once.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
# open http://127.0.0.1:5000
```

Tests: `.venv/bin/python -m pytest tests/ -q` (12 tests, all passing).

To re-download the data: `.venv/bin/python -m src.fetch_data`
(it only writes real downloads — a failed download is reported, never faked).

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
1981) and so is Reykjavik, at almost exactly the global rate — but Lahore
shows *no* significant temperature trend over the same 44 years, while its
rainfall has significantly increased. Same planet, different local stories:
that contrast is the whole point of the tool. The 2024 global anomaly was
+1.28 °C, the warmest year in the record.

## 60-second demo script

1. Open the page. "This is Trend Detective — it answers what is changing,
   where, how much, and whether it is significant, using real NASA data."
2. Select **Global → Temperature**. "The planet has warmed 0.21 °C per
   decade since 1981 — highly significant, p is essentially zero."
3. Switch region to **Reykjavik**. "Iceland warmed at almost exactly the
   global rate, +0.19 °C per decade."
4. Switch region to **Lahore**. "But Lahore shows no significant warming at
   all — minus 0.06 °C per decade, p = 0.52. Same planet, different story."
5. Switch variable to **Precipitation** (still Lahore). "What *is* changing
   in Lahore is the rain: significantly wetter, +0.15 mm per day per decade."
6. Scroll to the summary table. "Every finding, with its numbers, in one
   place — nothing here is invented, it all comes from NASA POWER and
   GISTEMP."

## Project layout

```
app.py            Flask app (routes + API)
src/
  data.py         loads the cached NASA datasets
  trends.py       OLS trend + p-value + plain-language verdict
  fetch_data.py   re-downloads the datasets from NASA
templates/        index.html (Chart.js chart, selectors, summary table)
static/           style.css
data/             cached real downloads (POWER JSON x2, GISTEMP CSV)
tests/            pytest: trend maths + Flask routes
```

## AI disclosure

This starter was built with AI assistance (Muse). All data is real NASA data
from the sources above; no numbers were invented. NASA Space Apps requires
disclosure of AI tools used — this line is that disclosure.

## Still to do for the hackathon

- Pick this challenge (Step 05) on the Space Apps site and form/join a team
  (Team Finder was not live at build time).
- Optional stretch if time allows: add a third region, monthly (seasonal)
  views, or a map view. Keep the numbers honest — every new series needs a
  real source and a significance test.
