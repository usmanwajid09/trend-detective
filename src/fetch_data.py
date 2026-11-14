"""Re-download the real NASA datasets into data/.

Run:  python -m src.fetch_data
Requires internet. If a download fails the script reports it and exits
non-zero — it never writes fake or placeholder data.
"""

import json
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")

POWER_BASE = (
    "https://power.larc.nasa.gov/api/temporal/monthly/point"
    "?parameters=T2M,PRECTOTCORR&community=RE&start=1981&end=2024&format=JSON"
)
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

JOBS = [
    ("Lahore POWER", POWER_BASE + "&latitude=31.52&longitude=74.36",
     "power_lahore.json"),
    ("Reykjavik POWER", POWER_BASE + "&latitude=64.15&longitude=-21.94",
     "power_reykjavik.json"),
    ("GISTEMP global", GISTEMP_URL, "gistemp_global.csv"),
]


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "nasa-trend-detective/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        data = resp.read()
    if len(data) < 1000:
        raise RuntimeError(f"suspiciously small download ({len(data)} bytes) for {url}")
    with open(dest, "wb") as fh:
        fh.write(data)
    return len(data)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    failed = []
    for name, url, fname in JOBS:
        dest = os.path.join(DATA_DIR, fname)
        try:
            size = fetch(url, dest)
            print(f"OK   {name}: {size} bytes -> {fname}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL {name}: {exc}", file=sys.stderr)
            failed.append(name)
    if failed:
        print(f"{len(failed)} download(s) failed: {', '.join(failed)}",
              file=sys.stderr)
        return 1
    print("All datasets downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
