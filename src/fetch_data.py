"""Re-download the real NASA datasets into data/.

Run:  python -m src.fetch_data            (point data: POWER + GISTEMP CSV)
       python -m src.fetch_data --grid     (gridded NetCDF into data/raw/)
Requires internet. If a download fails the script reports it and exits
non-zero — it never writes fake or placeholder data.
"""

import argparse
import gzip
import json
import os
import shutil
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

# Gridded datasets for the trend maps (large NetCDF, kept in data/raw/).
GRID_JOBS = [
    ("GISTEMP gridded",
     "https://data.giss.nasa.gov/pub/gistemp/gistemp250_GHCNv4.nc.gz",
     "gistemp250_GHCNv4.nc.gz"),
    ("GPCP v2.3",
     "https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc",
     "precip.mon.mean.nc"),
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-download real NASA datasets")
    parser.add_argument("--grid", action="store_true",
                        help="download the gridded NetCDF files into data/raw/")
    args = parser.parse_args(argv)
    if args.grid:
        return run_grid_jobs()
    return run_jobs(JOBS, DATA_DIR)


def run_jobs(jobs, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    failed = []
    for name, url, fname in jobs:
        dest = os.path.join(dest_dir, fname)
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


def run_grid_jobs():
    raw_dir = os.path.join(DATA_DIR, "raw")
    rc = run_jobs(GRID_JOBS, raw_dir)
    if rc:
        return rc
    # Decompress the GISTEMP archive so src.grid can stream it.
    gz = os.path.join(raw_dir, "gistemp250_GHCNv4.nc.gz")
    nc = os.path.join(raw_dir, "gistemp250_GHCNv4.nc")
    if os.path.exists(gz) and not os.path.exists(nc):
        with gzip.open(gz, "rb") as f_in, open(nc, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"OK   decompressed -> gistemp250_GHCNv4.nc "
              f"({os.path.getsize(nc)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
