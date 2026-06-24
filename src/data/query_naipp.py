#!/usr/bin/env python3
"""
Cache shard builder — thread pool stress test for the rolling refresh pipeline.
Fetches 2,400 NAIP crops in parallel and reports whether the result fits
inside a 10-minute epoch window.

Setup:
    pip install pystac-client planetary-computer rasterio numpy tqdm

Usage:
    python build_cache_shard.py              # 8 workers (default)
    python build_cache_shard.py --workers 4  # try other counts
"""

import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
N_CROPS       = 2_400
EPOCH_MIN     = 10.0       # your measured epoch duration
CACHE_SIZE    = 8_192
REFRESH_RATE  = 0.30
FULL_REFRESH  = int(CACHE_SIZE * REFRESH_RATE)   # 2,458

NYC_BBOX = [-74.05, 40.60, -73.70, 40.88]
LAT_DEG  = 250 / 111_000
LON_DEG  = 250 / (111_000 * 0.756)

OUTPUT = Path("naip_cache")
OUTPUT.mkdir(exist_ok=True)

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="2",
)


# ── Per-task result ───────────────────────────────────────────────────────────

@dataclass
class Result:
    success: bool
    stac_s:  float = 0.0
    cog_s:   float = 0.0
    error:   str   = ""


# ── Thread-local STAC client ──────────────────────────────────────────────────
# One client per thread avoids shared connection-pool contention.

_local = threading.local()

def get_catalog() -> pystac_client.Client:
    if not hasattr(_local, "catalog"):
        _local.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    return _local.catalog


# ── Worker ────────────────────────────────────────────────────────────────────

def fetch_one(idx: int) -> Result:
    lon = random.uniform(NYC_BBOX[0], NYC_BBOX[2] - LON_DEG)
    lat = random.uniform(NYC_BBOX[1], NYC_BBOX[3] - LAT_DEG)
    bbox = [lon, lat, lon + LON_DEG, lat + LAT_DEG]

    # ── STAC query ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        items = list(get_catalog().search(
            collections=["naip"],
            bbox=bbox,
            max_items=20,
        ).items())
    except Exception as e:
        return Result(success=False, error=f"stac:{e}")
    stac_s = time.perf_counter() - t0

    if not items:
        return Result(success=False, stac_s=stac_s, error="no_coverage")

    item  = random.choice(items)
    asset = item.assets.get("image") or item.assets.get("visual")
    if not asset:
        return Result(success=False, stac_s=stac_s, error="no_asset")

    # ── COG windowed read ─────────────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(asset.href) as src:
                native_bb = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
                window    = from_bounds(*native_bb, transform=src.transform)
                crop = src.read(
                    window=window,
                    out_shape=(src.count, 250, 250),
                    resampling=Resampling.bilinear,
                )
    except Exception as e:
        return Result(success=False, stac_s=stac_s, error=f"cog:{e}")
    cog_s = time.perf_counter() - t1

    np.save(OUTPUT / f"shard_{idx:05d}_{item.datetime:%Y%m%d}.npy", crop)
    return Result(success=True, stac_s=stac_s, cog_s=cog_s)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[Result], wall_s: float, n_workers: int):
    ok     = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not ok:
        print("All fetches failed — check connection or PC token.")
        return

    stac_times = [r.stac_s for r in ok]
    cog_times  = [r.cog_s  for r in ok]
    total_per  = [r.stac_s + r.cog_s for r in ok]

    def stats(vals):
        vals = sorted(vals)
        n = len(vals)
        return dict(
            avg = sum(vals) / n,
            p50 = vals[n // 2],
            p95 = vals[int(n * 0.95)],
            p99 = vals[int(n * 0.99)],
        )

    st = stats(stac_times)
    ct = stats(cog_times)
    tt = stats(total_per)

    crops_per_min   = len(ok) / (wall_s / 60)
    full_refresh_min = FULL_REFRESH / crops_per_min
    margin_min      = EPOCH_MIN - full_refresh_min

    # Effective parallelism: if threads overlapped perfectly,
    # wall_time = sequential_time / n_workers.
    # effective = sequential_time / wall_time  (ideally == n_workers)
    seq_equiv   = sum(total_per)
    effective_p = seq_equiv / wall_s

    print(f"\n── Shard build results  (workers={n_workers}) {'─'*35}")
    print(f"  Crops fetched : {len(ok):>5} / {N_CROPS}  ({100*len(ok)/N_CROPS:.1f}% success)")
    print(f"  Failed        : {len(failed):>5}")
    print(f"  Wall time     : {wall_s/60:.2f} min  ({wall_s:.1f}s)")
    print(f"  Throughput    : {crops_per_min:.1f} crops/min")
    print()
    print(f"  {'':20}  {'avg':>7}  {'p50':>7}  {'p95':>7}  {'p99':>7}")
    print(f"  {'STAC query':20}  {st['avg']:>6.2f}s  {st['p50']:>6.2f}s"
          f"  {st['p95']:>6.2f}s  {st['p99']:>6.2f}s")
    print(f"  {'COG read':20}  {ct['avg']:>6.2f}s  {ct['p50']:>6.2f}s"
          f"  {ct['p95']:>6.2f}s  {ct['p99']:>6.2f}s")
    print(f"  {'total per crop':20}  {tt['avg']:>6.2f}s  {tt['p50']:>6.2f}s"
          f"  {tt['p95']:>6.2f}s  {tt['p99']:>6.2f}s")
    print()
    print(f"  Effective parallelism : {effective_p:.1f}×  (ideal = {n_workers}×)")
    print()
    print(f"  ── Epoch fit check ─────────────────────────────────────────────")
    print(f"  Epoch duration        : {EPOCH_MIN:.0f} min")
    print(f"  Full 30% refresh      : {FULL_REFRESH} crops  →  {full_refresh_min:.1f} min")
    if margin_min >= 0:
        print(f"  Verdict               : ✓  fits  ({margin_min:.1f} min to spare)")
    else:
        print(f"  Verdict               : ✗  overruns by {-margin_min:.1f} min")
        needed = FULL_REFRESH / EPOCH_MIN
        print(f"  Need                  : ≥ {needed:.0f} crops/min  "
              f"→ try --workers {int(n_workers * (-margin_min / full_refresh_min)) + n_workers + 1}")
    print("─" * 62)

    if failed:
        err_counts: dict[str, int] = {}
        for r in failed:
            err_counts[r.error] = err_counts.get(r.error, 0) + 1
        print(f"\n  Errors: {dict(sorted(err_counts.items(), key=lambda x: -x[1]))}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NAIP cache shard builder")
    parser.add_argument("--workers", type=int, default=8,
                        help="Thread pool size (default: 8)")
    args = parser.parse_args()

    print(f"\nBuilding cache shard")
    print(f"  crops={N_CROPS}  workers={args.workers}  output=./{OUTPUT}/")
    print(f"  (epoch={EPOCH_MIN:.0f} min, cache={CACHE_SIZE}, refresh={REFRESH_RATE:.0%})\n")

    results: list[Result] = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, i): i for i in range(N_CROPS)}
        with tqdm(total=N_CROPS, unit="crop", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                ok = sum(1 for x in results if x.success)
                bar.set_postfix(ok=ok, fail=len(results) - ok, refresh=False)
                bar.update(1)

    wall_s = time.perf_counter() - t_start
    print_summary(results, wall_s, args.workers)


if __name__ == "__main__":
    main()