"""Throughput probe for municipal ortho services — the (a)/(b) decision for issue #36.

Answers one question with numbers instead of guesses: can we fetch a city's local
orthoimagery fast enough to predict *online* (fetch and infer in one pass), or
must crops be extracted to disk first?

Why the arithmetic already leans one way
----------------------------------------
Chicago at 200 buildings/tract x ~800 city tracts x 15 annual years is ~2.4M
crops. Stored as uint8 that is ~480 GB raw (224*224*4 B each), ~180 GB
PNG-compressed — against ~194 GB free on the WSL cache disk, with the only
roomier volume being NTFS-over-WSL, which CLAUDE.md already flags as slow for
many small writes. Meanwhile the one advantage offline extraction offered —
surviving a crash — ``prediction.predict_year_chunked`` already provides through
its manifest-guarded per-chunk parquets.

So the real unknown is not storage but *rate*: Planetary Computer is hyperscale
infrastructure that absorbed 64 concurrent workers with zero throttling, whereas
``gis.cookcountyil.gov`` is county infrastructure. If a polite rate cap puts us
far below the NAIP path's measured ~34 crops/s, Chicago becomes a multi-day fetch
and decoupling it from the GPU becomes attractive again. This probe measures
that, and reports the implied wall clock either way.

Politeness
----------
Escalation stops the moment throttling appears — the point is to find the polite
ceiling, not to discover the breaking point of a public service. Levels are tried
from the gentlest upward, each level is short, and any 429/503 ends the sweep.
Nothing here writes to disk or mutates production state.

Usage
-----
    python src/probe_ortho_throughput.py --city chicago --year 2023
    python src/probe_ortho_throughput.py --city chicago --workers 1,2,4,8 \
        --max-rps 5,10,20 --n 60
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from src.data import ortho_fetcher as of

# Crop geometry must match production or the numbers do not transfer:
# tau=100 m -> a 200 m window, rendered at image_size=224.
CROP_SIZE_M = 200.0
OUT_PIXELS = 224

# Bytes per crop as the model consumes it (uint8, 4 bands). Used only for the
# offline-storage arithmetic in the verdict.
BYTES_PER_CROP = OUT_PIXELS * OUT_PIXELS * 4


@dataclass
class LevelResult:
    workers: int
    max_rps: float
    n_ok: int
    n_fail: int
    n_throttled: int
    elapsed_s: float
    p50_ms: float
    p95_ms: float

    @property
    def throughput(self) -> float:
        return self.n_ok / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def line(self) -> str:
        rps = "unlimited" if self.max_rps <= 0 else f"{self.max_rps:g}"
        return (f"  workers={self.workers:<3} max_rps={rps:<9} "
                f"ok={self.n_ok:<4} fail={self.n_fail:<3} "
                f"throttled={self.n_throttled:<3} "
                f"{self.throughput:6.1f} crops/s   "
                f"p50={self.p50_ms:6.0f}ms p95={self.p95_ms:6.0f}ms")


def sample_points(city: str, n: int, *, seed: int = 825
                  ) -> list[tuple[float, float]]:
    """Real building centroids from the national index, as lon/lat.

    Real coordinates, not a synthetic grid: throughput depends on how many
    distinct source rasters the window touches, and a uniform grid over a
    bounding box would sample water, forest preserve and airfield in
    proportions no real building sample has.
    """
    import geopandas as gpd
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    from pyproj import Transformer

    from src.csa_event_study import CSA_CITIES
    from src.utils.paths import PROCESSED_DATA_DIR

    spec = CSA_CITIES[city]
    dset = ds.dataset(PROCESSED_DATA_DIR / "buildings_index" / f"state={spec.state}"
                      / "part.parquet")
    expr = None
    for prefix in spec.geoid_prefixes:
        e = pc.starts_with(ds.field("tract_id"), str(prefix))
        expr = e if expr is None else (expr | e)
    table = dset.to_table(columns=["cx", "cy", "tract_id"], filter=expr)
    df = table.to_pandas()
    if df.empty:
        raise RuntimeError(f"no buildings found for {city} in the index")

    df = df.sample(n=min(n, len(df)), random_state=seed)
    to_4326 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lons, lats = to_4326.transform(df["cx"].to_numpy(), df["cy"].to_numpy())
    if not np.isfinite(lons).all():
        raise RuntimeError(
            "projecting index centroids to EPSG:4326 gave non-finite values; "
            "if PROJ_NETWORK=ON and cdn.proj.org is unreachable, set "
            "PROJ_NETWORK=OFF and retry.")
    return list(zip(lons.tolist(), lats.tolist()))


def run_level(points, city: str, year: int, workers: int, max_rps: float
              ) -> LevelResult:
    """One (workers, max_rps) cell of the sweep."""
    of.reset_fetch_stats()
    of.set_max_rps(max_rps)
    latencies: list[float] = []
    n_ok = n_fail = 0

    def _one(pt):
        lon, lat = pt
        t0 = time.perf_counter()
        res = of.fetch_ortho(lon, lat, CROP_SIZE_M, nbands=4,
                             out_pixels=OUT_PIXELS, year_hint=year, city=city,
                             max_rps=max_rps)
        return res, (time.perf_counter() - t0) * 1000.0

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res, ms in pool.map(_one, points):
            latencies.append(ms)
            if res.crop is not None:
                n_ok += 1
            else:
                n_fail += 1
    elapsed = time.perf_counter() - t_start

    stats = of.get_fetch_stats()
    latencies.sort()
    return LevelResult(
        workers=workers, max_rps=max_rps, n_ok=n_ok, n_fail=n_fail,
        n_throttled=int(stats["throttled"]), elapsed_s=elapsed,
        p50_ms=statistics.median(latencies) if latencies else float("nan"),
        p95_ms=latencies[int(0.95 * (len(latencies) - 1))] if latencies else float("nan"),
    )


def verdict(results: list[LevelResult], *, n_crops_total: int) -> None:
    """Print the recommendation and the (a)-vs-(b) arithmetic."""
    usable = [r for r in results if r.n_ok and not r.n_throttled]
    print("\n" + "=" * 78)
    if not usable:
        print("VERDICT: every level either failed or hit throttling.")
        print("  Do not escalate. Check the service is up and reachable, then")
        print("  re-run at the gentlest level with a single worker.")
        return

    best = max(usable, key=lambda r: r.throughput)
    rps = "unlimited" if best.max_rps <= 0 else f"{best.max_rps:g}"
    print(f"Best untrottled level: workers={best.workers}, max_rps={rps} "
          f"-> {best.throughput:.1f} crops/s")

    hours = n_crops_total / best.throughput / 3600 if best.throughput else float("inf")
    gib = n_crops_total * BYTES_PER_CROP / (1024 ** 3)
    print(f"\nFor a {n_crops_total:,}-crop city pass:")
    print(f"  (b) online  : ~{hours:5.1f} h single process "
          f"(resumable per chunk, no bulk storage)")
    print(f"  (a) offline : same fetch time PLUS ~{gib:,.0f} GiB of uint8 crops "
          f"(~{gib * 0.38:,.0f} GiB PNG-compressed)")

    print("\nRecommendation:", end=" ")
    if hours <= 24:
        print("(b) ONLINE — fetch and predict in one pass.")
        print("  predict_year_chunked already resumes per chunk, so the only")
        print("  thing offline extraction would add is the storage bill above.")
    else:
        print("(b) online is still preferred on storage, but at "
              f"~{hours:.0f} h it is a multi-day run.")
        print("  Shard across processes (csa_predict --shard i/n), and DIVIDE")
        print("  max_rps by the shard count — the limiter is per-process, so")
        print("  N shards otherwise multiply the load the service sees by N.")
    budget = best.max_rps if best.max_rps > 0 else best.throughput
    print(f"\nTotal request budget to stay within: ~{budget:g} req/s across ALL "
          f"processes.")
    for n in (1, 2, 4):
        print(f"    --shard i/{n}  ->  --max-rps {budget / n:g} per process")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--city", default="chicago", choices=sorted(of.ORTHO_SOURCES))
    ap.add_argument("--year", type=int, default=None,
                    help="panel year to probe (default: latest available)")
    ap.add_argument("--n", type=int, default=48,
                    help="crops per level (keep small; this hits a public service)")
    ap.add_argument("--workers", default="1,2,4,8",
                    help="comma-separated concurrency levels, gentlest first")
    ap.add_argument("--max-rps", default="5,10,20",
                    help="comma-separated rate caps; 0 = unlimited")
    ap.add_argument("--total-crops", type=int, default=2_400_000,
                    help="crop count to extrapolate the verdict to")
    args = ap.parse_args()

    years = of.available_years(args.city)
    if not years:
        raise SystemExit(
            f"{args.city} has no local ortho service registered "
            f"(see ortho_fetcher.ORTHO_SOURCES) — it runs on NAIP.")
    year = args.year or max(years)
    if year not in years:
        raise SystemExit(f"{args.city} has no ortho for {year}; available: {years}")

    workers = [int(w) for w in args.workers.split(",")]
    rates = [float(r) for r in args.max_rps.split(",")]

    print(f"Probing {args.city} {year} — {args.n} crops per level, "
          f"{CROP_SIZE_M:g} m window at {OUT_PIXELS}px "
          f"({CROP_SIZE_M / OUT_PIXELS:.2f} m/px, production geometry)")
    source = of.ORTHO_SOURCES[args.city][year]
    print(f"  service: {source.export_url}")

    print("\nBand-order check (band 4 must be NIR)...")
    try:
        ok, ndvi = of.check_band_order(args.city, year)
        print(f"  median NDVI over vegetation = {ndvi:.3f} -> "
              f"{'OK' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit(
                "  band order is wrong; predictions from this service would be "
                "silently corrupt. Fix the band mapping before probing further.")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  could not run the band-order check: {exc}")

    points = sample_points(args.city, args.n)
    print(f"\nSampled {len(points)} real building centroids.\n")

    results: list[LevelResult] = []
    stop = False
    for rate in rates:
        for w in workers:
            r = run_level(points, args.city, year, w, rate)
            results.append(r)
            print(r.line())
            if r.n_throttled:
                print("    ⚠️ throttling detected — stopping escalation here. "
                      "This is a municipal service; back off rather than push.")
                stop = True
                break
        if stop:
            break

    verdict(results, n_crops_total=args.total_crops)


if __name__ == "__main__":
    main()
