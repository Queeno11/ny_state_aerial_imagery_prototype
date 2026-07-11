"""Opt-in NAIP coverage sweep: verify the dataset's NAIP queries are valid.

For every representative point across the US dataset, fetch a small NAIP crop and
record the outcome, so you can see exactly which (region, year) combinations lack
valid imagery. This mirrors what the training loader actually queries
(``fetch_naip`` at a real building centroid with ``year_hint`` = panel year).

Two modes (env ``NAIP_COVERAGE_MODE``):
  * ``per_cbsa`` (default) — one representative building per CBSA per year
    (~n_cbsa × 8 ≈ a few thousand queries; minutes).
  * ``per_tract``          — one representative building per tract per year
    (54,929 × 8 ≈ 440k queries; hours). The exhaustive check.

The sweep is concurrent (``fetch_naip`` is thread-safe: thread-local STAC client).
It writes a per-query CSV, prints a per-year / worst-region breakdown, AND emits
``data/processed/naip_unavailable.feather`` — the (region, year) gaps the ms_us
training/prediction pipeline auto-consumes (via ``build_dataset.load_income_dataset``)
to drop those pairs before any NAIP fetch, so the main model runs smoothly.
Only genuine coverage gaps (no_items/asset_missing) are written — transient API
errors are retried, never persisted. Runs union with any existing artifact, so a
coarse ``per_cbsa`` pass and a later ``per_tract`` pass accumulate.

Runs on the standard network opt-in (``per_cbsa`` by default = one per satellite
image / city-region per year)::

    python -m pytest src/tests/test_naip_coverage.py -s -q --run-network

The exhaustive per-tract check is opt-in via env::

    NAIP_COVERAGE_MODE=per_tract python -m pytest \
      src/tests/test_naip_coverage.py -s -q --run-network

Tunables (env): NAIP_COVERAGE_MODE (per_cbsa), NAIP_COVERAGE_WORKERS (8),
NAIP_COVERAGE_LIMIT (0 = no cap; total queries), NAIP_COVERAGE_MIN_OK (0.90),
NAIP_COVERAGE_MAX_READ_ERROR (0.05), NAIP_COVERAGE_RETRIES (4),
NAIP_COVERAGE_OUT_PIXELS (64).
"""
import csv
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from pyproj import Transformer

import src.build_dataset as bd
import src.geo_utils as geo_utils
from src.data import naip_fetcher as nf
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR

YEARS = list(range(2010, 2025, 2))
MODE = os.getenv("NAIP_COVERAGE_MODE", "per_cbsa")
WORKERS = int(os.getenv("NAIP_COVERAGE_WORKERS", "8"))
LIMIT = int(os.getenv("NAIP_COVERAGE_LIMIT", "0"))            # 0 = no cap
MIN_OK = float(os.getenv("NAIP_COVERAGE_MIN_OK", "0.90"))
MAX_READ_ERROR = float(os.getenv("NAIP_COVERAGE_MAX_READ_ERROR", "0.05"))
RETRIES = int(os.getenv("NAIP_COVERAGE_RETRIES", "4"))        # for transient errors
OUT_PIXELS = int(os.getenv("NAIP_COVERAGE_OUT_PIXELS", "64"))  # coverage needs
NBANDS = 3                                                     # existence, not res

# Genuine "no imagery here" outcomes vs. transient API/transport failures. Only
# the former are coverage gaps; the latter (e.g. a 502 from the STAC endpoint)
# are retried so they don't masquerade as missing coverage.
COVERAGE_GAP = {"no_items", "asset_missing"}
TRANSIENT = {"read_error", "unknown"}

pytestmark = [
    pytest.mark.network,   # skipped unless --run-network (see conftest.py)
    pytest.mark.skipif(
        not (PROCESSED_DATA_DIR / "buildings_index").exists(),
        reason="needs the real buildings_index (data/processed/buildings_index)",
    ),
]


def _representative_points(tmp_path, monkeypatch):
    """One representative building per region (CBSA or tract), with lon/lat.

    Builds the full slim buildings frame via the normalized helper but writes no
    artifacts (data/processed is read-only under the sandbox), then keeps the
    first building of each region — a stable, real query point from the dataset.
    """
    real = PROCESSED_DATA_DIR
    (tmp_path / "buildings_index").symlink_to(real / "buildings_index")
    for f in real.glob("us_metros_panel_*.feather"):
        (tmp_path / f.name).symlink_to(f)
    monkeypatch.setattr(bd, "PROCESSED_DATA_DIR", tmp_path)

    panel = bd.process_acs_panel()
    buildings = bd._build_buildings_frame(panel, states=None)

    key = "GEOID" if MODE == "per_tract" else "cbsa_code"
    reps = (
        buildings[["GEOID", "cbsa_code", "centroid_x", "centroid_y"]]
        .groupby(key, observed=True, sort=False).first().reset_index()
    )

    to4326 = Transformer.from_crs(geo_utils.METRIC_CRS, "EPSG:4326", always_xy=True)
    lon, lat = to4326.transform(reps["centroid_x"].to_numpy(),
                                reps["centroid_y"].to_numpy())
    reps["lon"], reps["lat"] = lon, lat
    reps["GEOID"] = reps["GEOID"].astype(str)
    reps["cbsa_code"] = reps["cbsa_code"].astype(str)
    return reps


def _open_csv():
    out_dir = RESULTS_DIR / "naip_coverage"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"coverage_{MODE}.csv"
        return path, open(path, "w", newline="")
    except OSError:
        # results/ is read-only in the sandbox — fall back to a temp file.
        import tempfile
        fd, name = tempfile.mkstemp(prefix=f"naip_coverage_{MODE}_", suffix=".csv")
        return name, os.fdopen(fd, "w", newline="")


def test_naip_coverage(tmp_path, monkeypatch):
    nf.reset_fetch_stats()
    reps = _representative_points(tmp_path, monkeypatch)

    queries = [
        (r.GEOID, r.cbsa_code, y, r.lon, r.lat)
        for r in reps.itertuples(index=False) for y in YEARS
    ]
    if LIMIT:
        queries = queries[:LIMIT]
    n_regions = reps[("cbsa_code" if MODE == "per_cbsa" else "GEOID")].nunique()
    crop_m = 2 * geo_utils.calculate_exact_tau(100, 224)[0]

    print(f"\n[NAIP coverage] mode={MODE} regions={n_regions:,} years={len(YEARS)} "
          f"queries={len(queries):,} workers={WORKERS} out_pixels={OUT_PIXELS}")

    def _q(item):
        geoid, cbsa, year, lon, lat = item
        # Retry transient (network/API) failures with backoff+jitter so a flaky
        # 502 is not misreported as a NAIP coverage gap. A deterministic
        # no_items/asset_missing is a real gap — return it immediately.
        for attempt in range(RETRIES + 1):
            res = nf.fetch_naip(lon, lat, crop_size_meters=crop_m, nbands=NBANDS,
                                out_pixels=OUT_PIXELS, year_hint=year)
            outcome = "ok" if res.crop is not None else (res.failure or "unknown")
            if outcome not in TRANSIENT or attempt == RETRIES:
                break
            time.sleep(min(8.0, 0.5 * 2 ** attempt) + random.uniform(0, 0.5))
        ok = res.crop is not None
        return {
            "geoid": geoid, "cbsa": cbsa, "year": year,
            "outcome": "ok" if ok else (res.failure or "unknown"),
            "actual_year": res.actual_year,
            "valid_shape": bool(ok and res.crop.shape == (NBANDS, OUT_PIXELS, OUT_PIXELS)),
            "nonblack": bool(ok and res.crop.max() > 0),
            "nir_padded": bool(res.nir_padded),
            "partial": bool(res.partial_coverage),
            "year_substituted": bool(ok and res.actual_year is not None
                                     and res.actual_year != year),
        }

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_q, it) for it in queries]
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % max(1, len(futures) // 20) == 0 or done == len(futures):
                ok_so_far = sum(r["outcome"] == "ok" for r in results)
                print(f"  {done:,}/{len(futures):,} "
                      f"({100*done/len(futures):.0f}%) ok={ok_so_far:,} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    # ── Persist per-query CSV ────────────────────────────────────────────────
    csv_path, fh = _open_csv()
    with fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ── Aggregate + report ───────────────────────────────────────────────────
    total = len(results)
    outcomes = Counter(r["outcome"] for r in results)
    ok = outcomes["ok"]
    ok_rate = ok / total
    gap = sum(v for k, v in outcomes.items() if k in COVERAGE_GAP)
    transient = sum(v for k, v in outcomes.items() if k in TRANSIENT)
    transient_rate = transient / total
    subst = sum(r["year_substituted"] for r in results)
    padded = sum(r["nir_padded"] for r in results)
    partial = sum(r["partial"] for r in results)

    per_year = defaultdict(lambda: [0, 0, 0])   # year -> [ok, gap, total]
    region_gaps = defaultdict(list)             # region -> [years with a TRUE gap]
    rkey = "cbsa" if MODE == "per_cbsa" else "geoid"
    for r in results:
        py = per_year[r["year"]]
        py[2] += 1
        if r["outcome"] == "ok":
            py[0] += 1
        elif r["outcome"] in COVERAGE_GAP:
            py[1] += 1
            region_gaps[r[rkey]].append(r["year"])

    print(f"\n{'='*66}\nNAIP COVERAGE REPORT ({MODE})\n{'='*66}")
    print(f"queries: {total:,} | valid: {ok:,} ({100*ok_rate:.2f}%) | "
          f"coverage gaps: {gap:,} | transient errors (after {RETRIES} retries): "
          f"{transient:,} | elapsed {time.time()-t0:.0f}s")
    print("outcomes: " + ", ".join(f"{k}={v:,}" for k, v in outcomes.most_common()))
    print(f"year substituted (actual != requested): {subst:,} "
          f"({100*subst/max(1,ok):.1f}% of valid) | nir_padded={padded:,} | partial={partial:,}")
    print("\nper-year (valid / coverage-gap / total):")
    for y in sorted(per_year):
        o, g, t = per_year[y]
        print(f"  {y}: {o:,} valid / {g:,} gap / {t:,}  ({100*o/t:.1f}% valid)")
    if region_gaps:
        worst = sorted(region_gaps.items(), key=lambda kv: -len(kv[1]))[:25]
        print(f"\nregions with genuine coverage gaps ({rkey}, top {len(worst)} of "
              f"{len(region_gaps):,}):")
        for reg, yrs in worst:
            print(f"  {reg}: no NAIP for {sorted(yrs)}")
    else:
        print("\nno genuine coverage gaps found 🎉")
    print(f"\nper-query CSV: {csv_path}")

    # ── Emit the exclusion artifact the ms_us pipeline auto-consumes ─────────
    # Only genuine coverage gaps (no_items/asset_missing) — never transient
    # errors — so the pipeline drops these (region, year) pairs before fetching.
    level = "tract" if MODE == "per_tract" else "cbsa"
    gaps = [{"level": level, "key": r[rkey], "year": r["year"]}
            for r in results if r["outcome"] in COVERAGE_GAP]
    if gaps and os.getenv("NAIP_COVERAGE_WRITE_EXCLUSIONS", "1") == "1":
        try:
            excl_path = bd.write_naip_unavailable(
                gaps, path=os.getenv("NAIP_COVERAGE_EXCLUDE_OUT"), merge=True)
            print(f"exclusion artifact ({len(gaps):,} rows, unioned): {excl_path}\n"
                  f"  → auto-masked by the ms_us pipeline on the next run.")
        except OSError as e:
            print(f"⚠️ could not write exclusion artifact ({e}); "
                  f"set NAIP_COVERAGE_EXCLUDE_OUT to a writable path.")
    print("=" * 66)

    # ── Assertions ───────────────────────────────────────────────────────────
    # 1. Query MECHANISM is healthy: transient API/transport failures must be
    #    rare AFTER retries (otherwise the coverage numbers can't be trusted).
    assert transient_rate <= MAX_READ_ERROR, (
        f"transient error rate {transient_rate:.2%} > {MAX_READ_ERROR:.2%} after "
        f"{RETRIES} retries — an API/rate-limit/network problem, not a coverage "
        f"gap. Lower NAIP_COVERAGE_WORKERS or retry later. See {csv_path}.")
    # 2. Every crop we DID get back is well-formed (right shape).
    bad_shape = [r for r in results if r["outcome"] == "ok" and not r["valid_shape"]]
    assert not bad_shape, f"{len(bad_shape)} valid crops had an unexpected shape"
    # 3. Coverage completeness (tunable): genuine region×year gaps drop this and
    #    are listed above + in the CSV.
    assert ok_rate >= MIN_OK, (
        f"valid-crop rate {ok_rate:.2%} < MIN_OK {MIN_OK:.2%}: {gap:,} genuine "
        f"coverage gaps + {transient:,} transient — inspect the per-year table "
        f"and {csv_path} (tune NAIP_COVERAGE_MIN_OK).")
