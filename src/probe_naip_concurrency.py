"""Concurrency probe for Planetary Computer NAIP access.

Purpose
-------
Prediction is fetch-bound: each chunk spends almost all its wall time pulling
NAIP crops from Planetary Computer (STAC search + windowed COG reads), while the
GPU sits idle. If the server tolerates more concurrent requests than we
currently issue, throughput scales ~linearly with worker count until we hit the
server's rate limit -- so the operating question is: *how many workers can we
run before Planetary Computer starts throttling (HTTP 429/503/SlowDown) or
latency blows up?*

This script answers that empirically. It ramps the worker count over a sweep
and, at each level, measures throughput, success rate, latency percentiles, and
throttle-response counts. It probes the two independently-throttled subsystems
separately, because they have different limits and the fix differs:

* ``reads``  -- concurrent windowed COG reads from Azure blob (the production
  bottleneck: search is cached per-tract, so steady-state load is almost all
  reads). Items are resolved once up front; the sweep hammers only open+read.
* ``search`` -- concurrent STAC ``/search`` calls against the PC STAC API (the
  first thing to rate-limit a cold run, before the per-tract cache warms).

It reuses the real :mod:`src.data.naip_fetcher` primitives, so a worker count
that is safe here is safe for ``predict_fetch_workers`` in ``src/main.py``. The
fetch functions are injectable so the sweep/summary/recommendation logic is unit
tested offline without touching the network.

Run (on the box that does the real fetching, e.g. the PC Hub)::

    python src/probe_naip_concurrency.py --mode both --levels 1,2,4,8,16,32,48,64

Nothing here writes to disk or mutates production state; it only issues reads.
It is deliberately *not* a pytest module (it hits the live network); the offline
logic tests live in ``src/tests/test_probe_naip_concurrency.py``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import ceil, cos, floor, pi, radians, sin, sqrt

from src.data import naip_fetcher

# ---------------------------------------------------------------------------
# Throttle classification
# ---------------------------------------------------------------------------
# Substrings that mark a *server-side rate-limit / overload* response, as
# opposed to a data problem (no coverage) or a client bug. GDAL surfaces blob
# throttling inside a RasterioIOError message; pystac raises requests errors
# whose text carries the HTTP status. We match on lower-cased text so both
# paths are covered. "403"/"401" are intentionally excluded: they mean auth /
# expired SAS, not throttling, and would falsely inflate the throttle count.
THROTTLE_TOKENS = (
    "429", "too many requests", "throttl", "slowdown", "slow down",
    "503", "server busy", "quota", "rate limit", "rate-limit",
)


def classify_throttle(message: str) -> bool:
    """True iff ``message`` looks like a server throttle / overload response."""
    if not message:
        return False
    low = message.lower()
    return any(tok in low for tok in THROTTLE_TOKENS)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class AttemptResult:
    """Outcome of one probe request (one search or one open+read)."""
    ok: bool
    latency: float          # seconds; 0.0 for attempts that never hit the net
    mode: str               # "ok" | failure-mode | short exception repr
    throttled: bool


@dataclass
class LevelResult:
    """Aggregate of one concurrency level (all attempts at ``workers``)."""
    workers: int
    total: int
    ok: int
    throttled: int
    wall_s: float
    p50_ms: float
    p95_ms: float
    fail_modes: Counter = field(default_factory=Counter)

    @property
    def fail(self) -> int:
        return self.total - self.ok

    @property
    def throughput(self) -> float:
        """Successful requests per second (0 if the level did no wall time)."""
        return self.ok / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def throttle_frac(self) -> float:
        return self.throttled / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Percentile / summary helpers (pure)
# ---------------------------------------------------------------------------
def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated ``q`` quantile (q in [0,1]) of ``values`` (or 0.0)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo, hi = floor(k), ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[AttemptResult], workers: int,
              wall_s: float) -> LevelResult:
    """Fold per-attempt results into one :class:`LevelResult`."""
    latencies = [r.latency for r in results if r.latency > 0]
    ok = sum(1 for r in results if r.ok)
    throttled = sum(1 for r in results if r.throttled)
    modes = Counter(r.mode for r in results if not r.ok)
    return LevelResult(
        workers=workers,
        total=len(results),
        ok=ok,
        throttled=throttled,
        wall_s=wall_s,
        p50_ms=_percentile(latencies, 0.50) * 1000.0,
        p95_ms=_percentile(latencies, 0.95) * 1000.0,
        fail_modes=modes,
    )


# ---------------------------------------------------------------------------
# Attempt functions (bind real primitives; injectable for tests)
# ---------------------------------------------------------------------------
def make_read_attempt(nbands: int, out_pixels: int,
                      open_fn=None, read_fn=None):
    """Return ``fn(ref) -> AttemptResult`` that opens a DOQQ and reads one crop.

    Mirrors the production read path (:func:`naip_fetcher.open_naip_src` +
    :func:`read_naip_crop_from_src`) exactly, but catches and *classifies* the
    exception text so blob throttling (HTTP 429/503/SlowDown raised by GDAL as a
    RasterioIOError) is counted rather than swallowed.
    """
    open_fn = open_fn or naip_fetcher.open_naip_src
    read_fn = read_fn or naip_fetcher.read_naip_crop_from_src

    def attempt(ref) -> AttemptResult:
        if getattr(ref, "failure", None) is not None:
            return AttemptResult(False, 0.0, ref.failure, False)
        t0 = time.perf_counter()
        try:
            with open_fn(ref.href, ref.use_cache) as src:
                crop, _nir, _partial, failure = read_fn(
                    src, ref.bbox, nbands, out_pixels)
            dt = time.perf_counter() - t0
            if failure is None:
                return AttemptResult(True, dt, "ok", False)
            return AttemptResult(False, dt, failure, False)
        except Exception as exc:  # open itself failed (throttle shows here)
            dt = time.perf_counter() - t0
            msg = f"{type(exc).__name__}: {exc}"
            return AttemptResult(False, dt, msg[:160], classify_throttle(msg))

    return attempt


def make_search_attempt(crop_size_meters: float, max_items: int = 50,
                        search_fn=None):
    """Return ``fn((lon, lat)) -> AttemptResult`` that runs one STAC search."""
    search_fn = search_fn or naip_fetcher._search_items

    def attempt(point) -> AttemptResult:
        lon, lat = point
        bbox = naip_fetcher.lonlat_bbox(lon, lat, crop_size_meters)
        t0 = time.perf_counter()
        try:
            items = search_fn(bbox, max_items=max_items, signed=True)
            dt = time.perf_counter() - t0
            if items:
                return AttemptResult(True, dt, "ok", False)
            return AttemptResult(False, dt, "no_items", False)
        except Exception as exc:
            dt = time.perf_counter() - t0
            msg = f"{type(exc).__name__}: {exc}"
            return AttemptResult(False, dt, msg[:160], classify_throttle(msg))

    return attempt


# ---------------------------------------------------------------------------
# Sweep engine (pure given an attempt_fn)
# ---------------------------------------------------------------------------
def run_level(work_items: list, workers: int, attempt_fn) -> LevelResult:
    """Run every item in ``work_items`` through ``attempt_fn`` at ``workers``."""
    results: list[AttemptResult] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(attempt_fn, item) for item in work_items]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall = time.perf_counter() - t0
    return summarize(results, workers, wall)


def sweep(levels: list[int], work_items: list, attempt_fn,
          early_stop_frac: float = 0.25, on_level=None) -> list[LevelResult]:
    """Run ``attempt_fn`` over ``work_items`` at each worker count in ``levels``.

    Stops early once a level's throttle fraction reaches ``early_stop_frac`` --
    once the server is refusing a quarter of requests, pushing harder only earns
    a ban, so there is no point ramping further. ``on_level`` (if given) is
    called with each :class:`LevelResult` as it completes (for live printing).
    """
    out: list[LevelResult] = []
    for workers in levels:
        res = run_level(list(work_items), workers, attempt_fn)
        out.append(res)
        if on_level is not None:
            on_level(res)
        if res.throttle_frac >= early_stop_frac:
            break
    return out


def recommend(results: list[LevelResult],
              throttle_ceiling: float = 0.05,
              knee_frac: float = 0.9) -> str:
    """One-line worker-count recommendation from a completed sweep.

    Considers only "clean" levels (throttle fraction under ``throttle_ceiling``
    and at least one success), finds the peak throughput among them, then picks
    the *knee*: the fewest workers that still reach ``knee_frac`` of that peak.
    Fewer workers at the same throughput is strictly better (less server load,
    smaller blast radius if PC tightens limits).
    """
    clean = [r for r in results
             if r.total and r.throttle_frac < throttle_ceiling and r.ok > 0]
    if not clean:
        return ("No clean concurrency level found -- even low worker counts hit "
                "throttling or failures. Investigate errors before scaling up.")
    peak = max(clean, key=lambda r: r.throughput)
    knee = min((r for r in clean if r.throughput >= knee_frac * peak.throughput),
               key=lambda r: r.workers)
    throttled_any = [r for r in results if r.throttle_frac >= throttle_ceiling]
    ceiling_note = (f" Throttling set in at {throttled_any[0].workers} workers."
                    if throttled_any else
                    " No throttling observed across the sweep -- the true ceiling "
                    "may be higher than the max level tested.")
    return (f"Peak clean throughput {peak.throughput:.1f} req/s at "
            f"{peak.workers} workers; knee at {knee.workers} workers "
            f"({knee.throughput:.1f} req/s, p95 {knee.p95_ms:.0f} ms). "
            f"Suggest predict_fetch_workers ~= {knee.workers}.{ceiling_note}")


# ---------------------------------------------------------------------------
# Horizontal-scaling probe (multi-PROCESS, not multi-thread)
# ---------------------------------------------------------------------------
# The thread sweep above finds a single process's ceiling (~16-32 workers).
# "Horizontal scaling" is the claim that the ceiling is PER PROCESS -- that
# running P independent processes (own GIL, own HTTP connection pools) yields
# ~P x the throughput because Planetary Computer is not enforcing an
# account/IP-wide rate limit. This probe tests that directly: it runs P
# processes at once, each doing its own read sweep at a fixed worker count on a
# DISJOINT set of tiles, and checks whether each process still achieves its
# single-process throughput. If per-process throughput holds as P grows,
# scaling is linear (shard chunks across processes); if it falls, a shared
# limit (the host NIC, or a real PC account cap) has been hit.
def _read_cpu_times() -> tuple:
    """(idle, total) jiffies from the aggregate ``/proc/stat`` cpu line."""
    with open("/proc/stat") as fh:
        parts = [int(x) for x in fh.readline().split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)  # idle + iowait
    return idle, sum(parts)


def _read_gpu_pct() -> "float | None":
    """Mean GPU utilization % via nvidia-smi, or None if no GPU / tool."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    vals = []
    for tok in out.stdout.split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return sum(vals) / len(vals) if vals else None


class ResourceSampler:
    """Background sampler of system-wide CPU% and GPU% over a code block.

    Context manager: samples every ``interval`` seconds on a daemon thread; after
    exit, ``cpu_pct`` / ``gpu_pct`` are the mean readings. CPU% is whole-host
    across all cores (0–100) from ``/proc/stat`` deltas, so it captures every
    child process; GPU% (via nvidia-smi) is here to *confirm the GPU stays idle*
    during pure fetching — it never runs the model — not because the fetch uses it.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._cpu: list = []
        self._gpu: list = []
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        idle0, total0 = _read_cpu_times()
        while not self._stop.wait(self.interval):
            idle1, total1 = _read_cpu_times()
            dt = total1 - total0
            if dt > 0:
                self._cpu.append(100.0 * (1.0 - (idle1 - idle0) / dt))
            idle0, total0 = idle1, total1
            gpu = _read_gpu_pct()
            if gpu is not None:
                self._gpu.append(gpu)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2)
        return False

    @property
    def cpu_pct(self) -> float:
        return sum(self._cpu) / len(self._cpu) if self._cpu else 0.0

    @property
    def gpu_pct(self) -> "float | None":
        return sum(self._gpu) / len(self._gpu) if self._gpu else None


@dataclass
class ScaleResult:
    """Aggregate of one process-count level in the horizontal sweep."""
    procs: int
    ok: int
    resolved: int
    throttled: int
    per_proc_thru: float    # mean successful reads/s across the P processes
    aggregate_thru: float   # total reads / slowest process's read wall
    cpu_pct: float = 0.0            # mean whole-host CPU% during the level
    gpu_pct: "float | None" = None  # mean GPU% (None if no GPU/monitor)

    def efficiency(self, base_per_proc: float) -> float:
        """Per-process throughput as a fraction of the single-process baseline."""
        return self.per_proc_thru / base_per_proc if base_per_proc > 0 else 0.0


def summarize_scaling(procs: int, child_results: list[dict],
                      cpu_pct: float = 0.0,
                      gpu_pct: "float | None" = None) -> ScaleResult:
    """Fold per-process result dicts into one :class:`ScaleResult`.

    Each child dict has keys ``ok``, ``wall_s``, ``throttled``, ``resolved``.
    Per-process throughput is the mean of each child's own ok/wall (so it is
    insensitive to processes starting a little out of step); aggregate is total
    successes over the slowest child's wall (the conservative concurrent rate).
    ``cpu_pct`` / ``gpu_pct`` are the whole-host utilization sampled during the
    level (0 / None when unmonitored).
    """
    walls = [c["wall_s"] for c in child_results if c["wall_s"] > 0]
    per = [c["ok"] / c["wall_s"] for c in child_results if c["wall_s"] > 0]
    ok = sum(c["ok"] for c in child_results)
    return ScaleResult(
        procs=procs,
        ok=ok,
        resolved=sum(c["resolved"] for c in child_results),
        throttled=sum(c["throttled"] for c in child_results),
        per_proc_thru=(sum(per) / len(per)) if per else 0.0,
        aggregate_thru=(ok / max(walls)) if walls else 0.0,
        cpu_pct=cpu_pct,
        gpu_pct=gpu_pct,
    )


def _scale_worker(spec: dict) -> dict:
    """One child process: resolve a disjoint tile pool, then read-sweep it once.

    Module-level and picklable so it survives a ``spawn`` start method (GDAL is
    not reliably fork-safe). Resolve (search) is done first and untimed; only the
    read phase's wall is reported, so the measurement is pure read throughput.
    """
    rank = spec["rank"]
    # Disjoint seed per rank -> each process hits different tiles (a shared CDN
    # cache would otherwise flatter the result).
    points = sample_points(spec["n_points"], seed=spec["seed"] + rank * 100003)
    refs = resolve_pool(points, spec["crop_m"], spec["year_hint"])
    if not refs:
        return {"rank": rank, "ok": 0, "wall_s": 0.0, "throttled": 0,
                "resolved": 0}
    attempt = make_read_attempt(spec["nbands"], spec["out_pixels"])
    res = run_level(refs, spec["workers"], attempt)
    return {"rank": rank, "ok": res.ok, "wall_s": res.wall_s,
            "throttled": res.throttled, "resolved": len(refs)}


def run_horizontal(process_counts: list[int], n_points: int, workers: int,
                   crop_m: float, out_pixels: int, nbands: int,
                   year_hint: int | None, seed: int = 0,
                   worker_fn=None, executor_factory=None,
                   sampler_factory=None) -> list[ScaleResult]:
    """Run the read sweep across P concurrent processes for each P in ``process_counts``.

    ``worker_fn`` / ``executor_factory`` / ``sampler_factory`` are injectable so
    the aggregation can be tested offline (a thread pool + fake worker + fake
    sampler) without spawning real processes or touching the network. Defaults
    spawn real processes via a ``spawn`` context. ``sampler_factory()`` returns a
    context manager exposing ``cpu_pct`` / ``gpu_pct`` after exit; when None, no
    utilization is recorded.
    """
    worker_fn = worker_fn or _scale_worker
    if executor_factory is None:
        ctx = mp.get_context("spawn")
        executor_factory = lambda mw: ProcessPoolExecutor(  # noqa: E731
            max_workers=mw, mp_context=ctx)
    out: list[ScaleResult] = []
    for procs in process_counts:
        specs = [dict(rank=r, n_points=n_points, seed=seed, crop_m=crop_m,
                      out_pixels=out_pixels, nbands=nbands, year_hint=year_hint,
                      workers=workers) for r in range(procs)]
        sampler = sampler_factory() if sampler_factory else None
        if sampler is not None:
            sampler.__enter__()
        try:
            with executor_factory(procs) as ex:
                child = list(ex.map(worker_fn, specs))
        finally:
            cpu_pct, gpu_pct = 0.0, None
            if sampler is not None:
                sampler.__exit__(None, None, None)
                cpu_pct, gpu_pct = sampler.cpu_pct, sampler.gpu_pct
        out.append(summarize_scaling(procs, child, cpu_pct=cpu_pct,
                                     gpu_pct=gpu_pct))
    return out


def recommend_scaling(results: list[ScaleResult],
                      efficiency_floor: float = 0.8) -> str:
    """Verdict on whether horizontal scaling holds across the process sweep."""
    if len(results) < 2:
        return "Need >=2 process counts to assess scaling."
    base = results[0].per_proc_thru
    if base <= 0:
        return "Single-process baseline did no successful reads; cannot assess."
    top = results[-1]
    eff = top.efficiency(base)
    if eff >= efficiency_floor:
        return (f"Horizontal scaling CONFIRMED: at {top.procs} processes each still "
                f"hits {eff * 100:.0f}% of single-process throughput "
                f"({top.aggregate_thru:.1f}/s aggregate vs {results[0].aggregate_thru:.1f}/s "
                f"at 1 process). PC is not account-throttling -- shard chunks across "
                f"processes/machines for ~linear speedup.")
    # Find where efficiency first dips below the floor.
    knee = next((r.procs for r in results if r.efficiency(base) < efficiency_floor),
                top.procs)
    return (f"Horizontal scaling SATURATES near {knee} processes: per-process "
            f"throughput falls to {eff * 100:.0f}% at {top.procs} processes "
            f"({top.aggregate_thru:.1f}/s aggregate). A shared limit (host NIC "
            f"bandwidth, or a PC account/IP cap) bounds you here -- add machines "
            f"on separate uplinks rather than more processes on one host.")


def _format_scale(res: ScaleResult, base_per_proc: float) -> str:
    gpu = "n/a" if res.gpu_pct is None else f"{res.gpu_pct:4.0f}%"
    return (f"  procs={res.procs:>2}  per_proc={res.per_proc_thru:6.1f}/s  "
            f"aggregate={res.aggregate_thru:7.1f}/s  "
            f"efficiency={res.efficiency(base_per_proc) * 100:5.0f}%  "
            f"cpu={res.cpu_pct:4.0f}%  gpu={gpu}  "
            f"ok={res.ok}  throttled={res.throttled}")


# ---------------------------------------------------------------------------
# Read-granularity probe: per-crop reads vs one big union read
# ---------------------------------------------------------------------------
# Does reading ONE big window covering many buildings beat reading each
# building's ~200 m window separately? Production already reads a per-tract union
# (open_naip_mosaic) and crops locally; this measures whether that pays off and
# how it degrades as the covered footprint grows (a bigger union decompresses
# more inter-building empty pixels, so its per-crop efficiency should fall once
# buildings thin out). Both strategies open the SAME DOQQ once *outside* the
# timer, so the comparison isolates read strategy -- not open/search overhead.
#
# Subtlety the numbers will settle: GDAL keeps a *decompressed* block cache
# (GDAL_CACHEMAX), so per-crop reads from one warm handle may already avoid
# re-decompressing shared blocks -- in which case the union's only remaining win
# is Python-level per-read + resample setup, and "bigger" barely helps. Measure,
# don't assume.
@dataclass
class GranularityResult:
    """One read strategy over a cluster of buildings sharing a single DOQQ."""
    strategy: str            # "per_crop" | "union"
    crops: int               # useful crops produced
    wall_s: float            # read wall time (open excluded)
    union_mpix: float = 0.0  # megapixels actually decompressed (union only)
    fell_back: bool = False  # union exceeded the cap -> would fall back in prod

    @property
    def crops_per_s(self) -> float:
        return self.crops / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def mpix_per_crop(self) -> float:
        return self.union_mpix / self.crops if self.crops else 0.0


def sample_cluster(n: int, center: tuple, radius_km: float,
                   seed: int = 0) -> list:
    """``n`` (lon, lat) points uniformly in a disk of ``radius_km`` about ``center``.

    Deterministic given ``seed``. Uniform-in-area (``r = R*sqrt(u)``) so density
    does not pile up at the center. A small radius keeps every point on one DOQQ;
    a larger one spreads them, which is how the sweep grows the union footprint.
    """
    clon, clat = center
    rng = random.Random(seed)
    dlat_per_km = 1.0 / 111.0
    dlon_per_km = 1.0 / (111.0 * max(0.1, cos(radians(clat))))
    pts = []
    for _ in range(n):
        rad = radius_km * sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * pi)
        pts.append((clon + rad * cos(theta) * dlon_per_km,
                    clat + rad * sin(theta) * dlat_per_km))
    return pts


def pick_shared_doqq(points: list, crop_size_meters: float,
                     year_hint: int | None, resolve_fn=None) -> tuple:
    """Resolve ``points`` and return ``(item_id, refs)`` for the most-common DOQQ.

    Restricting to one DOQQ means every read hits the same COG, so the per-crop
    vs union comparison isolates read granularity from tile diversity. Refs on
    one item share an href, so the caller opens it once for all reads.
    """
    resolve_fn = resolve_fn or naip_fetcher.resolve_naip_item
    refs = []
    for lon, lat in points:
        ref = resolve_fn(lon, lat, crop_size_meters, year_hint)
        if getattr(ref, "failure", None) is None and ref.item_id is not None:
            refs.append(ref)
    if not refs:
        return None, []
    top = Counter(r.item_id for r in refs).most_common(1)[0][0]
    return top, [r for r in refs if r.item_id == top]


def run_per_crop(refs: list, nbands: int, out_pixels: int,
                 open_fn=None, read_fn=None) -> GranularityResult:
    """Read each ref's window separately from one open handle (open untimed)."""
    open_fn = open_fn or naip_fetcher.open_naip_src
    read_fn = read_fn or naip_fetcher.read_naip_crop_from_src
    if not refs:
        return GranularityResult("per_crop", 0, 0.0)
    r0 = refs[0]
    with open_fn(r0.href, r0.use_cache) as src:
        t0 = time.perf_counter()
        crops = 0
        for ref in refs:
            _crop, _nir, _partial, failure = read_fn(
                src, ref.bbox, nbands, out_pixels)
            if failure is None:
                crops += 1
        wall = time.perf_counter() - t0
    return GranularityResult("per_crop", crops, wall)


def run_union(refs: list, nbands: int, out_pixels: int, max_union_pixels: int,
              open_fn=None, mosaic_fn=None, read_fn=None) -> GranularityResult:
    """Read the union of all refs once, then crop each locally (open untimed).

    Reports the union's decompressed megapixels; if the union exceeds
    ``max_union_pixels`` the mosaic yields None (as in production) and this is
    flagged ``fell_back`` with zero crops -- itself the signal that the read got
    too big to be worthwhile.
    """
    open_fn = open_fn or naip_fetcher.open_naip_src
    mosaic_fn = mosaic_fn or naip_fetcher.open_naip_mosaic
    read_fn = read_fn or naip_fetcher.read_naip_crop_from_src
    if not refs:
        return GranularityResult("union", 0, 0.0)
    r0 = refs[0]
    bboxes = [ref.bbox for ref in refs]
    with open_fn(r0.href, r0.use_cache) as src:
        t0 = time.perf_counter()
        crops = 0
        union_mpix = 0.0
        fell_back = False
        with mosaic_fn(src, bboxes, nbands, max_union_pixels) as mem:
            if mem is None:
                fell_back = True
            else:
                union_mpix = (mem.width * mem.height) / 1e6
                for ref in refs:
                    _crop, _nir, _partial, failure = read_fn(
                        mem, ref.bbox, nbands, out_pixels)
                    if failure is None:
                        crops += 1
        wall = time.perf_counter() - t0
    return GranularityResult("union", crops, wall, union_mpix, fell_back)


def choose_tract_ids(counts: dict, n_tracts: int, min_b: int, max_b: int,
                     seed: int = 0) -> list:
    """Pick ``n_tracts`` tract ids whose building count is in ``[min_b, max_b]``.

    Deterministic given ``seed``; returned sorted by building count ascending so
    the caller sweeps from sparse to dense tracts. Restricting the count band
    keeps per-tract read cost bounded while still spanning a density range.
    """
    eligible = sorted(g for g, c in counts.items() if min_b <= c <= max_b)
    if not eligible:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(n_tracts, len(eligible)))
    chosen.sort(key=lambda g: counts[g])
    return chosen


def load_tracts_from_index(index_path, n_tracts: int, min_b: int, max_b: int,
                           seed: int = 0) -> list:
    """Load real building centroids grouped by tract from a ``buildings_index``.

    Returns ``[(geoid, cx_array, cy_array), ...]`` (EPSG:5070 metres) for the
    chosen tracts. This is the production grouping: one tract == one mosaic
    group, at the real building layout/density.
    """
    import pandas as pd
    df = pd.read_parquet(index_path, columns=["cx", "cy", "tract_id"])
    counts = df.groupby("tract_id").size().to_dict()
    chosen = choose_tract_ids(counts, n_tracts, min_b, max_b, seed)
    out = []
    for geoid in chosen:
        sub = df[df["tract_id"] == geoid]
        out.append((str(geoid), sub["cx"].to_numpy(), sub["cy"].to_numpy()))
    return out


def load_chunk_from_index(index_path, chunk_size: int, year: int,
                          seed: int = 0) -> "object":
    """A GEOID-sorted, production-shaped chunk of real buildings.

    Returns a DataFrame with ``centroid_x/centroid_y`` (EPSG:5070), ``GEOID``,
    and ``year`` — the columns the production fetch reads. Buildings are sorted
    by tract (as ``predict_year_chunked`` does) and a contiguous window of
    ``chunk_size`` is taken at a seed-chosen offset, so the sample spans several
    real tracts at their true density — the input the grouped fetch is tuned for.
    """
    import numpy as np
    import pandas as pd
    df = pd.read_parquet(index_path, columns=["cx", "cy", "tract_id"])
    df = df.sort_values("tract_id", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, len(df) - chunk_size)))
    chunk = df.iloc[start:start + chunk_size].copy()
    chunk = chunk.rename(columns={"cx": "centroid_x", "cy": "centroid_y",
                                  "tract_id": "GEOID"})
    chunk["year"] = int(year)
    return chunk.reset_index(drop=True)


def resolve_tract_refs(geoid: str, lons, lats, crop_size_meters: float,
                       year_hint: int | None) -> tuple:
    """Resolve a tract's buildings to their shared DOQQ via a per-tract cache.

    Routes every building's STAC search through one :class:`TractSearchCache`
    keyed on the tract, so the whole tract costs ~one search (as in production),
    then returns the refs on the tract's dominant DOQQ.
    """
    from src.data.naip_fetcher import TractSearchCache, resolve_naip_item
    cache = TractSearchCache(max_entries=8)

    def _resolve(lon, lat, crop, yh):
        return resolve_naip_item(lon, lat, crop, yh, search_cache=cache,
                                 cache_key=geoid)

    points = list(zip(lons, lats))
    return pick_shared_doqq(points, crop_size_meters, year_hint,
                            resolve_fn=_resolve)


def _format_granularity_real(geoid: str, item_id: str, n_on_doqq: int,
                             n_total: int, pc: GranularityResult,
                             un: GranularityResult) -> str:
    head = (f"  tract={geoid}  buildings={n_total} "
            f"({n_on_doqq} on {item_id})")
    pc_line = (f"    per_crop: crops={pc.crops:>3}  wall={pc.wall_s:6.2f}s  "
               f"crops/s={pc.crops_per_s:6.1f}")
    if un.fell_back:
        un_line = "    union:    exceeded cap -> would fall back to per-crop"
        verdict = ""
    else:
        speedup = (un.crops_per_s / pc.crops_per_s) if pc.crops_per_s > 0 else 0.0
        un_line = (f"    union:    crops={un.crops:>3}  wall={un.wall_s:6.2f}s  "
                   f"crops/s={un.crops_per_s:6.1f}  "
                   f"union={un.union_mpix:5.1f}MP "
                   f"({un.mpix_per_crop:.3f}MP/crop)")
        verdict = f"    -> union {speedup:.2f}x per-crop"
    return "\n".join(x for x in (head, pc_line, un_line, verdict) if x)


def _format_granularity(radius_km: float, item_id: str, n: int,
                        pc: GranularityResult, un: GranularityResult) -> str:
    head = f"  radius={radius_km:>4}km  N={n:>3} on {item_id}"
    pc_line = (f"    per_crop: crops={pc.crops:>3}  wall={pc.wall_s:6.2f}s  "
               f"crops/s={pc.crops_per_s:6.1f}")
    if un.fell_back:
        un_line = (f"    union:    exceeded cap -> would fall back to per-crop "
                   f"(union too large)")
        verdict = ""
    else:
        speedup = (un.crops_per_s / pc.crops_per_s) if pc.crops_per_s > 0 else 0.0
        un_line = (f"    union:    crops={un.crops:>3}  wall={un.wall_s:6.2f}s  "
                   f"crops/s={un.crops_per_s:6.1f}  "
                   f"union={un.union_mpix:5.1f}MP "
                   f"({un.mpix_per_crop:.2f}MP/crop)")
        verdict = f"    -> union {speedup:.2f}x per-crop"
    return "\n".join(x for x in (head, pc_line, un_line, verdict) if x)


# ---------------------------------------------------------------------------
# Sample point generation (real metros -> guaranteed NAIP coverage)
# ---------------------------------------------------------------------------
# CONUS metro centers (lon, lat). Spread across the country so reads land on
# many distinct DOQQs (realistic tile diversity), and every point is over land
# with NAIP coverage (random CONUS points would hit water/no-coverage).
_METROS = [
    ("Phoenix", -112.074, 33.448),
    ("Los Angeles", -118.244, 34.052),
    ("Houston", -95.369, 29.760),
    ("Chicago", -87.632, 41.884),
    ("Atlanta", -84.388, 33.749),
    ("Denver", -104.991, 39.739),
    ("Seattle", -122.332, 47.606),
    ("Miami", -80.191, 25.762),
    ("New York", -73.968, 40.785),
    ("Minneapolis", -93.265, 44.978),
]


def sample_points(n: int, seed: int = 0, jitter_deg: float = 0.08) -> list:
    """``n`` (lon, lat) points, round-robined across metros with jitter.

    Deterministic given ``seed``. ``jitter_deg`` (~0.08 deg ~= 8 km) spreads
    points within each metro so distinct crops hit distinct COG byte ranges
    rather than re-reading one window.
    """
    rng = random.Random(seed)
    points = []
    for i in range(n):
        _name, lon, lat = _METROS[i % len(_METROS)]
        points.append((
            lon + rng.uniform(-jitter_deg, jitter_deg),
            lat + rng.uniform(-jitter_deg, jitter_deg),
        ))
    return points


def resolve_pool(points: list, crop_size_meters: float,
                 year_hint: int | None, resolve_fn=None) -> list:
    """Resolve ``points`` to NAIP item refs once (serially), dropping failures.

    Reads reuse these refs across every concurrency level, so the read sweep
    measures pure open+read scaling on an identical workload -- search cost and
    coverage gaps are factored out. Refs carry *signed* hrefs (no cache), valid
    for ~1h, which comfortably covers a sweep.
    """
    resolve_fn = resolve_fn or naip_fetcher.resolve_naip_item
    refs = []
    for lon, lat in points:
        ref = resolve_fn(lon, lat, crop_size_meters, year_hint)
        if getattr(ref, "failure", None) is None:
            refs.append(ref)
    return refs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _format_level(res: LevelResult) -> str:
    top = res.fail_modes.most_common(1)
    top_fail = f"{top[0][0]} x{top[0][1]}" if top else "-"
    return (f"  workers={res.workers:>3}  ok={res.ok:>3}/{res.total:<3}  "
            f"throttled={res.throttled:>3}  "
            f"thru={res.throughput:6.1f}/s  "
            f"p50={res.p50_ms:6.0f}ms  p95={res.p95_ms:7.0f}ms  "
            f"top_fail={top_fail}")


def _run_mode(label: str, levels: list[int], work_items: list, attempt_fn,
              early_stop_frac: float) -> list[LevelResult]:
    print(f"\n=== {label} sweep ({len(work_items)} requests/level) ===")
    results = sweep(levels, work_items, attempt_fn,
                    early_stop_frac=early_stop_frac,
                    on_level=lambda r: print(_format_level(r)))
    print(f"  -> {recommend(results)}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode",
                    choices=["reads", "search", "both", "scale", "granularity",
                             "real-granularity", "chunk"],
                    default="both",
                    help="reads/search/both = thread sweep on one process; "
                         "scale = multi-process horizontal-scaling test; "
                         "granularity = per-crop vs big-union (synthetic clusters); "
                         "real-granularity = same on real tracts from buildings_index; "
                         "chunk = benchmark the real production fetch_prediction_chunk")
    ap.add_argument("--levels", default="1,2,4,8,16,32,48,64",
                    help="comma-separated worker counts to sweep (thread modes)")
    ap.add_argument("--total", type=int, default=96,
                    help="requests issued per level (default: 96)")
    ap.add_argument("--procs", default="1,2,4",
                    help="scale mode: comma-separated process counts to test")
    ap.add_argument("--proc-workers", type=int, default=16,
                    help="scale mode: threads per process (default: the read knee, 16)")
    ap.add_argument("--proc-points", type=int, default=48,
                    help="scale mode: read requests per process (default: 48)")
    ap.add_argument("--radii-km", default="0.25,0.5,1,2,3",
                    help="granularity mode: cluster radii (km) to sweep")
    ap.add_argument("--cluster-points", type=int, default=200,
                    help="granularity mode: points sampled per cluster (filtered "
                         "to the shared DOQQ)")
    ap.add_argument("--union-cap-mpix", type=float, default=67.0,
                    help="granularity mode: max union megapixels before fallback "
                         "(prod cap is 16.8 = 4096^2; default 67 = 8192^2 to let "
                         "the sweep measure larger reads)")
    ap.add_argument("--center", default=None,
                    help="granularity mode: 'lon,lat' cluster center "
                         "(default: Phoenix)")
    ap.add_argument("--state", default="Arizona",
                    help="real-granularity mode: buildings_index state")
    ap.add_argument("--tracts", type=int, default=5,
                    help="real-granularity mode: number of tracts to test")
    ap.add_argument("--min-buildings", type=int, default=40,
                    help="real-granularity mode: min buildings/tract to include")
    ap.add_argument("--max-buildings", type=int, default=500,
                    help="real-granularity mode: max buildings/tract (bounds read cost)")
    ap.add_argument("--index-path", default=None,
                    help="real/chunk mode: override buildings_index parquet path")
    ap.add_argument("--chunk-size", type=int, default=512,
                    help="chunk mode: buildings per fetch_prediction_chunk call")
    ap.add_argument("--chunk-workers", default="8,16,32",
                    help="chunk mode: worker counts to benchmark the real fetch at")
    ap.add_argument("--crop-meters", type=float, default=200.0,
                    help="crop window size in meters (production: tau*2 = 200)")
    ap.add_argument("--out-pixels", type=int, default=224,
                    help="output crop resolution (production image_size)")
    ap.add_argument("--nbands", type=int, default=4)
    ap.add_argument("--year-hint", type=int, default=None,
                    help="preferred NAIP flight year (default: newest)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--early-stop-frac", type=float, default=0.25,
                    help="stop ramping once this throttle fraction is hit")
    args = ap.parse_args(argv)

    if args.mode == "scale":
        procs = [int(x) for x in args.procs.split(",") if x.strip()]
        print(f"NAIP horizontal-scaling probe | procs={procs} "
              f"workers/proc={args.proc_workers} points/proc={args.proc_points} "
              f"crop={args.crop_meters}m px={args.out_pixels} bands={args.nbands}")
        print(f"\n=== SCALE (multi-process reads) ===")
        results = run_horizontal(procs, args.proc_points, args.proc_workers,
                                 args.crop_meters, args.out_pixels, args.nbands,
                                 args.year_hint, seed=args.seed,
                                 sampler_factory=ResourceSampler)
        base = results[0].per_proc_thru if results else 0.0
        for res in results:
            print(_format_scale(res, base))
        print(f"  -> {recommend_scaling(results)}")
        return 0

    if args.mode == "granularity":
        if args.center:
            clon, clat = (float(x) for x in args.center.split(","))
            center = (clon, clat)
        else:
            center = (_METROS[0][1], _METROS[0][2])  # Phoenix
        radii = [float(x) for x in args.radii_km.split(",") if x.strip()]
        cap_pixels = int(args.union_cap_mpix * 1e6)
        print(f"NAIP read-granularity probe | center={center} radii={radii}km "
              f"cluster_points={args.cluster_points} crop={args.crop_meters}m "
              f"px={args.out_pixels} union_cap={args.union_cap_mpix}MP")
        print(f"\n=== GRANULARITY (per-crop vs union read) ===")
        for radius in radii:
            pts = sample_cluster(args.cluster_points, center, radius, seed=args.seed)
            item_id, refs = pick_shared_doqq(pts, args.crop_meters, args.year_hint)
            if len(refs) < 2:
                print(f"  radius={radius}km  only {len(refs)} pts on a shared "
                      f"DOQQ -- skip (widen cluster or check coverage)")
                continue
            pc = run_per_crop(refs, args.nbands, args.out_pixels)
            un = run_union(refs, args.nbands, args.out_pixels, cap_pixels)
            print(_format_granularity(radius, item_id, len(refs), pc, un))
        return 0

    if args.mode == "chunk":
        # Benchmark the REAL post-per-crop-change production fetch on a real
        # chunk, at several worker counts -> the sustained shared-handle
        # per-host read rate that the total-runtime estimate needs.
        from pathlib import Path as _Path

        from pyproj import Transformer

        from src.data.naip_fetcher import (TractSearchCache, get_fetch_stats,
                                           reset_fetch_stats)
        from src.prediction import fetch_prediction_chunk
        metric_crs = "EPSG:5070"
        if args.index_path:
            index_path = _Path(args.index_path)
        else:
            repo_root = _Path(__file__).resolve().parents[1]
            index_path = (repo_root / "data" / "processed" / "buildings_index"
                          / f"state={args.state}" / "part.parquet")
        year = int(args.year_hint) if args.year_hint else 2020
        workers = [int(x) for x in args.chunk_workers.split(",") if x.strip()]
        params = {"tau_meters": args.crop_meters / 2.0, "nbands": args.nbands,
                  "image_size": args.out_pixels, "search_grid_meters": 20000.0}
        df = load_chunk_from_index(index_path, args.chunk_size, year, args.seed)
        to_4326 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
        lons, lats = to_4326.transform(df["centroid_x"].to_numpy(),
                                       df["centroid_y"].to_numpy())
        df = df.assign(lon=lons, lat=lats)
        print(f"NAIP chunk-fetch benchmark | state={args.state} "
              f"chunk_size={len(df)} year={year} workers={workers} "
              f"(real production fetch_prediction_chunk)")
        print(f"\n=== CHUNK (production grouped per-crop fetch) ===")
        for w in workers:
            cache = TractSearchCache(max_entries=max(512, args.chunk_size))
            reset_fetch_stats()
            t0 = time.perf_counter()
            crops, _years, _fail = fetch_prediction_chunk(
                df, params, cache, to_4326, max_workers=w,
                sleep_fn=lambda s: None, verbose=False)
            wall = time.perf_counter() - t0
            ok = sum(1 for c in crops if c is not None)
            rate = ok / wall if wall > 0 else 0.0
            stats = get_fetch_stats()
            print(f"  workers={w:>3}  ok={ok:>4}/{len(df):<4}  wall={wall:6.1f}s  "
                  f"crops/s={rate:6.1f}  "
                  f"(no_items={stats['no_items']} read_err={stats['read_error']})")
        return 0

    if args.mode == "real-granularity":
        from pathlib import Path as _Path

        from pyproj import Transformer

        # buildings_index cx/cy are in METRIC_CRS = EPSG:5070 (src.geo_utils).
        # Use the literal here rather than importing geo_utils, whose import
        # chain (build_dataset -> paths) requires the IMAGERY_ROOT env var.
        metric_crs = "EPSG:5070"
        if args.index_path:
            index_path = _Path(args.index_path)
        else:
            repo_root = _Path(__file__).resolve().parents[1]
            index_path = (repo_root / "data" / "processed" / "buildings_index"
                          / f"state={args.state}" / "part.parquet")
        cap_pixels = int(args.union_cap_mpix * 1e6)
        print(f"NAIP real-tract granularity probe | state={args.state} "
              f"tracts={args.tracts} buildings/tract in "
              f"[{args.min_buildings},{args.max_buildings}] crop={args.crop_meters}m "
              f"px={args.out_pixels} union_cap={args.union_cap_mpix}MP")
        tracts = load_tracts_from_index(index_path, args.tracts,
                                        args.min_buildings, args.max_buildings,
                                        args.seed)
        if not tracts:
            print(f"  no tracts in count band -- check {index_path}")
            return 0
        to_4326 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
        print(f"\n=== REAL-GRANULARITY (per-crop vs union, real tracts) ===")
        for geoid, cx, cy in tracts:
            lons, lats = to_4326.transform(cx, cy)
            item_id, refs = resolve_tract_refs(geoid, lons, lats,
                                               args.crop_meters, args.year_hint)
            if len(refs) < 2:
                print(f"  tract={geoid}  only {len(refs)} refs on a shared DOQQ "
                      f"-- skip")
                continue
            pc = run_per_crop(refs, args.nbands, args.out_pixels)
            un = run_union(refs, args.nbands, args.out_pixels, cap_pixels)
            print(_format_granularity_real(geoid, item_id, len(refs), len(cx),
                                           pc, un))
        return 0

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    points = sample_points(args.total, seed=args.seed)
    print(f"NAIP concurrency probe | mode={args.mode} levels={levels} "
          f"total/level={args.total} crop={args.crop_meters}m "
          f"px={args.out_pixels} bands={args.nbands}")

    if args.mode in ("search", "both"):
        search_fn = make_search_attempt(args.crop_meters)
        _run_mode("SEARCH (STAC /search)", levels, points, search_fn,
                  args.early_stop_frac)

    if args.mode in ("reads", "both"):
        print(f"\nResolving {args.total} item refs for the read sweep ...")
        naip_fetcher.reset_fetch_stats()
        refs = resolve_pool(points, args.crop_meters, args.year_hint)
        print(f"  resolved {len(refs)}/{args.total} "
              f"(stats: {naip_fetcher.get_fetch_stats()})")
        if not refs:
            print("  no refs resolved -- cannot run read sweep.")
        else:
            read_fn = make_read_attempt(args.nbands, args.out_pixels)
            _run_mode("READS (COG open+read)", levels, refs, read_fn,
                      args.early_stop_frac)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
