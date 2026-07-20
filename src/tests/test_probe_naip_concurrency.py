"""Offline tests for the NAIP concurrency probe.

The probe hits the live Planetary Computer network when run for real; these
tests exercise its pure logic (classification, summary, sweep engine, knee
recommendation, sampling) and its attempt factories with injected fake fetch
functions, so nothing here touches the network.
"""

from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager

import pytest

from concurrent.futures import ThreadPoolExecutor

from src.probe_naip_concurrency import (
    AttemptResult,
    GranularityResult,
    LevelResult,
    ScaleResult,
    ResourceSampler,
    choose_tract_ids,
    classify_throttle,
    load_chunk_from_index,
    load_tracts_from_index,
    make_read_attempt,
    make_search_attempt,
    pick_shared_doqq,
    recommend,
    recommend_scaling,
    resolve_pool,
    run_horizontal,
    run_level,
    run_per_crop,
    run_union,
    sample_cluster,
    sample_points,
    summarize,
    summarize_scaling,
    sweep,
    _percentile,
)


# --------------------------------------------------------------------------
# classify_throttle
# --------------------------------------------------------------------------
@pytest.mark.parametrize("msg", [
    "RasterioIOError: HTTP response code: 429",
    "Too Many Requests",
    "server responded 503 Service Unavailable",
    "AzureError: SlowDown, request rate is too high",
    "quota exceeded for this account",
    "rate-limit reached",
])
def test_classify_throttle_positive(msg):
    assert classify_throttle(msg) is True


@pytest.mark.parametrize("msg", [
    "",
    "no_items",
    "RasterioIOError: HTTP response code: 403 (expired SAS token)",
    "read_error",
    "ConnectionResetError: connection aborted",
])
def test_classify_throttle_negative(msg):
    # 403 (auth/SAS) and generic errors must NOT be miscounted as throttling.
    assert classify_throttle(msg) is False


# --------------------------------------------------------------------------
# _percentile
# --------------------------------------------------------------------------
def test_percentile_empty_is_zero():
    assert _percentile([], 0.5) == 0.0


def test_percentile_single_value():
    assert _percentile([0.4], 0.95) == 0.4


def test_percentile_interpolates():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _percentile(vals, 0.5) == 2.0
    assert _percentile(vals, 0.0) == 0.0
    assert _percentile(vals, 1.0) == 4.0


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
def test_summarize_counts_and_modes():
    results = [
        AttemptResult(True, 0.10, "ok", False),
        AttemptResult(True, 0.20, "ok", False),
        AttemptResult(False, 0.05, "no_items", False),
        AttemptResult(False, 0.30, "RasterioIOError: 429", True),
    ]
    res = summarize(results, workers=4, wall_s=2.0)
    assert res.workers == 4
    assert res.total == 4
    assert res.ok == 2
    assert res.fail == 2
    assert res.throttled == 1
    assert res.throughput == pytest.approx(1.0)  # ok/wall = 2/2
    assert res.throttle_frac == pytest.approx(0.25)
    assert res.fail_modes == Counter({"no_items": 1, "RasterioIOError: 429": 1})
    # p50 over the four latencies (0.05,0.10,0.20,0.30) -> 0.15s = 150 ms
    assert res.p50_ms == pytest.approx(150.0)


def test_summarize_zero_wall_no_div_zero():
    res = summarize([AttemptResult(True, 0.1, "ok", False)], workers=1, wall_s=0.0)
    assert res.throughput == 0.0


def test_summarize_ignores_zero_latency_in_percentiles():
    # A ref that failed before hitting the net contributes latency 0.0 and must
    # not drag the percentile down.
    results = [
        AttemptResult(False, 0.0, "no_items", False),
        AttemptResult(True, 0.20, "ok", False),
    ]
    res = summarize(results, workers=2, wall_s=1.0)
    assert res.p50_ms == pytest.approx(200.0)


# --------------------------------------------------------------------------
# run_level / sweep engine (with a fake attempt_fn)
# --------------------------------------------------------------------------
def test_run_level_runs_every_item():
    seen = []

    def attempt(item):
        seen.append(item)
        return AttemptResult(True, 0.01, "ok", False)

    res = run_level(list(range(10)), workers=4, attempt_fn=attempt)
    assert res.total == 10
    assert res.ok == 10
    assert sorted(seen) == list(range(10))


def test_run_level_is_actually_concurrent():
    # 8 items each sleeping 50 ms at 8 workers should finish well under the
    # 400 ms serial time.
    def attempt(item):
        time.sleep(0.05)
        return AttemptResult(True, 0.05, "ok", False)

    t0 = time.perf_counter()
    run_level(list(range(8)), workers=8, attempt_fn=attempt)
    assert time.perf_counter() - t0 < 0.25


def test_sweep_early_stops_on_throttle():
    levels_run = []

    def attempt(item):
        # every attempt is throttled -> first level trips the early stop
        return AttemptResult(False, 0.01, "429", True)

    def on_level(res):
        levels_run.append(res.workers)

    out = sweep([1, 2, 4, 8], work_items=[0, 1, 2], attempt_fn=attempt,
                early_stop_frac=0.25, on_level=on_level)
    assert len(out) == 1
    assert levels_run == [1]


def test_sweep_runs_all_levels_when_clean():
    def attempt(item):
        return AttemptResult(True, 0.01, "ok", False)

    out = sweep([1, 2, 4], work_items=[0, 1], attempt_fn=attempt,
                early_stop_frac=0.25)
    assert [r.workers for r in out] == [1, 2, 4]


# --------------------------------------------------------------------------
# recommend
# --------------------------------------------------------------------------
def _level(workers, ok, total, throttled, wall, p95_ms=100.0):
    return LevelResult(workers=workers, total=total, ok=ok, throttled=throttled,
                       wall_s=wall, p50_ms=50.0, p95_ms=p95_ms)


def test_recommend_picks_knee_not_peak():
    # throughput: 10/s @8w, 11/s @16w (only +10%). Knee at 8 (>=90% of peak),
    # not the marginally-faster 16.
    results = [
        _level(1, ok=10, total=10, throttled=0, wall=10.0),   # 1.0/s
        _level(8, ok=100, total=100, throttled=0, wall=10.0),  # 10.0/s
        _level(16, ok=110, total=110, throttled=0, wall=10.0),  # 11.0/s (+10%)
    ]
    msg = recommend(results)
    assert "predict_fetch_workers ~= 8" in msg


def test_recommend_excludes_throttled_levels():
    results = [
        _level(8, ok=100, total=100, throttled=0, wall=10.0),   # clean 10/s
        _level(16, ok=150, total=200, throttled=50, wall=10.0),  # throttled 25%
    ]
    msg = recommend(results)
    # 16w is throttled (0.25 >= ceiling) -> excluded; knee stays at 8
    assert "~= 8" in msg
    assert "Throttling set in at 16 workers" in msg


def test_recommend_no_clean_level():
    results = [_level(1, ok=0, total=10, throttled=10, wall=1.0)]
    msg = recommend(results)
    assert "No clean concurrency level" in msg


def test_recommend_notes_no_throttling_headroom():
    results = [_level(64, ok=100, total=100, throttled=0, wall=5.0)]
    msg = recommend(results)
    assert "No throttling observed" in msg


# --------------------------------------------------------------------------
# sample_points
# --------------------------------------------------------------------------
def test_sample_points_deterministic():
    assert sample_points(20, seed=7) == sample_points(20, seed=7)


def test_sample_points_seed_varies():
    assert sample_points(20, seed=1) != sample_points(20, seed=2)


def test_sample_points_within_jitter_of_a_metro():
    from src.probe_naip_concurrency import _METROS
    pts = sample_points(40, seed=3, jitter_deg=0.08)
    assert len(pts) == 40
    for lon, lat in pts:
        near = min(abs(lon - m[1]) <= 0.0801 and abs(lat - m[2]) <= 0.0801
                   for m in _METROS)
        assert near is True or any(
            abs(lon - m[1]) <= 0.0801 and abs(lat - m[2]) <= 0.0801
            for m in _METROS)


# --------------------------------------------------------------------------
# make_read_attempt / make_search_attempt with injected fakes (no network)
# --------------------------------------------------------------------------
class _FakeRef:
    def __init__(self, failure=None, href="unsigned://x", use_cache=False,
                 bbox=(0, 0, 1, 1)):
        self.failure = failure
        self.href = href
        self.use_cache = use_cache
        self.bbox = bbox


def test_read_attempt_short_circuits_failed_ref():
    attempt = make_read_attempt(nbands=4, out_pixels=224)
    res = attempt(_FakeRef(failure="no_items"))
    assert res.ok is False
    assert res.mode == "no_items"
    assert res.latency == 0.0


def test_read_attempt_success_path():
    opened = {}

    @contextmanager
    def fake_open(href, use_cache):
        opened["href"] = href
        yield object()

    def fake_read(src, bbox, nbands, out_pixels):
        return ("crop", False, False, None)  # failure=None -> ok

    attempt = make_read_attempt(4, 224, open_fn=fake_open, read_fn=fake_read)
    res = attempt(_FakeRef(href="signed://tile"))
    assert res.ok is True
    assert res.mode == "ok"
    assert opened["href"] == "signed://tile"


def test_read_attempt_classifies_throttle_exception():
    @contextmanager
    def fake_open(href, use_cache):
        raise RuntimeError("HTTP response code: 429 Too Many Requests")
        yield  # pragma: no cover

    attempt = make_read_attempt(4, 224, open_fn=fake_open, read_fn=lambda *a: None)
    res = attempt(_FakeRef())
    assert res.ok is False
    assert res.throttled is True


def test_read_attempt_non_throttle_read_failure():
    @contextmanager
    def fake_open(href, use_cache):
        yield object()

    def fake_read(src, bbox, nbands, out_pixels):
        return (None, False, False, "read_error")

    attempt = make_read_attempt(4, 224, open_fn=fake_open, read_fn=fake_read)
    res = attempt(_FakeRef())
    assert res.ok is False
    assert res.throttled is False
    assert res.mode == "read_error"


def test_search_attempt_success_and_empty():
    def fake_search_hit(bbox, max_items, signed):
        return ["item"]

    def fake_search_empty(bbox, max_items, signed):
        return []

    hit = make_search_attempt(200.0, search_fn=fake_search_hit)((-112.0, 33.4))
    assert hit.ok is True and hit.mode == "ok"

    empty = make_search_attempt(200.0, search_fn=fake_search_empty)((-112.0, 33.4))
    assert empty.ok is False and empty.mode == "no_items"


def test_search_attempt_classifies_throttle():
    def fake_search(bbox, max_items, signed):
        raise RuntimeError("503 server busy")

    res = make_search_attempt(200.0, search_fn=fake_search)((-112.0, 33.4))
    assert res.ok is False and res.throttled is True


# --------------------------------------------------------------------------
# resolve_pool with injected resolver (no network)
# --------------------------------------------------------------------------
def test_resolve_pool_drops_failures():
    def fake_resolve(lon, lat, crop, year_hint):
        # fail every other point
        return _FakeRef(failure=None if int(lon) % 2 == 0 else "no_items")

    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    refs = resolve_pool(points, 200.0, None, resolve_fn=fake_resolve)
    assert len(refs) == 2
    assert all(r.failure is None for r in refs)


# --------------------------------------------------------------------------
# Horizontal-scaling: summarize_scaling
# --------------------------------------------------------------------------
def test_summarize_scaling_aggregates():
    # two processes, each 50 ok reads in 5s -> 10/s per proc, aggregate 100/5=20
    child = [
        {"rank": 0, "ok": 50, "wall_s": 5.0, "throttled": 0, "resolved": 50},
        {"rank": 1, "ok": 50, "wall_s": 5.0, "throttled": 1, "resolved": 50},
    ]
    res = summarize_scaling(2, child)
    assert res.procs == 2
    assert res.ok == 100
    assert res.resolved == 100
    assert res.throttled == 1
    assert res.per_proc_thru == pytest.approx(10.0)
    assert res.aggregate_thru == pytest.approx(20.0)  # 100 / max(wall)=5


def test_summarize_scaling_uses_slowest_wall_for_aggregate():
    child = [
        {"rank": 0, "ok": 40, "wall_s": 4.0, "throttled": 0, "resolved": 40},
        {"rank": 1, "ok": 60, "wall_s": 6.0, "throttled": 0, "resolved": 60},
    ]
    res = summarize_scaling(2, child)
    # aggregate uses the slowest (6s) -> conservative concurrent rate
    assert res.aggregate_thru == pytest.approx(100.0 / 6.0)
    # per-proc is mean of 10/s and 10/s
    assert res.per_proc_thru == pytest.approx(10.0)


def test_summarize_scaling_handles_dead_process():
    child = [{"rank": 0, "ok": 0, "wall_s": 0.0, "throttled": 0, "resolved": 0}]
    res = summarize_scaling(1, child)
    assert res.per_proc_thru == 0.0
    assert res.aggregate_thru == 0.0


def test_scale_result_efficiency():
    r = ScaleResult(procs=4, ok=1, resolved=1, throttled=0,
                    per_proc_thru=8.0, aggregate_thru=32.0)
    assert r.efficiency(10.0) == pytest.approx(0.8)
    assert r.efficiency(0.0) == 0.0


# --------------------------------------------------------------------------
# Horizontal-scaling: recommend_scaling
# --------------------------------------------------------------------------
def _scale(procs, per_proc, aggregate, throttled=0):
    return ScaleResult(procs=procs, ok=int(aggregate), resolved=int(aggregate),
                       throttled=throttled, per_proc_thru=per_proc,
                       aggregate_thru=aggregate)


def test_recommend_scaling_confirms_linear():
    # per-proc stays ~flat as procs grow -> linear scaling
    results = [
        _scale(1, per_proc=10.0, aggregate=10.0),
        _scale(2, per_proc=9.8, aggregate=19.6),
        _scale(4, per_proc=9.5, aggregate=38.0),
    ]
    msg = recommend_scaling(results)
    assert "CONFIRMED" in msg


def test_recommend_scaling_flags_saturation():
    # per-proc collapses at 4 procs -> shared bottleneck
    results = [
        _scale(1, per_proc=10.0, aggregate=10.0),
        _scale(2, per_proc=9.0, aggregate=18.0),
        _scale(4, per_proc=4.0, aggregate=16.0),
    ]
    msg = recommend_scaling(results)
    assert "SATURATES" in msg
    assert "4 processes" in msg


def test_recommend_scaling_needs_two_levels():
    assert "Need >=2" in recommend_scaling([_scale(1, 10.0, 10.0)])


# --------------------------------------------------------------------------
# Horizontal-scaling: run_horizontal with injected thread executor + fake worker
# (no real processes / network)
# --------------------------------------------------------------------------
def test_run_horizontal_with_injected_worker():
    calls = []

    def fake_worker(spec):
        calls.append(spec["rank"])
        # each "process" does 20 ok reads in 2s regardless of P -> perfect scaling
        return {"rank": spec["rank"], "ok": 20, "wall_s": 2.0,
                "throttled": 0, "resolved": 20}

    results = run_horizontal(
        [1, 2, 3], n_points=20, workers=16, crop_m=200.0, out_pixels=224,
        nbands=4, year_hint=None, seed=0, worker_fn=fake_worker,
        executor_factory=lambda mw: ThreadPoolExecutor(max_workers=mw),
    )
    assert [r.procs for r in results] == [1, 2, 3]
    # one worker call per process across all three levels: 1 + 2 + 3 = 6
    assert len(calls) == 6
    # per-proc throughput is flat (10/s) -> recommend confirms scaling
    assert all(r.per_proc_thru == pytest.approx(10.0) for r in results)
    assert "CONFIRMED" in recommend_scaling(results)


def test_run_horizontal_spawns_correct_process_counts():
    seen_workers = []

    class _RecordingExec:
        def __init__(self, mw):
            seen_workers.append(mw)
            self._ex = ThreadPoolExecutor(max_workers=mw)

        def __enter__(self):
            return self._ex

        def __exit__(self, *a):
            self._ex.shutdown()
            return False

    def fake_worker(spec):
        return {"rank": spec["rank"], "ok": 1, "wall_s": 1.0,
                "throttled": 0, "resolved": 1}

    run_horizontal([2, 4], n_points=5, workers=8, crop_m=200.0, out_pixels=224,
                   nbands=4, year_hint=None, worker_fn=fake_worker,
                   executor_factory=_RecordingExec)
    assert seen_workers == [2, 4]


def test_run_horizontal_propagates_sampler_readings():
    class _FakeSampler:
        entered = 0
        exited = 0

        def __enter__(self):
            _FakeSampler.entered += 1
            return self

        def __exit__(self, *a):
            _FakeSampler.exited += 1
            return False

        cpu_pct = 42.5
        gpu_pct = 7.0

    def fake_worker(spec):
        return {"rank": spec["rank"], "ok": 4, "wall_s": 1.0,
                "throttled": 0, "resolved": 4}

    results = run_horizontal(
        [1, 2], n_points=4, workers=8, crop_m=200.0, out_pixels=224,
        nbands=4, year_hint=None, worker_fn=fake_worker,
        executor_factory=lambda mw: ThreadPoolExecutor(max_workers=mw),
        sampler_factory=_FakeSampler,
    )
    # one sampler per level, entered and exited around the pool
    assert _FakeSampler.entered == 2 and _FakeSampler.exited == 2
    assert all(r.cpu_pct == pytest.approx(42.5) for r in results)
    assert all(r.gpu_pct == pytest.approx(7.0) for r in results)


def test_run_horizontal_without_sampler_defaults_to_zero():
    def fake_worker(spec):
        return {"rank": spec["rank"], "ok": 4, "wall_s": 1.0,
                "throttled": 0, "resolved": 4}

    (res,) = run_horizontal(
        [1], n_points=4, workers=8, crop_m=200.0, out_pixels=224,
        nbands=4, year_hint=None, worker_fn=fake_worker,
        executor_factory=lambda mw: ThreadPoolExecutor(max_workers=mw),
    )
    assert res.cpu_pct == 0.0
    assert res.gpu_pct is None


def test_resource_sampler_smoke():
    # Real sampler on this host: short interval, do a little work, expect a
    # sane CPU% (0-100) and gpu_pct either None (no nvidia-smi) or 0-100.
    with ResourceSampler(interval=0.02) as s:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.1:
            sum(i * i for i in range(1000))
    assert 0.0 <= s.cpu_pct <= 100.0
    assert s.gpu_pct is None or 0.0 <= s.gpu_pct <= 100.0


# --------------------------------------------------------------------------
# Read-granularity: sample_cluster
# --------------------------------------------------------------------------
def test_sample_cluster_deterministic_and_sized():
    a = sample_cluster(30, (-112.07, 33.45), 2.0, seed=1)
    b = sample_cluster(30, (-112.07, 33.45), 2.0, seed=1)
    assert a == b
    assert len(a) == 30


def test_sample_cluster_within_radius():
    from math import cos, radians, hypot
    center = (-112.07, 33.45)
    r_km = 2.0
    pts = sample_cluster(200, center, r_km, seed=2)
    for lon, lat in pts:
        dlat_km = (lat - center[1]) * 111.0
        dlon_km = (lon - center[0]) * 111.0 * cos(radians(center[1]))
        assert hypot(dlat_km, dlon_km) <= r_km + 1e-6


# --------------------------------------------------------------------------
# Read-granularity: pick_shared_doqq
# --------------------------------------------------------------------------
class _DoqqRef:
    def __init__(self, item_id, failure=None, bbox=(0, 0, 1, 1),
                 href="h", use_cache=False):
        self.item_id = item_id
        self.failure = failure
        self.bbox = bbox
        self.href = href
        self.use_cache = use_cache


def test_pick_shared_doqq_returns_largest_group():
    ids = ["A", "A", "B", "A", "B", None]

    def fake_resolve(lon, lat, crop, year_hint):
        i = int(lon)
        return _DoqqRef(ids[i], failure="no_items" if ids[i] is None else None)

    points = [(float(i), 0.0) for i in range(len(ids))]
    item_id, refs = pick_shared_doqq(points, 200.0, None, resolve_fn=fake_resolve)
    assert item_id == "A"
    assert len(refs) == 3
    assert all(r.item_id == "A" for r in refs)


def test_pick_shared_doqq_all_failures():
    def fake_resolve(lon, lat, crop, year_hint):
        return _DoqqRef(None, failure="no_items")

    item_id, refs = pick_shared_doqq([(0.0, 0.0)], 200.0, None,
                                     resolve_fn=fake_resolve)
    assert item_id is None
    assert refs == []


# --------------------------------------------------------------------------
# Read-granularity: run_per_crop / run_union with injected fakes
# --------------------------------------------------------------------------
def test_run_per_crop_reads_each_and_opens_once():
    opens = []

    @contextmanager
    def fake_open(href, use_cache):
        opens.append(href)
        yield object()

    reads = []

    def fake_read(src, bbox, nbands, out_pixels):
        reads.append(bbox)
        return ("crop", False, False, None)

    refs = [_DoqqRef("A", bbox=(0, 0, 1, 1)), _DoqqRef("A", bbox=(1, 1, 2, 2)),
            _DoqqRef("A", bbox=(2, 2, 3, 3))]
    res = run_per_crop(refs, 4, 224, open_fn=fake_open, read_fn=fake_read)
    assert res.strategy == "per_crop"
    assert res.crops == 3
    assert len(opens) == 1        # DOQQ opened exactly once
    assert len(reads) == 3        # one read per ref


def test_run_per_crop_counts_only_successes():
    @contextmanager
    def fake_open(href, use_cache):
        yield object()

    def fake_read(src, bbox, nbands, out_pixels):
        # fail the read whose bbox starts at 9
        if bbox[0] == 9:
            return (None, False, False, "read_error")
        return ("crop", False, False, None)

    refs = [_DoqqRef("A", bbox=(0, 0, 1, 1)), _DoqqRef("A", bbox=(9, 9, 10, 10))]
    res = run_per_crop(refs, 4, 224, open_fn=fake_open, read_fn=fake_read)
    assert res.crops == 1


def test_run_union_reports_mpix_and_crops():
    @contextmanager
    def fake_open(href, use_cache):
        yield object()

    class _FakeMem:
        width = 2000
        height = 1500  # 3.0 MP

    @contextmanager
    def fake_mosaic(src, bboxes, nbands, max_union_pixels):
        yield _FakeMem()

    def fake_read(src, bbox, nbands, out_pixels):
        return ("crop", False, False, None)

    refs = [_DoqqRef("A", bbox=(0, 0, 1, 1)) for _ in range(4)]
    res = run_union(refs, 4, 224, 67_000_000,
                    open_fn=fake_open, mosaic_fn=fake_mosaic, read_fn=fake_read)
    assert res.strategy == "union"
    assert res.crops == 4
    assert res.union_mpix == pytest.approx(3.0)
    assert res.mpix_per_crop == pytest.approx(0.75)
    assert res.fell_back is False


def test_run_union_flags_fallback_when_mosaic_none():
    @contextmanager
    def fake_open(href, use_cache):
        yield object()

    @contextmanager
    def fake_mosaic(src, bboxes, nbands, max_union_pixels):
        yield None  # union too large -> fall back

    refs = [_DoqqRef("A") for _ in range(3)]
    res = run_union(refs, 4, 224, 100,
                    open_fn=fake_open, mosaic_fn=fake_mosaic,
                    read_fn=lambda *a: ("c", False, False, None))
    assert res.fell_back is True
    assert res.crops == 0


def test_granularity_result_crops_per_s():
    r = GranularityResult("union", crops=50, wall_s=2.0, union_mpix=10.0)
    assert r.crops_per_s == pytest.approx(25.0)
    assert r.mpix_per_crop == pytest.approx(0.2)


def test_granularity_result_zero_wall_safe():
    r = GranularityResult("per_crop", crops=0, wall_s=0.0)
    assert r.crops_per_s == 0.0
    assert r.mpix_per_crop == 0.0


# --------------------------------------------------------------------------
# Real-tract granularity: choose_tract_ids
# --------------------------------------------------------------------------
def test_choose_tract_ids_filters_band_and_sorts_by_count():
    counts = {"a": 10, "b": 100, "c": 300, "d": 5, "e": 800}
    chosen = choose_tract_ids(counts, n_tracts=5, min_b=40, max_b=500, seed=0)
    # d(5), a(10) below min; e(800) above max -> only b,c; sorted by count asc
    assert chosen == ["b", "c"]


def test_choose_tract_ids_deterministic():
    counts = {str(i): i for i in range(50, 500, 10)}
    a = choose_tract_ids(counts, 5, 40, 500, seed=3)
    b = choose_tract_ids(counts, 5, 40, 500, seed=3)
    assert a == b
    assert len(a) == 5
    # returned ascending by count
    assert [counts[g] for g in a] == sorted(counts[g] for g in a)


def test_choose_tract_ids_empty_band():
    assert choose_tract_ids({"a": 10}, 5, 40, 500, seed=0) == []


def test_load_tracts_from_index_roundtrip(tmp_path):
    import numpy as np
    import pandas as pd
    # tract "T1": 3 buildings, "T2": 1 building (below min), "T3": 2 buildings
    df = pd.DataFrame({
        "cx": np.array([0.0, 1.0, 2.0, 5.0, 10.0, 11.0]),
        "cy": np.array([0.0, 1.0, 2.0, 5.0, 10.0, 11.0]),
        "tract_id": ["T1", "T1", "T1", "T2", "T3", "T3"],
    })
    p = tmp_path / "part.parquet"
    df.to_parquet(p)
    out = load_tracts_from_index(p, n_tracts=5, min_b=2, max_b=3, seed=0)
    ids = {geoid for geoid, _cx, _cy in out}
    assert ids == {"T1", "T3"}       # T2 (1 building) excluded
    by_id = {geoid: (cx, cy) for geoid, cx, cy in out}
    assert len(by_id["T1"][0]) == 3
    assert len(by_id["T3"][0]) == 2


def test_load_chunk_from_index_shape_and_columns(tmp_path):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({
        "cx": np.arange(20, dtype="float64"),
        "cy": np.arange(20, dtype="float64") * 2,
        "tract_id": [f"T{i // 4}" for i in range(20)],  # 5 tracts of 4
    })
    p = tmp_path / "part.parquet"
    df.to_parquet(p)
    chunk = load_chunk_from_index(p, chunk_size=8, year=2018, seed=1)
    assert len(chunk) == 8
    assert set(["centroid_x", "centroid_y", "GEOID", "year"]).issubset(chunk.columns)
    assert (chunk["year"] == 2018).all()
    # GEOID-sorted contiguous window -> at most a couple distinct tracts
    assert chunk["GEOID"].is_monotonic_increasing
