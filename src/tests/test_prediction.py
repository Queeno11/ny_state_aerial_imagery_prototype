"""Unit tests for the NAIP prediction engine (no GPU, no network, no sleeps).

Covers src/prediction.py + the backoff helper in src/data/naip_fetcher.py:
retry classification (transient vs permanent), cooldown escalation, chunk
resume, atomicity, the manifest guard, split typing, and output schema.
"""

import datetime
import json
import threading
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src import prediction
from src.data.naip_fetcher import NaipFetchResult, NaipItemRef, backoff_sleep


# ── Test doubles ─────────────────────────────────────────────────────────────

class IdentityTransformer:
    """Stands in for the METRIC_CRS -> EPSG:4326 pyproj Transformer; rows are
    then identified inside FakeFetch by lon == centroid_x == row id."""

    def transform(self, x, y):
        return x, y


class FakeFetch:
    """fetch_naip stand-in scripted per row id (int(lon)).

    ``script[row_id]`` is a list of outcomes consumed one per attempt:
    "ok", "ok@<year>", or a failure mode. Unscripted rows (or exhausted
    scripts) succeed. Thread-safe; records every call's row id.
    """

    def __init__(self, script=None, nbands=3, out_pixels=8, default_year=2019):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []
        self.exact_years = []            # exact_year flag seen per call
        self.nbands = nbands
        self.out_pixels = out_pixels
        self.default_year = default_year
        self._lock = threading.Lock()

    def _crop(self, row_id):
        return np.full((self.nbands, self.out_pixels, self.out_pixels),
                       row_id % 255, dtype=np.uint8)

    def __call__(self, lon, lat, crop_size_meters, nbands, out_pixels,
                 year_hint, search_cache=None, cache_key=None,
                 exact_year=False):
        row_id = int(round(lon))
        with self._lock:
            self.calls.append(row_id)
            self.exact_years.append(exact_year)
            outcomes = self.script.get(row_id)
            outcome = outcomes.pop(0) if outcomes else "ok"
        if outcome.startswith("ok"):
            year = int(outcome.split("@")[1]) if "@" in outcome else self.default_year
            return NaipFetchResult(self._crop(row_id), year)
        # Permanent modes may still know the item year (asset_missing does);
        # keep it None for simplicity, as with no_items/search_error.
        return NaipFetchResult(None, None, failure=outcome)

    def n_calls(self, row_id):
        return sum(1 for c in self.calls if c == row_id)


# Grouped-path doubles: resolve_fn / open_fn / read_fn seams. Rows are keyed by
# row id (int(lon) on resolve; int(bbox[0]) on read, since resolve stashes the
# row id in bbox). DOQQ grouping is row_id // group_size, so consecutive rows
# share an item and the open should be reused.

class FakeResolve:
    """resolve_naip_item stand-in: assigns each row a DOQQ item + scripts the
    resolve-phase failures (search_error / no_items / asset_missing)."""

    def __init__(self, group_size=4, script=None, default_year=2019):
        self.group_size = group_size
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []
        self.exact_years = []            # exact_year flag seen per call
        self.default_year = default_year
        self._lock = threading.Lock()

    def __call__(self, lon, lat, crop_size_meters, year_hint=None,
                 search_cache=None, cache_key=None, exact_year=False):
        row_id = int(round(lon))
        with self._lock:
            self.calls.append(row_id)
            self.exact_years.append(exact_year)
            outcomes = self.script.get(row_id)
            outcome = outcomes.pop(0) if outcomes else "ok"
        item_id = str(row_id // self.group_size)
        bbox = [float(row_id)] * 4          # row id smuggled to read_fn
        if outcome == "ok":
            return NaipItemRef(item_id, f"href-{item_id}", True,
                               self.default_year, bbox, None)
        # asset_missing knows the item/year; no_items/search_error don't.
        if outcome == "asset_missing":
            return NaipItemRef(item_id, None, True, self.default_year, bbox, outcome)
        return NaipItemRef(None, None, True, None, bbox, outcome)

    def n_calls(self, row_id):
        return sum(1 for c in self.calls if c == row_id)


class FakeOpen:
    """open_naip_src stand-in: a context manager recording every opened href so
    tests can assert open-reuse. Hrefs in ``fail_hrefs`` raise on enter."""

    def __init__(self, fail_hrefs=()):
        self.opens = []
        self.fail_hrefs = set(fail_hrefs)
        self._lock = threading.Lock()

    @contextmanager
    def __call__(self, href, use_cache):
        with self._lock:
            self.opens.append(href)
        if href in self.fail_hrefs:
            raise IOError(f"open failed: {href}")
        yield ("src", href)

    def n_opens(self):
        return len(self.opens)


class FakeRead:
    """read_naip_crop_from_src stand-in: scripts per-row read outcomes and
    returns a crop encoding the row id (mean = row_id, so TinyModel = row_id/255)."""

    def __init__(self, script=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.reads = []
        self._lock = threading.Lock()

    def __call__(self, src, bbox, nbands, out_pixels):
        row_id = int(round(bbox[0]))
        with self._lock:
            self.reads.append(row_id)
            outcomes = self.script.get(row_id)
            outcome = outcomes.pop(0) if outcomes else "ok"
        if outcome == "ok":
            crop = np.full((nbands, out_pixels, out_pixels), row_id % 255,
                           dtype=np.uint8)
            return crop, False, False, None
        return None, False, False, outcome

    def n_reads(self, row_id):
        return sum(1 for r in self.reads if r == row_id)


class TinyModel(nn.Module):
    """Deterministic stand-in: prediction = mean of the (scaled) crop."""

    def forward(self, x, metadata=None):
        return x.mean(dim=(1, 2, 3))


def _eval_transform(t):
    return t.float() / 255.0


def make_df(n, year=2020, cbsa="35620", nan_rows=(), geoid_group=4):
    """Synthetic per-year frame; centroid_x doubles as the row id."""
    rel = np.linspace(-1, 1, n).astype("float32")
    rel[list(nan_rows)] = np.nan
    return pd.DataFrame({
        "building_id": np.arange(n, dtype="int64"),
        "GEOID": [f"36061{i // geoid_group:06d}" for i in range(n)],
        "cbsa_code": [cbsa] * n,
        "year": np.full(n, year, dtype="int64"),
        "type": ["test"] * n,
        "Rel_Score": rel,
        "dist_to_center": np.zeros(n, dtype="float32"),
        "centroid_x": np.arange(n, dtype="float64"),
        "centroid_y": np.zeros(n, dtype="float64"),
    })


PARAMS = {
    "tau_meters": 100,
    "image_size": 8,
    "nbands": 3,
    "batch_size": 2,          # eval batch = 16
    "predict_chunk_size": 4,
    "indicator": "W2_r5",
    "footprints_source": "ms_us",
    "states": None,
    "predict_split": "test",
    "years": [2020],
}

DEVICE = torch.device("cpu")


def run_year(df, tmp_path, fetch, *, max_passes=5, sleeps=None, **kw):
    sleep_log = sleeps if sleeps is not None else []
    return prediction.predict_year_chunked(
        TinyModel(), df, 2020, PARAMS, DEVICE, _eval_transform,
        tmp_path / "chunks", tmp_path / "2020_predictions.csv",
        search_cache=object(), to_4326=IdentityTransformer(),
        max_passes=max_passes, fetch_fn=fetch,
        sleep_fn=sleep_log.append, verbose=False, **kw,
    )


# ── backoff_sleep ────────────────────────────────────────────────────────────

def test_backoff_sleep_escalates_and_caps():
    for _ in range(20):
        slept = []
        durations = [backoff_sleep(a, base_s=30, cap_s=900, sleep_fn=slept.append)
                     for a in range(6)]
        assert slept == durations
        for attempt, d in enumerate(durations):
            nominal = min(900, 30 * 2 ** attempt)
            assert 0.75 * nominal <= d <= 1.25 * nominal
        assert durations[5] <= 1125  # capped: 900 * 1.25


# ── fetch_prediction_chunk retry classification ──────────────────────────────

def test_fetch_chunk_retries_transient_only():
    df = make_df(8)
    fetch = FakeFetch(script={4: ["no_items"], 5: ["asset_missing"],
                              6: ["search_error", "ok"],
                              7: ["read_error", "read_error", "ok"]})
    sleeps = []
    crops, actual_years, failures = prediction.fetch_prediction_chunk(
        df, PARAMS, object(), IdentityTransformer(),
        max_passes=5, fetch_fn=fetch, sleep_fn=sleeps.append, verbose=False)

    assert fetch.n_calls(4) == 1 and fetch.n_calls(5) == 1  # permanent: once
    assert failures[4] == "no_items" and failures[5] == "asset_missing"
    assert crops[6] is not None and crops[7] is not None    # transient: recovered
    assert failures[6] is None and failures[7] is None
    assert all(crops[i] is not None for i in range(4))
    assert len(sleeps) == 2  # one cooldown before pass 1 and pass 2
    assert actual_years[0] == 2019


# ── failure triage: isolated transients fast-fail, widespread keep ladder ────

def test_fast_fail_isolated_persistent_transient_no_cooldown():
    # 1 failing row out of 200 (0.5% <= 1%): one immediate retry, no sleeps,
    # then recorded and dropped — not walked through the cooldown ladder.
    df = make_df(200)
    fetch = FakeFetch(script={5: ["read_error"] * 10})
    sleeps = []
    crops, _, failures = prediction.fetch_prediction_chunk(
        df, PARAMS, object(), IdentityTransformer(),
        max_passes=5, fetch_fn=fetch, sleep_fn=sleeps.append, verbose=False)
    assert sleeps == []                      # never slept
    assert fetch.n_calls(5) == 2             # pass 0 + one immediate retry
    assert crops[5] is None and failures[5] == "read_error"
    assert sum(c is not None for c in crops) == 199


def test_fast_fail_isolated_transient_recovers_on_immediate_retry():
    df = make_df(200)
    fetch = FakeFetch(script={5: ["read_error", "ok"]})
    sleeps = []
    crops, _, failures = prediction.fetch_prediction_chunk(
        df, PARAMS, object(), IdentityTransformer(),
        max_passes=5, fetch_fn=fetch, sleep_fn=sleeps.append, verbose=False)
    assert sleeps == [] and crops[5] is not None and failures[5] is None


def test_widespread_transients_still_use_cooldown_ladder():
    # 50% failing is the throttle signature: the jittered ladder must engage
    # (fast-fail is only for isolated, row-specific problems).
    df = make_df(8)
    fetch = FakeFetch(script={i: ["read_error", "ok"] for i in range(4)})
    sleeps = []
    crops, _, _ = prediction.fetch_prediction_chunk(
        df, PARAMS, object(), IdentityTransformer(),
        max_passes=5, fetch_fn=fetch, sleep_fn=sleeps.append, verbose=False)
    assert len(sleeps) == 1                  # one cooldown before the retry
    assert all(c is not None for c in crops)


# ── eval batch size: predict_batch_size + sticky OOM halving ────────────────

class SpyModel(nn.Module):
    """TinyModel that records every batch size it sees; optionally raises a
    CUDA OOM whenever the batch exceeds ``oom_above`` (raisable on CPU too —
    it's just an exception class)."""

    def __init__(self, oom_above=None):
        super().__init__()
        self.batches = []
        self.oom_above = oom_above

    def forward(self, x, metadata=None):
        if self.oom_above is not None and x.shape[0] > self.oom_above:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        self.batches.append(x.shape[0])
        return x.mean(dim=(1, 2, 3))


def test_predict_batch_size_param_overrides_legacy(tmp_path, monkeypatch):
    monkeypatch.setitem(prediction._OOM_BATCH_CAP, "cap", None)
    model = SpyModel()
    prediction.predict_year_chunked(
        model, make_df(4), 2020, dict(PARAMS, predict_batch_size=3),
        DEVICE, _eval_transform, tmp_path / "chunks", tmp_path / "2020.csv",
        search_cache=object(), to_4326=IdentityTransformer(),
        fetch_fn=FakeFetch(), sleep_fn=lambda s: None, verbose=False)
    assert model.batches == [3, 1]           # one 4-row chunk in batches of 3


def test_oom_halves_batch_and_sticks(monkeypatch):
    monkeypatch.setitem(prediction._OOM_BATCH_CAP, "cap", None)
    crops = [np.zeros((3, 8, 8), dtype=np.uint8)] * 8
    model = SpyModel(oom_above=2)            # anything > 2 "OOMs"
    preds = prediction.predict_chunk_rows(
        model, crops, [0.0] * 8, DEVICE, _eval_transform, batch_size=8)
    assert len(preds) == 8
    assert model.batches == [2, 2, 2, 2]     # 8 -> 4 -> 2, then steady
    assert prediction._OOM_BATCH_CAP["cap"] == 2
    # A later chunk starts directly at the discovered cap, even if asked for 8.
    model2 = SpyModel(oom_above=2)
    prediction.predict_chunk_rows(
        model2, crops, [0.0] * 8, DEVICE, _eval_transform, batch_size=8)
    assert model2.batches == [2, 2, 2, 2]


# ── depth-N fetch pipeline ───────────────────────────────────────────────────

def test_pipeline_depth_preserves_output_and_order(tmp_path):
    # Deeper pipelines change scheduling only: chunk results are consumed in
    # order and the assembled CSV is identical to the depth-1 run.
    outs = {}
    for depth in (1, 3):
        df = make_df(20)                     # 5 chunks of 4
        root = tmp_path / f"d{depth}"
        prediction.predict_year_chunked(
            TinyModel(), df, 2020, dict(PARAMS, predict_fetch_pipeline=depth),
            DEVICE, _eval_transform, root / "chunks", root / "2020.csv",
            search_cache=object(), to_4326=IdentityTransformer(),
            fetch_fn=FakeFetch(), sleep_fn=lambda s: None, verbose=False)
        outs[depth] = pd.read_csv(root / "2020.csv")
    pd.testing.assert_frame_equal(outs[1], outs[3])


# ── exact-year (no flight-year substitution) ─────────────────────────────────

def test_exact_year_default_true_reaches_both_paths():
    # per-row path (fetch_fn seam)
    fetch = FakeFetch()
    prediction.fetch_prediction_chunk(
        make_df(4), PARAMS, object(), IdentityTransformer(),
        max_passes=1, fetch_fn=fetch, sleep_fn=lambda s: None, verbose=False)
    assert fetch.exact_years and all(fetch.exact_years)
    # grouped path (resolve_fn seam)
    resolve = FakeResolve()
    prediction.fetch_prediction_chunk(
        make_df(4), PARAMS, object(), IdentityTransformer(),
        max_passes=1, max_workers=2, fetch_fn=None, resolve_fn=resolve,
        open_fn=FakeOpen(), read_fn=FakeRead(), sleep_fn=lambda s: None,
        verbose=False)
    assert resolve.exact_years and all(resolve.exact_years)


def test_exact_year_can_be_disabled_via_params():
    params = dict(PARAMS, predict_exact_year=False)
    fetch = FakeFetch()
    prediction.fetch_prediction_chunk(
        make_df(4), params, object(), IdentityTransformer(),
        max_passes=1, fetch_fn=fetch, sleep_fn=lambda s: None, verbose=False)
    assert fetch.exact_years and not any(fetch.exact_years)


def test_no_year_match_is_permanent_never_retried():
    fetch = FakeFetch(script={2: ["no_year_match"]})
    crops, actual_years, failures = prediction.fetch_prediction_chunk(
        make_df(4), PARAMS, object(), IdentityTransformer(),
        max_passes=5, fetch_fn=fetch, sleep_fn=lambda s: None, verbose=False)
    assert crops[2] is None and failures[2] == "no_year_match"
    assert fetch.n_calls(2) == 1          # permanent: no retry passes
    assert all(crops[i] is not None for i in (0, 1, 3))


def test_fingerprint_pins_exact_year(tmp_path):
    # Chunks from an exact-year run must never be assembled with chunks from a
    # substitution-era run: the same row can hold a prediction from different
    # imagery. The manifest fingerprint therefore pins the flag.
    model_path = tmp_path / "best.pth"
    model_path.write_bytes(b"w")
    fp_default = prediction.prediction_fingerprint(PARAMS, model_path)
    assert fp_default["predict_exact_year"] is True   # default is exact
    fp_legacy = prediction.prediction_fingerprint(
        dict(PARAMS, predict_exact_year=False), model_path)
    assert fp_legacy["predict_exact_year"] is False
    assert fp_default != fp_legacy


# ── select_prediction_rows (evaluation sampling design) ──────────────────────

def make_sel_df(spec, year=2020):
    """Frame from ``spec`` rows of (geoid, cbsa, type, n_buildings)."""
    rows = []
    bid = 0
    for geoid, cbsa, typ, n in spec:
        for _ in range(n):
            rows.append((bid, geoid, cbsa, typ))
            bid += 1
    df = pd.DataFrame(rows, columns=["building_id", "GEOID", "cbsa_code", "type"])
    df["year"] = np.int64(year)
    df["Rel_Score"] = np.float32(0.5)
    df["dist_to_center"] = np.float32(0.0)
    df["centroid_x"] = np.arange(len(df), dtype="float64")
    df["centroid_y"] = 0.0
    return df


# Sampling knobs shrunk to test scale; NYC bypass off unless a test opts in.
SEL_PARAMS = dict(PARAMS, predict_full_universe_geoid_prefixes=(),
                  predict_tract_sample_frac=0.10,
                  predict_tract_sample_min=2,
                  predict_buildings_per_tract=5)


def test_select_tract_floor_and_frac():
    # CBSA A: 40 tracts -> max(ceil(0.1*40), 2) = 4; CBSA B: 2 tracts -> both.
    spec = ([(f"01001{i:06d}", "A", "test", 1) for i in range(40)]
            + [(f"02001{i:06d}", "B", "test", 1) for i in range(2)])
    out = prediction.select_prediction_rows(make_sel_df(spec), SEL_PARAMS,
                                            verbose=False)
    per_cbsa = out.groupby("cbsa_code")["GEOID"].nunique()
    assert per_cbsa["A"] == 4 and per_cbsa["B"] == 2


def test_select_building_cap_per_tract():
    params = dict(SEL_PARAMS, predict_tract_sample_frac=None,
                  predict_tract_sample_min=None)
    spec = [("01001000001", "A", "test", 12), ("01001000002", "A", "test", 3)]
    out = prediction.select_prediction_rows(make_sel_df(spec), params,
                                            verbose=False)
    sizes = out.groupby("GEOID").size()
    assert sizes["01001000001"] == 5 and sizes["01001000002"] == 3


def test_select_stable_across_years_and_row_order():
    # The longitudinal design needs the SAME buildings in every panel year,
    # whatever the frame's year value or row order.
    spec = ([(f"01001{i:06d}", "A", "test", 30) for i in range(25)]
            + [(f"02001{i:06d}", "B", "test", 7) for i in range(3)])
    a = prediction.select_prediction_rows(make_sel_df(spec, year=2010),
                                          SEL_PARAMS, verbose=False)
    shuffled = (make_sel_df(spec, year=2022)
                .sample(frac=1.0, random_state=7).reset_index(drop=True))
    b = prediction.select_prediction_rows(shuffled, SEL_PARAMS, verbose=False)
    assert set(a["building_id"]) == set(b["building_id"])
    assert set(a["GEOID"]) == set(b["GEOID"])


def test_select_full_universe_bypasses_split_and_sampling():
    # NYC borough rows are train-typed and exceed the building cap, yet must
    # survive whole; the test-CBSA tract is still capped.
    params = dict(SEL_PARAMS,
                  predict_full_universe_geoid_prefixes=("36061",))
    spec = [("36061000100", "35620", "train", 12),
            ("01001000001", "A", "test", 12)]
    out = prediction.select_prediction_rows(make_sel_df(spec), params,
                                            verbose=False)
    sizes = out.groupby("GEOID").size()
    assert sizes["36061000100"] == 12          # full universe: uncapped
    assert sizes["01001000001"] == 5           # sampled tier: capped
    assert set(out.loc[out["GEOID"] == "36061000100", "type"]) == {"train"}


def test_select_split_filter():
    spec = [("01001000001", "A", "test", 4), ("02001000001", "B", "val", 4)]
    df = make_sel_df(spec)
    off = dict(SEL_PARAMS, predict_tract_sample_frac=None,
               predict_tract_sample_min=None, predict_buildings_per_tract=None)
    test_only = prediction.select_prediction_rows(df, off, verbose=False)
    assert set(test_only["GEOID"]) == {"01001000001"}
    everything = prediction.select_prediction_rows(
        df, dict(off, predict_split="all"), verbose=False)
    assert len(everything) == len(df)


def test_select_all_knobs_none_reproduces_legacy_filter():
    spec = [("01001000001", "A", "test", 6), ("02001000001", "B", "train", 6)]
    df = make_sel_df(spec)
    off = dict(SEL_PARAMS, predict_tract_sample_frac=None,
               predict_tract_sample_min=None, predict_buildings_per_tract=None)
    out = prediction.select_prediction_rows(df, off, verbose=False)
    pd.testing.assert_frame_equal(out, df[df["type"] == "test"])


def test_fingerprint_pins_sampling_design(tmp_path):
    model_path = tmp_path / "best.pth"
    model_path.write_bytes(b"w")
    fp = prediction.prediction_fingerprint(PARAMS, model_path)
    assert fp["predict_tract_sample_frac"] == 0.10       # spec defaults
    assert fp["predict_tract_sample_min"] == 100
    assert fp["predict_buildings_per_tract"] == 100
    assert fp["predict_full_universe_geoid_prefixes"] == [
        "36005", "36047", "36061", "36081", "36085"]     # NYC boroughs
    fp2 = prediction.prediction_fingerprint(
        dict(PARAMS, predict_buildings_per_tract=None), model_path)
    assert fp2["predict_buildings_per_tract"] is None and fp != fp2
    json.dumps(fp2)                                       # manifest-serializable


def test_fetch_chunk_halts_on_sustained_failure():
    df = make_df(6)
    fetch = FakeFetch(script={i: ["read_error"] * 10 for i in range(6)})
    sleeps = []
    with pytest.raises(RuntimeError, match="sustained exhaustion"):
        prediction.fetch_prediction_chunk(
            df, PARAMS, object(), IdentityTransformer(),
            max_passes=3, fetch_fn=fetch, sleep_fn=sleeps.append, verbose=False)
    assert len(sleeps) == 2  # max_passes - 1 cooldowns
    assert fetch.n_calls(0) == 3


def test_fetch_chunk_tolerates_minority_leftover_failures():
    # 1/8 still transient after all passes (< halt rate): row-level, not fatal.
    df = make_df(8)
    fetch = FakeFetch(script={3: ["read_error"] * 10})
    crops, _, failures = prediction.fetch_prediction_chunk(
        df, PARAMS, object(), IdentityTransformer(),
        max_passes=3, fetch_fn=fetch, sleep_fn=lambda s: None, verbose=False)
    assert crops[3] is None and failures[3] == "read_error"
    assert sum(c is not None for c in crops) == 7


# ── predict_chunk_rows (on-device transform) ─────────────────────────────────

def test_predict_chunk_rows_empty():
    out = prediction.predict_chunk_rows(
        TinyModel(), [], [], DEVICE, _eval_transform, batch_size=4)
    assert out.shape == (0,) and out.dtype == np.float32


def test_predict_chunk_rows_batches_and_scales():
    # TinyModel returns the scaled crop mean = crop_value / 255; batching (here
    # 3 crops in batches of 2) must not change the per-row result or order.
    crops = [np.full((3, 8, 8), v, dtype=np.uint8) for v in (10, 200, 55)]
    out = prediction.predict_chunk_rows(
        TinyModel(), crops, [0.0, 0.0, 0.0], DEVICE, _eval_transform, batch_size=2)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, np.array([10, 200, 55]) / 255.0, atol=1e-5)


# ── grouped (per-DOQQ open-reuse) fetch path ─────────────────────────────────

def _grouped_chunk(df, resolve, opener, reader, *, max_workers=16, max_passes=5):
    return prediction.fetch_prediction_chunk(
        df, PARAMS, object(), IdentityTransformer(),
        max_passes=max_passes, max_workers=max_workers, fetch_fn=None,
        resolve_fn=resolve, open_fn=opener, read_fn=reader,
        sleep_fn=lambda s: None, verbose=False)


def test_grouped_opens_once_per_doqq_not_per_building():
    # 8 rows, 2 DOQQs of 4. One shard (max_workers=1) => one open per DOQQ.
    df = make_df(8)
    resolve, opener, reader = FakeResolve(group_size=4), FakeOpen(), FakeRead()
    crops, years, failures = _grouped_chunk(df, resolve, opener, reader,
                                            max_workers=1)
    assert opener.n_opens() == 2                     # not 8
    assert all(c is not None for c in crops)
    assert sorted(reader.reads) == list(range(8))    # every row read once
    assert all(y == 2019 for y in years)
    # Predictions map to the right rows (crop mean = row_id).
    assert crops[5].mean() == pytest.approx(5, abs=1e-6)


def test_grouped_parallel_shards_reuse_open_within_shard():
    # 3 DOQQs of 4; 3 workers => 3 contiguous shards, one open each.
    df = make_df(12)
    resolve, opener, reader = FakeResolve(group_size=4), FakeOpen(), FakeRead()
    crops, _, _ = _grouped_chunk(df, resolve, opener, reader, max_workers=3)
    assert opener.n_opens() == 3
    assert all(c is not None for c in crops)


def test_grouped_retries_transient_only_and_permanent_once():
    df = make_df(8)
    resolve = FakeResolve(group_size=4, script={
        4: ["no_items"], 5: ["asset_missing"], 6: ["search_error"]})
    reader = FakeRead(script={7: ["read_error", "read_error"]})
    opener = FakeOpen()
    crops, years, failures = _grouped_chunk(df, resolve, opener, reader)

    assert failures[4] == "no_items" and failures[5] == "asset_missing"
    assert crops[4] is None and crops[5] is None
    assert resolve.n_calls(4) == 1 and resolve.n_calls(5) == 1   # permanent: once
    assert crops[6] is not None and crops[7] is not None         # transient recovered
    assert failures[6] is None and failures[7] is None
    assert years[5] == 2019                                      # asset_missing keeps year


def test_grouped_open_failure_counts_as_transient_read_error():
    # A DOQQ whose COG can't be opened => every row in it is a retryable
    # read_error; a second, openable DOQQ in the same chunk still succeeds.
    df = make_df(8)                              # item "0"=rows0-3, "1"=rows4-7
    resolve = FakeResolve(group_size=4)
    opener = FakeOpen(fail_hrefs={"href-0"})     # only the first DOQQ is unreadable
    # max_passes=1: no retry; 4/8 = 50% failures is not > 0.5 halt rate.
    crops, _, failures = _grouped_chunk(df, resolve, opener, FakeRead(),
                                        max_workers=1, max_passes=1)
    assert all(crops[i] is None and failures[i] == "read_error" for i in range(4))
    assert all(crops[i] is not None for i in range(4, 8))


def test_grouped_halts_on_sustained_failure():
    df = make_df(6)
    resolve = FakeResolve()
    reader = FakeRead(script={i: ["read_error"] * 10 for i in range(6)})
    with pytest.raises(RuntimeError, match="sustained exhaustion"):
        _grouped_chunk(df, resolve, FakeOpen(), reader, max_passes=3)


def test_grouped_end_to_end_via_predict_year(tmp_path):
    df = make_df(10)
    resolve, opener, reader = FakeResolve(group_size=4), FakeOpen(), FakeRead()
    produced = prediction.predict_year_chunked(
        TinyModel(), df, 2020, PARAMS, DEVICE, _eval_transform,
        tmp_path / "chunks", tmp_path / "2020_predictions.csv",
        search_cache=object(), to_4326=IdentityTransformer(),
        fetch_fn=None, resolve_fn=resolve, open_fn=opener, read_fn=reader,
        sleep_fn=lambda s: None, verbose=False)
    assert produced is True
    csv = pd.read_csv(tmp_path / "2020_predictions.csv")
    assert len(csv) == 10
    by_bid = csv.set_index("building_id")
    # TinyModel returns scaled crop mean = row_id / 255.
    assert by_bid.loc[3, "predicted_value"] == pytest.approx(3 / 255, abs=1e-5)
    assert (csv["actual_year"] == 2019).all()
    assert (csv["year"] == 2020).all()


def test_grouped_reads_each_crop_from_shared_doqq_handle():
    # 2 DOQQs of 4, each split into 2 tracts of 2 → 4 (DOQQ, tract) runs. One
    # shard (max_workers=1): each DOQQ opened once, every crop read directly
    # from that shared handle (no mosaic materialization).
    df = make_df(8, geoid_group=2)
    resolve, opener, reader = FakeResolve(group_size=4), FakeOpen(), FakeRead()
    crops, _, _ = prediction.fetch_prediction_chunk(
        df, PARAMS, object(), IdentityTransformer(),
        max_passes=5, max_workers=1, fetch_fn=None, resolve_fn=resolve,
        open_fn=opener, read_fn=reader,
        sleep_fn=lambda s: None, verbose=False)
    assert opener.n_opens() == 2                          # one open per DOQQ
    assert all(c is not None for c in crops)
    assert sorted(reader.reads) == list(range(8))         # every row read once
    assert crops[5].mean() == pytest.approx(5, abs=1e-6)  # crop maps to its row


def test_resolve_interleaves_rows_across_search_cells():
    # GEOID-sorted rows must be resolved round-robin across search cells, so the
    # in-flight window spans distinct cells (concurrent searches) instead of
    # piling max_workers same-cell rows onto one single-flight lock. With one
    # worker the call order is exactly the interleaving. grid=3 maps centroid_x
    # 0..11 to cells i//3, i.e. one cell per make_df tract-of-3.
    df = make_df(12, geoid_group=3)                 # 4 cells of 3 buildings
    resolve = FakeResolve(group_size=100)           # DOQQ grouping irrelevant here
    prediction.fetch_prediction_chunk(
        df, dict(PARAMS, search_grid_meters=3), object(), IdentityTransformer(),
        max_passes=5, max_workers=1, fetch_fn=None, resolve_fn=resolve,
        open_fn=FakeOpen(), read_fn=FakeRead(), sleep_fn=lambda s: None,
        verbose=False)
    assert resolve.calls == [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]


def test_grouped_warms_one_stac_search_per_cell(monkeypatch):
    # The cold-resume stall: GEOID-sorted rows all hit one cell's single-flight
    # lock at once. Phase 1a must fire exactly one STAC search per DISTINCT search
    # cell (not per building), so a real TractSearchCache warms in parallel.
    # grid=3 maps centroid_x 0..11 to 4 cells of 3.
    from src.data import naip_fetcher as nf
    searches = []
    monkeypatch.setattr(nf, "_search_items",
                        lambda bbox, **kw: searches.append(bbox) or [])
    df = make_df(12, geoid_group=3)          # 12 buildings across 4 cells
    cache = nf.TractSearchCache()
    resolve, opener, reader = FakeResolve(group_size=6), FakeOpen(), FakeRead()
    prediction.fetch_prediction_chunk(
        df, dict(PARAMS, search_grid_meters=3), cache, IdentityTransformer(),
        max_workers=8, fetch_fn=None, resolve_fn=resolve, open_fn=opener,
        read_fn=reader, sleep_fn=lambda s: None, verbose=False)
    assert len(searches) == 4                 # one per cell, not one per building


def test_warm_covers_full_tract_extent_so_spread_rows_dont_research(monkeypatch):
    # A tract whose buildings span far more than the cache's 1500 m buffer: the
    # warm must search the tract's FULL extent, so the real resolver then hits
    # the cache for every row instead of re-searching the far ones under the
    # single-flight lock (the residual serial stall a representative-only warm
    # left behind).
    from src.data import naip_fetcher as nf

    class _Item:
        bbox = [-1.0, -1.0, 1.0, 1.0]
        # Flight year matches the frame's panel year (2020): this test drives
        # the REAL resolver, which under predict_exact_year (the default)
        # rejects off-year items as no_year_match.
        datetime = datetime.datetime(2020, 6, 1)
        id = "doqq"
        assets = {"image": type("A", (), {"href": "h"})()}

    searches = []
    monkeypatch.setattr(nf, "_search_items",
                        lambda bbox, **kw: searches.append(bbox) or [_Item()])

    n = 10
    df = pd.DataFrame({
        "building_id": np.arange(n, dtype="int64"),
        "GEOID": ["36061000001"] * n,                 # all one tract
        "cbsa_code": ["35620"] * n,
        "year": np.full(n, 2020, dtype="int64"),
        "type": ["test"] * n,
        "Rel_Score": np.linspace(-1, 1, n).astype("float32"),
        "dist_to_center": np.zeros(n, dtype="float32"),
        "centroid_x": np.linspace(0.0, 0.05, n),      # ~5.5 km spread (>> 1500 m)
        "centroid_y": np.zeros(n, dtype="float64"),
    })
    crops, _, _ = prediction.fetch_prediction_chunk(
        df, PARAMS, nf.TractSearchCache(), IdentityTransformer(), max_workers=8,
        fetch_fn=None, open_fn=FakeOpen(), read_fn=FakeRead(),
        sleep_fn=lambda s: None, verbose=False)
    assert len(searches) == 1                          # one full-extent search, no re-search
    assert all(c is not None for c in crops)


def test_search_cache_sized_to_chunk_avoids_eviction_thrash(monkeypatch, tmp_path):
    # A chunk can span more search cells than the cache's default 512-entry LRU.
    # If the cache is too small, the parallel warm fills it then evicts its own
    # earliest cells before the resolve reaches them → they re-search serially.
    # predict_year_chunked must size the cache to the chunk so warm-then-resolve
    # is exactly one search per cell. 600 cells here (> 512) would thrash a
    # default cache.
    from src.data import naip_fetcher as nf

    class _Item:
        bbox = [-180.0, -90.0, 180.0, 90.0]
        datetime = datetime.datetime(2018, 6, 1)
        id = "d"
        assets = {"image": type("A", (), {"href": "h"})()}

    searches = []
    monkeypatch.setattr(nf, "_search_items",
                        lambda bbox, **kw: searches.append(bbox) or [_Item()])
    nf.reset_fetch_stats()

    n_cells, per = 600, 3
    n = n_cells * per
    cell = np.arange(n) // per
    # grid=1 with integer lon/lat → each cell is its own (floor(x), floor(y));
    # a 60x10 layout gives 600 distinct cells, all valid lon/lat.
    lon = (cell % 60).astype("float64")
    lat = (cell // 60).astype("float64")
    df = pd.DataFrame({
        "building_id": np.arange(n, dtype="int64"),
        "GEOID": [f"T{c:05d}" for c in cell],
        "cbsa_code": ["X"] * n,
        "year": np.full(n, 2018, dtype="int64"),
        "type": ["test"] * n,
        "Rel_Score": np.full(n, 0.1, dtype="float32"),
        "dist_to_center": np.zeros(n, dtype="float32"),
        "centroid_x": lon,
        "centroid_y": lat,
    })
    # One chunk of 600 cells; grid=1 → one cell per T-group; no disk writes.
    params = dict(PARAMS, predict_chunk_size=n, search_grid_meters=1,
                  persist_search_cache=False)
    prediction.predict_year_chunked(
        TinyModel(), df, 2018, params, DEVICE, _eval_transform,
        tmp_path / "chunks", tmp_path / "2018.csv",
        to_4326=IdentityTransformer(), open_fn=FakeOpen(), read_fn=FakeRead(),
        sleep_fn=lambda s: None, verbose=False)
    # Exactly one STAC search per cell — no eviction-driven re-search.
    assert len(searches) == n_cells


# ── predict_year_chunked: resume & atomicity ─────────────────────────────────

def test_predict_year_end_to_end_and_resume(tmp_path):
    df = make_df(10)
    fetch = FakeFetch()
    assert run_year(df, tmp_path, fetch) is True
    csv_path = tmp_path / "2020_predictions.csv"
    first = pd.read_csv(csv_path)
    assert len(first) == 10
    assert len(fetch.calls) == 10

    # Delete the CSV only: chunks remain, so a rerun must fetch nothing and
    # reassemble the identical CSV.
    csv_path.unlink()
    fetch2 = FakeFetch()
    assert run_year(df, tmp_path, fetch2) is True
    assert fetch2.calls == []
    pd.testing.assert_frame_equal(first, pd.read_csv(csv_path))


def test_year_complete_csv_short_circuits(tmp_path):
    (tmp_path / "2020_predictions.csv").write_text("already done")
    fetch = FakeFetch()
    assert run_year(make_df(10), tmp_path, fetch) is True
    assert fetch.calls == []
    assert (tmp_path / "2020_predictions.csv").read_text() == "already done"


def test_crash_mid_year_resumes_without_refetch(tmp_path):
    df = make_df(8)  # 2 chunks of 4
    # Chunk 2 (rows 4..7) fails hard on the first run: every fetch transient.
    fetch = FakeFetch(script={i: ["read_error"] * 10 for i in range(4, 8)})
    with pytest.raises(RuntimeError):
        run_year(df, tmp_path, fetch, max_passes=2)
    year_dir = tmp_path / "chunks" / "year=2020"
    assert (year_dir / "chunk_000000000_000000004.parquet").exists()
    assert not (year_dir / "chunk_000000004_000000008.parquet").exists()
    assert not list(year_dir.glob("*.tmp"))

    # "Server recovered": rerun refetches ONLY the failed chunk's rows.
    fetch2 = FakeFetch()
    assert run_year(df, tmp_path, fetch2) is True
    assert sorted(set(fetch2.calls)) == [4, 5, 6, 7]
    assert len(pd.read_csv(tmp_path / "2020_predictions.csv")) == 8


def test_changed_dataset_size_refuses_stale_chunks(tmp_path):
    run_year(make_df(8), tmp_path, FakeFetch())
    (tmp_path / "2020_predictions.csv").unlink()
    with pytest.raises(RuntimeError, match="dataset changed"):
        run_year(make_df(9), tmp_path, FakeFetch())


# ── manifest guard ───────────────────────────────────────────────────────────

def test_manifest_mismatch_raises(tmp_path):
    model_path = tmp_path / "best.pth"
    model_path.write_bytes(b"weights")
    fp = prediction.prediction_fingerprint(PARAMS, model_path)
    root = tmp_path / "chunks"
    prediction.init_chunk_root(root, fp)
    prediction.init_chunk_root(root, fp)  # idempotent

    changed = dict(fp, indicator="income")
    with pytest.raises(RuntimeError, match="different"):
        prediction.init_chunk_root(root, changed)

    # A retrained checkpoint (new size/mtime) must also refuse.
    model_path.write_bytes(b"retrained weights!")
    with pytest.raises(RuntimeError, match="different"):
        prediction.init_chunk_root(root, prediction.prediction_fingerprint(PARAMS, model_path))
    assert json.loads((root / "manifest.json").read_text()) == fp


def test_manifest_legacy_years_key_ignored(tmp_path):
    # Manifests written before the year SET was dropped from the fingerprint
    # carry a "years" key; ADDING years to a run must not orphan its finished
    # chunks (per-year row identity is _check_year_meta's job).
    model_path = tmp_path / "best.pth"
    model_path.write_bytes(b"weights")
    fp = prediction.prediction_fingerprint(PARAMS, model_path)
    assert "years" not in fp
    root = tmp_path / "chunks"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(dict(fp, years=[2010, 2012])))
    prediction.init_chunk_root(root, fp)           # no raise: years ignored
    with pytest.raises(RuntimeError, match="different"):
        prediction.init_chunk_root(root, dict(fp, indicator="income"))


# ── output schema ────────────────────────────────────────────────────────────

def test_output_schema_failed_rows_and_actual_year(tmp_path):
    df = make_df(4)
    fetch = FakeFetch(script={1: ["no_items"], 2: ["ok@2017"]})
    run_year(df, tmp_path, fetch)

    chunk = pd.read_parquet(
        tmp_path / "chunks" / "year=2020" / "chunk_000000000_000000004.parquet")
    assert list(chunk.columns) == prediction.LEGACY_CSV_COLUMNS + [
        "actual_year", "fetch_failure"]
    failed = chunk[chunk["building_id"] == 1].iloc[0]
    assert failed["fetch_failure"] == "no_items"
    assert np.isnan(failed["predicted_value"])

    csv = pd.read_csv(tmp_path / "2020_predictions.csv")
    assert list(csv.columns) == prediction.LEGACY_CSV_COLUMNS + ["actual_year"]
    assert 1 not in set(csv["building_id"])          # failed row excluded
    assert len(csv) == 3
    by_bid = csv.set_index("building_id")
    assert by_bid.loc[2, "actual_year"] == 2017      # substituted flight year
    assert by_bid.loc[0, "actual_year"] == 2019
    assert (csv["year"] == 2020).all()               # requested year preserved
    assert csv["predicted_value"].notna().all()
    # TinyModel returns the scaled crop mean = row_id/255.
    assert by_bid.loc[3, "predicted_value"] == pytest.approx(3 / 255, abs=1e-5)


def test_nan_rel_score_rows_never_fetched(tmp_path):
    df = make_df(8, nan_rows=(2, 5))
    fetch = FakeFetch()
    run_year(df, tmp_path, fetch)
    assert 2 not in fetch.calls and 5 not in fetch.calls
    csv = pd.read_csv(tmp_path / "2020_predictions.csv")
    assert set(csv["building_id"]) == {0, 1, 3, 4, 6, 7}


def test_all_nan_year_returns_false(tmp_path):
    df = make_df(4, nan_rows=(0, 1, 2, 3))
    fetch = FakeFetch()
    assert run_year(df, tmp_path, fetch) is False
    assert fetch.calls == []
    assert not (tmp_path / "2020_predictions.csv").exists()


# ── assign_prediction_types ──────────────────────────────────────────────────

@pytest.mark.parametrize("categorical", [True, False])
def test_assign_prediction_types(categorical):
    codes = ["10100", "10100", "20200", "30300", "40400"]
    df = pd.DataFrame({
        "cbsa_code": pd.Categorical(codes) if categorical else codes,
        "Rel_Score": np.ones(5, dtype="float32"),
    })
    split_map = {"10100": "train", "20200": "val", "30300": "test"}
    holdout = {"10100": 2020}

    out = prediction.assign_prediction_types(df.copy(), 2020, split_map, holdout)
    assert list(out["type"]) == ["val_temporal", "val_temporal", "val", "test",
                                 "unassigned"]

    # Off the holdout year, train cities stay "train".
    out = prediction.assign_prediction_types(df.copy(), 2018, split_map, holdout)
    assert list(out["type"]) == ["train", "train", "val", "test", "unassigned"]

    # No split map -> untouched (no type column added).
    out = prediction.assign_prediction_types(df.copy(), 2020, None, holdout)
    assert "type" not in out.columns
