"""NAIP prediction engine: resumable, rate-limit-robust building inference.

Replaces the zarr-era ``predict_buildings_chunked`` for the NAIP/ms_us
pipeline. Design:

  - Crops come from Planetary Computer via :func:`fetch_naip`, routed through
    one :class:`TractSearchCache` per year, keyed by a coarse ~20 km grid cell
    (:func:`_search_key_of`) so one STAC search serves every tract in the cell —
    a handful of searches per chunk instead of one per tract. The cache is
    disk-backed (``persist_search_cache``), so reruns reload prior searches
    instead of re-querying STAC. Each year's frame is stable-sorted by GEOID
    before chunking so consecutive buildings share a cell and a DOQQ.
  - The unit of work and of resume is a CHUNK (``predict_chunk_size`` rows).
    Each finished chunk is written atomically (``.tmp`` -> ``os.replace``) as
    ``pred_chunks/{predict_split}/year={y}/chunk_{start}_{stop}.parquet``; a
    restart skips chunks already on disk and redoes only the one in flight.
  - Transient Planetary Computer failures (throttled search, dropped COG
    reads) are retried per chunk with a jittered exponential cooldown between
    passes; windows with no imagery (``no_items`` / ``asset_missing``) are
    recorded once and never retried. Sustained exhaustion raises so the run
    can be restarted later and resume at the same chunk.
  - Once every chunk of a year exists, the legacy per-year CSV
    (``{year}_predictions.csv``) is assembled atomically — its existence is
    the year-completion marker, and downstream consumers (evaluation.py,
    diagnose_city_coding.py, the tract dissolve in run()) are unchanged. The
    CSV gains one additive column, ``actual_year`` (the NAIP flight year the
    crop actually came from). By default (``predict_exact_year=True``)
    flight-year substitution is FORBIDDEN: a building whose panel year has no
    same-year NAIP flight fails permanently as ``no_year_match`` (NaN
    prediction in the chunk parquet, row dropped from the CSV) instead of
    silently reading the closest year's imagery — so on success
    ``actual_year == year`` always. Set ``predict_exact_year=False`` to
    restore the legacy closest-year behavior.
  - ``manifest.json`` at the chunk root fingerprints the run config + model
    checkpoint so a changed config or a retrained model can never silently
    mix outputs with an old run.

Row selection (:func:`select_prediction_rows`) implements the evaluation
sampling design: per test CBSA, ``max(ceil(10% of tracts), 100)`` tracts and
up to 100 buildings per sampled tract — enough to bound every tract mean's
95% CI at ±0.098 for σ_within = 0.5 (±0.196 worst-case σ_within = 1.0, given
the loss anchors the global cross-sectional SD at 1.0), which is invisible to
rank-order metrics. Selection is a deterministic hash of the IDs (never RNG
state), so the SAME tracts and buildings are predicted in every panel year
(the longitudinal ICC/stability metrics require it) and across restarts. The
NYC five boroughs bypass both the split filter and the sampling entirely: the
Callaway–Sant'Anna event study needs the full city universe (treatment-scale
construction is rare; sampling would underpower the DiD) and DoITT change
data is city-only.

No label substitution and no holdout guards here — those are training
concerns. Rows whose ``Rel_Score`` is NaN (unlabeled tracts, NAIP-unavailable
(region, year) gaps masked by the pair table) are skipped before any fetch,
matching the legacy predictor.
"""

import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import geo_utils
from src.utils.paths import CACHE_DIR
from src.data.naip_fetcher import (
    RETRYABLE_FAILURES,
    TractSearchCache,
    backoff_sleep,
    fetch_naip,
    get_fetch_stats,
    open_naip_src,
    read_naip_crop_from_src,
    record_failure,
    resolve_naip_item,
)

# Legacy CSV schema (order preserved); actual_year is appended after these.
LEGACY_CSV_COLUMNS = ["Rel_Score", "predicted_value", "building_id",
                      "GEOID", "year", "type"]


# --------------------------------------------------------------------------- #
# Run identity / resume layout                                                 #
# --------------------------------------------------------------------------- #

def _knob(params: dict, key: str, default, cast):
    """Resolve a nullable sampling knob: params value (or default), cast —
    None stays None (that tier disabled)."""
    v = params.get(key, default)
    return None if v is None else cast(v)


def prediction_fingerprint(params: dict, model_path: Path) -> dict:
    """Config + checkpoint identity for a prediction run.

    Model identity via file size + mtime_ns: cheap, and catches the real
    hazard (the best checkpoint overwritten by more training between runs).
    """
    st = Path(model_path).stat()
    states = params.get("states")
    return {
        "indicator": params.get("indicator"),
        "footprints_source": params.get("footprints_source"),
        "states": sorted(str(s) for s in states) if states else None,
        "tau_meters": float(params.get("tau_meters", 100)),
        "image_size": int(params["image_size"]),
        "nbands": int(params["nbands"]),
        "predict_split": params.get("predict_split", "test"),
        "predict_chunk_size": int(params.get("predict_chunk_size", 4096)),
        # Exact-year runs must never mix chunks with substitution-era runs:
        # the same row can carry a prediction from different imagery.
        "predict_exact_year": bool(params.get("predict_exact_year", True)),
        # The sampling design changes which rows exist per chunk — sampled
        # and unsampled (or differently sampled) runs must never mix.
        "predict_tract_sample_frac": _knob(
            params, "predict_tract_sample_frac", TRACT_SAMPLE_FRAC_DEFAULT, float),
        "predict_tract_sample_min": _knob(
            params, "predict_tract_sample_min", TRACT_SAMPLE_MIN_DEFAULT, int),
        "predict_buildings_per_tract": _knob(
            params, "predict_buildings_per_tract", BUILDINGS_PER_TRACT_DEFAULT, int),
        "predict_full_universe_geoid_prefixes": sorted(
            str(p) for p in (params.get("predict_full_universe_geoid_prefixes",
                                        FULL_UNIVERSE_GEOID_PREFIXES_DEFAULT)
                             or ())),
        # Deliberately NOT fingerprinted: the year SET. A chunk's content
        # depends only on its own year's frame (guarded per-year by
        # _check_year_meta), so ADDING years to a run must not invalidate
        # finished ones.
        "model_size": st.st_size,
        "model_mtime_ns": st.st_mtime_ns,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def init_chunk_root(chunk_root: Path, fingerprint: dict) -> None:
    """Create the chunk root, guarding against mixing runs.

    An existing manifest with a different fingerprint aborts: partial chunks
    from another config or another checkpoint must never be assembled into
    this run's CSVs.
    """
    chunk_root = Path(chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)
    manifest_path = chunk_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        # Legacy manifests fingerprinted the year set; it never affected
        # chunk content (per-year identity is _check_year_meta's job), so
        # ignore it — otherwise adding years would orphan finished chunks.
        existing.pop("years", None)
        if existing != fingerprint:
            diff = {k for k in set(existing) | set(fingerprint)
                    if existing.get(k) != fingerprint.get(k)}
            raise RuntimeError(
                f"Prediction chunk cache at {chunk_root} belongs to a different "
                f"run (mismatched: {sorted(diff)}). Delete that directory to "
                f"start over, or restore the original config/model."
            )
        return
    _atomic_write_text(manifest_path, json.dumps(fingerprint, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- #
# Split typing                                                                 #
# --------------------------------------------------------------------------- #

def assign_prediction_types(df_year: pd.DataFrame, year, split_map: dict | None,
                            holdout_years: dict | None = None) -> pd.DataFrame:
    """Stamp the ``type`` column from the CBSA split map (in place).

    train / val / test per ``split_map`` (keyed by cbsa_code str); train-city
    rows at their temporal-holdout year become ``val_temporal``; CBSAs absent
    from the map become ``unassigned``. When the column is categorical the
    mapping runs over the (small) category set, never the 71.8M rows.
    """
    if not split_map or df_year.empty:
        return df_year
    holdout = {str(k): int(v) for k, v in (holdout_years or {}).items()
               if not pd.isna(v)}
    year = int(year)

    def _type_of(code):
        split = split_map.get(str(code), "unassigned")
        if split == "train" and holdout.get(str(code)) == year:
            return "val_temporal"
        return split

    col = df_year["cbsa_code"]
    if isinstance(col.dtype, pd.CategoricalDtype):
        mapping = {c: _type_of(c) for c in col.cat.categories}
        df_year["type"] = col.map(mapping).astype("category")
    else:
        df_year["type"] = col.map(_type_of).astype("category")
    return df_year


# --------------------------------------------------------------------------- #
# Row selection: split filter + evaluation sampling design                      #
# --------------------------------------------------------------------------- #

# Defaults enacting the evaluation design (see select_prediction_rows).
TRACT_SAMPLE_FRAC_DEFAULT = 0.10       # per CBSA: ceil(frac * n_tracts) ...
TRACT_SAMPLE_MIN_DEFAULT = 100         # ... floored at this many tracts
BUILDINGS_PER_TRACT_DEFAULT = 100      # per sampled tract
# NYC's five borough county FIPS (Bronx, Kings, New York, Queens, Richmond):
# the CSA event study runs on the FULL city universe — treatment-scale
# construction is rare, sampling would underpower the DiD, and the DoITT
# cadastral change data it merges against is city-only.
FULL_UNIVERSE_GEOID_PREFIXES_DEFAULT = (
    "36005", "36047", "36061", "36081", "36085")


def _stable_hash(values: pd.Series) -> np.ndarray:
    """Deterministic per-row uint64 hash, stable across processes and reruns.

    ``pd.util.hash_pandas_object`` with its fixed default key — NOT Python's
    per-process-salted ``hash()`` and NOT RNG state — so the sampled subset is
    identical in every panel year (the longitudinal metrics track the same
    buildings across years) and across restarts of a resumable run. Caveat:
    the hash is a pandas implementation detail, so don't upgrade pandas
    mid-run (the manifest fingerprint can't see library versions).
    """
    return pd.util.hash_pandas_object(values, index=False).to_numpy()


def select_prediction_rows(df_year: pd.DataFrame, params: dict,
                           verbose: bool = True) -> pd.DataFrame:
    """Split filter + deterministic evaluation sampling for one year's frame.

    Three tiers (all driven by ``params``, all in the run fingerprint):

    * **Full-universe bypass** — rows whose GEOID starts with any
      ``predict_full_universe_geoid_prefixes`` prefix (default: the NYC five
      boroughs) are ALWAYS kept, regardless of split type and untouched by
      sampling. The CSA event study needs every building of the city.
    * **Split filter** — with ``predict_split='test'`` only test-CBSA rows
      survive (plus the bypass above); ``'all'`` keeps every row.
    * **Sampling** — per CBSA, the ``max(ceil(frac * n_tracts), min_tracts)``
      lowest-hash tracts (``predict_tract_sample_frac`` /
      ``predict_tract_sample_min``); within each sampled tract the
      ``predict_buildings_per_tract`` lowest-hash buildings. With the global
      cross-sectional SD anchored at 1.0 by the loss, 100 buildings bound the
      tract mean's 95% CI at ±0.098 for σ_within = 0.5 (±0.196 worst case) —
      invisible to Spearman/rank metrics. Set a knob to None to disable that
      tier (all three None + empty prefixes reproduces the legacy
      predict-everything behavior).

    Selection is a pure function of the IDs (hash-ranked, ID tiebreak), so it
    is identical across panel years, restarts, and row order. Tract ranks are
    computed within the tracts PRESENT in this year's frame: if a tract lacks
    a label in some year, its slot slides to the next-ranked tract that year —
    building membership within a tract never changes, so longitudinal metrics
    (which inner-join buildings across years) are unaffected.
    """
    if df_year.empty:
        return df_year

    prefixes = params.get("predict_full_universe_geoid_prefixes",
                          FULL_UNIVERSE_GEOID_PREFIXES_DEFAULT)
    prefixes = tuple(str(p) for p in (prefixes or ()))
    frac = params.get("predict_tract_sample_frac", TRACT_SAMPLE_FRAC_DEFAULT)
    min_tracts = params.get("predict_tract_sample_min", TRACT_SAMPLE_MIN_DEFAULT)
    per_tract = params.get("predict_buildings_per_tract",
                           BUILDINGS_PER_TRACT_DEFAULT)

    geoid = df_year["GEOID"].astype(str)
    full = (geoid.str.startswith(prefixes) if prefixes
            else pd.Series(False, index=df_year.index))
    if params.get("predict_split", "test") == "test":
        base = (df_year["type"].astype(str) == "test") & ~full
    else:
        base = ~full

    pool = pd.DataFrame({
        "GEOID": geoid[base],
        "cbsa_code": df_year.loc[base, "cbsa_code"].astype(str),
        "building_id": df_year.loc[base, "building_id"],
    })

    # Tract tier: rank tracts by hash within their CBSA, keep the top
    # max(ceil(frac * n), min) (clipped to n by construction of the rank).
    if (frac is not None or min_tracts is not None) and len(pool):
        tracts = pool[["cbsa_code", "GEOID"]].drop_duplicates()
        tracts["_h"] = _stable_hash(tracts["GEOID"])
        tracts = tracts.sort_values(["cbsa_code", "_h", "GEOID"], kind="stable")
        pos = tracts.groupby("cbsa_code", sort=False).cumcount()
        n = tracts.groupby("cbsa_code", sort=False)["GEOID"].transform("size")
        k = np.maximum(np.ceil(float(frac or 0.0) * n.to_numpy()),
                       int(min_tracts or 0))
        selected = set(tracts.loc[(pos.to_numpy() < k), "GEOID"])
        pool = pool[pool["GEOID"].isin(selected)]

    # Building tier: rank buildings by hash within their tract, keep the top
    # per_tract. Membership depends only on the IDs — exact across years.
    if per_tract is not None and len(pool):
        pool = pool.assign(_h=_stable_hash(pool["building_id"]))
        pool = pool.sort_values(["GEOID", "_h", "building_id"], kind="stable")
        pos = pool.groupby("GEOID", sort=False).cumcount()
        pool = pool[pos.to_numpy() < int(per_tract)]

    mask = full.copy()
    mask.loc[pool.index] = True
    out = df_year[mask]
    if verbose:
        print(f"    row selection: {len(df_year):,} -> {len(out):,} rows "
              f"({int(full.sum()):,} full-universe, "
              f"{len(pool):,} sampled across "
              f"{pool['GEOID'].nunique():,} tracts)")
    return out


# --------------------------------------------------------------------------- #
# Fetch with retry/cooldown                                                    #
# --------------------------------------------------------------------------- #

def _cache_key_of(row) -> str | None:
    """Tract GEOID — the read-grouping key (a tract's rows share a DOQQ handle,
    opened once). None when absent/blank."""
    geoid = row.get("GEOID")
    return (str(geoid) if geoid is not None and not pd.isna(geoid)
            and str(geoid) != "" else None)


def _search_key_of(row, grid_meters: float) -> str | None:
    """Coarse metric-grid cell — the *search* cache key.

    Snaps the building's projected centroid to a ``grid_meters`` cell so every
    tract in that cell shares ONE STAC search (a ~20 km cell spans ~10 DOQQs),
    collapsing the per-tract searches that dominated the resolve phase to a
    handful per chunk. Deliberately coarser than :func:`_cache_key_of`: the
    search wants few large queries, the read grouping wants per-tract DOQQ runs.
    None when the centroid is missing (routes to a direct, uncached search).
    """
    x, y = row.get("centroid_x"), row.get("centroid_y")
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    return f"g{int(np.floor(x / grid_meters))}_{int(np.floor(y / grid_meters))}"


def _split_contiguous(items: list, n: int) -> list[list]:
    """Split ``items`` into ``n`` contiguous, near-equal shards (order kept).

    Applied to a list already ordered by DOQQ, so each shard sees runs of same
    item and opens each COG once; only a group straddling a shard boundary is
    opened twice (≤ n-1 extra opens). ``n`` is clamped to ``[1, len]`` so the
    thread pool never gets empty shards.
    """
    if not items:
        return []
    n = max(1, min(n, len(items)))
    k, r = divmod(len(items), n)
    shards, start = [], 0
    for j in range(n):
        size = k + (1 if j < r else 0)
        shards.append(items[start:start + size])
        start += size
    return shards


def _grouped_fetch_pass(
    pending: list, records: list, to_4326, crop_size_meters: float,
    nbands: int, out_pixels: int, search_cache, resolve_fn, open_fn, read_fn,
    max_workers: int, grid_meters: float, exact_year: bool,
) -> dict:
    """One resolve-then-grouped-read pass over ``pending`` row indices.

    Returns ``{i: (crop_or_None, actual_year, failure)}``. Phase 1 resolves
    every pending row to its DOQQ (parallel; STAC search is TractSearchCache-
    backed, so this is mostly cache hits). Phase 2 orders the resolvable rows by
    (DOQQ, tract), splits them into ``max_workers`` contiguous shards, and reads
    each shard on one thread. Within a shard the COG is opened once per DOQQ and
    every crop is read directly from that shared handle. GDAL's decompressed-
    block cache (``GDAL_CACHEMAX``) already serves the overlapping ~200 m windows
    in a tract without re-downloading, so per-crop reads don't re-fetch shared
    blocks — and they beat materializing a per-tract mosaic (measured 1.6–4× on
    real tracts: ``src/probe_naip_concurrency.py --mode real-granularity``),
    because the mosaic paid an in-RAM GeoTIFF encode/decode round-trip and
    decompressed the empty space between buildings. Grouping still keeps each
    DOQQ open across its tracts (opened once), and consecutive groups in a shard
    share that handle.
    """
    # Phase 1 — resolve which DOQQ each row reads from (no pixels).
    # Resolve every centroid to lon/lat WITHOUT touching a pyproj Transformer on
    # this (background fetch) thread — Transformers are thread-affine and return
    # garbage lon/lat when used off their creation thread (→ every crop misses
    # its cell → 100% no_items). predict_year_chunked precomputes "lon"/"lat"
    # columns main-side; use those. Only when they're absent (a direct
    # fetch_prediction_chunk caller, e.g. the unit tests with a thread-safe
    # IdentityTransformer) do we transform here, in one batched call.
    if pending and "lon" in records[pending[0]]:
        lonlat = {i: (float(records[i]["lon"]), float(records[i]["lat"]))
                  for i in pending}
    else:
        xs = [records[i]["centroid_x"] for i in pending]
        ys = [records[i]["centroid_y"] for i in pending]
        _lons, _lats = to_4326.transform(xs, ys) if pending else ([], [])
        lonlat = {i: (float(_lons[k]), float(_lats[k]))
                  for k, i in enumerate(pending)}

    def _resolve_one(i):
        row = records[i]
        lon, lat = lonlat[i]
        return i, resolve_fn(
            lon=lon, lat=lat, crop_size_meters=crop_size_meters,
            year_hint=int(row["year"]), search_cache=search_cache,
            cache_key=_search_key_of(row, grid_meters),
            exact_year=exact_year)

    # Phase 1a — pre-warm the STAC search cache in parallel, one search per
    # DISTINCT search cell (coarse ~grid_meters grid; see _search_key_of), over
    # that cell's FULL extent. Rows are GEOID-sorted, so every worker in the
    # resolve pool below lands on the SAME cell at once; the cache's per-key
    # single-flight lock then lets exactly ONE cold search run while the others
    # block, collapsing the pool to serial-over-cells — the dominant, network-
    # idle stall on a cold/resumed run. Warming distinct keys up front (distinct
    # keys ⇒ no lock contention ⇒ workers search different cells at once) fixes
    # that, but ONLY if each warm search covers the whole cell: its buildings
    # span the cell, well over the cache's 1500 m buffer, so warming off a single
    # representative would leave far rows missing → re-searching under the key
    # lock → serial again. So we search each cell's centroid bounding box, which
    # (after the buffer) contains every crop in it, making the resolve pass all
    # lock-free cache hits. Talks to the cache directly: no resolve_fn
    # perturbation, and a no-op for the unit tests' cache stub (no get_items).
    if hasattr(search_cache, "get_items"):
        extents: dict = {}
        for i in pending:
            key = _search_key_of(records[i], grid_meters)
            if key is None:
                continue
            lon, lat = lonlat[i]
            e = extents.get(key)
            if e is None:
                extents[key] = [lon, lat, lon, lat]
            else:
                e[0], e[1] = min(e[0], lon), min(e[1], lat)
                e[2], e[3] = max(e[2], lon), max(e[3], lat)

        def _warm(item):
            key, e = item                 # e is the tract's [w, s, e, n] centroid box
            try:
                search_cache.get_items(key, e)   # get_items buffers it by 1500 m
            except Exception:
                pass  # a real failure resurfaces (and is counted) in the resolve pass

        if extents:
            with ThreadPoolExecutor(max_workers=min(8, len(extents))) as pool:
                for _ in pool.map(_warm, extents.items()):
                    pass

    # Resolve in a cell-interleaved order. Rows are GEOID-sorted, so mapping in
    # order puts ~max_workers rows of the SAME search cell in flight together —
    # and if its warm entry is missing (a search that failed or was too slow to
    # cache in time under sustained Planetary Computer load), all of them pile
    # onto that cell's single-flight lock while one re-searches and the rest idle
    # (the residual resolve-phase stall the warm alone can't guarantee away).
    # Round-robin across cells so the in-flight window spans max_workers DISTINCT
    # cells: their searches run concurrently, and one slow/failed search blocks
    # only its own cell's rows, never the whole pool.
    by_cell: dict = {}
    for i in pending:
        by_cell.setdefault(_search_key_of(records[i], grid_meters), []).append(i)
    interleaved = [i for rnd in itertools.zip_longest(*by_cell.values())
                   for i in rnd if i is not None]

    refs: dict = {}
    with ThreadPoolExecutor(max_workers=min(8, len(pending))) as pool:
        for i, ref in pool.map(_resolve_one, interleaved):
            refs[i] = ref

    results: dict = {}
    to_read: list = []
    for i in pending:
        ref = refs[i]
        if ref.failure is not None or ref.href is None:
            results[i] = (None, ref.actual_year, ref.failure)
        else:
            to_read.append(i)

    # Phase 2 — read crops. Collapse rows into one (DOQQ, tract) GROUP per
    # contiguous run, then shard WHOLE groups across workers. Grouping keeps a
    # tract's rows together so the DOQQ they share is opened once and every crop
    # is read from that one handle; splitting the flat row list instead (as an
    # earlier version did) scattered a tract across many shards, each re-opening
    # its DOQQ. Groups are ordered by DOQQ, so consecutive groups in a shard
    # share a COG handle (opened once).
    to_read.sort(key=lambda i: (refs[i].item_id, _cache_key_of(records[i]) or ""))
    groups: list = []
    k = 0
    while k < len(to_read):
        i0 = to_read[k]
        gkey = (refs[i0].item_id, _cache_key_of(records[i0]))
        group = [i0]
        k += 1
        while k < len(to_read) and (refs[to_read[k]].item_id,
                                    _cache_key_of(records[to_read[k]])) == gkey:
            group.append(to_read[k])
            k += 1
        groups.append(group)
    group_shards = _split_contiguous(groups, max_workers)

    def _read_group(out, src, group):
        """Read every crop in one (DOQQ, tract) group from the open handle."""
        for i in group:
            crop, _nir, _partial, failure = read_fn(
                src, refs[i].bbox, nbands, out_pixels)
            out.append((i, (crop, refs[i].actual_year, failure)))

    def _read_shard(shard):
        out = []
        cur_id, cm, src = None, None, None

        def _close():
            nonlocal cm, src, cur_id
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
            cm, src, cur_id = None, None, None

        try:
            for group in shard:
                item_id = refs[group[0]].item_id
                if src is None or item_id != cur_id:
                    _close()
                    try:
                        cm = open_fn(refs[group[0]].href, refs[group[0]].use_cache)
                        src = cm.__enter__()
                        cur_id = item_id
                    except Exception:
                        # Open failed before any per-window read counted it —
                        # fail this group's rows as a retryable read_error.
                        _close()
                        for i in group:
                            record_failure("read_error")
                            out.append((i, (None, refs[i].actual_year, "read_error")))
                        continue
                _read_group(out, src, group)
        finally:
            _close()
        return out

    if group_shards:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(group_shards))) as pool:
            for shard_out in pool.map(_read_shard, group_shards):
                for i, res in shard_out:
                    results[i] = res
    return results


def fetch_prediction_chunk(
    df_chunk: pd.DataFrame,
    params: dict,
    search_cache: "TractSearchCache | None",
    to_4326,
    *,
    max_passes: int = 5,
    backoff_base_s: float = 5.0,
    backoff_cap_s: float = 900.0,
    halt_failure_rate: float = 0.5,
    max_workers: int = 16,
    fetch_fn=None,
    resolve_fn=resolve_naip_item,
    open_fn=open_naip_src,
    read_fn=read_naip_crop_from_src,
    sleep_fn=time.sleep,
    verbose: bool = True,
):
    """Fetch every row's crop, retrying transient failures with cooldowns.

    Returns three lists aligned to ``df_chunk`` rows:
    ``(crops, actual_years, failures)`` — crop is (C, H, W) uint8 or None;
    failure is None on success, else the last failure mode.

    Two fetch mechanisms share one retry/halt orchestration:

    * default (``fetch_fn is None``): the **grouped** path — resolve every row
      to its DOQQ, then read crops grouped by DOQQ so each COG is opened once
      per shard instead of once per building (see :func:`_grouped_fetch_pass`).
    * ``fetch_fn`` given: the legacy **per-row** path (one ``fetch_naip``-style
      call per building), used by the unit tests.

    ``params["predict_exact_year"]`` (default True) forbids flight-year
    substitution on both paths: rows whose panel year has no same-year NAIP
    item fail permanently as ``no_year_match``.

    Pass 0 fetches all rows. Permanent failures (no imagery for the window,
    or none from the requested year under ``predict_exact_year``)
    are final after one attempt; retryable ones (rate-limit signature) are
    re-fetched on passes 1..max_passes-1, each preceded by a jittered
    exponential cooldown — UNLESS the failing fraction is at most
    ``params["predict_retry_fast_fail_frac"]`` (default 1%), in which case the
    stragglers get one immediate cooldown-free retry and are then recorded and
    dropped: an isolated persistent failure is a row-specific data problem,
    not throttling, and sleeping the ladder for it stalls the whole chunk.
    If, after the final pass, more than
    ``halt_failure_rate`` of the chunk is still failing transiently, raises
    RuntimeError — sustained exhaustion is an operator problem, and a restart
    resumes at exactly this chunk.
    """
    records = df_chunk.to_dict("records")
    n = len(records)
    crops: list = [None] * n
    actual_years: list = [None] * n
    failures: list = [None] * n
    if n == 0:
        return crops, actual_years, failures

    crop_size_meters = float(params.get("tau_meters", 100)) * 2
    nbands = int(params["nbands"])
    out_pixels = int(params["image_size"])
    grid_meters = float(params.get("search_grid_meters", 20000.0))
    # Never predict a panel year off another year's imagery (default): a
    # window with no same-year flight fails as no_year_match -> NaN, instead
    # of silently substituting the closest flight year.
    exact_year = bool(params.get("predict_exact_year", True))

    def _perrow_pass(pending):
        def _fetch_one(i):
            row = records[i]
            lon, lat = to_4326.transform(row["centroid_x"], row["centroid_y"])
            res = fetch_fn(
                lon=lon, lat=lat, crop_size_meters=crop_size_meters,
                nbands=nbands, out_pixels=out_pixels, year_hint=int(row["year"]),
                search_cache=search_cache, cache_key=_cache_key_of(row),
                exact_year=exact_year)
            return i, (res.crop, res.actual_year, res.failure)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
            return dict(pool.map(_fetch_one, pending))

    def _run_pass(pending):
        if fetch_fn is not None:
            return _perrow_pass(pending)
        return _grouped_fetch_pass(
            pending, records, to_4326, crop_size_meters, nbands, out_pixels,
            search_cache, resolve_fn, open_fn, read_fn, max_workers,
            grid_meters, exact_year)

    # Failure triage: a SMALL fraction of transient failures is almost never
    # throttling (which hits broadly) but a row-specific problem — a corrupt
    # tile, a flaky asset. Those get ONE immediate cooldown-free retry, then
    # are recorded and dropped: walking them through the full exponential
    # ladder costs ~a minute of pure sleep per affected chunk for rows that
    # rarely recover. A LARGE fraction is the throttle signature and keeps
    # the jittered cooldown ladder.
    fast_fail_frac = float(params.get("predict_retry_fast_fail_frac", 0.01))

    pending = list(range(n))
    fast_retries = 0
    for pass_idx in range(max_passes):
        if pass_idx > 0:
            small = len(pending) / n <= fast_fail_frac
            if small and fast_retries >= 1:
                break     # isolated stragglers already retried once: data
                          # problem, not throttling — record and move on
            if verbose:
                print(f"    ↻ pass {pass_idx}: {len(pending)}/{n} transient "
                      f"fetch failures — "
                      f"{'retrying immediately (isolated)' if small else 'cooling down before retry'}...")
            if small:
                fast_retries += 1
            else:
                backoff_sleep(pass_idx - 1, backoff_base_s, backoff_cap_s, sleep_fn)
        results = _run_pass(pending)
        still_pending = []
        for i in pending:
            crop, actual_year, failure = results[i]
            actual_years[i] = actual_year
            if crop is not None:
                crops[i] = crop
                failures[i] = None
            else:
                failures[i] = failure
                if failure in RETRYABLE_FAILURES:
                    still_pending.append(i)
        pending = still_pending
        if not pending:
            break

    if len(pending) / n > halt_failure_rate:
        from collections import Counter
        modes = dict(Counter(failures[i] for i in pending))
        sample = next((str(df_chunk.iloc[i].get("GEOID")) for i in pending
                       if df_chunk.iloc[i].get("GEOID") is not None), "?")
        raise RuntimeError(
            f"Planetary Computer still failing on {len(pending)}/{n} rows "
            f"({len(pending) / n:.0%}) after {max_passes} passes with cooldowns "
            f"— halting as sustained exhaustion. Failure modes: {modes}. "
            f"Example GEOID: {sample}. Restart later; the run resumes at this chunk."
        )
    return crops, actual_years, failures


# --------------------------------------------------------------------------- #
# GPU inference                                                                #
# --------------------------------------------------------------------------- #

# Process-wide sticky cap discovered by OOM-halving (see predict_chunk_rows):
# once a batch size OOMs under the VRAM limit, every later chunk starts at the
# size that worked, so the halving cost is paid at most a few times per run.
_OOM_BATCH_CAP = {"cap": None}


def predict_chunk_rows(model, crops: list, dist_to_center: list, device,
                       eval_transform, batch_size: int) -> np.ndarray:
    """Model predictions for a list of successfully fetched crops.

    ``crops`` must contain only valid (C, H, W) uint8 arrays; alignment with
    failed rows is the caller's job.

    Same math as the legacy consumer (stack uint8 -> scale + ImageNet
    normalize -> autocast (cuda only) -> ``model(batch, metadata=metas)``) but
    the scale+normalize runs **on-device**: the batch is stacked uint8, moved to
    the GPU as uint8 (¼ the bytes of float32), then ``eval_transform`` casts and
    normalizes there. This removes the single-threaded CPU float conversion and
    shrinks the H2D copy 4×. Outputs accumulate on-device and sync once at the
    end, not once per batch — the prediction pass is GPU-bound, so the old
    per-batch ``.cpu()`` stalls were serializing the pipeline.

    ``batch_size`` may be set aggressively: a CUDA OOM halves the batch (min
    1), remembers the working size process-wide, and retries — so an
    over-ambitious ``predict_batch_size`` costs a few wasted batches once,
    instead of crashing a multi-hour run.
    """
    if not crops:
        return np.empty(0, dtype="float32")
    model.eval()
    use_cuda = device.type == "cuda"
    bs = int(batch_size)
    if _OOM_BATCH_CAP["cap"] is not None:
        bs = min(bs, _OOM_BATCH_CAP["cap"])
    preds = []
    s = 0
    with torch.no_grad():
        while s < len(crops):
            try:
                # One contiguous uint8 stack, transferred to device before the
                # (now on-device) scale + normalize. pin_memory makes the copy
                # genuinely async under non_blocking on cuda.
                batch = torch.from_numpy(np.stack(crops[s:s + bs]))
                if use_cuda:
                    batch = batch.pin_memory()
                batch = eval_transform(batch.to(device, non_blocking=use_cuda))
                metas = torch.tensor(dist_to_center[s:s + bs],
                                     dtype=torch.float32).unsqueeze(1).to(
                                         device, non_blocking=use_cuda)
                if use_cuda:
                    with torch.autocast(device_type="cuda"):
                        out = model(batch, metadata=metas)
                else:
                    out = model(batch, metadata=metas)
                preds.append(out.view(-1).float())
                s += bs
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                _OOM_BATCH_CAP["cap"] = bs
                if use_cuda:
                    torch.cuda.empty_cache()
                print(f"    ⚠️ eval batch OOM under the VRAM cap — halving to "
                      f"{bs} (sticky for the rest of the run)")
    return torch.cat(preds).cpu().numpy()


# --------------------------------------------------------------------------- #
# Year orchestration                                                           #
# --------------------------------------------------------------------------- #

def _chunk_path(year_dir: Path, start: int, stop: int) -> Path:
    return year_dir / f"chunk_{start:09d}_{stop:09d}.parquet"


def _check_year_meta(year_dir: Path, n_rows: int, chunk_size: int) -> None:
    """Guard chunk-boundary identity across restarts.

    Chunk files are keyed by row offsets into the filtered + sorted frame; if
    the underlying dataset changed size between runs, old chunk files would
    silently misalign. Refuse to mix them.
    """
    meta_path = year_dir / "_rows.json"
    meta = {"n_rows": int(n_rows), "chunk_size": int(chunk_size)}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing != meta:
            raise RuntimeError(
                f"Existing prediction chunks in {year_dir} were built from a "
                f"frame with {existing} but this run has {meta} — the dataset "
                f"changed. Delete {year_dir} to recompute this year."
            )
        return
    _atomic_write_text(meta_path, json.dumps(meta))


def assemble_year_csv(year_dir: Path, bounds: list, output_csv: Path) -> pd.DataFrame:
    """Concatenate the year's chunk parquets into the legacy CSV (atomic).

    Rows whose fetch permanently failed are dropped (they have no
    prediction); the ``fetch_failure`` column stays in the chunk parquets for
    diagnostics but not in the CSV. Column order: legacy six + actual_year.
    """
    parts = [pd.read_parquet(_chunk_path(year_dir, s, e)) for s, e in bounds]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=LEGACY_CSV_COLUMNS + ["actual_year", "fetch_failure"])
    df = df[df["fetch_failure"].isna()].drop(columns=["fetch_failure"])
    df = df[LEGACY_CSV_COLUMNS + ["actual_year"]]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_csv.with_name(output_csv.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, output_csv)
    return df


def predict_year_chunked(
    model,
    df_year: pd.DataFrame,
    year: int,
    params: dict,
    device,
    eval_transform,
    chunk_root: Path,
    output_csv: Path,
    *,
    search_cache: "TractSearchCache | None" = None,
    to_4326=None,
    max_passes: int = 5,
    backoff_base_s: float = 5.0,
    backoff_cap_s: float = 900.0,
    halt_failure_rate: float = 0.5,
    max_workers: int = 16,
    fetch_fn=None,
    resolve_fn=resolve_naip_item,
    open_fn=open_naip_src,
    read_fn=read_naip_crop_from_src,
    sleep_fn=time.sleep,
    verbose: bool = True,
) -> bool:
    """Predict one year's buildings via NAIP, chunk by chunk, resumably.

    Returns True when ``output_csv`` exists at the end (year complete —
    possibly from a previous run), False when the year has no predictable
    rows. See the module docstring for the resume contract.
    """
    output_csv = Path(output_csv)
    year = int(year)
    if output_csv.exists():
        if verbose:
            print(f"✅ {output_csv.name} already exists — year {year} complete, skipping.")
        return True

    df = df_year[df_year["Rel_Score"].notna()]
    if df.empty:
        if verbose:
            print(f"No labeled rows to predict for {year}.")
        return False
    # Stable GEOID sort: consecutive rows share a tract (hence a search cell and
    # usually a DOQQ), so the TractSearchCache answers almost every STAC search
    # from one coarse-grid query, and each shared DOQQ is opened once per tract.
    df = df.sort_values("GEOID", kind="stable").reset_index(drop=True)

    n = len(df)
    chunk_size = int(params.get("predict_chunk_size", 4096))
    # Eval batch: explicit predict_batch_size wins; else the legacy 8x the
    # training batch. Oversized values are safe — predict_chunk_rows halves
    # on OOM and remembers the working size.
    eval_bs = params.get("predict_batch_size")
    batch_size = (int(eval_bs) if eval_bs
                  else int(params.get("batch_size", 32)) * 8)
    bounds = [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]

    year_dir = Path(chunk_root) / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    _check_year_meta(year_dir, n, chunk_size)
    todo = [(s, e) for s, e in bounds if not _chunk_path(year_dir, s, e).exists()]

    if verbose:
        print(f"Year {year}: {n:,} rows -> {len(bounds)} chunks "
              f"({len(bounds) - len(todo)} already done, {len(todo)} to go)")

    if todo:
        if search_cache is None:
            # Size the in-memory tier to hold at least a whole chunk's search
            # cells. A chunk has at most chunk_size rows, so at most chunk_size
            # distinct cells; if the LRU is smaller, the parallel pre-warm in
            # _grouped_fetch_pass fills it and then evicts its own earliest cells
            # before the resolve reaches them — every evicted cell then re-searches
            # serially under its single-flight lock, the exact stall this cache
            # exists to prevent. Disk persistence (below) makes reruns after the
            # first nearly search-free: NAIP item lists for a fixed grid cell are
            # stable, so a resumed run reloads them instead of re-querying STAC.
            grid_meters = float(params.get("search_grid_meters", 20000.0))
            cache_dir = params.get("search_cache_dir")
            if cache_dir is None and params.get("persist_search_cache", True):
                cache_dir = CACHE_DIR / "naip_search_cache" / f"grid{int(grid_meters)}"
            search_cache = TractSearchCache(max_entries=max(512, chunk_size),
                                            cache_dir=cache_dir)
        if to_4326 is None:
            from pyproj import Transformer
            to_4326 = Transformer.from_crs(geo_utils.METRIC_CRS, "EPSG:4326",
                                           always_xy=True)

        # Project ALL centroids to lon/lat HERE, in the main thread, once. The
        # per-chunk fetch runs on a background thread, and a pyproj Transformer is
        # thread-affine — using this one across threads returns garbage lon/lat
        # (→ every crop misses its cell → 100% no_items). Precomputing the columns
        # main-side keeps pyproj single-threaded; the fetch path then just reads
        # them. (One vectorized call; adds ~nothing.)
        if "lon" not in df.columns:
            lons, lats = to_4326.transform(df["centroid_x"].to_numpy(),
                                           df["centroid_y"].to_numpy())
            df = df.assign(lon=np.asarray(lons, dtype="float64"),
                           lat=np.asarray(lats, dtype="float64"))

        def _fetch(bound):
            s, e = bound
            return fetch_prediction_chunk(
                df.iloc[s:e], params, search_cache, to_4326,
                max_passes=max_passes, backoff_base_s=backoff_base_s,
                backoff_cap_s=backoff_cap_s, halt_failure_rate=halt_failure_rate,
                max_workers=max_workers, fetch_fn=fetch_fn, resolve_fn=resolve_fn,
                open_fn=open_fn, read_fn=read_fn,
                sleep_fn=sleep_fn, verbose=verbose,
            )

        stats_before = get_fetch_stats()
        # Depth-N fetch pipeline: ``predict_fetch_pipeline`` background threads
        # each fetch one upcoming chunk while the main thread runs the model
        # and writes chunk k. Depth 1 reproduces the old two-in-flight
        # behavior; deeper pipelines smooth the huge per-chunk fetch variance
        # (an all-``no_year_match`` chunk takes seconds, a dense covered chunk
        # a minute-plus, a retry-cooldown chunk mostly sleeps) so the network
        # stays busy through slow chunks instead of the whole line stalling.
        # Results are consumed strictly in chunk order regardless of
        # completion order. RAM: each in-flight chunk holds up to
        # chunk_size * nbands * image_size^2 bytes of crops (~0.8 GB at the
        # 4096/4/224 defaults).
        depth = max(1, int(params.get("predict_fetch_pipeline", 3)))
        with ThreadPoolExecutor(max_workers=depth) as fetch_pool:
            in_flight = {j: fetch_pool.submit(_fetch, todo[j])
                         for j in range(min(depth, len(todo)))}
            pbar = tqdm(total=len(todo), desc=f"Predicting {year}", leave=False,
                        disable=not verbose)
            # Per-chunk pipeline timing, averaged and reset each stats print:
            # ``fetch_wait`` is how long the main thread *blocked* on the
            # background fetch (≈0 ⇒ fetch fully hid under GPU ⇒ GPU-bound;
            # large ⇒ fetch is the ceiling); ``gpu`` is model+write time.
            t_wait_sum = t_gpu_sum = 0.0
            since = 0
            for k, (s, e) in enumerate(todo):
                _t = time.perf_counter()
                crops, actual_years, failures = in_flight.pop(k).result()
                t_wait_sum += time.perf_counter() - _t
                nxt = k + len(in_flight) + 1
                if nxt < len(todo):
                    in_flight[nxt] = fetch_pool.submit(_fetch, todo[nxt])

                _t = time.perf_counter()
                chunk = df.iloc[s:e]
                ok_idx = [i for i, c in enumerate(crops) if c is not None]
                preds = predict_chunk_rows(
                    model, [crops[i] for i in ok_idx],
                    [float(chunk.iloc[i].get("dist_to_center", 0.0) or 0.0)
                     for i in ok_idx],
                    device, eval_transform, batch_size,
                )
                pred_full = np.full(len(chunk), np.nan, dtype="float32")
                pred_full[ok_idx] = preds

                out = pd.DataFrame({
                    "Rel_Score": chunk["Rel_Score"].to_numpy(dtype="float32"),
                    "predicted_value": pred_full,
                    "building_id": chunk["building_id"].to_numpy(),
                    "GEOID": chunk["GEOID"].astype(str).to_numpy(),
                    "year": np.full(len(chunk), year, dtype="int64"),
                    "type": chunk["type"].astype(str).to_numpy(),
                    "actual_year": pd.array(actual_years, dtype="Int64"),
                    "fetch_failure": pd.array(failures, dtype="object"),
                })
                path = _chunk_path(year_dir, s, e)
                tmp = path.with_name(path.name + ".tmp")
                out.to_parquet(tmp)
                os.replace(tmp, path)
                t_gpu_sum += time.perf_counter() - _t
                since += 1
                pbar.update(1)

                if verbose and (k % 25 == 24 or k == len(todo) - 1):
                    stats = get_fetch_stats()
                    delta = {key: stats[key] - stats_before.get(key, 0)
                             for key in stats if stats[key] != stats_before.get(key, 0)}
                    print(f"    fetch stats so far ({year}): {delta}")
                    print(f"    timing/chunk (last {since}): fetch-wait "
                          f"{t_wait_sum / since:5.1f}s | gpu+write "
                          f"{t_gpu_sum / since:5.1f}s  "
                          f"(bottleneck: "
                          f"{'GPU' if t_gpu_sum > t_wait_sum else 'FETCH'})")
                    t_wait_sum = t_gpu_sum = 0.0
                    since = 0
            pbar.close()

    df_result = assemble_year_csv(year_dir, bounds, output_csv)
    if verbose:
        print(f"✅ Year {year}: {len(df_result):,} predictions "
              f"({n - len(df_result):,} rows dropped for failed fetches) "
              f"-> {output_csv}")
    return True
