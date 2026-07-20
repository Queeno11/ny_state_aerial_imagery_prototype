##############      Configuración      ##############

### Main libraries
import os
import gc
import ctypes
import math
import time
import json
import shutil
import random
import logging
import warnings
import queue
import threading
import xarray as xr
import pandas as pd
import seaborn as sns
import geopandas as gpd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

### My modules
import src.custom_models as custom_models
import src.build_dataset as build_dataset
import src.geo_utils as geo_utils
from src.data import indicators
from src.data import cbsa_brackets
from src.data.pair_table import LazyPairTable
from src.data.tract_sampling import GradientHardnessRegistry, stable_geoid_hash
from src.debug_batch_dump import BatchImageDumper
from src.data.process_acs import PANEL_YEARS as ACS_PANEL_YEARS
from src.utils.paths import PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR, IMAGERY_ROOT
pd.set_option("display.max_columns", None)

### ML libraries
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from torch.amp import autocast, GradScaler # Add this to the top of main.py
from torch.utils.data import Dataset, DataLoader

### HARDCODED PARAMETERS
# Set a VRAM hard limit (e.g., 6.0GB to be safe and avoid driver crashes on 8GB GPUs)
limit_in_bytes = 7.0 * 1024**3 
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # Cap at a maximum of 80% to leave sufficient overhead for OS and other applications
    fraction = min(0.80, limit_in_bytes / total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"--- 🛡️ Dynamic VRAM safety limit set to {fraction:.2%} ({limit_in_bytes / 1024**3:.2f} GB) of total {total_memory / 1024**3:.2f} GB ---")

os.environ['WANDB_API_KEY'] = os.getenv("WANDB_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Define a subset of the data that will comfortably fit in RAM cache
CACHE_SIZE = 2048*4 # Around 8000k images (4 batch size)

def generate_savename(run_id=None):
    """Short run identifier: ``params["run_id"]`` if provided, else ``run_{YYYYMMDD}``.

    The full config already lives in wandb (``config=params``), so the savename no
    longer encodes hyperparameters. Note the couplings: wandb resumes on
    ``id=savename`` and checkpoints/split feathers are keyed by it — to resume a
    run started on another day (or a pre-refactor run), pass its exact string as
    ``params["run_id"]``.
    """
    if run_id:
        return str(run_id)
    return f"run_{datetime.now().strftime('%Y%m%d')}"


def _cache_marker_path(cache_dir=None):
    """Path of the marker recording which run (savename) built the shard caches."""
    return Path(cache_dir or CACHE_DIR) / "cache_run_marker.json"


def read_cache_marker(cache_dir=None):
    """Savename recorded by the last cache-building run, or None if absent/unreadable."""
    try:
        return json.loads(_cache_marker_path(cache_dir).read_text()).get("savename")
    except (OSError, ValueError):
        return None


def write_cache_marker(savename, cache_dir=None):
    """Record which run owns the shard caches (read back to allow crash-resume)."""
    path = _cache_marker_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"savename": savename,
                                "written_at": datetime.now().isoformat()}))


def _wandb_id_is_unusable(exc):
    """True if wandb.init failed because the requested run id can't be reused.

    wandb permanently retires a run id once it's been deleted from the web UI —
    resume="allow" can't revive it. Depending on timing this surfaces either as
    the HTTP 409 message itself ("previously created and deleted") or, because
    wandb retries the 409 internally with backoff, as an init timeout after
    init_timeout seconds. Both mean the same thing for us: pick a fresh id.
    """
    msg = str(exc)
    return "previously created and deleted" in msg or "timed out" in msg


def override_optimizer_lr(optimizer, lr):
    """Force ``lr`` on every optimizer param group; no-op when ``lr`` is None.

    Used to decay the learning rate across a resume: loading the optimizer
    state restores the checkpoint's lr, so params["learning_rate"] alone has no
    effect on resumed runs. Only the lr is touched — momenta, weight-decay and
    step counts stay intact, so training continues without a reset.
    """
    if lr is None:
        return
    old = sorted({pg.get("lr") for pg in optimizer.param_groups})
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    print(f"⚙️ Resume lr override: {old} -> {lr} on {len(optimizer.param_groups)} param group(s).")


def init_wandb_run(savename, params, wandb_resuming):
    """wandb.init keyed to savename, falling back to a fresh id if it's unusable.

    Resuming runs reuse ``id=savename`` so the chart continues; fresh runs get a
    timestamp-suffixed id so they never collide with a previously deleted one.
    savename itself still governs checkpoints/cache, so falling back to a new
    wandb id never affects local resume logic.
    """
    wandb_id = savename if wandb_resuming else f"{savename}_{datetime.now().strftime('%H%M%S')}"
    try:
        return wandb.init(
            project="urban-income-prediction",
            name=savename, config=params,
            id=wandb_id, resume="allow"
        )
    except wandb.errors.CommError as e:
        if not _wandb_id_is_unusable(e):
            raise
        fallback_id = f"{savename}_{datetime.now().strftime('%H%M%S')}_retry"
        print(f"⚠️ wandb run '{wandb_id}' is unusable ({e}); "
              f"starting a new wandb run '{fallback_id}' instead (local checkpoints/cache unaffected).")
        return wandb.init(
            project="urban-income-prediction",
            name=savename, config=params,
            id=fallback_id, resume="allow"
        )

# ══════════════════════════════════════════════════════════════════════════════
# ZARR CHUNK MANAGEMENT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class ZarrChunkCache:
    """
    Thread-safe LRU cache for zarr chunks. Holds recently-loaded chunks in RAM
    to enable reuse across buildings that span multiple chunks.

    A threading.Lock guards all mutations so multiple extraction threads can
    call get/put concurrently without data races.

    When cache exceeds max_memory_gb, oldest chunks are evicted.
    """
    def __init__(self, max_memory_gb=5.0, max_chunks=None):
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.max_chunks = max_chunks
        self.cache = {}          # {(dataset_name, chunk_row, chunk_col): numpy_array}
        self.access_order = []   # LRU order
        self.total_bytes = 0
        self.stats = {'hits': 0, 'misses': 0, 'loads': 0}
        self._lock = threading.Lock()   # protects all mutable state

    def key(self, dataset_name, chunk_row, chunk_col):
        return (dataset_name, chunk_row, chunk_col)

    def get(self, dataset_name, chunk_row, chunk_col):
        """Return cached chunk or None if not cached."""
        k = self.key(dataset_name, chunk_row, chunk_col)
        with self._lock:
            if k in self.cache:
                self.stats['hits'] += 1
                self.access_order.remove(k)
                self.access_order.append(k)
                return self.cache[k]
            self.stats['misses'] += 1
            return None

    def put(self, dataset_name, chunk_row, chunk_col, chunk_array):
        """Store chunk in cache, evicting old entries if needed.

        If another thread already inserted this key between our cache-miss check
        and this put(), we skip silently to avoid double-counting bytes.
        """
        k = self.key(dataset_name, chunk_row, chunk_col)
        with self._lock:
            if k in self.cache:
                return  # already inserted by a racing thread — skip
            chunk_bytes = chunk_array.nbytes
            self.cache[k] = chunk_array
            self.access_order.append(k)
            self.total_bytes += chunk_bytes
            self.stats['loads'] += 1
            # Evict oldest chunks until under memory limit
            while self.total_bytes > self.max_memory_bytes and len(self.cache) > 1:
                oldest_k = self.access_order.pop(0)
                evicted_bytes = self.cache[oldest_k].nbytes
                del self.cache[oldest_k]
                self.total_bytes -= evicted_bytes

    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.total_bytes = 0

    def get_stats(self):
        """Return cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            return {
                'hit_rate_pct': hit_rate,
                'total_bytes_mb': self.total_bytes / 1024**2,
                'num_chunks': len(self.cache),
                **self.stats
            }


def get_zarr_chunks_for_image(row, chunk_dims, image_size):
    """
    Compute which zarr chunks are needed to extract an image.
    
    Args:
        row: dataframe row with row_start, col_start
        chunk_dims: (nbands, chunk_h, chunk_w) from zarr metadata
        image_size: side length in pixels (assumes square images)
    
    Returns:
        List of (chunk_row, chunk_col) tuples needed to extract this image.
    """
    nbands, chunk_h, chunk_w = chunk_dims
    row_start = int(row["row_start"])
    col_start = int(row["col_start"])
    row_stop = row_start + image_size
    col_stop = col_start + image_size
    
    # Compute which chunks overlap with [row_start:row_stop, col_start:col_stop]
    chunk_rows = set()
    chunk_cols = set()
    
    # Row chunks
    first_row_chunk = row_start // chunk_h
    last_row_chunk = (row_stop - 1) // chunk_h
    for cr in range(first_row_chunk, last_row_chunk + 1):
        chunk_rows.add(cr)
    
    # Column chunks
    first_col_chunk = col_start // chunk_w
    last_col_chunk = (col_stop - 1) // chunk_w
    for cc in range(first_col_chunk, last_col_chunk + 1):
        chunk_cols.add(cc)
    
    return [(cr, cc) for cr in chunk_rows for cc in chunk_cols]


def extract_image_from_chunks(row, cache, zarr_array, all_years_datasets, chunk_dims, image_size, nbands):
    """
    Extract a full image by loading required zarr chunks from cache (or loading
    on cache-miss and storing the result).

    Both the single-chunk and multi-chunk paths now go through the shared
    ZarrChunkCache, so the preloading done by preload_zarr_chunks_async is
    actually used instead of bypassed.  The cache is thread-safe, so this
    function can be called from multiple threads concurrently.

    Args:
        row: dict-like row with row_start, col_start, dataset, building_id …
        cache: ZarrChunkCache instance (thread-safe)
        zarr_array: unused — kept for API compatibility (resolved via all_years_datasets)
        all_years_datasets: dict of all zarr datasets
        chunk_dims: (nbands, chunk_h, chunk_w)
        image_size: square image side length in pixels
        nbands: number of bands to extract

    Returns:
        numpy array of shape (nbands, image_size, image_size), or None on error.
        Returns a *copy* of the slice so callers don't hold references into the
        large cached chunk arrays.
    """
    dataset_name = row.get("dataset")
    zarr_array   = all_years_datasets[dataset_name]["value"]
    _, chunk_h, chunk_w = chunk_dims

    row_start = int(row["row_start"])
    col_start = int(row["col_start"])
    row_stop  = row_start + image_size
    col_stop  = col_start + image_size

    required_chunks = get_zarr_chunks_for_image(row, chunk_dims, image_size)

    def _load_chunk_into_cache(cr, cc):
        """Check cache; load from zarr and populate cache on miss."""
        chunk = cache.get(dataset_name, cr, cc)
        if chunk is None:
            try:
                raw = zarr_array[:nbands,
                                 cr * chunk_h:(cr + 1) * chunk_h,
                                 cc * chunk_w:(cc + 1) * chunk_w]
                chunk = raw.to_numpy()
                cache.put(dataset_name, cr, cc, chunk)
            except Exception as e:
                logging.error(
                    f"Failed to load zarr chunk ({dataset_name},{cr},{cc}): {e}"
                )
                return None
        return chunk

    if len(required_chunks) == 1:
        # ── Single-chunk path ─────────────────────────────────────────────────
        # Previously bypassed the cache entirely — now reads from cache first.
        cr, cc = required_chunks[0]
        chunk = _load_chunk_into_cache(cr, cc)
        if chunk is None:
            return None

        chunk_row_start = cr * chunk_h
        chunk_col_start = cc * chunk_w
        local_r0 = row_start - chunk_row_start
        local_c0 = col_start - chunk_col_start

        try:
            tile = chunk[:, local_r0:local_r0 + image_size,
                            local_c0:local_c0 + image_size]
            if tile.shape == (nbands, image_size, image_size):
                return tile.copy()   # copy so we don't pin the full chunk
        except Exception as e:
            logging.error(
                f"Failed single-chunk extract for building_id {row.get('building_id', '?')}: {e}"
            )
        return None

    else:
        # ── Multi-chunk path ──────────────────────────────────────────────────
        try:
            out = np.zeros((nbands, image_size, image_size), dtype=np.uint8)

            for cr, cc in required_chunks:
                chunk = _load_chunk_into_cache(cr, cc)
                if chunk is None:
                    return None

                chunk_row_start = cr * chunk_h
                chunk_col_start = cc * chunk_w

                overlap_r0 = max(row_start, chunk_row_start)
                overlap_r1 = min(row_stop,  chunk_row_start + chunk_h)
                overlap_c0 = max(col_start, chunk_col_start)
                overlap_c1 = min(col_stop,  chunk_col_start + chunk_w)

                out[:,
                    overlap_r0 - row_start : overlap_r1 - row_start,
                    overlap_c0 - col_start : overlap_c1 - col_start
                ] = chunk[:,
                          overlap_r0 - chunk_row_start : overlap_r1 - chunk_row_start,
                          overlap_c0 - chunk_col_start : overlap_c1 - chunk_col_start]

            return out
        except Exception as e:
            logging.error(
                f"Failed multi-chunk extract for building_id {row.get('building_id', '?')}: {e}"
            )
        return None


def assign_groupby_chunk_ids(df, image_size, groupby_chunk_size=2000):
    """
    Assign each building to a groupby chunk based on its image centroid.
    
    Groupby chunks are logical, large tiles (default 2000x2000 px) that organize
    the prediction workflow. Buildings are grouped by which logical chunk their
    centroid falls into. This allows handling boundary-crossing buildings efficiently.
    
    Args:
        df: DataFrame with row_start, col_start columns
        image_size: side length of image (px)
        groupby_chunk_size: logical chunk size (default 2000x2000)
    
    Returns:
        DataFrame with added 'groupby_chunk_id' column (tuple: (chunk_row, chunk_col))
    """
    df = df.copy()
    
    # Compute image centroid for each building
    df['centroid_row'] = df['row_start'] + image_size / 2.0
    df['centroid_col'] = df['col_start'] + image_size / 2.0
    
    # Assign to groupby chunk using integer division
    df['groupby_chunk_id'] = df.apply(
        lambda row: (
            int(row['centroid_row'] // groupby_chunk_size),
            int(row['centroid_col'] // groupby_chunk_size)
        ),
        axis=1
    )
    
    return df


def preload_zarr_chunks_async(df_group, all_years_datasets, cache, chunk_dims, 
                              max_workers=4):
    """
    Asynchronously preload all zarr chunks needed for a groupby group.
    
    Identifies all unique zarr chunks required to extract all buildings in the group,
    then loads them in parallel using a thread pool.
    
    Args:
        df_group: DataFrame subset for one groupby chunk (all buildings in that group)
        all_years_datasets: dict of all zarr datasets
        cache: ZarrChunkCache instance
        chunk_dims: (nbands, chunk_h, chunk_w) from zarr metadata
        max_workers: number of parallel loader threads
    
    Returns:
        cache (updated with preloaded chunks)
    """
    # Identify all unique zarr chunks needed
    required_chunks_set = set()
    for _, row in df_group.iterrows():
        dataset_name = row.get("dataset")
        image_size = int(row["row_stop"] - row["row_start"])
        chunks_for_building = get_zarr_chunks_for_image(row, chunk_dims, image_size)
        for cr, cc in chunks_for_building:
            required_chunks_set.add((dataset_name, cr, cc))
    
    required_chunks = list(required_chunks_set)
    
    # Filter to chunks not already in cache
    chunks_to_load = [
        (dataset_name, cr, cc) for dataset_name, cr, cc in required_chunks
        if cache.get(dataset_name, cr, cc) is None
    ]
    
    if not chunks_to_load:
        return cache  # All chunks already cached
    
    def _load_chunk(item):
        dataset_name, cr, cc = item
        zarr_array = all_years_datasets[dataset_name]["value"]
        chunk_nbands, chunk_h, chunk_w = chunk_dims
        
        try:
            chunk_row_start = cr * chunk_h
            chunk_col_start = cc * chunk_w
            chunk = zarr_array[:chunk_nbands,
                              chunk_row_start:chunk_row_start + chunk_h,
                              chunk_col_start:chunk_col_start + chunk_w]
            return dataset_name, cr, cc, chunk.to_numpy()
        except Exception as e:
            logging.error(f"Failed to load zarr chunk ({dataset_name}, {cr}, {cc}): {e}")
            return dataset_name, cr, cc, None
    
    # Load chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for dataset_name, cr, cc, chunk_array in pool.map(_load_chunk, chunks_to_load):
            if chunk_array is not None:
                cache.put(dataset_name, cr, cc, chunk_array)
    
    return cache


class CyclicCacheManager:
    def __init__(
        self,
        df,
        all_years_datasets,
        params,
        cache_dir,
        num_shards=5,
        shard_size=20480,
        single_shard_mode=False,
        type="train",
        clear_cache=False,
        max_jitter=10,
        sat_data="aerial",
        holdout_years=None,
        hardness_registry=None,
    ):
        import pandas as pd
        self.df = df
        self.sat_data = sat_data
        # Gradient hard-tract mining (train + lazy path only): supplies the
        # per-tract sampling weights for materialize_tract_sample. None or
        # tract_sampling=False keeps the legacy cyclic building traversal.
        self.hardness_registry = hardness_registry
        # {cbsa_code (int) -> temporal-holdout year} for TRAIN cities (#28).
        # NAIP's actual-year substitution must not erode the per-city holdout:
        # train shards reject crops landing ON the holdout year; val_temporal
        # shards reject crops landing OFF it.
        self.holdout_years = holdout_years or {}

        self._is_lazy = isinstance(df, LazyPairTable)

        # Group temporal twins together so they end up in the same shard.
        # LazyPairTable is already building-major (shuffled at split time), so
        # only materialized DataFrames need the sort; the old categorical trick
        # is O(n_buildings) memory and impossible at 71.8M buildings anyway.
        if type == "train" and not self._is_lazy:
            print("Sorting dataframe to group building_ids for temporal sampling...")
            bids = self.df['building_id'].unique()
            np.random.shuffle(bids)  # Shuffle groups to maintain randomness
            cat_type = pd.CategoricalDtype(categories=bids, ordered=True)
            self.df['building_id_cat'] = self.df['building_id'].astype(cat_type)
            self.df = self.df.sort_values(['building_id_cat', 'year']).reset_index(drop=True)
            self.df.drop('building_id_cat', axis=1, inplace=True)

        self.all_years_datasets = all_years_datasets
        self.params = params
        self.max_jitter = max_jitter
        self.max_jitter_pixels = geo_utils.meters_to_pixels(self.max_jitter, 0.5, epsg_code=geo_utils.METRIC_EPSG)
        self.nbands = params["nbands"]

        if self.sat_data == "NAIP":
            from pyproj import Transformer
            self._to_4326 = Transformer.from_crs(geo_utils.METRIC_CRS, "EPSG:4326", always_xy=True)
            self.crop_size_meters = params.get("tau_meters", 100) * 2
            self.params = {**params, "subsample_step": 1}
            self.image_size = params["image_size"]

            # ACS lookup for actual-year substitution, scoped to the SELECTED
            # indicator only. (Iterating every Rel_Score_* column would also pick
            # up the other wealth variants and mis-parse their year suffix.)
            from src.build_dataset import process_acs_panel, PANEL_GEOID_COL
            from src.data import indicators
            self.indicator = params.get("indicator", indicators.DEFAULT_INDICATOR)
            panel = process_acs_panel()
            geoids = panel[PANEL_GEOID_COL].to_numpy()
            self.panel_years = sorted(
                yr for yr in ACS_PANEL_YEARS
                if indicators.score_col(self.indicator, yr) in panel.columns
            )
            self.acs_lookup = {}
            for yr in self.panel_years:
                vals = panel[indicators.score_col(self.indicator, yr)].to_numpy()
                self.acs_lookup.update({(g, yr): v for g, v in zip(geoids, vals)})
            # Counters for the actual-year -> panel-year label substitution
            # (+ crops rejected by the temporal-holdout guards)
            self._year_sub_lock = threading.Lock()
            self.year_sub_counts = {"exact": 0, "nearest_fallback": 0, "miss": 0,
                                    "holdout_reject": 0}
            # One STAC search per tract instead of per crop — Planetary
            # Computer throttles the search endpoint under per-crop load
            # (~23% dropped fetches in training). Shared across the extract
            # workers; thread-safe.
            from src.data.naip_fetcher import TractSearchCache
            self._naip_search_cache = TractSearchCache()
        else:
            if self._is_lazy:
                raise NotImplementedError(
                    "LazyPairTable (ms_us) requires sat_data='NAIP'; the zarr "
                    "path needs a materialized dataframe with row/col indices."
                )
            self.image_size = int((df["row_stop"] - df["row_start"]).min())  # Assuming all images have the same size in the raw zarr array


        self.type = type
        self.cache_dir = cache_dir / f"{self.type}_cache"
        self.num_shards = num_shards
        self.shard_size = shard_size
        self.single_shard_mode = single_shard_mode
        self.clear_cache = clear_cache
        self.active_shards = []
        self.next_shard_idx = 0
        self.bg_thread = None
        self._pending_k = 0
        self._bg_error = None       # (shard_id, exception) captured off the bg thread
        self._bg_completed = []     # shard ids fully written by the in-flight batch
        self._bg_start_idx = 0      # first shard id of the in-flight batch
        self._bg_started_at = None  # wall-clock start of the in-flight batch
        self._bg_stall_steps = 0    # consecutive step() calls with the thread still alive
        self.last_swap_count = 0    # shards swapped in by the latest step()
        self._is_initialized = False
        self.progress = 0.0
 
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Stale .pt.tmp files are partial writes from a crashed run — never valid.
        for f in self.cache_dir.glob("*.pt.tmp"):
            f.unlink()
        if self.clear_cache:
            for f in self.cache_dir.glob("*.pt"):
                f.unlink()
        else:
            self._load_existing_shards()

        # NOTE: the old df-level year/building groupby indices were removed —
        # they were write-only (HybridBatchSampler uses the SHARD-level dicts
        # rebuilt in InBatchRankingDataset.refresh()) and cost many GB at
        # 71.8M buildings.
        if self.type == "train":
            print(f"Train pair universe: {len(self.df):,} building-year rows")

    def _load_existing_shards(self):
        existing_shards = sorted(
            self.cache_dir.glob("shard_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not existing_shards:
            return

        self.active_shards = existing_shards
        self.next_shard_idx = max(
            int(p.stem.split("_")[1]) for p in existing_shards
        ) + 1
 
    def _extract_raw_image(self, row, n_bands=None, pad=0):
        if self.sat_data == "NAIP":
            from src.data.naip_fetcher import fetch_naip
            cx, cy = row["centroid_x"], row["centroid_y"]
            lon, lat = self._to_4326.transform(cx, cy)

            pad_meters = pad * 0.5 * self.params.get("subsample_step", 1)
            out_pixels = self.image_size + 2 * pad

            # Tract-keyed STAC search cache: rows without a GEOID fall back
            # to the direct per-crop search (cache_key=None disables it).
            geoid = row.get("GEOID")
            cache_key = str(geoid) if geoid is not None and not pd.isna(geoid) \
                and str(geoid) != "" else None

            res = fetch_naip(
                lon=lon, lat=lat,
                crop_size_meters=self.crop_size_meters + 2 * pad_meters,
                nbands=n_bands or self.nbands,
                out_pixels=out_pixels,
                year_hint=int(row["year"]),
                search_cache=self._naip_search_cache,
                cache_key=cache_key,
            )
            if res.crop is None:
                return None
            if res.nir_padded and self.params.get("reject_padded_nir", False):
                return None
            return (res.crop, res.actual_year)

        dataset_name = row.get("dataset")
        zarr_array = self.all_years_datasets[dataset_name]["value"]  # raw zarr array

        # Expand the extraction window by `pad` zarr pixels on each side (for RandomCrop jitter)
        tile_size = self.image_size + 2 * pad
        row_start = max(0, int(row["row_start"]) - pad)
        row_stop  = row_start + tile_size
        col_start = max(0, int(row["col_start"]) - pad)
        col_stop  = col_start + tile_size
        
        try:
            tile = zarr_array[:n_bands, row_start:row_stop, col_start:col_stop]
            if (
                tile.shape[0] == self.nbands and
                tile.shape[1] == tile_size and
                tile.shape[2] == tile_size
            ):                
                return tile.to_numpy()  # Convert from zarr array to numpy array for processing
            else:
                raise ValueError(f"Extracted tile has invalid shape: {tile.shape}. Expected ({self.nbands}, {tile_size}, {tile_size}).")
            
        except Exception as e:
            logging.error(f"Failed for building_id {row.get('building_id', '?')}: {e}")

        return None

    def _batch_subsample_and_convert(self, raw_images):
        """
        Subsample a list of raw (C, H, W) numpy arrays by taking every Nth pixel,
        where N = params["subsample_step"]. This avoids ALL interpolation/antialiasing
        artifacts by reading exact pixel values from the zarr.

        The caller must ensure raw tiles are sized so that H/N and W/N produce
        the desired model-resolution output (image_size or image_size + jitter).
        """
        step = self.params["subsample_step"]

        # Stack into (N_batch, C, H, W) uint8 and subsample
        batch = torch.stack([
            torch.from_numpy(img) for img in raw_images
        ])  # (N_batch, C, H, W) — still uint8, no float conversion needed

        subsampled = batch[:, :, ::step, ::step]  # exact pixel picking, zero interpolation

        return list(subsampled)  # list of (C, H//step, W//step) uint8 tensors

    def _shard_source_df(self, shard_id):
        """Rows feeding one shard.

        Train + lazy ms_us path with ``tract_sampling`` on: a fresh tract-first
        draw per shard (uniform over tracts ~ population-weighted), optionally
        importance-weighted by the gradient-hardness registry — sampling by
        per-example gradient norm (Katharopoulos & Fleuret 2018), lambdas per
        Burges (2010); see src/data/tract_sampling.py. Otherwise: the legacy
        cyclic building-major slice (building-count-weighted by construction).
        """
        start_idx = (shard_id * self.shard_size) % len(self.df)
        end_idx = start_idx + self.shard_size
        if self._is_lazy and self.type == "train" and self.params.get("tract_sampling", False):
            weight_lookup = (
                self.hardness_registry.weights_for
                if self.hardness_registry is not None else None
            )
            return self.df.materialize_tract_sample(
                self.shard_size,
                seed=(int(self.params.get("sampling_seed", 825)), int(shard_id)),
                weight_lookup=weight_lookup,
            )
        if self._is_lazy:
            # LazyPairTable synthesizes the flat slice on demand (cyclic).
            return self.df.materialize(start_idx, end_idx)
        if end_idx > len(self.df):
            return pd.concat([self.df.iloc[start_idx:], self.df.iloc[:end_idx % len(self.df)]])
        return self.df.iloc[start_idx:end_idx]

    def _worker_generate(self, shard_id, show_progress=False):
        step = self.params["subsample_step"]
        # Snapshot the (cumulative) holdout-reject counter so this shard's rejects
        # can be excluded from its API failure rate below.
        holdout_rejects_before = (
            self.year_sub_counts["holdout_reject"] if self.sat_data == "NAIP" else 0
        )
        jitter_pad = math.ceil(self.max_jitter_pixels / step) * step if self.type == "train" else 0

        sampled_df = self._shard_source_df(shard_id)

        if sampled_df.empty: return

        items_to_extract = [(row, pos) for pos, (_, row) in enumerate(sampled_df.iterrows())]

        valid_images, valid_scores, valid_geoids = [], [], []
        valid_years, valid_building_ids, valid_change = [], [], []
        valid_metas, valid_score_bins, valid_cbsas = [], [], []

        def _cbsa_int(r):
            """CBSA code as int for the shard tensor (0 when absent/NaN)."""
            try:
                return int(r.get("cbsa_code", 0))
            except (TypeError, ValueError):
                return 0

        CHUNK_SIZE = 8
        MAX_EXTRACT_WORKERS = 16 if self.sat_data == "NAIP" else 2

        def _extract(item):
            row, pos = item
            if pd.isna(row["Rel_Score"]): return None
            res = self._extract_raw_image(row, n_bands=self.params["nbands"], pad=jitter_pad)
            if res is None: return None
            
            if self.sat_data == "NAIP":
                raw_img, actual_year = res

                # ── Temporal-holdout guards (#28) ──
                # NAIP substitutes flight years, so the EFFECTIVE year of a crop
                # can differ from the requested one. Keep the per-city holdout
                # exact: train never sees holdout-year imagery of its own city,
                # and val_temporal only contains holdout-year imagery.
                if self.holdout_years:
                    eff_year = int(actual_year) if actual_year is not None else int(row["year"])
                    holdout = self.holdout_years.get(_cbsa_int(row))
                    if holdout is not None and (
                        (self.type == "train" and eff_year == holdout)
                        or (self.type == "val_temporal" and eff_year != holdout)
                    ):
                        with self._year_sub_lock:
                            self.year_sub_counts["holdout_reject"] += 1
                        return None

                if actual_year is not None and actual_year != int(row["year"]):
                    # NAIP returned a different flight year than requested: relabel
                    # with that year's ACS score, falling back to the NEAREST panel
                    # year when the exact one is missing (never keep the requested
                    # year's label for an image from another year).
                    geoid = row.get("GEOID", "")
                    label_year = actual_year
                    kind = "exact"
                    if (geoid, label_year) not in self.acs_lookup:
                        label_year = min(self.panel_years, key=lambda y: abs(y - actual_year))
                        kind = "nearest_fallback"
                    acs_key = (geoid, label_year)
                    new_score = self.acs_lookup.get(acs_key)
                    if new_score is not None and not pd.isna(new_score):
                        row = dict(row)
                        row["Rel_Score"] = new_score
                        row["year"] = actual_year
                    else:
                        kind = "miss"
                    with self._year_sub_lock:
                        self.year_sub_counts[kind] += 1
            else:
                raw_img = res

            return raw_img, row

        with ThreadPoolExecutor(max_workers=MAX_EXTRACT_WORKERS) as pool:
            futures = [pool.submit(_extract, item) for item in items_to_extract]
            iterator = tqdm(as_completed(futures), total=len(futures), desc=f"Generating shard {shard_id}") if show_progress else as_completed(futures)

            raw_chunk, chunk_meta = [], []
            for future in iterator:
                res = future.result()
                if res is None: continue
                raw_img, row = res
                
                raw_chunk.append(raw_img)
                chunk_meta.append(row)

                if len(raw_chunk) >= CHUNK_SIZE:
                    processed = self._batch_subsample_and_convert(raw_chunk)
                    for img_tensor, r in zip(processed, chunk_meta):
                        if img_tensor.max() > 0:
                            valid_images.append(img_tensor)
                            valid_scores.append(r['Rel_Score'])
                            # stable_geoid_hash (not builtin hash): process-salt-
                            # free, so the loss's per-tract lambdas key back to
                            # the same tract at shard generation and on resume.
                            valid_geoids.append(stable_geoid_hash(r.get('GEOID', '')))
                            valid_years.append(r['year'])
                            valid_building_ids.append(r['building_id'])
                            valid_change.append(r.get('Valid_Structural_Change', 0)) # Save change flag!
                            valid_metas.append(r.get("dist_to_center", 0.0))
                            valid_score_bins.append(r.get("score_bin", 0))
                            valid_cbsas.append(_cbsa_int(r))
                    raw_chunk.clear(); chunk_meta.clear()

            # Process remainder
            if raw_chunk:
                processed = self._batch_subsample_and_convert(raw_chunk)
                for img_tensor, r in zip(processed, chunk_meta):
                    if img_tensor.max() > 0:
                        valid_images.append(img_tensor)
                        valid_scores.append(r['Rel_Score'])
                        valid_geoids.append(stable_geoid_hash(r.get('GEOID', '')))
                        valid_years.append(r['year'])
                        valid_building_ids.append(r['building_id'])
                        valid_change.append(r.get('Valid_Structural_Change', 0))
                        valid_metas.append(r.get("dist_to_center", 0.0))
                        valid_score_bins.append(r.get("score_bin", 0))
                        valid_cbsas.append(_cbsa_int(r))

        # ── NAIP API failure rate check + observability ──
        total_attempted = len(items_to_extract)
        total_succeeded = len(valid_images)
        total_failed = total_attempted - total_succeeded

        if self.sat_data == "NAIP" and total_attempted > 0:
            # Holdout-guard rejects are deliberate filtering (train/val_temporal
            # discard crops on the wrong effective year), NOT fetch failures:
            # exclude them from the rate or a val_temporal shard where most
            # flight years miss the holdout year trips the rate-limit halt.
            holdout_rejects = self.year_sub_counts["holdout_reject"] - holdout_rejects_before
            api_attempted = max(total_attempted - holdout_rejects, 1)
            failure_rate = (total_failed - holdout_rejects) / api_attempted
            fetch_stats = {}
            try:
                from src.data.naip_fetcher import get_fetch_stats
                fetch_stats = get_fetch_stats()
            except Exception:
                pass
            stats_msg = (
                f"shard {shard_id}: {total_succeeded}/{total_attempted} ok "
                f"({failure_rate:.1%} failed of {api_attempted} non-holdout-rejected, "
                f"{holdout_rejects} holdout rejects) | fetcher totals: {fetch_stats} | "
                f"year substitution: {dict(self.year_sub_counts)}"
            )
            print(f"[NAIP] {stats_msg}")
            try:
                if wandb.run is not None:
                    wandb.log({
                        "naip/shard_failure_rate": failure_rate,
                        **{f"naip/fetch_{k}": v for k, v in fetch_stats.items()},
                        **{f"naip/year_sub_{k}": v for k, v in self.year_sub_counts.items()},
                    })
            except Exception:
                pass
            if failure_rate > 0.50:
                # Option (a): Hard halt
                raise RuntimeError(
                    f"🚨 NAIP {stats_msg}. "
                    f"Likely Planetary Computer API rate limit exhaustion. "
                    f"Halting training — restart after cooldown."
                )
                # Option (b): Sleep and retry
                # import logging
                # logging.error(f"⚠️ NAIP shard {shard_id}: {failure_rate:.0%} failure rate. Sleeping 5 min...")
                # import time
                # time.sleep(300)
                # return self._worker_generate(shard_id, show_progress)


        # Write to a .tmp then atomically rename: a crash mid-save can never leave
        # a truncated shard_*.pt that a resumed run would pick up as valid.
        shard_path = self.cache_dir / f"shard_{shard_id}.pt"
        tmp_path = shard_path.with_suffix(".pt.tmp")
        if valid_images:
            torch.save({
                "images": torch.stack(valid_images),
                "scores": torch.tensor(valid_scores, dtype=torch.float32),
                "geoids": torch.tensor(valid_geoids, dtype=torch.int64),
                "years": torch.tensor(valid_years, dtype=torch.int64),
                "building_ids": torch.tensor(valid_building_ids, dtype=torch.int64),
                "structural_change": torch.tensor(valid_change, dtype=torch.int64), # Replaces twin/pair logic
                "metas": torch.tensor(valid_metas, dtype=torch.float32).unsqueeze(1),
                "score_bins": torch.tensor(valid_score_bins, dtype=torch.int64),
                "cbsa_ids": torch.tensor(valid_cbsas, dtype=torch.int64),
            }, tmp_path)
        else:
            torch.save({"images": torch.empty(0), "scores": torch.empty(0)}, tmp_path)
        os.replace(tmp_path, shard_path)

    def build_initial_cache(self):
        """Synchronously generates the initial num_shards shards. Call once before training.

        Resume-aware: pre-existing shards (clear_cache=False) are kept and only the
        missing ones are generated, so a run that crashed mid-build doesn't refetch
        the NAIP tiles already on disk.
        """
        if self.active_shards and not self.clear_cache:
            if len(self.active_shards) >= self.num_shards:
                print(
                    f"Using existing {len(self.active_shards)} pre-built cache shard(s) in {self.cache_dir}"
                )
                self._is_initialized = True
                return
            print(
                f"Resuming partial cache: {len(self.active_shards)}/{self.num_shards} "
                f"shard(s) already in {self.cache_dir}, generating the rest..."
            )
        else:
            print(f"Building initial {self.num_shards} cache shards...")

        while len(self.active_shards) < self.num_shards:
            # True: Show progress bar during initial blocking setup
            self._worker_generate(self.next_shard_idx, show_progress=True)
            self.active_shards.append(self.cache_dir / f"shard_{self.next_shard_idx}.pt")
            self.next_shard_idx += 1
        self._is_initialized = True
        print("Initial cache ready.")
 
    def _worker_generate_k(self, start_idx, k, show_progress=False):
        """Generates k consecutive shards sequentially in a background thread.

        Progress is recorded in ``self._bg_completed`` and any exception is
        captured in ``self._bg_error``: a daemon thread dies silently
        otherwise, and step() would then swap in shards that were never
        written (deleting good ones in the process).
        """
        for i in range(k):
            shard_id = start_idx + i
            try:
                self._worker_generate(shard_id, show_progress)
            except Exception as e:
                self._bg_error = (shard_id, e)
                print(f"❌ Background generation of shard {shard_id} failed: {e!r}")
                return
            self._bg_completed.append(shard_id)

    def _launch_background(self, k):
        """(Re)start the background thread for the next k shards."""
        self._bg_completed = []
        self._bg_error = None
        self._bg_start_idx = self.next_shard_idx
        self._bg_started_at = time.time()
        self._bg_stall_steps = 0
        self._pending_k = k
        self.bg_thread = threading.Thread(
            target=self._worker_generate_k, args=(self.next_shard_idx, k, False), daemon=True
        )
        self.bg_thread.start()

    def start_background_generation(self, k=2):
        """Kicks off generation of the next k shards on a background thread before training begins."""
        if not self._is_initialized:
            raise RuntimeError("Call build_initial_cache() before starting background generation.")
        self._launch_background(k)

    def step(self, k=2, progress=0.0):
        """Non-blocking cache rotation. Background generates next k shards, then swaps them safely on completion."""
        self.progress = progress
        if not self._is_initialized:
            raise RuntimeError("Call build_initial_cache() before calling step().")

        if self.single_shard_mode:
            return False 

        self.last_swap_count = 0

        if self.bg_thread is None:
            # Kick off the very first background generation
            self._launch_background(k)
            return False

        if self.bg_thread.is_alive():
            # Still generating. Every network call is now bounded (see
            # naip_fetcher.GDAL_ENV / STAC_TIMEOUT_S), so staying alive for
            # several epochs means generation is outpaced or wedged — say so
            # loudly instead of silently training on stale shards forever.
            self._bg_stall_steps += 1
            if self._bg_stall_steps >= 2:
                mins = (time.time() - self._bg_started_at) / 60.0
                print(
                    f"⚠️ Cache rotation stalled: background batch (shards "
                    f"{self._bg_start_idx}..{self._bg_start_idx + self._pending_k - 1}) "
                    f"still running after {self._bg_stall_steps} epochs "
                    f"({mins:.0f} min, {len(self._bg_completed)}/{self._pending_k} done). "
                    f"Training is reusing stale shards."
                )
                try:
                    if wandb.run is not None:
                        wandb.log({"naip/cache_stall_epochs": self._bg_stall_steps})
                except Exception:
                    pass
            return False

        # Thread FINISHED (normally or after an error): swap in ONLY the
        # shards that were actually written to disk.
        if self._bg_error is not None:
            shard_id, err = self._bg_error
            print(
                f"❌ Background shard generation died at shard {shard_id}: {err!r}. "
                f"Swapping in the {len(self._bg_completed)} completed shard(s) "
                f"and retrying from shard {shard_id}."
            )
            try:
                if wandb.run is not None:
                    wandb.log({"naip/bg_generation_errors": 1})
            except Exception:
                pass

        swapped = 0
        for shard_id in self._bg_completed:
            new_shard = self.cache_dir / f"shard_{shard_id}.pt"
            if not new_shard.exists():
                print(f"⚠️ Completed shard {shard_id} missing on disk; not swapping it in.")
                continue
            oldest_shard = self.active_shards.pop(0)
            try: oldest_shard.unlink()
            except FileNotFoundError: pass
            self.active_shards.append(new_shard)
            swapped += 1
        self.last_swap_count = swapped

        # Resume AFTER the last completed shard: on error the failed shard id
        # (same pair-table slice) is retried instead of silently skipped.
        self.next_shard_idx = self._bg_start_idx + len(self._bg_completed)

        # Immediately kick off the next batch cycle
        self._launch_background(k)

        return swapped > 0
 

class InBatchRankingDataset(Dataset):
    """
    Loads individual images with rich metadata from training shards.
    The HybridBatchSampler constructs valid hybrid batches at iteration time
    using the exposed metadata (year_to_idxs, building_to_idxs).
    """
    def __init__(self, cache_manager: CyclicCacheManager, transform=None, se_lookup=None):
        if not cache_manager._is_initialized:
            raise RuntimeError(
                "Call cache_manager.build_initial_cache() before instantiating InBatchRankingDataset."
            )
        self.cache_manager = cache_manager
        self.transform = transform
        self.se_lookup = se_lookup
        self.images = None
        self.scores = None
        self.geoids = None
        self.years = None
        self.building_ids = None
        self.structural_change = None
        self.metas = None
        self.score_bins = None
        self.cbsa_ids = None

        # Lookups for the HybridBatchSampler (built after loading)
        self.year_to_idxs = {}
        self.yearcbsa_to_idxs = {}
        self.building_to_idxs = {}
        self.refresh()

    def refresh(self):
        """Drops the current in-RAM tensors and reloads from active shards."""
        if not self.cache_manager.active_shards:
            raise RuntimeError("No active shards to load. Call build_initial_cache() first.")

        self.images = None
        self.scores = None
        gc.collect()

        img_list, sc_list, geo_list, yr_list = [], [], [], []
        did_list, ch_list, mt_list, sb_list, cb_list = [], [], [], [], []

        for shard_path in self.cache_manager.active_shards:
            data = torch.load(shard_path, weights_only=False)
            shard_len = len(data["images"])
            img_list.append(data["images"])
            sc_list.append(data["scores"])
            geo_list.append(data.get("geoids", torch.zeros(shard_len, dtype=torch.int64)))
            yr_list.append(data.get("years", torch.zeros(shard_len, dtype=torch.int64)))
            did_list.append(data.get("building_ids", torch.zeros(shard_len, dtype=torch.int64)))
            ch_list.append(data.get("structural_change", torch.zeros(shard_len, dtype=torch.int64)))
            mt_list.append(data.get("metas", torch.zeros(shard_len, 1)))
            sb_list.append(data.get("score_bins", torch.zeros(shard_len, dtype=torch.int64)))
            # Old shards lack cbsa_ids -> zeros, which collapses to per-year pools
            # (single-city behavior), keeping backward compatibility.
            cb_list.append(data.get("cbsa_ids", torch.zeros(shard_len, dtype=torch.int64)))

        self.images           = torch.cat(img_list)
        self.scores           = torch.cat(sc_list)
        self.geoids           = torch.cat(geo_list)
        self.years            = torch.cat(yr_list)
        self.building_ids        = torch.cat(did_list)
        self.structural_change = torch.cat(ch_list)
        self.metas            = torch.cat(mt_list)
        self.score_bins       = torch.cat(sb_list)
        self.cbsa_ids         = torch.cat(cb_list)

        if getattr(self, 'se_lookup', None) is not None and len(self.se_lookup) > 0:
            se_list = []
            geoids_list = self.geoids.tolist()
            years_list = self.years.tolist()
            for g, y in zip(geoids_list, years_list):
                se_list.append(self.se_lookup.get((g, y), 0.151))
            self.ses = torch.tensor(se_list, dtype=torch.float32)
        else:
            self.ses = None

        # Build lookups for the batch sampler
        self.year_to_idxs = {}
        self.yearcbsa_to_idxs = {}
        self.building_to_idxs = {}
        for i in range(len(self.images)):
            yr = self.years[i].item()
            did = self.building_ids[i].item()
            cb = self.cbsa_ids[i].item()
            self.year_to_idxs.setdefault(yr, []).append(i)
            self.yearcbsa_to_idxs.setdefault((yr, cb), []).append(i)
            self.building_to_idxs.setdefault(did, []).append(i)

        n_stable = (self.structural_change == 0).sum().item()
        n_change = (self.structural_change == 1).sum().item()
        print(f"[InBatchRankingDataset] Loaded {len(self.images)} images "
              f"(Stable: {n_stable} | Change: {n_change}) "
              f"from {len(self.cache_manager.active_shards)} shards.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        se_val = self.ses[idx] if self.ses is not None else 0.0
        return (img, self.scores[idx], self.geoids[idx], self.years[idx],
                self.building_ids[idx], self.structural_change[idx], self.metas[idx], self.score_bins[idx], se_val)


class HybridBatchSampler(torch.utils.data.Sampler):
    """
    Constructs valid hybrid batches for In-Batch Pairwise Ranking.

    Each batch contains:
      - B_cs cross-sectional images from the SAME (year, CBSA): per-CBSA z-scores
        are only comparable within a metro, so cross-metro pairs would inject
        noise into the ranking loss. Pool choice is weighted by pool size so big
        metros are visited proportionally to their data.
      - Their temporal twins from other years of the SAME buildings, split
        between stable and change buildings per the temporal_fraction semantics
        (50/50 budget, backfilled from the other side when one is short). The
        loss gates lambda_s to stable twins; change twins carry no loss term
        (they feed the changed-MASD validation metrics), so including them
        never contaminates the stability penalty.

    Old shards without cbsa_ids collapse to cbsa=0 pools == the previous
    per-year (single-city) behavior.
    """
    def __init__(self, dataset: InBatchRankingDataset, batch_size_cs: int, max_temporal_per_batch: int = 32,
                 batch_dumper=None):
        self.dataset = dataset
        self.batch_size_cs = batch_size_cs
        self.max_temporal = max_temporal_per_batch
        self.num_batches = max(1, len(dataset) // batch_size_cs)
        self._warned_fallback = False
        # Optional BatchImageDumper (fresh runs only): persists the first N yielded
        # batches for the visual QA notebook. None on resumed runs.
        self.batch_dumper = batch_dumper

    def _pools(self):
        """(year, cbsa) pools big enough for meaningful in-batch ranking."""
        min_pool = max(2, self.batch_size_cs // 4)
        pools = {k: v for k, v in self.dataset.yearcbsa_to_idxs.items()
                 if len(v) >= min_pool}
        # Degenerate fallback (tiny/misconfigured cache): use per-year pools.
        if not pools:
            pools = {(yr, 0): v for yr, v in self.dataset.year_to_idxs.items()}
            if not self._warned_fallback and bool((self.dataset.cbsa_ids != 0).any()):
                self._warned_fallback = True
                print(
                    "⚠️ [HybridBatchSampler] all (year, CBSA) pools below min size — "
                    "falling back to per-year pools; batches may MIX CBSAs. "
                    "Increase shard size or check cbsa_ids."
                )
        return pools

    def __iter__(self):
        pools = self._pools()
        keys = list(pools.keys())
        weights = [len(pools[k]) for k in keys]

        for _ in range(self.num_batches):
            year, _cbsa = key = random.choices(keys, weights=weights, k=1)[0]
            pool = pools[key]

            # Sample cross-sectional core (single year x single CBSA)
            cs_size = min(self.batch_size_cs, len(pool))
            cs_idxs = random.sample(pool, cs_size)

            batch = list(cs_idxs)

            # Temporal twins: same building, different year. Budget split 50/50
            # between stable and change buildings; backfill if one side is short.
            stable_budget = self.max_temporal // 2
            change_budget = self.max_temporal - stable_budget
            stable_twins, change_twins = [], []
            for idx in cs_idxs:
                if len(stable_twins) + len(change_twins) >= self.max_temporal:
                    break
                bid = self.dataset.building_ids[idx].item()
                twin_candidates = [
                    i for i in self.dataset.building_to_idxs.get(bid, [])
                    if i != idx and self.dataset.years[i].item() != year
                ]
                if not twin_candidates:
                    continue
                twin = random.choice(twin_candidates)
                if self.dataset.structural_change[idx].item() == 0:
                    stable_twins.append(twin)
                else:
                    change_twins.append(twin)

            n_stable = min(len(stable_twins), stable_budget)
            n_change = min(len(change_twins), change_budget)
            spare = self.max_temporal - n_stable - n_change
            if spare > 0:   # backfill unused budget from whichever side has extras
                extra = min(len(stable_twins) - n_stable, spare)
                n_stable += extra
                spare -= extra
                n_change += min(len(change_twins) - n_change, spare)
            batch.extend(stable_twins[:n_stable])
            batch.extend(change_twins[:n_change])

            if self.batch_dumper is not None and not self.batch_dumper.finished:
                self.batch_dumper.dump_batch(
                    self.dataset, batch, cs_size=cs_size,
                    anchor_year=year, anchor_cbsa=_cbsa,
                )

            yield batch

    def __len__(self):
        return self.num_batches


class StaticShardedDataset(Dataset):
    """
    Evaluates cleanly across many pre-computed shards.
    Only keeps one shard in RAM at a time to prevent OOM when the validation set is massive.
    Validation shards store single images using the keys 'images' and 'scores'.
    """
    def __init__(self, active_shards, transform=None, verbose=True):
        if not active_shards:
            raise RuntimeError("No active shards to load.")
        self.shard_paths = active_shards
        self.transform = transform
        self.current_shard_idx = -1
        self.images = None
        self.scores = None
        self.metas  = None
        self.building_ids = None
        self.years = None
        self.structural_change = None
        self.cbsa_ids = None

        if verbose:
            print(f"Loading metadata for {len(self.shard_paths)} shards to establish dataset sizes...")
        self.shard_lengths = []
        for p in self.shard_paths:
            data = torch.load(p, weights_only=False)
            self.shard_lengths.append(len(data["images"]))

        self.total_length = sum(self.shard_lengths)
        if verbose:
            print(f"Discovered {self.total_length} total validation items across shards.")

        # To make DataLoader indexing completely seamless
        self.idx_mapping = []
        for shard_idx, length in enumerate(self.shard_lengths):
            for i in range(length):
                self.idx_mapping.append((shard_idx, i))

    def _load_shard(self, shard_idx):
        if self.current_shard_idx == shard_idx:
            return
        # Drops the old tensor, loads the new one
        data = torch.load(self.shard_paths[shard_idx], weights_only=False)
        self.images  = data["images"]
        self.scores  = data["scores"]
        self.metas   = data.get("metas")
        self.building_ids = data.get("building_ids")
        self.years   = data.get("years")
        self.structural_change = data.get("structural_change")
        self.cbsa_ids = data.get("cbsa_ids")   # absent in pre-US shards -> 0 per item
        self.current_shard_idx = shard_idx

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        shard_idx, item_idx = self.idx_mapping[idx]
        self._load_shard(shard_idx)

        img  = self.images[item_idx]
        lbl  = self.scores[item_idx]
        meta = self.metas[item_idx] if self.metas is not None else torch.zeros(1)
        bldg_id = self.building_ids[item_idx] if self.building_ids is not None else 0
        year = self.years[item_idx] if self.years is not None else 0
        structural_change = self.structural_change[item_idx] if self.structural_change is not None else 0
        cbsa_id = self.cbsa_ids[item_idx] if self.cbsa_ids is not None else 0

        if self.transform:
            img = self.transform(img)

        return img, lbl, meta, bldg_id, year, structural_change, cbsa_id
                                        
class PhotometricAugmentation:
    def __init__(self):
        self.cj = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        self.gr = transforms.RandomGrayscale(p=0.1)
        
    def __call__(self, img):
        is_batch = img.ndim == 4
        c_dim = 1 if is_batch else 0
        
        n_channels = img.shape[c_dim]
        if n_channels >= 3:
            if is_batch:
                rgb = img[:, :3]
                rgb = self.gr(self.cj(rgb))
                if n_channels == 3:
                    return rgb
                return torch.cat([rgb, img[:, 3:]], dim=c_dim)
            else:
                rgb = img[:3]
                rgb = self.gr(self.cj(rgb))
                if n_channels == 3:
                    return rgb
                return torch.cat([rgb, img[3:]], dim=c_dim)
        return img

def _subsample_val_buildings(df_val, n_buildings_per_tract=1, seed=825):
    """Subsample a validation dataframe by BUILDING, keeping all years.

    Picks up to ``n_buildings_per_tract`` building_ids per GEOID and keeps every
    year-row of the chosen buildings (deterministic). Sampling whole buildings —
    instead of rows per (tract, year) as before — guarantees multi-year
    buildings survive, so stable/changed MASD and directional accuracy are
    computable on every val set (replaces the old val_spatial_temporal set).
    Default is 1 building/tract: labels are tract-level, so extra buildings in
    a tract add near-zero metric information — tract coverage is what matters.

    The result is shuffled at BUILDING level (rows of each building stay
    contiguous, year-sorted) so the MAX_VAL_SHARDS truncation downstream keeps
    a random cross-city sample of buildings instead of the head of the input
    order (which could concentrate on a few cities).
    """
    picks = (
        df_val[["GEOID", "building_id"]]
        .drop_duplicates()
        .groupby("GEOID", group_keys=False)[["GEOID", "building_id"]]
        .apply(lambda g: g.sample(n=min(len(g), n_buildings_per_tract), random_state=seed))
    )
    out = df_val[df_val["building_id"].isin(set(picks["building_id"]))].copy()
    # Building-level shuffle: random building order, all years of a building
    # kept adjacent so no building straddles the shard-cap cutoff.
    shuffled_bids = (
        out["building_id"].drop_duplicates().sample(frac=1, random_state=seed)
    )
    codes = pd.Categorical(
        out["building_id"], categories=shuffled_bids, ordered=True
    ).codes
    order = np.lexsort((out["year"].to_numpy(), codes)) if "year" in out.columns else np.argsort(codes, kind="stable")
    return out.iloc[order].reset_index(drop=True)


def build_se_lookup(indicator_token):
    import pandas as pd
    from src.data.process_acs import PROCESSED_DATA_DIR, PANEL_YEARS
    from src.data.indicators import token_to_var
    se_dict = {}
    panel_path = PROCESSED_DATA_DIR / "us_metros_panel_2011_2023.feather"
    if not panel_path.exists():
        print(f"⚠️ Warning: Panel {panel_path} not found. MOE filter will fallback to score bins.")
        return se_dict
    
    var = token_to_var(indicator_token)
    if not var: 
        return se_dict
    prefix = f"Rel_SE_{var}"
    
    df = pd.read_feather(panel_path)
    se_cols = [c for c in df.columns if c.startswith(prefix)]
    if not se_cols: return se_dict
    
    available_years = [int(c.split("_")[-1]) for c in se_cols]
    if not available_years: return se_dict

    def get_closest(yr):
        return min(available_years, key=lambda x: abs(x - yr))
    
    for _, row in df.iterrows():
        geo = int(row["geoid_2023"])
        for yr in PANEL_YEARS:
            closest_yr = get_closest(yr)
            val = row[f"{prefix}_{closest_yr}"]
            if pd.notna(val):
                se_dict[(geo, yr)] = val
    return se_dict


def setup_dataloaders(df_train, dfs_val_dict, df_test, all_years_datasets, params, train_cache_manager=None, val_cache_manager=None, batch_dumper=None):
    print("--- Initializing PyTorch Datasets ---")
    string_val_lengths = "| ".join([f"{name}: {len(df)}" for name, df in dfs_val_dict.items()])
    print(f"Train: Cyclical Cache | {string_val_lengths} | Test: {len(df_test)}")

    batch_size = params.get("batch_size", 32)
    nbands = params.get("nbands", 4)

    # ImageNet mean/std for first 3 bands, 0.5 for remaining NIR/multispectral bands
    mean = [0.485, 0.456, 0.406] + [0.5] * max(0, nbands - 3)
    std = [0.229, 0.224, 0.225] + [0.5] * max(0, nbands - 3)

    # TRAIN
    image_size = params.get("image_size", 224)
    # CPU: Fast spatial crops and flips on uint8
    train_transform_cpu = transforms.Compose([
        transforms.RandomCrop(image_size),   # Realizes per-batch spatial jitter from the padded shard tiles
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    # GPU: Photometric augmentation, float scaling, and normalization
    train_transform_gpu = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        PhotometricAugmentation(),           # 🔴 Handle photometric augmentation securely on GPU
        transforms.Normalize(mean=mean, std=std) # 🔴 ImageNet Normalization
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std) # 🔴 ImageNet Normalization
    ])

    # Build SE lookup for MOE filter
    indicator = params.get("indicator", "W2_r5")
    se_lookup = build_se_lookup(indicator)

    # In-Batch Pairwise Ranking: individual images + metadata, grouped by HybridBatchSampler
    train_dataset = InBatchRankingDataset(
        cache_manager=train_cache_manager,
        transform=train_transform_cpu,
        se_lookup=se_lookup,
    )
    hybrid_sampler = HybridBatchSampler(
        dataset=train_dataset,
        batch_size_cs=batch_size,
        max_temporal_per_batch=max(1, int(batch_size * params.get("temporal_fraction", 0.20))),
        batch_dumper=batch_dumper,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=hybrid_sampler, num_workers=0)
    train_loader.gpu_transform = train_transform_gpu
    
    # VAL
    val_loaders = {}
    if val_cache_manager is not None:
        for name, df_val in dfs_val_dict.items():
            if name in val_cache_manager and val_cache_manager[name] is not None:
                val_dataset = StaticShardedDataset(
                    active_shards=val_cache_manager[name].active_shards,
                    transform=eval_transform # No random augmentations
                )
                val_loaders[name] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # TEST

    # test_dataset = EvalSatelliteDataset(
    #     df=df_test, all_years_datasets=all_years_datasets,
    #     params=params, transform=eval_transform, mode="lazy_eval"
    # )
    test_loader = None # TODO
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
   
    return train_loader, val_loaders, test_loader

def validate_parameters(params, default_params):
    
    for key, value in params.items():
        if key not in default_params.keys():
            raise ValueError("Invalid parameter: %s" % key)

    try:
        sat_data = params["sat_data"]
    except:
        print("No parameters are being validated, as sat_data is not defined...")
        return

    nbands = params["nbands"]
    years = params["years"]
    image_size = params["image_size"]
    weights = params["weights"]

    sat_options = ["aerial", "pleiades", "landsat", "NAIP"]
    if sat_data not in sat_options:
        raise ValueError("Invalid sat_data type. Expected one of: %s" % sat_options)

    if sat_data == "pleiades":
        if (nbands != 3) and (nbands != 4):
            raise ValueError("nbands for pleiades dataset must be 3 or 4.")

        # if len(years) > 3:
        #     raise ValueError("Pleiades data only available in 2013, 2018 and 2022.")
        # elif not all(year in [2013, 2018, 2022] for year in years):
        #     raise ValueError("Pleiades data only available in 2013, 2018 and 2022.")

        if image_size > 1024:
            warnings.warn(
                "Warning: image_size greater than 1024 might encompass an area much bigger than the census tracts..."
            )

    elif sat_data == "landsat":
        if nbands > 10:
            raise ValueError("nbands for pleiades dataset must be less than 11.")

        if years != [2013]:
            raise ValueError("Landsat data only available in 2013.")

        if image_size > 32:
            warnings.warn(
                "Warning: image_size greater than 32 might encompass an area much bigger than the census tracts..."
            )

    return


def fill_params_defaults(params):

    default_params = {
        "model_name": "effnet_v2S",
        "kind": "reg",
        "weights": None,
        "image_size": 256,
        "nbands": 4,
        "batch_size": 32,
        "small_sample": False,
        "n_epochs": 100,
        "learning_rate": 0.0001,
        "sat_data": "pleiades",
        "years": [2013],
        "naip_coverage_csv": None,  # naip_coverage_audit.py CSV for per-city holdout years (#28); None = auto-probe ~/outputs/naip_coverage.csv
        "extra": "",
        "run_id": None,       # short custom savename; default is run_{YYYYMMDD}
        "resume_lr_override": None,  # force this lr after restoring optimizer state on resume (None = keep checkpoint lr)
        "selection_metric": "within",  # "within" (mean per-city-year spearman, honest) | "pooled" (legacy set-level)
        "indicator": indicators.DEFAULT_INDICATOR,  # training label (see src/data/indicators.py)
        "footprints_source": "ms_us",  # "ms_us" (national MS index) | "doitt_nyc" (legacy)
        "states": None,       # optional list of state stems to subset the MS index
        "reject_padded_nir": False,  # drop NAIP crops whose NIR band is zero-padded
        "predict_split": "test",     # "test" (main run) | "all" (every selected city: train+val+test)
        "predict_chunk_size": 4096,  # rows per resumable prediction chunk (NAIP path)
        "predict_exact_year": True,  # forbid flight-year substitution: no same-year NAIP flight => no_year_match -> NaN, never the closest year's imagery. In the fingerprint (exact/substituted chunks never mix).
        "predict_tract_sample_frac": 0.10,   # per CBSA: sample max(ceil(frac*n_tracts), min) tracts (None = all tracts). Deterministic hash of GEOID — same tracts every year/rerun.
        "predict_tract_sample_min": 100,     # tract floor per CBSA: bounds small-city 95% CIs
        "predict_buildings_per_tract": 100,  # <=100 buildings per sampled tract (None = all): tract-mean 95% CI +-0.098 at sigma_within=0.5 (+-0.196 worst case) given the loss anchors global SD=1 — invisible to rank metrics
        "predict_full_universe_geoid_prefixes": ("36005", "36047", "36061", "36081", "36085"),  # NYC's 5 boroughs: FULL universe (bypasses split filter + sampling) for the CSA event study; DoITT change data is city-only
        "predict_fetch_workers": 16,  # NAIP fetch shards per chunk. Each shard opens each (DOQQ, tract) run's COG once and reads its crops per-window from that shared handle (GDAL block cache dedupes overlaps; per-crop beat the old mosaic 1.6-4x on real tracts — src/probe_naip_concurrency.py --mode real-granularity). Keep near the tract count, not maxed out. PC is not request-count throttling (0 throttles to 64 workers); throughput is a per-process saturation ceiling that PEAKS ~16-32 workers and DEGRADES past it (measured chunk fetch ~34 crops/s flat 8->32w). To go faster, shard chunks across processes/machines, not raise this.
        "predict_batch_size": 512,   # eval-only batch (None = legacy batch_size*8). Set near the VRAM cap: predict_chunk_rows halves on OOM and remembers the working size, so aggressive is safe.
        "predict_fetch_pipeline": 3,  # chunks fetched concurrently ahead of the GPU (RAM ~0.8GB each at 4096/4/224): smooths the fast/slow chunk variance so the network stays busy. 1 = legacy two-in-flight.
        "predict_retry_fast_fail_frac": 0.01,  # transient-failure fraction at or below which stragglers get ONE immediate retry then drop (isolated data problems), instead of the full cooldown ladder (throttling signature).
        "search_grid_meters": 20000,  # STAC search cache cell (EPSG:5070 m): one search per ~20km cell serves all its tracts, not one per tract. Larger = fewer/bigger searches.
        "persist_search_cache": True, # cache STAC search results to CACHE_DIR/naip_search_cache so reruns are ~search-free (NAIP item lists per cell are stable).
        "tau_meters": 100,
        "subsample_step": 1,
        "max_jitter": 10,
        # In-Batch Pairwise Ranking hyperparameters
        "m_min": 0.1,         # Hard floor for pairwise margin (prevents collapse)
        "m_base": 1.0,        # Margin scale factor: m_kl = max(m_min, m_base * |Z_l - Z_k|)
        "lambda_s": 1.0,      # Weight for temporal stability L1 penalty
        "lambda_var": 1.0,    # Weight for cross-sectional variance regularizer (prevents collapse)
        "temporal_fraction": 0.20,  # Fraction of batch reserved for temporal auxiliary (split 50/50 stable/change)
        # Tract-first shard sampling + gradient hard-tract mining (train, lazy
        # ms_us path only; see src/data/tract_sampling.py for rationale + refs)
        "tract_sampling": False,     # sample tracts (~population-weighted), not buildings; False = legacy cyclic building traversal
        "grad_mining_alpha": 0.0,    # hardness mixture weight in tract weights: (1-a) + a*min(ema/mean, cap); 0 = uniform tract sampling
        "grad_mining_beta": 0.9,     # per-tract |lambda| EMA decay (stale hardness fades as the model improves)
        "grad_mining_cap": 10.0,     # hardness ratio cap: irreducibly-hard tracts can't monopolize shards
        "sampling_seed": 825,        # base seed for per-shard tract draws (combined with shard_id)
    }
    validate_parameters(params, default_params)

    # Merge default and provided hyperparameters (keep from params)
    updated_params = {**default_params, **params}
    print("-" * 40)
    print("Runtime Parameters:")
    for k, v in updated_params.items():
        print(f"  {k}: {v}")
    print("-" * 40)

    return updated_params


class InBatchPairwiseRankingLoss(nn.Module):
    """In-Batch Pairwise Ranking Loss with Score-Based Margins (Ordinal Framework).

    Computes two decoupled objectives from a single forward pass:
      1. L_cross:  Smooth logistic surrogate (RankNet) over all valid unique cross-sectional pairs
                   T = max(m_min, m_base * σ_batch) — temperature scales with score spread, NOT ACS magnitude
      2. L_stable: L1 temporal invariance for stable twins

    L_total = L_cross + lambda_s * L_stable + L_var

    Change twins carry NO loss term (the directional L_change was removed, issue #30):
    they are still sampled by HybridBatchSampler and feed the changed-MASD / directional-
    accuracy validation metrics, but L_stable is masked to stable twins only.

    Key design: margins carry zero economic information. They scale with the model's own output
    distribution (σ_batch) to prevent collapse at initialization, never with ACS label magnitudes.
    This preserves the ordinal framework: supervision is purely directional (sign), never cardinal.

    Returns (loss, diagnostics_dict) so the training loop can log internals to W&B.
    """
    def __init__(self, m_base=1.0, m_min=0.1, lambda_s=1.0, lambda_var=1.0,
                 hardness_registry=None):
        super().__init__()
        self.m_base = m_base
        self.m_min = m_min
        self.lambda_s = lambda_s
        self.lambda_var = lambda_var  # CS-only variance regularizer weight
        # Optional GradientHardnessRegistry: forward() harvests the per-sample
        # ranking gradients lambda_i = m * dL_cross/ds_i (LambdaRank's lambdas,
        # Burges 2010) and folds tract-level |lambda| EMAs into it, driving
        # hard-tract importance sampling at shard generation (Katharopoulos &
        # Fleuret 2018). Harvested AFTER the MOE pair filter, so hardness only
        # accumulates where the label sign is statistically reliable (noise
        # guard in the spirit of RHO-loss, Mindermann et al. 2022).
        self.hardness_registry = hardness_registry

    def forward(self, scores, labels, geoids, years, building_ids, structural_change, score_bins=None, current_epoch=None, current_step=None, se_values=None):
        scores = scores.squeeze(-1)
        labels = labels.float()
        
        # Because HybridBatchSampler puts CS core first, the first year is the CS anchor year
        anchor_year = years[0].item()
        
        cs_mask = (years == anchor_year)
        temp_mask = (years != anchor_year)

        # ──────────────────────────────────────────────────────────
        # 1. CROSS-SECTIONAL RANKING (L_cross)
        # ──────────────────────────────────────────────────────────
        cs_scores = scores[cs_mask]
        cs_labels = labels[cs_mask]
        cs_geoids = geoids[cs_mask]
        B_cs = cs_scores.shape[0]

        L_cross = torch.tensor(0.0, device=scores.device)
        n_valid_pairs = 0
        cross_hinge_active, avg_margin = 0.0, 0.0

        if B_cs >= 2:
            idx_k, idx_l = torch.triu_indices(B_cs, B_cs, offset=1, device=scores.device)
            valid_mask = cs_geoids[idx_k] != cs_geoids[idx_l]
            
            # --- Margin of Error (MOE) Filtering ---
            if se_values is not None and (se_values > 0).any():
                cs_se = se_values[cs_mask]
                se_k, se_l = cs_se[idx_k], cs_se[idx_l]
                z_k, z_l = cs_labels[idx_k], cs_labels[idx_l]
                
                moe = 1.645 * torch.sqrt(se_k**2 + se_l**2)
                delta_z = torch.abs(z_l - z_k)
                valid_mask &= (delta_z >= moe)
            elif current_epoch is not None and score_bins is not None:
                cs_bins = score_bins[cs_mask]
                bin_diff = torch.abs(cs_bins[idx_k] - cs_bins[idx_l])
                
                if current_epoch < 60:
                    valid_mask &= (bin_diff >= 3)
                elif current_epoch < 150:
                    valid_mask &= (bin_diff >= 2)
                else:
                    # PERMANENT FILTER: Never train on bin_diff == 0. 
                    # The ACS margin of error makes intra-quintile rankings pure noise.
                    # Forcing a strict margin on them causes variance collapse.
                    valid_mask &= (bin_diff >= 1)

            if valid_mask.any():
                idx_k, idx_l = idx_k[valid_mask], idx_l[valid_mask]
                n_valid_pairs = idx_k.shape[0]

                z_k, z_l = cs_labels[idx_k], cs_labels[idx_l]
                y_kl = torch.sign(z_l - z_k)

                # [FIXED] Fixed margin for Smooth RankNet. 
                # We removed sigma_batch because RankNet doesn't need dynamic margins,
                # and the variance_penalty already anchors the scale to 1.0.
                m_scalar = self.m_base
                avg_margin = m_scalar

                r_k, r_l = cs_scores[idx_k], cs_scores[idx_l]
                delta = y_kl * (r_l - r_k)
                # Standard smooth logistic loss
                L_cross = F.softplus(-delta / m_scalar).mean()

                with torch.no_grad():
                    # Diagnostic: proportion of pairs that would violate a hard margin
                    cross_hinge_active = (-delta + m_scalar > 0).float().mean().item()

                    # ── Gradient hard-tract mining: harvest per-sample lambdas ──
                    # Per valid pair, |grad| of its RankNet term w.r.t. either
                    # score is sigma(-delta/m)/m; signed accumulation gives
                    # lambda_i = m * dL_cross_sum/ds_i for free (no extra
                    # backward). |lambda|/degree in [0, 1] is the per-sample
                    # hardness; tract aggregation happens in the registry.
                    if self.hardness_registry is not None and n_valid_pairs > 0:
                        contrib = torch.sigmoid(-delta / m_scalar) * y_kl
                        lam = torch.zeros(B_cs, device=scores.device)
                        lam.index_add_(0, idx_k, contrib)
                        lam.index_add_(0, idx_l, -contrib)
                        deg = torch.zeros(B_cs, device=scores.device)
                        ones = torch.ones_like(contrib)
                        deg.index_add_(0, idx_k, ones)
                        deg.index_add_(0, idx_l, ones)
                        in_pairs = deg > 0
                        hardness = lam[in_pairs].abs() / deg[in_pairs]
                        self.hardness_registry.update(
                            cs_geoids[in_pairs].cpu().numpy(),
                            hardness.cpu().numpy(),
                        )

        # ──────────────────────────────────────────────────────────
        # 2. TEMPORAL PENALTY (L_stable) — masked to stable twins.
        #    Change twins carry no loss term; they only feed val metrics (#30).
        # ──────────────────────────────────────────────────────────
        L_stable = torch.tensor(0.0, device=scores.device)
        n_stable, n_change = 0, 0

        if temp_mask.any():
            temp_scores = scores[temp_mask]
            temp_building_ids = building_ids[temp_mask]
            temp_change = structural_change[temp_mask]

            cs_building_ids = building_ids[cs_mask]
            cs_scores_full = scores[cs_mask]

            # Vectorized dynamic pairing: match temporal building_id to cross-sectional building_id
            matches = (temp_building_ids.unsqueeze(1) == cs_building_ids.unsqueeze(0))
            has_match, match_idx_in_cs = matches.max(dim=1)

            # Filter to twins that successfully matched a CS anchor
            valid_temp = has_match
            if valid_temp.any():
                v_temp_scores = temp_scores[valid_temp]
                v_temp_change = temp_change[valid_temp]

                v_partner_scores = cs_scores_full[match_idx_in_cs[valid_temp]]

                # STABLE Penalty
                stable_idx = (v_temp_change == 0)
                if stable_idx.any():
                    L_stable = ((v_temp_scores[stable_idx] - v_partner_scores[stable_idx]) ** 2).mean()
                    n_stable = stable_idx.sum().item()

                # Change twins: counted only (sampler-health diagnostic; no loss term)
                n_change = (v_temp_change == 1).sum().item()

        # CS-only variance regularizer: targets the exact distribution L_cross operates on.
        # Eliminates the tug-of-war with L_stable (temporal scores are excluded).
        # Internally consistent: both m_scalar and variance_penalty reference cs_scores.std().
        variance_penalty = torch.tensor(0.0, device=scores.device)
        if B_cs > 1:
            variance_penalty = (cs_scores.std() - 1.0).pow(2)
            
        loss = L_cross + self.lambda_s * L_stable + self.lambda_var * variance_penalty

        # --- Compute gradient norms of each loss component w.r.t predictions ---
        # Only compute every 10 steps to avoid 3x extra backward traversals per step
        grad_cross, grad_stable, grad_variance = 0.0, 0.0, 0.0
        compute_grad_norms = (current_step is not None and current_step % 10 == 0)
        if compute_grad_norms:
            try:
                if L_cross.requires_grad and L_cross.item() > 0:
                    g_c = torch.autograd.grad(L_cross, scores, retain_graph=True)[0]
                    if g_c is not None: grad_cross = g_c.norm(p=2).item() * 10

                if L_stable.requires_grad and L_stable.item() > 0:
                    g_s = torch.autograd.grad(self.lambda_s * L_stable, scores, retain_graph=True)[0]
                    if g_s is not None: grad_stable = g_s.norm(p=2).item() * 10

                if variance_penalty.requires_grad and variance_penalty.item() > 0:
                    g_v = torch.autograd.grad(variance_penalty, scores, retain_graph=True)[0]
                    if g_v is not None: grad_variance = g_v.norm(p=2).item() * 10
            except Exception:
                pass # Fail gracefully (e.g. if graph is disconnected or autograd anomalies)

        # Diagnostics — all detached, zero overhead on the backward pass
        with torch.no_grad():
            diag = {
                "loss/L_cross":              L_cross.item(),
                "loss/L_stable":             L_stable.item(),
                "grad_norm/cross":           grad_cross,
                "grad_norm/stable":          grad_stable,
                "grad_norm/variance":        grad_variance,
                "loss/cross_valid_pairs":    n_valid_pairs,
                "loss/cross_hinge_active":   cross_hinge_active,
                "loss/avg_margin":           avg_margin,
                "loss/n_stable":             n_stable,
                "loss/n_change":             n_change,
                "loss/score_std":            scores.std().item() if scores.shape[0] > 1 else 0.0,
                "loss/cs_score_std":         cs_scores.std().item() if B_cs > 1 else 0.0,
                "loss/score_mean":           scores.mean().item(),
                "loss/score_min":            scores.min().item(),
                "loss/score_max":            scores.max().item(),
            }
            if self.hardness_registry is not None:
                diag.update(self.hardness_registry.stats())

        return loss, diag


def set_model_and_loss_function(
    model_name: str, kind: str, image_size: int, bands: int = 4, weights: str = None, meta_dim: int = 0,
    m_base: float = 1.0, m_min: float = 0.1, lambda_s: float = 1.0, lambda_var: float = 1.0,
    hardness_registry=None,
):
    """
    Initializes the PyTorch model and appropriate loss function.
    """
    print(f"--- Initializing Model: {model_name} ---")
    
    # Validación de parámetros
    assert kind in ["reg", "cla"], "kind must be either 'reg' or 'cla'"

    # 1. Instantiate the model dynamically using our new registry!
    model = custom_models.get_model(
        name=model_name, 
        image_size=image_size, 
        bands=bands, 
        kind=kind,
        meta_dim=meta_dim
    )

    # Load weights if provided
    if weights is not None and weights not in ["imagenet"]:
        # PyTorch uses state_dicts to load weights
        model.load_state_dict(torch.load(weights))
        print(f"\n--- 🚀 Successfully loaded custom weights from: {weights} ---")

    # 2. Set loss functions
    if kind == "reg":
        # In-Batch Pairwise Ranking Loss with economic-distance-proportional margins.
        loss_fn = InBatchPairwiseRankingLoss(
            m_base=m_base, m_min=m_min, lambda_s=lambda_s, lambda_var=lambda_var,
            hardness_registry=hardness_registry,
        )
        
    elif kind == "cla":
        # CrossEntropyLoss expects raw logits (no softmax in the model output)
        loss_fn = nn.CrossEntropyLoss()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded and moved to: {device}")

    # We return just model and loss. 
    # Metrics and Optimizers are instantiated right before the training loop in PyTorch.
    return model, loss_fn

def generate_parameters_log(params, savename):

    os.makedirs(f"{MODELS_DIR}/{savename}", exist_ok=True)
    filename = f"{MODELS_DIR}/{savename}/{savename}_logs.txt"

    with open(filename, "w") as file:
        json.dump(params, file)

    print(f"Created parameters log at: {filename}")
    return

def check_feature_importance(model):
    """
    Extracts and logs the relative weight of the image features vs. metadata features
    from the final fusion layer.
    """
    # Look for the fusion layer (final_head) which might be nested in ScaleMAE
    fusion_layer = None
    if hasattr(model, "final_head"):
        fusion_layer = model.final_head
    elif hasattr(model, "head") and hasattr(model.head, "final_head"):
        fusion_layer = model.head.final_head

    if not hasattr(model, "meta_dim") or model.meta_dim == 0 or fusion_layer is None:
        return

    # Get absolute weights from the final linear layer (squeeze to 1D)
    weights = fusion_layer.weight.data.abs().squeeze()
    
    # Slice weights based on meta_dim
    image_weights = weights[:-model.meta_dim]
    meta_weights = weights[-model.meta_dim:]
    
    image_avg_strength = image_weights.mean().item()
    commute_strength = meta_weights.mean().item()
    ratio = commute_strength / image_avg_strength if image_avg_strength > 0 else 0
    
    print(f"\n[Feature Weights] Commute: {commute_strength:.4f} | Avg Image: {image_avg_strength:.4f} | Ratio: {ratio:.2f}x")
    
    # Log to W&B if a run is active
    if wandb.run is not None:
        wandb.log({
            "weights/commute_strength": commute_strength,
            "weights/image_avg_strength": image_avg_strength,
            "weights/commute_to_image_ratio": ratio
        }, commit=False) # commit=False ties it to the next step log

# Validation metrics live in src/utils/metrics.py, shared with evaluation.py so
# training-time and post-hoc evaluation can never drift apart.
from src.utils.metrics import (
    compute_val_metrics,
    rank_autocorrelation as _rank_autocorrelation,
    within_city_cells as _within_city_cells,
)


def train_model(
    model,
    train_loader,
    val_loaders,
    loss_fn,
    optimizer,
    scheduler,
    epochs,
    device,
    savename,
    cache_manager,
    start_epoch=0,
    initial_best_val_loss=None,
    val_cache_managers=None,
    cbsa_meta=None,
    selection_metric="within",
):
    print("--- Starting PyTorch Training Loop ---")
    # "within": checkpoint on the mean per-(city, year) within-city Spearman —
    # the quantity the ordinal loss actually optimizes. "pooled": legacy
    # set-level pooled Spearman (kept for continuity with pre-US runs).
    sel_key = "within_spearman" if selection_metric == "within" else "spearman"
    best_val_spearman = (
        initial_best_val_loss if initial_best_val_loss is not None else float(-1)
    )
    # Per-city split metadata for the #31 breakdowns (None -> base metrics only)
    top_cities = cbsa_brackets.top_cbsas(cbsa_meta) if cbsa_meta is not None else None

    # Pointwise MSE for validation — training uses in-batch pairwise ranking but validation
    # evaluates individual predictions against their labels for interpretable tracking.
    val_loss_fn = nn.MSELoss()
    
    # Ensure the save directory exists
    save_dir = MODELS_DIR / "models_by_epoch" / savename
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Mixed Precision Scaler
    scaler = GradScaler()
    accumulation_steps = 8  # Accumulate gradients (e.g., batch_size 8 * 4 steps = effective batch size 32)

    for epoch in range(start_epoch, epochs):
               
        # ==========================
        # 1. TRAINING PHASE
        # ==========================
        
        # Start training loop
        model.train() # Set model to training mode (enables dropout, batchnorm updates)
        running_train_loss = 0.0
        running_train_correct = 0.0 # Tracking Pairwise Accuracy
        
        # Accumulator for diagnostic metrics — reset each epoch
        diag_accum = {}
        diag_steps = 0
        total_pairs_epoch = 0  # Track total cross-sectional pairs trained on (after curriculum mask)
        total_acc_pairs_epoch = 0 # Track total cross-sectional pairs evaluated for accuracy
        
        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)
        
        train_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch [{epoch+1}/{epochs}] Train",
            leave=False   # clears the bar when done, keeping output clean
        )

        t_batch_start = time.perf_counter()  # start timer before first batch
        for batch_idx, (images, scores, geoids, years, building_ids, structural_change, metas, score_bins, se_values) in train_bar:
            t_data_end = time.perf_counter()  # data is ready; measure load time

            # Move all to device
            images     = images.to(device)
            if hasattr(train_loader, "gpu_transform") and train_loader.gpu_transform is not None:
                with torch.no_grad():
                    images = train_loader.gpu_transform(images)
            
            scores_lbl = scores.to(device)
            geoids     = geoids.to(device)
            years      = years.to(device)
            building_ids  = building_ids.to(device)
            structural_change = structural_change.to(device)
            metas      = metas.to(device)
            score_bins = score_bins.to(device)
            se_values  = se_values.to(device)
            
            # Single unified forward pass over the entire hybrid batch
            t_forward_start = time.perf_counter()

            with autocast(device_type='cuda'):
                # One forward pass — all images (CS + temporal) through same backbone
                outputs = model(images, metadata=metas)
                
                # In-Batch Pairwise Ranking Loss (3 decoupled objectives)
                loss, diag = loss_fn(
                    outputs, scores_lbl, geoids, years, building_ids, structural_change, 
                    score_bins=score_bins, current_epoch=epoch, current_step=batch_idx, se_values=se_values
                )

                # Scale loss to account for accumulation
                loss = loss / accumulation_steps
                
            # Accumulate diagnostics for epoch-level averaging
            for k, v in diag.items():
                diag_accum[k] = diag_accum.get(k, 0.0) + v
            diag_steps += 1
            total_pairs_epoch += diag.get("loss/cross_valid_pairs", 0)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Only step the optimizer every `accumulation_steps`
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Unscale first so the clip threshold applies to TRUE grads,
                # then cap them: the variance penalty's std() gradient blows
                # up as scores collapse to a constant (std -> 0), which NaN'd
                # the weights within a few epochs on run_20260710.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for g in optimizer.param_groups for p in g["params"]), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # ── Per-step W&B perf metrics (logged every 50 steps to reduce overhead) ──
            if batch_idx % 50 == 0:
                # Sync GPU before stopping the timer so we measure real GPU time, not just kernel launch
                torch.cuda.synchronize()
                t_forward_end = time.perf_counter()
                
                data_ms   = (t_data_end   - t_batch_start) * 1000
                forward_ms = (t_forward_end - t_forward_start) * 1000
                total_ms   = data_ms + forward_ms
                batch_sz   = images.size(0)
                
                step_log = {
                    "perf/data_load_ms":    data_ms,
                    "perf/forward_ms":      forward_ms,
                    "perf/gpu_util_ratio":  forward_ms / total_ms if total_ms > 0 else 0,
                    "perf/samples_per_sec": batch_sz / (total_ms / 1000) if total_ms > 0 else 0,
                    # Live loss internals at this step (not averaged — useful for spotting instability)
                    **{f"step/{k.split('/')[1]}": v for k, v in diag.items()},
                }
                wandb.log(step_log)
            
            t_batch_start = time.perf_counter()  # reset for next batch's data-load window
            
            # De-scale loss for logging
            step_loss = loss.item() * accumulation_steps
            running_train_loss += step_loss * images.size(0)
            
            # Pairwise Concordance: fraction of valid cross-sectional pairs correctly ranked
            with torch.no_grad():
                anchor_year = years[0].item()
                cs_mask = (years == anchor_year)
                cs_out = outputs.squeeze(-1)[cs_mask]
                cs_lbl = scores_lbl[cs_mask]
                cs_geo = geoids[cs_mask]
                B_cs = cs_out.shape[0]
                if B_cs >= 2:
                    idx_k, idx_l = torch.triu_indices(B_cs, B_cs, offset=1, device=device)
                    # [NEW] Exclude same-tract comparisons from accuracy!
                    valid_acc_mask = cs_geo[idx_k] != cs_geo[idx_l]
                    if valid_acc_mask.any():
                        idx_k = idx_k[valid_acc_mask]
                        idx_l = idx_l[valid_acc_mask]
                        pred_sign = torch.sign(cs_out[idx_l] - cs_out[idx_k])
                        true_sign = torch.sign(cs_lbl[idx_l] - cs_lbl[idx_k])
                        correct = (pred_sign == true_sign).float().sum()
                        running_train_correct += correct.item()
                        total_acc_pairs_epoch += idx_k.shape[0]
            
            samples_seen = (batch_idx + 1) * images.size(0)
            running_avg = running_train_loss / max(samples_seen, 1)

            # Live loss update in the bar
            train_bar.set_postfix(
                loss=f"{step_loss:.4f}",
                pairs=f"{diag.get('loss/cross_valid_pairs', 0)}"
            )

        epoch_train_loss = running_train_loss / max(len(train_loader.dataset), 1)
        epoch_train_acc = running_train_correct / max(total_acc_pairs_epoch, 1)
        
        # ── Epoch-averaged loss diagnostics ──
        if diag_steps > 0:
            epoch_diag = {k: v / diag_steps for k, v in diag_accum.items()}
            epoch_diag["loss/total_pairs_epoch"] = total_pairs_epoch
            # Check health and warn in console
            har = epoch_diag.get("loss/cross_hinge_active", 0)
            if har < 0.30:
                tqdm.write(f"⚠️  Cross hinge active rate {har:.2f} < 0.30 — m_base may be too low.")
            elif har > 0.80:
                tqdm.write(f"⚠️  Cross hinge active rate {har:.2f} > 0.80 — m_base may be too high, training may be unstable.")
        else:
            epoch_diag = {}
            
        # Always step scheduler at end of epoch
        # Scheduler step moved to after validation for ReduceLROnPlateau compatibility
        # if scheduler:
        #     scheduler.step()

        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        val_losses = {}
        val_spearmans = {}
        val_set_metrics = {}
        if len(val_loaders.values()) > 0:
            for val_name, val_loader in val_loaders.items():
                model.eval() # Set model to eval mode (disables dropout)
                running_val_loss = 0.0
                all_preds = []
                all_labels = []
                all_building_ids = []
                all_years = []
                all_changes = []
                all_cbsas = []
                val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] {val_name}", leave=False)

                # Disable gradient calculation for validation (saves RAM and compute)
                with torch.no_grad():
                    for images, labels, metas, building_ids, val_years, val_structural_changes, val_cbsa_ids in val_bar:
                        images, labels, metas = images.to(device), labels.to(device), metas.to(device)
                        if images.dtype != torch.float32 or images.max() > 10.0:
                            raise RuntimeError(
                                f"\n[NORMALIZATION BUG CAUGHT]\n"
                                f"Image dtype: {images.dtype} | Max pixel value: {images.max().item()}\n"
                                f"ScaleMAE requires float32 tensors with standard ImageNet normalization. "
                                f"If the model receives raw 0-255 uint8 arrays, its attention layers will output pure noise and refuse to learn."
                            )

                        with autocast(device_type='cuda'):
                            outputs = model(images, metadata=metas)
                            if outputs.shape != labels.shape:
                                outputs = outputs.view(labels.shape)
                            # Pointwise MSE for validation so per-image accuracy is interpretable
                            loss = val_loss_fn(outputs, labels)
                        running_val_loss += loss.item() * images.size(0)

                        all_preds.extend(outputs.view(-1).cpu().numpy())
                        all_labels.extend(labels.view(-1).cpu().numpy())
                        if building_ids is not None:
                            all_building_ids.extend(building_ids.cpu().numpy())
                        if val_years is not None:
                            all_years.extend(val_years.cpu().numpy())
                        if val_structural_changes is not None:
                            all_changes.extend(val_structural_changes.cpu().numpy())
                        if val_cbsa_ids is not None:
                            all_cbsas.extend(val_cbsa_ids.cpu().numpy())

                        val_bar.set_postfix(loss=f"{loss.item():.4f}")

                val_losses[val_name] = running_val_loss / len(val_loader.dataset)

                # ── Metrics: spearman, MASD (stable vs changed), DA, pred moments,
                #    plus per-population-bracket and top-city breakdowns (#31) ──
                n = len(all_preds)
                val_df = pd.DataFrame({
                    'building_id': all_building_ids if len(all_building_ids) == n else np.arange(n),
                    'year': all_years if len(all_years) == n else np.zeros(n, dtype=int),
                    'pred': all_preds,
                    'label': all_labels,
                    'change': all_changes if len(all_changes) == n else np.zeros(n, dtype=int),
                    'cbsa': all_cbsas if len(all_cbsas) == n else np.zeros(n, dtype=int),
                })
                set_metrics = compute_val_metrics(val_df, cbsa_meta=cbsa_meta,
                                                  top_cities=top_cities)
                val_set_metrics[val_name] = set_metrics
                if sel_key in set_metrics:
                    val_spearmans[val_name] = set_metrics[sel_key]
                elif "spearman" in set_metrics:
                    # selection metric unavailable (e.g. every within cell < 5
                    # buildings) — fall back to pooled rather than skip the set
                    tqdm.write(f"⚠️ {val_name}: '{sel_key}' unavailable, "
                               f"using pooled spearman for selection this epoch.")
                    val_spearmans[val_name] = set_metrics["spearman"]

                # Display: Spearman + whatever temporal metrics this set supports
                parts = []
                if "spearman" in set_metrics:
                    parts.append(f"Spearman: {set_metrics['spearman']:.4f}")
                if "within_spearman" in set_metrics:
                    parts.append(f"Within-ρ: {set_metrics['within_spearman']:.4f}")
                if "stable_masd" in set_metrics:
                    parts.append(f"S-MASD: {set_metrics['stable_masd']:.4f}")
                if "changed_masd" in set_metrics:
                    parts.append(f"C-MASD: {set_metrics['changed_masd']:.4f}")
                if "changed_da" in set_metrics:
                    parts.append(f"C-DA: {set_metrics['changed_da']:.2%}")
                tqdm.write(f"{val_name} {' | '.join(parts)}")

            # Use mean validation spearman for early stopping / best model checkpointing
            epoch_val_spearman = sum(val_spearmans.values()) / max(len(val_spearmans), 1)
        else:
            epoch_val_spearman = float('-inf')  # No validation data

        # Step the scheduler (ReduceLROnPlateau needs the metric)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_spearman)
            else:
                scheduler.step()

        # ==========================
        # 3. LOGGING & CHECKPOINTING
        # ==========================
        val_display = " | ".join([f"{k} Spearman: {v:.4f}" for k, v in val_spearmans.items()])
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | {val_display}")

        # 🎯 Log metrics directly to W&B cloud!
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "train_pairwise_acc": epoch_train_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            **epoch_diag,
        }
        if len(val_loaders.values())>0:
            # Flat legacy-style keys for the base metrics; slash keys for the
            # per-bracket / per-city breakdowns and predicted-score moments (#31).
            _base_keys = ("spearman", "stable_masd", "changed_masd", "changed_da")
            for val_name in val_loaders.keys():
                log_dict[f"{val_name}_mse"] = val_losses[val_name]
                for key, value in val_set_metrics.get(val_name, {}).items():
                    if key == "mse":
                        continue  # already logged from the running loss
                    if key in _base_keys:
                        log_dict[f"{val_name}_{key}"] = value
                    else:
                        log_dict[f"{val_name}/{key}"] = value

        wandb.log(log_dict)

        # Model Checkpoint logic (Maximize Spearman)
        if epoch_val_spearman > best_val_spearman:
            tqdm.write(f"⭐ Val Spearman improved from {best_val_spearman:.4f} to {epoch_val_spearman:.4f}. Saving...")
            tqdm.write(f"   📊 [Loss Components] L_cross: {epoch_diag.get('loss/L_cross', 0.0):.4f} | L_stable: {epoch_diag.get('loss/L_stable', 0.0):.4f}")
            best_val_spearman = epoch_val_spearman
            
            model_path = save_dir / f"{savename}_best.pth"
            # Save LoRA adapter + regression head separately.
            # state_dict() tensors are always detached — requires_grad is never set on them,
            # so a filter like `if v.requires_grad` produces an empty dict.
            lora_dir = save_dir / f"{savename}_best_lora"
            model.backbone.save_pretrained(lora_dir)         # saves adapter_config.json + adapter_model.safetensors
            torch.save(model.head.state_dict(), model_path)  # saves the regression head weights

            # Tell wandb to track these files
            wandb.save(str(model_path))
            wandb.save(str(lora_dir / "adapter_model.safetensors"))

            # Report feature importance for the best model
            check_feature_importance(model)


        # Free unused cached VRAM and run garbage collection to prevent fragmentation/OOM/crashes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==========================
        # 4. ROTATE CACHE FOR NEXT EPOCH
        # ==========================
        if cache_manager:
            k = 4
            progress = epoch / max(1, epochs)
            cache_updated = cache_manager.step(k=k, progress=progress)
            if cache_updated:
                swapped = cache_manager.last_swap_count
                if swapped < k:
                    tqdm.write(f"⚠️ Cache rotation degraded: only {swapped}/{k} new train shard(s) swapped in this epoch (generation error or slow fetches — see ❌/[NAIP] lines above).")
                else:
                    tqdm.write(f"🔄 {swapped} new train background shard(s) ready! Reloading pre-packaged pairs from disk...")
                # InBatchRankingDataset.refresh() reloads flat images + metadata tensors from disk
                train_loader.dataset.refresh()

        if val_cache_managers:
            for val_name, v_manager in val_cache_managers.items():
                if v_manager is not None:
                    v_updated = v_manager.step()
                    if v_updated and val_name in val_loaders:
                        pass  # StaticShardedDataset loads one shard at a time lazily — no full refresh needed

        # Save checkpoint after each epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_spearman": best_val_spearman,
            # currency of best_val_spearman: a resume under a different
            # selection metric must reset the tracker (values not comparable)
            "selection_metric": selection_metric,
        }
        torch.save(checkpoint, save_dir / f"{savename}_last.pth")

        # Persist the hard-tract mining state next to the checkpoint so a
        # resumed run keeps its learned tract weights (cold registry = uniform).
        registry = getattr(loss_fn, "hardness_registry", None)
        if registry is not None and len(registry) > 0:
            registry.save(save_dir / f"{savename}_hardness.json")

    # Return model for caller chaining (especially if we loaded/checkpointed externally)
    return model

import queue
import threading
import pandas as pd

def predict_buildings_chunked(model, df, all_years_datasets, params,
                              device, output_path, eval_transform, verbose=True):
    """
    Generate predictions using a producer-consumer pipeline that keeps the GPU
    near 100% utilisation.

    Architecture
    ────────────
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PRODUCER THREAD                                                        │
    │                                                                         │
    │  For each spatial groupby-chunk K:                                      │
    │    ① Wait for group K's zarr preload to finish  (SSD → RAM)           │
    │    ② Kick off group K+1's preload in background (overlaps with ③④)    │
    │    ③ Extract + subsample images in parallel     (EXTRACT_WORKERS)      │
    │    ④ Pack full batches → push to batch_queue                           │
    └──────────────────────────────┬──────────────────────────────────────────┘
                                   │  queue.Queue (bounded = QUEUE_DEPTH)
                                   │  (blocks producer when GPU is slow →
                                   │   provides back-pressure automatically)
    ┌──────────────────────────────▼──────────────────────────────────────────┐
    │  CONSUMER  (main thread)                                                │
    │    ① Pull batch from queue                                              │
    │    ② Move tensors to GPU with non_blocking=True                        │
    │    ③ autocast forward pass                                              │
    │    ④ Append results to CSV                                              │
    └─────────────────────────────────────────────────────────────────────────┘

    The GPU is always either running inference or waiting for the very next
    batch.  Because the queue is pre-filled (QUEUE_DEPTH deep) the typical
    case is that a new batch is ready the moment the previous one finishes.

    Args:
        model:              PyTorch model (will be set to eval mode)
        df:                 DataFrame with all buildings to predict
        all_years_datasets: dict of zarr datasets keyed by dataset name
        params:             config dict — batch_size, subsample_step, nbands
        device:             torch.device
        output_path:        CSV file path for results
        eval_transform:     torchvision transform (scale + normalise)
        verbose:            print summary stats when done
    """
    if all_years_datasets is None:
        raise ValueError(
            "predict_buildings_chunked is zarr-only; the NAIP/ms_us path uses "
            "src.prediction.predict_year_chunked instead."
        )
    model.eval()

    # ── Hyperparameters ───────────────────────────────────────────────────────
    batch_size      = params.get("batch_size", 32) * 8   # scale up for eval
    image_size      = int((df["row_stop"] - df["row_start"]).min())
    nbands          = params["nbands"]
    subsample_step  = params["subsample_step"]
    chunk_dims      = (nbands, 2500, 2500)
    groupby_chunk_size = 2500

    # Number of threads for each stage.  These are I/O or numpy-bound so
    # threads (not processes) are the right tool — numpy releases the GIL.
    PRELOAD_WORKERS = 8    # SSD → RAM (zarr → numpy), fully I/O-bound
    EXTRACT_WORKERS = 16   # RAM slice + subsample, numpy-bound (releases GIL)
    QUEUE_DEPTH     = 12   # batches buffered; tune down if RAM is tight

    # ── Setup ─────────────────────────────────────────────────────────────────
    df = assign_groupby_chunk_ids(df, image_size, groupby_chunk_size)
    cache = ZarrChunkCache(max_memory_gb=5.0)   # thread-safe after our rewrite

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    groups      = list(df.groupby('groupby_chunk_id'))
    batch_queue = queue.Queue(maxsize=QUEUE_DEPTH)

    if verbose:
        print(f"Processing {len(df)} buildings across {len(groups)} groupby chunks...")
        print(f"  Pipeline: PRELOAD_WORKERS={PRELOAD_WORKERS}  "
              f"EXTRACT_WORKERS={EXTRACT_WORKERS}  "
              f"QUEUE_DEPTH={QUEUE_DEPTH}  "
              f"batch_size={batch_size}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _preload_group(df_grp):
        """Load all zarr chunks needed by df_grp into the shared cache."""
        rows_list = df_grp.to_dict('records')   # avoids slow iterrows()

        # Collect unique (dataset, chunk_row, chunk_col) triples
        needed = set()
        for row in rows_list:
            img_sz = int(row["row_stop"] - row["row_start"])
            for cr, cc in get_zarr_chunks_for_image(row, chunk_dims, img_sz):
                needed.add((row["dataset"], cr, cc))

        to_load = [(ds, cr, cc) for ds, cr, cc in needed
                   if cache.get(ds, cr, cc) is None]
        if not to_load:
            return

        def _load_one(item):
            ds, cr, cc = item
            zarr_arr = all_years_datasets[ds]["value"]
            _, ch, cw = chunk_dims
            try:
                arr = zarr_arr[:nbands,
                               cr * ch:(cr + 1) * ch,
                               cc * cw:(cc + 1) * cw]
                return ds, cr, cc, arr.to_numpy()
            except Exception as e:
                logging.error(f"Preload failed ({ds},{cr},{cc}): {e}")
                return ds, cr, cc, None

        with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as pool:
            for ds, cr, cc, arr in pool.map(_load_one, to_load):
                if arr is not None:
                    cache.put(ds, cr, cc, arr)

    def _extract_one(row):
        """Extract, validate, and subsample one image. Returns (tensor, row) or None."""
        if pd.isna(row.get("Rel_Score")):
            return None
        raw = extract_image_from_chunks(
            row, cache, None, all_years_datasets, chunk_dims, image_size, nbands
        )
        if raw is None:
            return None
        # Subsample on CPU (cheap numpy op); keep as uint8 until eval_transform
        tensor = torch.from_numpy(raw)[:, ::subsample_step, ::subsample_step]
        return tensor, row

    def _pack_batch(items):
        """Stack a list of (tensor, row) into GPU-ready tensors + metadata lists."""
        tensors, rows = zip(*items)
        # eval_transform: uint16→float32 scale + ImageNet normalise
        batch_tensor = eval_transform(torch.stack(tensors))
        metas_tensor = torch.tensor(
            [r.get("dist_to_center", 0.0) for r in rows],
            dtype=torch.float32
        ).unsqueeze(1)
        return (
            batch_tensor,
            metas_tensor,
            [r["Rel_Score"]          for r in rows],
            [r.get("building_id", 0)    for r in rows],
            [str(r.get("GEOID", "")) for r in rows],
            [r.get("year", 0)        for r in rows],
            [str(r.get("type", ""))  for r in rows],
        )

    # ── Producer thread ───────────────────────────────────────────────────────
    def _producer():
        """
        Iterates spatial groups.  For each group:
          1. Waits for that group's preload to finish.
          2. Immediately fires off the *next* group's preload in background.
          3. Extracts images in parallel and pushes packed batches to the queue.

        The queue's maxsize creates automatic back-pressure: if the GPU falls
        behind, put() blocks here, throttling extraction so we don't waste RAM
        storing thousands of pre-built batches.
        """
        # One persistent thread for background preloading of the next group
        preload_executor = ThreadPoolExecutor(max_workers=1)
        extract_executor = ThreadPoolExecutor(max_workers=EXTRACT_WORKERS)

        try:
            # Kick off preload for group 0 before the loop starts
            pending_preload = preload_executor.submit(_preload_group, groups[0][1])

            for i, (chunk_id, df_grp) in enumerate(groups):

                # ① Block until this group's zarr chunks are in RAM
                pending_preload.result()

                # ② Fire preload for next group immediately (overlaps with ③)
                if i + 1 < len(groups):
                    pending_preload = preload_executor.submit(
                        _preload_group, groups[i + 1][1]
                    )

                # ③ Submit all extractions for this group concurrently
                rows_list = df_grp.to_dict('records')
                futures   = [extract_executor.submit(_extract_one, r) for r in rows_list]

                # ④ As results arrive, accumulate and push full batches
                pending = []
                for fut in as_completed(futures):
                    result = fut.result()
                    if result is None:
                        continue
                    pending.append(result)
                    if len(pending) >= batch_size:
                        batch_queue.put(_pack_batch(pending))
                        pending = []

                # Flush any remainder from this group
                if pending:
                    batch_queue.put(_pack_batch(pending))

        except Exception as e:
            logging.error(f"Producer thread crashed: {e}", exc_info=True)
        finally:
            preload_executor.shutdown(wait=False)
            extract_executor.shutdown(wait=False)
            batch_queue.put(None)   # sentinel: tells consumer we're done

    producer_thread = threading.Thread(target=_producer, daemon=True,
                                       name="zarr-producer")
    producer_thread.start()

    # ── Consumer (main thread): GPU inference + CSV write ─────────────────────
    first_write  = True
    total_approx = max(1, len(df) // batch_size)

    with torch.no_grad():
        pbar = tqdm(total=total_approx, desc="Generating predictions", leave=False)
        while True:
            item = batch_queue.get()
            if item is None:
                break   # producer finished

            batch_tensor, metas_tensor, labels, building_ids, geoids, years, types = item

            # non_blocking=True overlaps H2D transfer with prior GPU work
            batch_tensor = batch_tensor.to(device, non_blocking=True)
            metas_tensor = metas_tensor.to(device, non_blocking=True)

            with autocast(device_type='cuda'):
                outputs = model(batch_tensor, metadata=metas_tensor)

            preds = outputs.view(-1).cpu().numpy()

            pd.DataFrame({
                "Rel_Score":       labels,
                "predicted_value": preds,
                "building_id":        building_ids,
                "GEOID":           geoids,
                "year":            years,
                "type":            types,
            }).to_csv(output_path,
                      mode='w' if first_write else 'a',
                      header=first_write,
                      index=False)
            first_write = False
            pbar.update(1)

        pbar.close()

    producer_thread.join()

    if verbose:
        stats = cache.get_stats()
        print(f"\n✅ Predictions saved to {output_path}")
        print(f"   Cache — chunks: {stats['num_chunks']} | "
              f"hit rate: {stats['hit_rate_pct']:.1f}% | "
              f"RAM used: {stats['total_bytes_mb']:.0f} MB | "
              f"loads: {stats['loads']}")


def run(
    params=None,
    train=True,
    compute_loss=True,
    generate_predictions=False,
    retrain=False,
    evaluate=False,
):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    params = fill_params_defaults(params)

    model_name = params["model_name"]
    kind = params["kind"]
    weights = params["weights"]
    image_size = params["image_size"]
    nbands = params["nbands"]
    small_sample = params["small_sample"]
    n_epochs = params["n_epochs"]
    learning_rate = params["learning_rate"]
    sat_data = params["sat_data"]
    years = params["years"]
    batch_size = params["batch_size"]
    tau_meters_requested = params.get("tau_meters", 100)
    max_jitter = params.get("max_jitter", 10)

    # Override tau_meters with the exact value that makes the raw zarr tile an exact
    # multiple of image_size.  This lets us subsample every Nth pixel instead of
    # interpolating, completely eliminating antialiasing artifacts.
    tau_meters, subsample_step = geo_utils.calculate_exact_tau(
        tau_meters_requested, image_size
    )
    params["tau_meters"] = tau_meters
    params["subsample_step"] = subsample_step
    print(f"📐 Exact tau override: {tau_meters_requested}m → {tau_meters:.2f}m  "
          f"(subsample step N={subsample_step}, raw tile = {subsample_step * image_size}px)")

    savename = generate_savename(params.get("run_id"))
    log_dir = f"{LOGS_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    generate_parameters_log(params, savename)
    
    print("\n" + "="*80)
    print(f"🚀 STARTING RUN: {savename}")
    print("="*80 + "\n")
    print(f"- Train: {train} \n- Compute Loss: {compute_loss} \n- Generate Predictions: {generate_predictions} \n- Retrain: {retrain} \n- Evaluate: {evaluate}\n")
    print("="*84 + "\n")

    if train:

        all_years_datasets, all_years_extents, df_train, df_vals_dict, df_test, df_dead_zone = build_dataset.generate_datasets(
            savename, sat_data, years, small_sample=small_sample, tau_meters=tau_meters,
            indicator=params["indicator"], footprints_source=params["footprints_source"], states=params["states"],
            naip_coverage_csv=params.get("naip_coverage_csv"),
        )

        # Per-city split metadata (#28/#31): temporal-holdout years feed the NAIP
        # actual-year guards below; brackets/populations feed the val metrics.
        cbsa_meta = cbsa_brackets.load_city_split()
        holdout_years = cbsa_brackets.holdout_year_map(cbsa_meta)

        #### 1. Setup resume logic for model/cache
        checkpoint_dir = MODELS_DIR / "models_by_epoch" / savename
        best_checkpoint_path = checkpoint_dir / f"{savename}_best.pth"
        last_checkpoint_path = checkpoint_dir / f"{savename}_last.pth"

        resume_cache = False    
        resume_model_checkpoint = None

        if not retrain and checkpoint_dir.exists():
            if last_checkpoint_path.exists():
                resume_model_checkpoint = last_checkpoint_path
                resume_cache = True
                print(f"🟢 Resuming from checkpoint: {last_checkpoint_path}")
            elif best_checkpoint_path.exists():
                resume_model_checkpoint = best_checkpoint_path
                resume_cache = True
                print(f"🟡 Found best weights only: {best_checkpoint_path} (resume with no optimizer state)")
        else:
            print(f"{checkpoint_dir} does not exist. Starting fresh training run.")

        # Cache-only resume: a previous run with the SAME savename may have crashed
        # while still building the NAIP shard caches, i.e. before any model
        # checkpoint existed. The marker records which run owns the shards on disk;
        # if it matches, keep them and let build_initial_cache() top up the missing
        # ones instead of refetching everything. retrain=True still forces a full
        # rebuild, and a different savename (new run_id / split) clears as before.
        if not retrain and not resume_cache and read_cache_marker() == savename:
            resume_cache = True
            print(f"🟡 No model checkpoint, but cache marker matches '{savename}': "
                  f"resuming partial shard cache instead of rebuilding.")
        write_cache_marker(savename)

        print("Building Initial Cache...")
        if small_sample:
            num_shards = 1
            current_shard_size = len(df_train) # E.g., ~150 images per shard
        else:
            num_shards = 10
            current_shard_size = CACHE_SIZE // 10 # E.g., 20,480 images per shard

        # Gradient hard-tract mining registry (see src/data/tract_sampling.py):
        # written by the loss (per-tract |lambda| EMAs), read by the train
        # shard generator as tract sampling weights. Requires tract_sampling;
        # alpha=0 disables the bias (registry still logs hardness diagnostics).
        hardness_registry = None
        if params.get("tract_sampling", False):
            hardness_registry = GradientHardnessRegistry(
                alpha=params.get("grad_mining_alpha", 0.0),
                beta=params.get("grad_mining_beta", 0.9),
                cap=params.get("grad_mining_cap", 10.0),
            )
            hardness_path = checkpoint_dir / f"{savename}_hardness.json"
            if resume_model_checkpoint is not None and hardness_registry.load(hardness_path):
                print(f"🟢 Restored hardness registry ({len(hardness_registry):,} tracts) "
                      f"from {hardness_path}")

        print("\n[TRAIN] Building cyclic training cache...")
        train_cache_manager = CyclicCacheManager(
            df=df_train,
            all_years_datasets=all_years_datasets,
            params=params,
            cache_dir=CACHE_DIR,
            single_shard_mode=small_sample, # If small_sample is True, keep only one shard to speed up testing
            num_shards=num_shards,
            shard_size=current_shard_size,
            type="train",
            clear_cache=not resume_cache,
            max_jitter=max_jitter,
            sat_data=sat_data,
            holdout_years=holdout_years,
            hardness_registry=hardness_registry,
        )
        train_cache_manager.build_initial_cache()

        print("\n[VAL] Building static validation cache...")
        vals_cache_manager_dict = {}
        for val_name, df_val in df_vals_dict.items():
            if df_val.shape[0] > 0:
                # Subsample buildings (not rows) so every kept building retains ALL
                # its years: multi-year buildings are what make the stable/changed
                # MASD and directional-accuracy metrics computable inside val_cities.
                # (Lazy ms_us vals arrive pre-sampled at 1/tract, so the per-tract
                # pick is a no-op there; the building-level shuffle still applies.)
                df_val = _subsample_val_buildings(df_val)
                # Cap the static val cache at MAX_VAL_SHARDS shards: full coverage
                # would need ceil(len/shard_size) shards — far more val imagery than
                # the metrics need. df_val arrives building-shuffled, so truncation
                # to MAX_VAL_SHARDS * shard_size rows is a random building sample.
                # 3 shards ≈ 60k rows ≈ 3.8k tracts → per-year Spearman CI ≈ ±0.03,
                # tighter on the multi-year composite — enough for epoch tracking.
                MAX_VAL_SHARDS = 3
                val_num_shards = max(1, (len(df_val) + current_shard_size - 1) // current_shard_size)
                val_num_shards = min(val_num_shards, MAX_VAL_SHARDS)
                val_cache_manager = CyclicCacheManager(
                    df=df_val, # Or full df_val
                    all_years_datasets=all_years_datasets,
                    params=params,
                    cache_dir=CACHE_DIR,
                    num_shards=val_num_shards,  # Build multiple shards to cover the whole val set
                    shard_size=current_shard_size,  # Smaller shard size avoids RAM explosion during generation
                    single_shard_mode=True,     # IMPORTANT: This disables active rotation / generation
                    type=val_name,
                    clear_cache=not resume_cache,
                    sat_data=sat_data,
                    holdout_years=holdout_years,
                )
                val_cache_manager.build_initial_cache() # This will build and show a progress bar
                vals_cache_manager_dict[val_name] = val_cache_manager
            else:
                val_cache_manager = None

        #### 2. PyTorch Data Pipeline Setup
        print("Setting up data generators...")

        # Fresh runs only: dump the first 30 training batches (raw shard images +
        # metadata) for visual QA of the ingested data. Inspect the dump with
        # src/notebooks/visualize_debug_batches.ipynb.
        batch_dumper = None
        if resume_model_checkpoint is None:
            batch_dumper = BatchImageDumper(RESULTS_DIR / savename / "debug_batches")
            batch_dumper.write_run_info(params, savename)
            ref_source = (df_train.building_reference()
                          if isinstance(df_train, LazyPairTable) else df_train)
            batch_dumper.write_building_reference(ref_source, cbsa_meta=cbsa_meta)
            print(f"📸 Fresh run: dumping first {batch_dumper.max_batches} training batches to {batch_dumper.out_dir}")

        train_loader, val_loaders, test_loader = setup_dataloaders(
            df_train=df_train, dfs_val_dict=df_vals_dict, df_test=df_test,
            all_years_datasets=all_years_datasets, params=params,
            train_cache_manager=train_cache_manager, val_cache_manager=vals_cache_manager_dict,
            batch_dumper=batch_dumper,
        )

        del df_train, df_test, all_years_extents
        gc.collect() # Force Python to free up memory from large objects we no longer need
        
        train_cache_manager.start_background_generation() # Starts generating shard 6 for Epoch 1

        print("Data Pipeline Ready!")

        #### 3. Model Initialization
        # meta_dim=0: dist_to_center disabled. With meta_dim=1 the ordinal loss +
        # variance regularizer were minimizable through the time-invariant scalar
        # covariate alone, and weight decay ground the image pathway to zero
        # (frozen val spearman, stable MASD == 0). Force the model to rank from pixels.
        model, loss_fn = set_model_and_loss_function(
            model_name=model_name,
            kind=kind,
            bands=nbands,
            image_size=image_size,
            weights=weights,
            meta_dim=0,
            m_base=params.get("m_base", 1.0),
            m_min=params.get("m_min", 0.1),
            lambda_s=params.get("lambda_s", 1.0),
            lambda_var=params.get("lambda_var", 1.0),
            hardness_registry=hardness_registry,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='max', factor=0.5, patience=30, min_lr=learning_rate/20
        # )
        scheduler = None
        # If we found a saved checkpoint from a previous run, resume from it
        start_epoch = 0
        initial_best_val_loss = None
        if not retrain and resume_model_checkpoint is not None and resume_model_checkpoint.exists():
            print(f"➡️ Resuming model/optimizer from checkpoint: {resume_model_checkpoint}")
            checkpoint = torch.load(resume_model_checkpoint, map_location=device, weights_only=False)
            
            # Restore all states
            if "model_state_dict" in checkpoint:
                # Full training checkpoint (_last.pth): restore model + optimizer + scheduler
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    # The checkpoint restores the old lr; apply the explicit decay
                    # (params["resume_lr_override"]) without touching moments/epoch.
                    override_optimizer_lr(optimizer, params.get("resume_lr_override"))
                if scheduler and checkpoint.get("scheduler_state_dict"):
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                    # Reset patience counter to wait a full 20 epochs in the new run before reducing LR
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.num_bad_epochs = 0
                        print("Reset ReduceLROnPlateau num_bad_epochs to 0 for the new run.")

                start_epoch = checkpoint.get("epoch", 0)
                initial_best_val_loss = checkpoint.get("best_val_spearman")
                # best_val_spearman is denominated in the checkpoint's selection
                # metric; if the metric changed (e.g. pooled -> within), the old
                # best is not comparable — reset so _best.pth saving restarts.
                ckpt_sel = checkpoint.get("selection_metric", "pooled")
                if ckpt_sel != params["selection_metric"] and initial_best_val_loss is not None:
                    print(f"🔁 Selection metric changed ('{ckpt_sel}' -> "
                          f"'{params['selection_metric']}'): resetting best-Spearman tracker "
                          f"(was {initial_best_val_loss:.4f}).")
                    initial_best_val_loss = None

            else:
                # Best-model checkpoint (_best.pth): head weights only; load LoRA adapter separately
                model.head.load_state_dict(checkpoint)
                lora_dir = resume_model_checkpoint.parent / f"{resume_model_checkpoint.stem}_lora"
                if lora_dir.exists():
                    from peft import PeftModel
                    model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model, lora_dir)
                    print(f"✅ LoRA adapter loaded from {lora_dir}")
                start_epoch = 0
                initial_best_val_loss = None
            print(f"✅ Resumed at epoch {start_epoch+1} with best_val_loss={initial_best_val_loss}...")

        # Only ask wandb to resume a specific run id when we're actually resuming a
        # model checkpoint. Otherwise there's nothing to resume on the wandb side,
        # and passing id=savename risks colliding with a run id that was deleted on
        # the web UI — wandb then retries the resulting HTTP 409 internally with
        # backoff for minutes before giving up, which just hangs a fresh run.
        wandb_resuming = not retrain and resume_model_checkpoint is not None and resume_model_checkpoint.exists()
        init_wandb_run(savename, params, wandb_resuming)

        #### 4. Run PyTorch Model
        model = train_model(
            model=model, train_loader=train_loader, val_loaders=val_loaders,
            loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, 
            epochs=n_epochs, device=device, savename=savename,
            cache_manager=train_cache_manager,
            start_epoch=start_epoch,
            initial_best_val_loss=initial_best_val_loss,
            val_cache_managers=vals_cache_manager_dict,
            cbsa_meta=cbsa_meta,
            selection_metric=params["selection_metric"],
        )
        
        wandb.finish()
        print("Fin del entrenamiento")
    
    if generate_predictions:
        print("Generando predicciones...")

        if params["predict_split"] not in ("test", "all"):
            raise ValueError(
                f"predict_split must be 'test' or 'all', got {params['predict_split']!r}"
            )

        pred_pair_table = None
        pred_split_map = None
        pred_holdout_years = None
        if params["footprints_source"] == "ms_us" and not small_sample:
            # Normalized path: ONE lazy table over the full universe; per-year
            # frames are materialized inside the year loop below (the legacy
            # all-splits concat is ~575M rows at US scale and OOMs).
            pred_pair_table = build_dataset.load_income_dataset(
                years, tau_meters=tau_meters, indicator=params["indicator"],
                footprints_source="ms_us", states=params["states"],
            )
            all_years_datasets = None  # NAIP-only path
            try:
                city_split = cbsa_brackets.load_city_split()
                pred_split_map = dict(zip(
                    city_split["cbsa_code"].astype(str), city_split["split"]
                ))
                pred_holdout_years = cbsa_brackets.holdout_year_map(city_split)
            except FileNotFoundError:
                if params["predict_split"] == "test":
                    raise RuntimeError(
                        "predict_split='test' needs cbsa_splits.feather to know "
                        "which CBSAs are the test set — generate the split first, "
                        "or use predict_split='all'."
                    )
                print("⚠️ cbsa_splits.feather not found — prediction 'type' column will be 'all'.")
        else:
            all_years_datasets, all_years_extents, df_train, df_vals_dict, df_test, df_dead_zone = build_dataset.generate_datasets(
                savename, sat_data, years, small_sample=small_sample, tau_meters=tau_meters,
                indicator=params["indicator"], footprints_source=params["footprints_source"], states=params["states"],
                naip_coverage_csv=params.get("naip_coverage_csv"),
            )

            # Combine all dataframes
            val_dfs = list(df_vals_dict.values())
            df_all = pd.concat([df_train, df_test, df_dead_zone] + val_dfs, ignore_index=True)
            # df_all = pd.concat(val_dfs, ignore_index=True)
            del df_train, df_test, df_dead_zone, val_dfs, df_vals_dict
            gc.collect()

        # 1. Load the Best PyTorch Model
        model, _ = set_model_and_loss_function(
            model_name=model_name,
            kind=kind,
            bands=nbands, 
            image_size=image_size,
            meta_dim=0,  # Must match training (dist_to_center covariate disabled)
        )
        
        best_model_path = MODELS_DIR / "models_by_epoch" / savename / f"{savename}_best.pth"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Cannot generate predictions: The best model weights at {best_model_path} do not exist. Did the model finish training?")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Best-model loading: Load head + Load LoRA separately
        model.head.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        lora_dir = best_model_path.parent / f"{best_model_path.stem}_lora"
        if lora_dir.exists():
            from peft import PeftModel
            # Wrap the base model (unwrapped backbone) with the best lora adapter
            model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model, lora_dir)
            print(f"✅ Best LoRA adapter loaded from {lora_dir}")
        
        model.to(device)

        # 3. Setup the Evaluation Transforms
        # MUST match the pipeline used during training/validation exactly:
        # scale uint16→float32, then apply ImageNet normalization.
        nbands = params.get("nbands", 4)
        mean = [0.485, 0.456, 0.406] + [0.5] * max(0, nbands - 3)
        std = [0.229, 0.224, 0.225] + [0.5] * max(0, nbands - 3)
        eval_transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std)  # 🔴 ImageNet Normalization — must match training
        ])
        
        # 4. Process each year. NAIP path: resumable chunked prediction with
        # one manifest-guarded chunk cache per predict_split (src/prediction.py).
        if pred_pair_table is not None:
            from src import prediction
            # Chunk cache on WSL-native disk (CACHE_DIR, ext4): the resumable
            # chunk parquets are many small writes, which are slow through the
            # /mnt/c NTFS bridge. Durable outputs ({year}_predictions.csv)
            # still land in RESULTS_DIR.
            pred_chunk_root = CACHE_DIR / "pred_chunks" / savename / params["predict_split"]
            prediction.init_chunk_root(
                pred_chunk_root,
                prediction.prediction_fingerprint(params, best_model_path),
            )
            pred_years = years
        else:
            pred_years = [2016, 2018, 2020, 2022, 2024, 2010, 2012, 2014]

        # Return freed heap pages to the OS between years. gc.collect() alone
        # only frees at the Python level; repeatedly building/freeing the
        # ~71.8M-row per-year frames fragments glibc's malloc arenas so RSS
        # creeps up every year until materialize_year OOM-kills the process
        # (observed mid-run). malloc_trim(0) hands the freed pages back.
        def _release_memory():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass  # non-glibc platform; gc.collect() above is the fallback

        for year in pred_years:
            if year<=2015:
                continue  # Skip pre-2016 years for now; already computed
            print(f"\n{'='*80}")
            print(f"🚀 Processing Predictions for Year: {year}")
            print(f"{'='*80}")

            _release_memory()  # reclaim last year's frames before the next big alloc

            if pred_pair_table is not None:
                df_year = pred_pair_table.materialize_year(year)
                df_year = prediction.assign_prediction_types(
                    df_year, year, pred_split_map, pred_holdout_years
                )
                # Split filter + evaluation sampling (max(10%,100) tracts per
                # CBSA x <=100 buildings per tract, deterministic and
                # year-stable) + the NYC full-universe bypass for the CSA
                # event study. See prediction.select_prediction_rows.
                df_year = prediction.select_prediction_rows(df_year, params)
            else:
                df_year = df_all[(df_all["year"] == year)].copy()
            if df_year.empty:
                print(f"No data for year {year}, skipping.")
                continue

            print(f"Found {len(df_year)} buildings to predict for {year}")

            output_path = RESULTS_DIR / f"{savename}/{year}_predictions.csv"

            if pred_pair_table is not None:
                produced = prediction.predict_year_chunked(
                    model=model,
                    df_year=df_year,
                    year=year,
                    params=params,
                    device=device,
                    eval_transform=eval_transform,
                    chunk_root=pred_chunk_root,
                    output_csv=output_path,
                    max_workers=int(params.get("predict_fetch_workers", 16)),
                    verbose=True,
                )
                if not produced:
                    print(f"No labeled rows for year {year}, skipping.")
                    continue
            else:
                # Legacy zarr path (doitt_nyc / small_sample only)
                predict_buildings_chunked(
                    model=model,
                    df=df_year,
                    all_years_datasets=all_years_datasets,
                    params=params,
                    device=device,
                    output_path=output_path,
                    eval_transform=eval_transform,
                    verbose=True
                )

            df_result = pd.read_csv(output_path)

            if len(df_result) == 0:
                print(f"No valid predictions generated for year {year}.")
                continue

            # Save georeferenced predictions. Polygon geometries only exist for the
            # legacy DoITT source; the MS index is centroid-only (polygons live in
            # the cold buildings_polygons store) so we skip the polygon join there.
            df_result = df_result.set_index("building_id")
            if params["footprints_source"] == "doitt_nyc":
                tag = f"{params['footprints_source']}_{params['indicator']}_epsg{geo_utils.METRIC_EPSG}"
                gdf = gpd.read_parquet(
                    PROCESSED_DATA_DIR / f"building_geometries_{tag}_years{min(years)}-{max(years)}.parquet"
                )
                gdf = gdf.join(df_result, how="inner")
                gdf.to_parquet(RESULTS_DIR / f"{savename}/predictions_{year}.parquet")

            # Dissolve by census tract
            df_result_tracts = df_result.groupby("GEOID").agg({
                "Rel_Score": "mean",
                "predicted_value": ["mean", "std"]
            }).reset_index()
            df_result_tracts.columns = ["GEOID", "Rel_Score", "predicted_value", "predicted_value_std"]

            df_result_tracts.to_parquet(RESULTS_DIR / f"{savename}/predictions_by_tract_{year}.parquet")

            print(f"Finished evaluating {len(df_result)} valid buildings for year {year} at {RESULTS_DIR}")

            # Drop large per-year objects before the next iteration — repeated
            # buildup of df_year/df_result/tract frames across ~8 years is the
            # likely cause of the WSL OOM kill observed mid-run.
            del df_year, df_result, df_result_tracts
            if params["footprints_source"] == "doitt_nyc":
                del gdf
            _release_memory()

    if evaluate:
        # Must run AFTER generate_predictions — it consumes the per-year
        # prediction CSVs / tract parquets written above. Imported lazily so an
        # evaluation-only import error can never break training/prediction.
        print("\n" + "=" * 80 + "\n📊 EVALUATION\n" + "=" * 80)
        try:
            from src.evaluation import run_evaluation
            run_evaluation(savename, params)
        except Exception:
            import traceback
            traceback.print_exc()
            print("⚠️ Evaluation failed; prediction artifacts are intact — rerun with: "
                  f"python -m src.evaluation --savename {savename}")

if __name__ == "__main__":

    variable = "avg_hh_income"

    # Selection of parameters
    params = {
        "model_name": "scalemae",
        "kind": "reg",
        "weights": None,
        "image_size": 224,
        "tau_meters": 100,
        "nbands": 3,
        "batch_size": 8,
        "small_sample": False,
        "n_epochs": 750,
        "learning_rate": 0.0001,
        "sat_data": "NAIP",
        # Every year the data can support, derived from the annual ACS panel
        # (2011-2023) plus one clamped edge year each side (2010/2024 labels
        # clamp to the nearest vintage). NOT the legacy even-years grid — that
        # was a relic of the biennial NYC orthos; NAIP availability is
        # discovered per (region, year) by the exact-year resolve
        # (no same-year flight => no_year_match -> NaN).
        "years": list(range(min(ACS_PANEL_YEARS) - 1, max(ACS_PANEL_YEARS) + 2)),
        # US-scale data selection
        "indicator": "W2_r5",          # W2 occupant wealth, rho=0.05 (r_k=0.045)
        "footprints_source": "ms_us",  # Microsoft US Building Footprints index
        "states": None,                # None = all states present in buildings_index
        "predict_split": "test",       # main run predicts test CBSAs only; "all" = every selected city
        "predict_exact_year": True,    # forbid flight-year substitution: no same-year NAIP flight => no_year_match -> NaN, never the closest year's imagery.
        "predict_tract_sample_frac": 0.10,   # per CBSA: max(ceil(frac*n_tracts), min) tracts, hash-deterministic (None = all)
        "predict_tract_sample_min": 100,     # tract floor per CBSA (bounds small-city CIs)
        "predict_buildings_per_tract": 100,  # <=100 buildings per sampled tract (None = all); CI +-0.098 at sigma_within=0.5
        "predict_full_universe_geoid_prefixes": ("36005", "36047", "36061", "36081", "36085"),  # NYC boroughs: full universe for the CSA event study
        "predict_fetch_workers": 16,   # NAIP fetch shards per chunk. Each shard opens each (DOQQ, tract) run's COG once and reads crops per-window from that shared handle (per-crop beat the old mosaic 1.6-4x on real tracts). Keep near the tract count, not maxed out. Measured knee (src/probe_naip_concurrency.py): PC is not request-count throttling; read throughput peaks ~16-32 workers and DEGRADES past it. Scale out across processes/machines to go faster, not up.
        "predict_batch_size": 512,     # eval-only batch (None = batch_size*8); OOM auto-halves and sticks, so aggressive is safe
        "predict_fetch_pipeline": 3,   # chunks fetched concurrently ahead of the GPU (~0.8GB RAM each); 1 = legacy two-in-flight
        "predict_retry_fast_fail_frac": 0.01,  # <=1% transient failures: one immediate retry then drop (data problem), no cooldown ladder
        "search_grid_meters": 20000,   # STAC search cache cell (EPSG:5070 m): one search per ~20km cell serves all its tracts. Larger = fewer/bigger searches.
        "persist_search_cache": True,  # persist STAC searches to CACHE_DIR/naip_search_cache so reruns are ~search-free.
        # In-Batch Ranking hyperparameters
        "m_base": 1.0,
        "m_min": 0.05,
        "lambda_s": 0.3,
        "lambda_var": 1.5,
        "temporal_fraction": 0.4,
        # Tract-first shards + gradient hard-tract mining (NYC-collapse fix:
        # building-weighted sampling starved intra-city pairs — boroughs are
        # 47% of NYC-metro tracts but 13% of buildings, Manhattan 0.17%).
        "tract_sampling": True,
        "grad_mining_alpha": 0.3,
        "run_id": "run_20260710",  # default run_{YYYYMMDD}: same-day restarts share a savename and resume the shard cache
        "resume_lr_override": 3e-5,  # lr decay on resume (plateau since ~ep 175 at 1e-4); optimizer state otherwise untouched
        "selection_metric": "within",  # checkpoint on mean within-city (CBSA x year) spearman; resets best tracker on first resume
    }

    # Run full pipeline
    run(params, train=True, retrain=False, compute_loss=False, generate_predictions=True, evaluate=True)