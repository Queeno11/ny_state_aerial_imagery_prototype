##############      Configuración      ##############

### Main libraries
import os
import gc
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
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

### My modules
import src.custom_models as custom_models
import src.build_dataset as build_dataset
import src.geo_utils as geo_utils
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

def generate_savename(
    model_name, image_size, learning_rate, years, extra
):
    years_str = "-".join(map(str, years))
    # stacked_images hardcoded to [1], so no stacking
    savename = (
        f"{model_name}_lr{learning_rate}_size{image_size}_y{years_str}{extra}"
    )

    return savename

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
        row: dict-like row with row_start, col_start, dataset, DOITT_ID …
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
                f"Failed single-chunk extract for DOITT_ID {row.get('DOITT_ID', '?')}: {e}"
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
                f"Failed multi-chunk extract for DOITT_ID {row.get('DOITT_ID', '?')}: {e}"
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
    ):
        self.df = df
        
        # [NEW] Group temporal twins together so they end up in the same shard
        if type == "train":
            print("Sorting dataframe to group DOITT_IDs for temporal sampling...")
            doitts = self.df['DOITT_ID'].unique()
            np.random.shuffle(doitts)  # Shuffle groups to maintain randomness
            cat_type = pd.CategoricalDtype(categories=doitts, ordered=True)
            self.df['DOITT_ID_cat'] = self.df['DOITT_ID'].astype(cat_type)
            self.df = self.df.sort_values(['DOITT_ID_cat', 'year']).reset_index(drop=True)
            self.df.drop('DOITT_ID_cat', axis=1, inplace=True)
        
        self.all_years_datasets = all_years_datasets
        self.params = params
        self.max_jitter = max_jitter
        self.max_jitter_pixels = geo_utils.meters_to_pixels(self.max_jitter, 0.5, epsg_code=6539) # TODO: I should be able to extract this from the zarr
        self.image_size = int((df["row_stop"] - df["row_start"]).min())  # Assuming all images have the same size in the raw zarr array
        self.nbands = params["nbands"]

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
        self._is_initialized = False
        self.progress = 0.0
 
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.clear_cache:
            for f in self.cache_dir.glob("*.pt"):
                f.unlink()
        else:
            self._load_existing_shards()

        if self.type == "train":
            print("Pre-computing indices for fast pairwise sampling...")
            # 1. Fast lookup for Spatial Pairs (by year)
            self.year_to_idxs = self.df.groupby('year').groups

            # 2. Fast lookup for Temporal Pairs (by building)
            # Find buildings that exist in at least 2 different years
            counts = self.df['DOITT_ID'].value_counts()
            valid_doitts = counts[counts >= 2].index

            valid_temporal_df = self.df[self.df['DOITT_ID'].isin(valid_doitts)]
            self.doitt_to_idxs = valid_temporal_df.groupby('DOITT_ID').groups

            # 3. Separate them into Stable (0) and Change (1) pools
            doitt_meta = valid_temporal_df[['DOITT_ID', 'Valid_Structural_Change']].drop_duplicates('DOITT_ID')
            self.stable_doitts = doitt_meta[doitt_meta['Valid_Structural_Change'] == 0]['DOITT_ID'].values
            self.change_doitts = doitt_meta[doitt_meta['Valid_Structural_Change'] == 1]['DOITT_ID'].values
            print(f"  Spatial years: {len(self.year_to_idxs)} | Stable buildings: {len(self.stable_doitts)} | Change buildings: {len(self.change_doitts)}")

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
            logging.error(f"Failed for DOITT_ID {row.get('DOITT_ID', '?')}: {e}")

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

    def _worker_generate(self, shard_id, show_progress=False):
        step = self.params["subsample_step"]
        jitter_pad = math.ceil(self.max_jitter_pixels / step) * step if self.type == "train" else 0

        # Grab sequential chunk (twins are naturally adjacent now)
        start_idx = (shard_id * self.shard_size) % len(self.df)
        end_idx = start_idx + self.shard_size
        if end_idx > len(self.df):
            sampled_df = pd.concat([self.df.iloc[start_idx:], self.df.iloc[:end_idx % len(self.df)]])
        else:
            sampled_df = self.df.iloc[start_idx:end_idx]

        if sampled_df.empty: return

        items_to_extract = [(row, pos) for pos, (_, row) in enumerate(sampled_df.iterrows())]

        valid_images, valid_scores, valid_geoids = [], [], []
        valid_years, valid_doitt_ids, valid_change = [], [], []
        valid_metas, valid_score_bins = [], []

        CHUNK_SIZE = 8
        MAX_EXTRACT_WORKERS = 2

        def _extract(item):
            row, pos = item
            if pd.isna(row["Rel_Score"]): return None
            raw_img = self._extract_raw_image(row, n_bands=self.params["nbands"], pad=jitter_pad)
            if raw_img is None: return None
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
                            valid_geoids.append(hash(str(r.get('GEOID', ''))) % (2**31))
                            valid_years.append(r['year'])
                            valid_doitt_ids.append(r['DOITT_ID'])
                            valid_change.append(r.get('Valid_Structural_Change', 0)) # Save change flag!
                            valid_metas.append(r.get("dist_to_center", 0.0))
                            valid_score_bins.append(r.get("score_bin", 0))
                    raw_chunk.clear(); chunk_meta.clear()

            # Process remainder
            if raw_chunk:
                processed = self._batch_subsample_and_convert(raw_chunk)
                for img_tensor, r in zip(processed, chunk_meta):
                    if img_tensor.max() > 0:
                        valid_images.append(img_tensor)
                        valid_scores.append(r['Rel_Score'])
                        valid_geoids.append(hash(str(r.get('GEOID', ''))) % (2**31))
                        valid_years.append(r['year'])
                        valid_doitt_ids.append(r['DOITT_ID'])
                        valid_change.append(r.get('Valid_Structural_Change', 0))
                        valid_metas.append(r.get("dist_to_center", 0.0))
                        valid_score_bins.append(r.get("score_bin", 0))

        shard_path = self.cache_dir / f"shard_{shard_id}.pt"
        if valid_images:
            torch.save({
                "images": torch.stack(valid_images),
                "scores": torch.tensor(valid_scores, dtype=torch.float32),
                "geoids": torch.tensor(valid_geoids, dtype=torch.int64),
                "years": torch.tensor(valid_years, dtype=torch.int64),
                "doitt_ids": torch.tensor(valid_doitt_ids, dtype=torch.int64),
                "structural_change": torch.tensor(valid_change, dtype=torch.int64), # Replaces twin/pair logic
                "metas": torch.tensor(valid_metas, dtype=torch.float32).unsqueeze(1),
                "score_bins": torch.tensor(valid_score_bins, dtype=torch.int64),
            }, shard_path)
        else:
            torch.save({"images": torch.empty(0), "scores": torch.empty(0)}, shard_path)

    def build_initial_cache(self):
        """Synchronously generates the initial num_shards shards. Call once before training."""
        if self.active_shards and not self.clear_cache:
            print(
                f"Using existing {len(self.active_shards)} pre-built cache shard(s) in {self.cache_dir}"
            )
            self._is_initialized = True
            return

        print(f"Building initial {self.num_shards} cache shards...")
        for i in range(self.num_shards):
            # True: Show progress bar during initial blocking setup
            self._worker_generate(self.next_shard_idx, show_progress=True) 
            self.active_shards.append(self.cache_dir / f"shard_{self.next_shard_idx}.pt")
            self.next_shard_idx += 1
        self._is_initialized = True
        print("Initial cache ready.")
 
    def _worker_generate_k(self, start_idx, k, show_progress=False):
        """Generates k consecutive shards sequentially in a background thread."""
        for i in range(k):
            self._worker_generate(start_idx + i, show_progress)
 
    def start_background_generation(self, k=2):
        """Kicks off generation of the next k shards on a background thread before training begins."""
        if not self._is_initialized:
            raise RuntimeError("Call build_initial_cache() before starting background generation.")
        self.bg_thread = threading.Thread(
            target=self._worker_generate_k, args=(self.next_shard_idx, k, False), daemon=True
        )
        self._pending_k = k
        self.bg_thread.start()

    def step(self, k=2, progress=0.0):
        """Non-blocking cache rotation. Background generates next k shards, then swaps them safely on completion."""
        self.progress = progress
        if not self._is_initialized:
            raise RuntimeError("Call build_initial_cache() before calling step().")

        if self.single_shard_mode:
            return False 

        if self.bg_thread is None:
            # Kick off the very first background generation
            self.bg_thread = threading.Thread(
                target=self._worker_generate_k, args=(self.next_shard_idx, k, False), daemon=True
            )
            self._pending_k = k
            self.bg_thread.start()
            return False
            
        else:
            if self.bg_thread.is_alive():
                return False  # Still generating in background
                
            # Background thread FINISHED: Swap the shards securely!
            for _ in range(self._pending_k):
                oldest_shard = self.active_shards.pop(0)
                try: oldest_shard.unlink() 
                except FileNotFoundError: pass
        
                self.active_shards.append(self.cache_dir / f"shard_{self.next_shard_idx}.pt")
                self.next_shard_idx += 1
    
            # Immediately kick off the next batch cycle
            self.bg_thread = threading.Thread(
                target=self._worker_generate_k, args=(self.next_shard_idx, k, False), daemon=True
            )
            self._pending_k = k
            self.bg_thread.start()
            
            return True 
        return False
 

class InBatchRankingDataset(Dataset):
    """
    Loads individual images with rich metadata from training shards.
    The HybridBatchSampler constructs valid hybrid batches at iteration time
    using the exposed metadata (year_to_idxs, doitt_to_idxs).
    """
    def __init__(self, cache_manager: CyclicCacheManager, transform=None):
        if not cache_manager._is_initialized:
            raise RuntimeError(
                "Call cache_manager.build_initial_cache() before instantiating InBatchRankingDataset."
            )
        self.cache_manager = cache_manager
        self.transform = transform
        self.images = None
        self.scores = None
        self.geoids = None
        self.years = None
        self.doitt_ids = None
        self.structural_change = None
        self.metas = None
        self.score_bins = None

        # Lookups for the HybridBatchSampler (built after loading)
        self.year_to_idxs = {}
        self.doitt_to_idxs = {}
        self.refresh()

    def refresh(self):
        """Drops the current in-RAM tensors and reloads from active shards."""
        if not self.cache_manager.active_shards:
            raise RuntimeError("No active shards to load. Call build_initial_cache() first.")

        self.images = None
        self.scores = None
        gc.collect()

        img_list, sc_list, geo_list, yr_list = [], [], [], []
        did_list, ch_list, mt_list, sb_list = [], [], [], []

        for shard_path in self.cache_manager.active_shards:
            data = torch.load(shard_path, weights_only=False)
            shard_len = len(data["images"])
            img_list.append(data["images"])
            sc_list.append(data["scores"])
            geo_list.append(data.get("geoids", torch.zeros(shard_len, dtype=torch.int64)))
            yr_list.append(data.get("years", torch.zeros(shard_len, dtype=torch.int64)))
            did_list.append(data.get("doitt_ids", torch.zeros(shard_len, dtype=torch.int64)))
            ch_list.append(data.get("structural_change", torch.zeros(shard_len, dtype=torch.int64)))
            mt_list.append(data.get("metas", torch.zeros(shard_len, 1)))
            sb_list.append(data.get("score_bins", torch.zeros(shard_len, dtype=torch.int64)))

        self.images           = torch.cat(img_list)
        self.scores           = torch.cat(sc_list)
        self.geoids           = torch.cat(geo_list)
        self.years            = torch.cat(yr_list)
        self.doitt_ids        = torch.cat(did_list)
        self.structural_change = torch.cat(ch_list)
        self.metas            = torch.cat(mt_list)
        self.score_bins       = torch.cat(sb_list)

        # Build lookups for the batch sampler
        self.year_to_idxs = {}
        self.doitt_to_idxs = {}
        for i in range(len(self.images)):
            yr = self.years[i].item()
            did = self.doitt_ids[i].item()
            self.year_to_idxs.setdefault(yr, []).append(i)
            self.doitt_to_idxs.setdefault(did, []).append(i)

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
        return (img, self.scores[idx], self.geoids[idx], self.years[idx],
                self.doitt_ids[idx], self.structural_change[idx], self.metas[idx], self.score_bins[idx])


class HybridBatchSampler(torch.utils.data.Sampler):
    """
    Constructs valid hybrid batches for In-Batch Pairwise Ranking.

    Each batch contains:
      - B_cs cross-sectional images from the SAME year (maximizes pairwise supervision)
      - Their temporal twins (stable + change) from the loaded data
    """
    def __init__(self, dataset: InBatchRankingDataset, batch_size_cs: int, max_temporal_per_batch: int = 32):
        self.dataset = dataset
        self.batch_size_cs = batch_size_cs
        self.max_temporal = max_temporal_per_batch
        self.num_batches = max(1, len(dataset) // batch_size_cs)

    def __iter__(self):
        for _ in range(self.num_batches):
            # Pick random year from loaded data
            year = random.choice(list(self.dataset.year_to_idxs.keys()))
            pool = self.dataset.year_to_idxs[year]

            # Sample cross-sectional core
            cs_size = min(self.batch_size_cs, len(pool))
            cs_idxs = random.sample(pool, cs_size)

            batch = list(cs_idxs)

            # Find temporal twins for CS buildings (if they exist in loaded data)
            temporal_added = 0
            for idx in cs_idxs:
                if temporal_added >= self.max_temporal:
                    break
                doitt = self.dataset.doitt_ids[idx].item()
                # [PATCH] Avoid sampling changed tracts for now
                twin_candidates = [
                    i for i in self.dataset.doitt_to_idxs.get(doitt, [])
                    if i != idx and self.dataset.years[i].item() != year and self.dataset.structural_change[i].item() == 0
                ]
                if twin_candidates:
                    twin = random.choice(twin_candidates)
                    batch.append(twin)
                    temporal_added += 1

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
        self.doitt_ids = None
        self.years = None
        self.structural_change = None

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
        self.doitt_ids = data.get("doitt_ids")
        self.years   = data.get("years")
        self.structural_change = data.get("structural_change")
        self.current_shard_idx = shard_idx

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        shard_idx, item_idx = self.idx_mapping[idx]
        self._load_shard(shard_idx)

        img  = self.images[item_idx]
        lbl  = self.scores[item_idx]
        meta = self.metas[item_idx] if self.metas is not None else torch.zeros(1)
        doitt_id = self.doitt_ids[item_idx] if self.doitt_ids is not None else 0
        year = self.years[item_idx] if self.years is not None else 0
        structural_change = self.structural_change[item_idx] if self.structural_change is not None else 0

        if self.transform:
            img = self.transform(img)

        return img, lbl, meta, doitt_id, year, structural_change
                                        
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

def setup_dataloaders(df_train, dfs_val_dict, df_test, all_years_datasets, params, train_cache_manager=None, val_cache_manager=None):
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

    # In-Batch Pairwise Ranking: individual images + metadata, grouped by HybridBatchSampler
    train_dataset = InBatchRankingDataset(
        cache_manager=train_cache_manager,
        transform=train_transform_cpu,
    )
    hybrid_sampler = HybridBatchSampler(
        dataset=train_dataset,
        batch_size_cs=batch_size,
        max_temporal_per_batch=max(1, int(batch_size * params.get("temporal_fraction", 0.20))),
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

    sat_options = ["aerial", "pleiades", "landsat"]
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
        "test_years": [],
        "test_column": None,
        "extra": "",
        "tau_meters": 100,
        "subsample_step": 1,
        "max_jitter": 10,
        # In-Batch Pairwise Ranking hyperparameters
        "m_min": 0.1,         # Hard floor for pairwise margin (prevents collapse)
        "m_base": 1.0,        # Margin scale factor: m_kl = max(m_min, m_base * |Z_l - Z_k|)
        "lambda_s": 1.0,      # Weight for temporal stability L1 penalty
        "lambda_c": 0.0,      # [PATCH] Weight for temporal change ranking hinge (temporarily disabled)
        "temporal_fraction": 0.20,  # Fraction of batch reserved for temporal auxiliary (split 50/50 stable/change)
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

    Computes three decoupled objectives from a single forward pass:
      1. L_cross:  Smooth logistic surrogate (RankNet) over all valid unique cross-sectional pairs
                   T = max(m_min, m_base * σ_batch) — temperature scales with score spread, NOT ACS magnitude
      2. L_stable: L1 temporal invariance for stable twins
      3. L_change: Smooth logistic surrogate for directional rank mobility of changed twins

    L_total = L_cross + lambda_s * L_stable + lambda_c * L_change

    Key design: margins carry zero economic information. They scale with the model's own output
    distribution (σ_batch) to prevent collapse at initialization, never with ACS label magnitudes.
    This preserves the ordinal framework: supervision is purely directional (sign), never cardinal.

    Returns (loss, diagnostics_dict) so the training loop can log internals to W&B.
    """
    def __init__(self, m_base=1.0, m_min=0.1, lambda_s=1.0, lambda_c=1.0):
        super().__init__()
        self.m_base = m_base
        self.m_min = m_min
        self.lambda_s = lambda_s
        self.lambda_c = lambda_c

    def forward(self, scores, labels, geoids, years, doitt_ids, structural_change, score_bins=None, current_epoch=None, current_step=None):
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
            
            # --- Easy-to-Hard Curriculum Filtering ---
            if current_epoch is not None and score_bins is not None:
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

        # ──────────────────────────────────────────────────────────
        # 2 & 3. TEMPORAL PENALTIES (L_stable & L_change)
        # ──────────────────────────────────────────────────────────
        L_stable = torch.tensor(0.0, device=scores.device)
        L_change = torch.tensor(0.0, device=scores.device)
        n_stable, n_change, change_hinge_active = 0, 0, 0.0

        if temp_mask.any():
            temp_scores, temp_labels = scores[temp_mask], labels[temp_mask]
            temp_doitts = doitt_ids[temp_mask]
            temp_change = structural_change[temp_mask]
            
            cs_doitts = doitt_ids[cs_mask]
            cs_scores_full = scores[cs_mask]
            cs_labels_full = labels[cs_mask]
            
            # Vectorized dynamic pairing: match temporal doitt to cross-sectional doitt
            matches = (temp_doitts.unsqueeze(1) == cs_doitts.unsqueeze(0))
            has_match, match_idx_in_cs = matches.max(dim=1)
            
            # Filter to twins that successfully matched a CS anchor
            valid_temp = has_match
            if valid_temp.any():
                v_temp_scores, v_temp_labels = temp_scores[valid_temp], temp_labels[valid_temp]
                v_temp_change = temp_change[valid_temp]
                
                v_partner_scores = cs_scores_full[match_idx_in_cs[valid_temp]]
                v_partner_labels = cs_labels_full[match_idx_in_cs[valid_temp]]

                # STABLE Penalty
                stable_idx = (v_temp_change == 0)
                if stable_idx.any():
                    L_stable = ((v_temp_scores[stable_idx] - v_partner_scores[stable_idx]) ** 2).mean()
                    n_stable = stable_idx.sum().item()

                # CHANGE Ranking
                change_idx = (v_temp_change == 1)
                if change_idx.any():
                    c_temp_scores = v_temp_scores[change_idx]
                    c_partner_scores = v_partner_scores[change_idx]
                    
                    y_change = torch.sign(v_temp_labels[change_idx] - v_partner_labels[change_idx])
                    
                    m_change = self.m_base
                    delta_change = y_change * (c_temp_scores - c_partner_scores)
                    L_change = F.softplus(-delta_change / m_change).mean()
                    n_change = change_idx.sum().item()
                    with torch.no_grad():
                        # Diagnostic: proportion of pairs that would violate a hard margin
                        change_hinge_active = (-delta_change + m_change > 0).float().mean().item()

        # CS-only variance regularizer: targets the exact distribution L_cross operates on.
        # Eliminates the tug-of-war with L_stable (temporal scores are excluded).
        # Internally consistent: both m_scalar and variance_penalty reference cs_scores.std().
        variance_penalty = torch.tensor(0.0, device=scores.device)
        if B_cs > 1:
            variance_penalty = (cs_scores.std() - 1.0).pow(2)
            
        loss = L_cross + self.lambda_s * L_stable + self.lambda_c * L_change + 1.0 * variance_penalty
        
        # --- Compute gradient norms of each loss component w.r.t predictions ---
        # Only compute every 10 steps to avoid 4x extra backward traversals per step
        grad_cross, grad_stable, grad_change, grad_variance = 0.0, 0.0, 0.0, 0.0
        compute_grad_norms = (current_step is not None and current_step % 10 == 0)
        if compute_grad_norms:
            try:
                if L_cross.requires_grad and L_cross.item() > 0:
                    g_c = torch.autograd.grad(L_cross, scores, retain_graph=True)[0]
                    if g_c is not None: grad_cross = g_c.norm(p=2).item() * 10
                    
                if L_stable.requires_grad and L_stable.item() > 0:
                    g_s = torch.autograd.grad(self.lambda_s * L_stable, scores, retain_graph=True)[0]
                    if g_s is not None: grad_stable = g_s.norm(p=2).item() * 10
                    
                if L_change.requires_grad and L_change.item() > 0:
                    g_ch = torch.autograd.grad(self.lambda_c * L_change, scores, retain_graph=True)[0]
                    if g_ch is not None: grad_change = g_ch.norm(p=2).item() * 10
                    
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
                "loss/L_change":             L_change.item(),
                "grad_norm/cross":           grad_cross,
                "grad_norm/stable":          grad_stable,
                "grad_norm/change":          grad_change,
                "grad_norm/variance":        grad_variance,
                "loss/cross_valid_pairs":    n_valid_pairs,
                "loss/cross_hinge_active":   cross_hinge_active,
                "loss/change_hinge_active":  change_hinge_active,
                "loss/avg_margin":           avg_margin,
                "loss/n_stable":             n_stable,
                "loss/n_change":             n_change,
                "loss/score_std":            scores.std().item() if scores.shape[0] > 1 else 0.0,
                "loss/cs_score_std":         cs_scores.std().item() if B_cs > 1 else 0.0,
                "loss/score_mean":           scores.mean().item(),
                "loss/score_min":            scores.min().item(),
                "loss/score_max":            scores.max().item(),
            }

        return loss, diag


def set_model_and_loss_function(
    model_name: str, kind: str, image_size: int, bands: int = 4, weights: str = None, meta_dim: int = 0,
    m_base: float = 1.0, m_min: float = 0.1, lambda_s: float = 1.0, lambda_c: float = 1.0
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
            m_base=m_base, m_min=m_min, lambda_s=lambda_s, lambda_c=lambda_c
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
):
    print("--- Starting PyTorch Training Loop ---")
    best_val_spearman = (
        initial_best_val_loss if initial_best_val_loss is not None else float(-1)
    )

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
        for batch_idx, (images, scores, geoids, years, doitt_ids, structural_change, metas, score_bins) in train_bar:
            t_data_end = time.perf_counter()  # data is ready; measure load time

            # Move all to device
            images     = images.to(device)
            if hasattr(train_loader, "gpu_transform") and train_loader.gpu_transform is not None:
                with torch.no_grad():
                    images = train_loader.gpu_transform(images)
            
            scores_lbl = scores.to(device)
            geoids     = geoids.to(device)
            years      = years.to(device)
            doitt_ids  = doitt_ids.to(device)
            structural_change = structural_change.to(device)
            metas      = metas.to(device)
            score_bins = score_bins.to(device)
            
            # Single unified forward pass over the entire hybrid batch
            t_forward_start = time.perf_counter()

            with autocast(device_type='cuda'):
                # One forward pass — all images (CS + temporal) through same backbone
                outputs = model(images, metadata=metas)
                
                # In-Batch Pairwise Ranking Loss (3 decoupled objectives)
                loss, diag = loss_fn(
                    outputs, scores_lbl, geoids, years, doitt_ids, structural_change, 
                    score_bins=score_bins, current_epoch=epoch, current_step=batch_idx
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
        val_stable_masd = {}
        val_changed_masd = {}
        val_changed_da = {}
        if len(val_loaders.values()) > 0:
            for val_name, val_loader in val_loaders.items():
                model.eval() # Set model to eval mode (disables dropout)
                running_val_loss = 0.0
                all_preds = []
                all_labels = []
                all_doitts = []
                all_years = []
                all_changes = []
                val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] {val_name}", leave=False)

                # Disable gradient calculation for validation (saves RAM and compute)
                with torch.no_grad():
                    for images, labels, metas, doitt_ids, val_years, val_structural_changes in val_bar:
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
                        if doitt_ids is not None:
                            all_doitts.extend(doitt_ids.cpu().numpy())
                        if val_years is not None:
                            all_years.extend(val_years.cpu().numpy())
                        if val_structural_changes is not None:
                            all_changes.extend(val_structural_changes.cpu().numpy())
                        
                        val_bar.set_postfix(loss=f"{loss.item():.4f}")
                
                from scipy.stats import spearmanr
                spearman_corr, _ = spearmanr(all_preds, all_labels)
                
                val_losses[val_name] = running_val_loss / len(val_loader.dataset)
                # Exclude val_spatial_temporal: that spearman is inconsistent as it compares geoids across time and space!
                if val_name != "val_spatial_temporal":
                    val_spearmans[val_name] = spearman_corr
                
                # ── Temporal metrics: MASD (stable vs changed) + Directional Accuracy ──
                if len(all_doitts) > 0 and len(all_doitts) == len(all_preds):
                    val_df = pd.DataFrame({
                        'DOITT_ID': all_doitts,
                        'year': all_years,
                        'pred': all_preds,
                        'label': all_labels,
                        'change': all_changes
                    })
                    
                    val_counts = val_df['DOITT_ID'].value_counts()
                    valid_doitts = val_counts[val_counts >= 2].index
                    
                    if len(valid_doitts) > 0:
                        valid_df = val_df[val_df['DOITT_ID'].isin(valid_doitts)].sort_values(['DOITT_ID', 'year'])
                        
                        # Per-building: compute MASD and directional accuracy over all year-ordered pairs
                        building_change_flag = valid_df.groupby('DOITT_ID')['change'].first()
                        stable_ids = building_change_flag[building_change_flag == 0].index
                        changed_ids = building_change_flag[building_change_flag == 1].index
                        
                        stable_abs_disps = []
                        changed_abs_disps = []
                        da_correct = 0
                        da_total = 0
                        
                        for doitt_id, grp in valid_df.groupby('DOITT_ID'):
                            preds_arr = grp['pred'].values
                            labels_arr = grp['label'].values
                            is_changed = grp['change'].iloc[0]
                            n = len(preds_arr)
                            # All ordered pairs (j > i by year)
                            for i in range(n):
                                for j in range(i + 1, n):
                                    pred_d = preds_arr[j] - preds_arr[i]
                                    label_d = labels_arr[j] - labels_arr[i]
                                    if is_changed == 0:
                                        stable_abs_disps.append(abs(pred_d))
                                    else:
                                        changed_abs_disps.append(abs(pred_d))
                                        if label_d != 0:  # skip tied labels
                                            da_total += 1
                                            if (pred_d > 0) == (label_d > 0):
                                                da_correct += 1
                        
                        if stable_abs_disps:
                            val_stable_masd[val_name] = float(np.mean(stable_abs_disps))
                        if changed_abs_disps:
                            val_changed_masd[val_name] = float(np.mean(changed_abs_disps))
                        if da_total > 0:
                            val_changed_da[val_name] = da_correct / da_total

                # Build display string (minimal: Spearman for spatial sets, MASD/DA for temporal set)
                if val_name == "val_spatial_temporal":
                    parts = []
                    if val_name in val_stable_masd:
                        parts.append(f"S-MASD: {val_stable_masd[val_name]:.4f}")
                    if val_name in val_changed_masd:
                        parts.append(f"C-MASD: {val_changed_masd[val_name]:.4f}")
                    if val_name in val_changed_da:
                        parts.append(f"C-DA: {val_changed_da[val_name]:.2%}")
                    tqdm.write(f"{val_name} {' | '.join(parts)}")
                else:
                    tqdm.write(f"{val_name} Spearman: {spearman_corr:.4f}")
            
            # Use mean validation spearman for early stopping / best model checkpointing
            epoch_val_spearman = sum(val_spearmans.values()) / len(val_spearmans)
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
        val_display = " | ".join([f"{k} Spearman: {v:.4f}" for k, v in val_spearmans.items() if k != "val_spatial_temporal"])
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
            for val_name, val_loader in val_loaders.items():
                log_dict[f"{val_name}_mse"] = val_losses[val_name]
                if val_name in val_spearmans:
                    log_dict[f"{val_name}_spearman"] = val_spearmans[val_name]
                if val_name in val_stable_masd:
                    log_dict[f"{val_name}_stable_masd"] = val_stable_masd[val_name]
                if val_name in val_changed_masd:
                    log_dict[f"{val_name}_changed_masd"] = val_changed_masd[val_name]
                if val_name in val_changed_da:
                    log_dict[f"{val_name}_changed_da"] = val_changed_da[val_name]

        wandb.log(log_dict)

        # Model Checkpoint logic (Maximize Spearman)
        if epoch_val_spearman > best_val_spearman:
            tqdm.write(f"⭐ Val Spearman improved from {best_val_spearman:.4f} to {epoch_val_spearman:.4f}. Saving...")
            tqdm.write(f"   📊 [Loss Components] L_cross: {epoch_diag.get('loss/L_cross', 0.0):.4f} | L_stable: {epoch_diag.get('loss/L_stable', 0.0):.4f} | L_change: {epoch_diag.get('loss/L_change', 0.0):.4f}")
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
                if cache_manager._pending_k != k:
                    tqdm.write(f"⚠️ The cache is falling behind the GPU! {cache_manager._pending_k} new train background shard(s) ready, but {cache_manager._pending_k / k * 100}% of the cache is stale. Consider increasing k.")
                else:
                    tqdm.write(f"🔄 {cache_manager._pending_k} new train background shard(s) ready! Reloading pre-packaged pairs from disk...")
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
        }
        torch.save(checkpoint, save_dir / f"{savename}_last.pth")

    # Return model for caller chaining (especially if we loaded/checkpointed externally)
    return model

import queue
import threading
import pandas as pd

class FastPredictLoader:
    def __init__(self, df, all_years_datasets, params, batch_size, eval_transform):
        self.df = df
        self.all_years_datasets = all_years_datasets
        self.params = params
        self.batch_size = batch_size
        self.eval_transform = eval_transform
        
        self.image_size = int((self.df["row_stop"] - self.df["row_start"]).min())
        self.nbands = params["nbands"]
        self.step = params["subsample_step"]
        
        # Fake dataset to satisfy print statements in predict_buildings
        class FakeDataset:
            def __init__(self, length):
                self.length = length
            def __len__(self):
                return self.length
        self.dataset = FakeDataset(len(self.df))
        
        self.queue = queue.Queue(maxsize=10) # 10 batches buffered ahead keeping GPU fed
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _extract_raw_image(self, row):
        dataset_name = row.get("dataset")
        zarr_array = self.all_years_datasets[dataset_name]["value"]
        
        row_start = int(row["row_start"])
        row_stop  = row_start + self.image_size
        col_start = int(row["col_start"])
        col_stop  = col_start + self.image_size
        
        try:
            tile = zarr_array[:self.nbands, row_start:row_stop, col_start:col_stop]
            if tile.shape[0] == self.nbands and tile.shape[1] == self.image_size and tile.shape[2] == self.image_size:
                return tile.to_numpy()
        except:
            pass
        return None

    def _worker(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import torch

        def _extract(item):
            _, row = item
            label = row["Rel_Score"]
            dist_to_center = row.get("dist_to_center", 0.0)
            return self._extract_raw_image(row), label, dist_to_center, row

        raw_chunk, label_chunk, meta_chunk, row_chunk = [], [], [], []
        rows = list(self.df.iterrows())
        MAX_EXTRACT_WORKERS = 18

        # KEY FIX: Only allow this many images to be in-flight at once.
        # Without this, pool.map() submits ALL rows as futures simultaneously,
        # storing every extracted numpy tile in memory before any are consumed.
        MAX_IN_FLIGHT = self.batch_size * 4  # e.g., 4 batches worth (~1GB for 224px tiles)

        with ThreadPoolExecutor(max_workers=MAX_EXTRACT_WORKERS) as pool:
            i = 0
            while i < len(rows):
                # Submit only a bounded window of futures at a time
                chunk_end = min(i + MAX_IN_FLIGHT, len(rows))
                futures = {pool.submit(_extract, rows[j]): j for j in range(i, chunk_end)}
                i = chunk_end

                for future in as_completed(futures):
                    raw_img, label, meta, row = future.result()

                    if raw_img is None or pd.isna(label):
                        continue

                    raw_chunk.append(raw_img)
                    label_chunk.append(label)
                    meta_chunk.append(meta)
                    row_chunk.append(row)

                    if len(raw_chunk) >= self.batch_size:
                        batch = torch.stack([torch.from_numpy(img) for img in raw_chunk])
                        batch = batch[:, :, ::self.step, ::self.step]

                        labels_tensor = torch.tensor(label_chunk, dtype=torch.float32)
                        metas_tensor = torch.tensor(meta_chunk, dtype=torch.float32).unsqueeze(1)
                        doitt_ids = torch.tensor([r.get("DOITT_ID", 0) for r in row_chunk])
                        geoids = [str(r.get("GEOID", "")) for r in row_chunk]
                        years_t = torch.tensor([r.get("year", 0) for r in row_chunk])
                        types = [str(r.get("type", "")) for r in row_chunk]

                        # queue.put() blocks here if GPU is slow — this is your backpressure valve
                        self.queue.put((
                            self.eval_transform(batch), labels_tensor, metas_tensor,
                            doitt_ids, geoids, years_t, types
                        ))

                        raw_chunk.clear(); label_chunk.clear()
                        meta_chunk.clear(); row_chunk.clear()

        # Flush the last partial batch
        if raw_chunk:
            batch = torch.stack([torch.from_numpy(img) for img in raw_chunk])
            batch = batch[:, :, ::self.step, ::self.step]

            labels_tensor = torch.tensor(label_chunk, dtype=torch.float32)
            metas_tensor = torch.tensor(meta_chunk, dtype=torch.float32).unsqueeze(1)
            doitt_ids = torch.tensor([r.get("DOITT_ID", 0) for r in row_chunk])
            geoids = [str(r.get("GEOID", "")) for r in row_chunk]
            years_t = torch.tensor([r.get("year", 0) for r in row_chunk])
            types = [str(r.get("type", "")) for r in row_chunk]

            self.queue.put((
                self.eval_transform(batch), labels_tensor, metas_tensor,
                doitt_ids, geoids, years_t, types
            ))

        self.queue.put(None)

    def __iter__(self):
        return self
    
    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch
        
    def __len__(self):
        return max(1, len(self.df) // self.batch_size)


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
            [r.get("DOITT_ID", 0)    for r in rows],
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

            batch_tensor, metas_tensor, labels, doitt_ids, geoids, years, types = item

            # non_blocking=True overlaps H2D transfer with prior GPU work
            batch_tensor = batch_tensor.to(device, non_blocking=True)
            metas_tensor = metas_tensor.to(device, non_blocking=True)

            with autocast(device_type='cuda'):
                outputs = model(batch_tensor, metadata=metas_tensor)

            preds = outputs.view(-1).cpu().numpy()

            pd.DataFrame({
                "Rel_Score":       labels,
                "predicted_value": preds,
                "DOITT_ID":        doitt_ids,
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


def predict_buildings(model, dataloader, device, output_path, verbose=True):
    """
    Streams predictions directly to CSV in chunks — no full-dataset RAM accumulation.
    """
    model.eval()
    first_write = True

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Starting inference on {len(dataloader.dataset)} items...")

    with torch.no_grad():
        iterator = enumerate(dataloader)
        if verbose:
            iterator = tqdm(iterator, total=len(dataloader), desc="Predicting Batches")

        for batch_idx, (batch_images, batch_labels, batch_metas, batch_doitt_ids, batch_geoids, batch_years, batch_types) in iterator:
            batch_images = batch_images.to(device)
            batch_metas = batch_metas.to(device)

            with autocast(device_type='cuda'):
                outputs = model(batch_images, metadata=batch_metas)

            preds = outputs.view(-1).cpu().numpy()

            chunk_df = pd.DataFrame({
                "Rel_Score":       batch_labels.cpu().numpy(),
                "predicted_value": preds,
                "DOITT_ID":        batch_doitt_ids.cpu().numpy(),
                "GEOID":           batch_geoids,
                "year":            batch_years.cpu().numpy(),
                "type":            batch_types,
            })

            # Write header only on first chunk, then append
            mode = 'w' if first_write else 'a'
            chunk_df.to_csv(output_path, mode=mode, header=first_write, index=False)
            first_write = False

    if verbose:
        print(f"Predictions saved to {output_path}")

def run(
    params=None,
    train=True,
    compute_loss=True,
    generate_predictions=False,
    retrain=False,
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
    test_years = params["test_years"]
    test_column = params["test_column"]
    extra = params["extra"]
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

    savename = generate_savename(
        model_name, image_size, learning_rate, years, extra
    )
    log_dir = f"{LOGS_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    generate_parameters_log(params, savename)
    
    print("\n" + "="*80)
    print(f"🚀 STARTING RUN: {savename}")
    print("="*80 + "\n")
    print(f"- Train: {train} \n- Compute Loss: {compute_loss} \n- Generate Predictions: {generate_predictions} \n- Retrain: {retrain}\n")
    print("="*84 + "\n")

    if train:

        all_years_datasets, all_years_extents, df_train, df_vals_dict, df_test, df_dead_zone = build_dataset.generate_datasets(
            savename, sat_data, years, test_years, test_column, small_sample, max_jitter, tau_meters
        )

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

        print("Building Initial Cache...")
        if small_sample:
            num_shards = 1
            current_shard_size = len(df_train) # E.g., ~150 images per shard
        else:
            num_shards = 10
            current_shard_size = CACHE_SIZE // 10 # E.g., 20,480 images per shard

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
        )
        train_cache_manager.build_initial_cache()

        print("\n[VAL] Building static validation cache...")
        vals_cache_manager_dict = {}
        for val_name, df_val in df_vals_dict.items():
            if df_val.shape[0] > 0:
                # Take up to 5 buildings per shard for validation to keep RAM usage low, since we load the whole shard at once during validation
                if val_name != "val_spatial_temporal":
                    df_val = df_val.groupby(['GEOID', 'year'], group_keys=False)[df_val.columns].apply(lambda x: x.sample(n=min(len(x), 2)))
                val_num_shards = max(1, (len(df_val) + current_shard_size - 1) // current_shard_size)
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
                )
                val_cache_manager.build_initial_cache() # This will build and show a progress bar
                vals_cache_manager_dict[val_name] = val_cache_manager
            else:
                val_cache_manager = None
       
        #### 2. PyTorch Data Pipeline Setup
        print("Setting up data generators...")
        train_loader, val_loaders, test_loader = setup_dataloaders(
            df_train=df_train, dfs_val_dict=df_vals_dict, df_test=df_test,
            all_years_datasets=all_years_datasets, params=params,
            train_cache_manager=train_cache_manager, val_cache_manager=vals_cache_manager_dict
        )            

        del df_train, df_test, df_val, all_years_extents
        gc.collect() # Force Python to free up memory from large objects we no longer need
        
        train_cache_manager.start_background_generation() # Starts generating shard 6 for Epoch 1

        print("Data Pipeline Ready!")

        #### 3. Model Initialization
        # meta_dim=1 for the 'dist_to_center' covariate
        model, loss_fn = set_model_and_loss_function(
            model_name=model_name, 
            kind=kind,
            bands=nbands, 
            image_size=image_size, 
            weights=weights,
            meta_dim=1,
            m_base=params.get("m_base", 1.0),
            m_min=params.get("m_min", 0.1),
            lambda_s=params.get("lambda_s", 1.0),
            lambda_c=params.get("lambda_c", 1.0),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=100, min_lr=learning_rate/20
        )

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
                if scheduler and checkpoint.get("scheduler_state_dict"):
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    
                    # Reset patience counter to wait a full 20 epochs in the new run before reducing LR
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.num_bad_epochs = 0
                        print("Reset ReduceLROnPlateau num_bad_epochs to 0 for the new run.")

                    # Force learning rate override to 0.000064 on resume
                    override_lr = 0.000025
                    print(f"Forcing learning rate override to {override_lr}...")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = override_lr

                start_epoch = checkpoint.get("epoch", 0)
                initial_best_val_loss = checkpoint.get("best_val_spearman")

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

        wandb.init(
            project="urban-income-prediction", 
            name=savename, config=params,
            id=savename, resume="allow"
        )

        #### 4. Run PyTorch Model
        model = train_model(
            model=model, train_loader=train_loader, val_loaders=val_loaders,
            loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, 
            epochs=n_epochs, device=device, savename=savename,
            cache_manager=train_cache_manager,
            start_epoch=start_epoch,
            initial_best_val_loss=initial_best_val_loss,
            val_cache_managers=vals_cache_manager_dict
        )
        
        wandb.finish()
        print("Fin del entrenamiento")
    
    if generate_predictions:
        print("Generando predicciones...")

        all_years_datasets, all_years_extents, df_train, df_vals_dict, df_test, df_dead_zone = build_dataset.generate_datasets(
            savename, sat_data, years, test_years, test_column, small_sample, max_jitter, tau_meters
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
            meta_dim=1,  # Set back to 1 for dist_to_center
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
        
        # 4. Process each year using the new chunk-grouped strategy
        for year in [2016, 2018, 2020, 2022, 2024, 2010, 2012, 2014]:

            print(f"\n{'='*80}")
            print(f"🚀 Processing Predictions for Year: {year}")
            print(f"{'='*80}")
            
            df_year = df_all[(df_all["year"] == year)].copy()
            if df_year.empty:
                print(f"No data for year {year}, skipping.")
                continue

            print(f"Found {len(df_year)} buildings to predict for {year}")

            output_path = RESULTS_DIR / f"{savename}/{year}_predictions.csv"
            
            # Use new chunk-grouped prediction strategy
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

            # Save georeferenced predictions
            gdf = gpd.read_parquet(PROCESSED_DATA_DIR / f"building_geometries_years{min(years)}-{max(years)}.parquet")
            df_result = df_result.set_index("DOITT_ID")
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
        "n_epochs": 700,
        "learning_rate": 0.0001,
        "sat_data": "aerial",
        "years": [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
        "test_years": [2016],
        "test_column": None,
        "extra": "_ranknet_mining_lambda_s_05",
        # In-Batch Ranking hyperparameters
        "m_base": 1.0,
        "m_min": 0.05,
        "lambda_s": 0.3,
        "lambda_c": 0.2,
        "temporal_fraction": 0.4,
    } 

    # Run full pipeline
    run(params, train=False, retrain=False, compute_loss=False, generate_predictions=True)