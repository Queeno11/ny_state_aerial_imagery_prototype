import os
import time
import dotenv
import zarr
import xarray as xr
import numpy as np

# Load environment variables
dotenv.load_dotenv()
imagery_root = os.getenv("IMAGERY_ROOT")
zarr_path = os.path.join(imagery_root, "nyc_2016.zarr")

print(f"Opening Zarr at: {zarr_path}")

# 1. Open with Xarray (original method)
t0 = time.time()
ds_xr = xr.open_dataset(zarr_path, engine="zarr", mask_and_scale=False)
xarr = ds_xr["value"]
t_open_xr = time.time() - t0
print(f"Xarray open time: {t_open_xr:.4f}s")

# 2. Open with Raw Zarr (no cache)
t0 = time.time()
z_group = zarr.open(zarr_path, mode="r")
zarr_arr = z_group["value"]
t_open_zarr = time.time() - t0
print(f"Raw Zarr open time: {t_open_zarr:.4f}s")

# 3. Open with Raw Zarr + LRU Store Cache (512MB)
t0 = time.time()
store = zarr.DirectoryStore(zarr_path)
cache_store = zarr.LRUStoreCache(store, max_size=512 * 1024 * 1024)
z_group_cached = zarr.open(cache_store, mode="r")
zarr_arr_cached = z_group_cached["value"]
t_open_cached = time.time() - t0
print(f"Cached Zarr open time: {t_open_cached:.4f}s")

# Define slice coordinates (sequential and spatial)
nbands = 3
image_size = 224

# Let's read 100 consecutive small slices that fall into the same or adjacent chunks
# In our dataset, consecutive buildings are close. Let's simulate this by moving slightly.
print("\n--- Running sequential slices (simulating spatial locality) ---")

# Warm up / read first chunk
_ = xarr[:nbands, 10000:10224, 10000:10224].to_numpy()
_ = zarr_arr[:nbands, 10000:10224, 10000:10224]
_ = zarr_arr_cached[:nbands, 10000:10224, 10000:10224]

# Benchmark 1: Xarray
t0 = time.time()
for i in range(100):
    offset = i * 10 # consecutive steps close to each other
    tile = xarr[:nbands, 10000+offset:10224+offset, 10000+offset:10224+offset].to_numpy()
t_xr_seq = time.time() - t0
print(f"Xarray (100 slices): {t_xr_seq:.4f}s ({1000 * t_xr_seq / 100:.2f} ms/slice)")

# Benchmark 2: Raw Zarr (no cache)
t0 = time.time()
for i in range(100):
    offset = i * 10
    tile = zarr_arr[:nbands, 10000+offset:10224+offset, 10000+offset:10224+offset]
t_zarr_seq = time.time() - t0
print(f"Raw Zarr (100 slices): {t_zarr_seq:.4f}s ({1000 * t_zarr_seq / 100:.2f} ms/slice)")

# Benchmark 3: Cached Zarr
t0 = time.time()
for i in range(100):
    offset = i * 10
    tile = zarr_arr_cached[:nbands, 10000+offset:10224+offset, 10000+offset:10224+offset]
t_cached_seq = time.time() - t0
print(f"Cached Zarr (100 slices): {t_cached_seq:.4f}s ({1000 * t_cached_seq / 100:.2f} ms/slice)")

print("\n--- Running widely spaced slices (crossing chunks) ---")
# Benchmark 1: Xarray
t0 = time.time()
for i in range(5):
    offset = i * 5000 # widely spaced to hit different chunks
    tile = xarr[:nbands, 10000+offset:10224+offset, 10000+offset:10224+offset].to_numpy()
t_xr_wide = time.time() - t0
print(f"Xarray (5 wide slices): {t_xr_wide:.4f}s ({1000 * t_xr_wide / 5:.2f} ms/slice)")

# Benchmark 2: Raw Zarr
t0 = time.time()
for i in range(5):
    offset = i * 5000
    tile = zarr_arr[:nbands, 10000+offset:10224+offset, 10000+offset:10224+offset]
t_zarr_wide = time.time() - t0
print(f"Raw Zarr (5 wide slices): {t_zarr_wide:.4f}s ({1000 * t_zarr_wide / 5:.2f} ms/slice)")

# Benchmark 3: Cached Zarr
t0 = time.time()
for i in range(5):
    offset = i * 5000
    tile = zarr_arr_cached[:nbands, 10000+offset:10224+offset, 10000+offset:10224+offset]
t_cached_wide = time.time() - t0
print(f"Cached Zarr (5 wide slices): {t_cached_wide:.4f}s ({1000 * t_cached_wide / 5:.2f} ms/slice)")
