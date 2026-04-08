##############      Configuración      ##############

### Main libraries
import os
import gc
import time
import json
import shutil
import random
import logging
import warnings
import threading
import xarray as xr
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

### My modules
import src.true_metrics as true_metrics
import src.custom_models as custom_models
import src.build_dataset as build_dataset
import src.grid_predictions as  grid_predictions
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
# Set a VRAM hard limit (e.g., 7.5GB to be safe)
limit_in_bytes = 7.5 * 1024**3 
torch.cuda.set_per_process_memory_fraction(0.95, device=0)

os.environ['WANDB_API_KEY'] = os.getenv("WANDB_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Define a subset of the data that will comfortably fit in RAM cache
CACHE_SIZE = 2048*8 # Around 16000k images (8 batch size)

def generate_savename(
    model_name, image_size, learning_rate, years, extra
):
    years_str = "-".join(map(str, years))
    # stacked_images hardcoded to [1], so no stacking
    savename = (
        f"{model_name}_lr{learning_rate}_size{image_size}_y{years_str}{extra}"
    )

    return savename


def open_datasets(sat_data="aerial", years=[2013, 2018, 2022], tau_meters=100):

    ### Open dataframe with files and labels
    print("Reading dataset...")
    df = build_dataset.load_income_dataset(years, tau_meters=tau_meters)

    year_cols = []
    if sat_data == "aerial":
        datasets_all_years, extents_all_years = build_dataset.load_satellite_datasets(
            years=years
        )
    elif sat_data == "landsat":
        raise NotImplementedError("Landsat support not implemented yet.")
        sat_imgs_datasets, extents = build_dataset.load_landsat_datasets()

    df = build_dataset.assign_datasets_to_gdf(df, datasets_all_years, extents_all_years, years=years, verbose=True, save_plot=True)

    print("Datasets loaded!")

    return datasets_all_years, extents_all_years, df

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
        self._is_initialized = False
 
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.clear_cache:
            for f in self.cache_dir.glob("*.pt"):
                f.unlink()
        else:
            self._load_existing_shards()

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
 
    def _extract_raw_image(self, row, n_bands=None):
        dataset_name = row.get("dataset")
        zarr_array = self.all_years_datasets[dataset_name]["value"]  # raw zarr array

        # Apply jitter in pixel space — much cheaper than CRS perturbation
        jitter_row = np.random.randint(-self.max_jitter_pixels, self.max_jitter_pixels)
        jitter_col = np.random.randint(-self.max_jitter_pixels, self.max_jitter_pixels)
        row_start = max(0, int(row["row_start"]) + jitter_row)
        row_stop  = row_start + self.image_size
        col_start = max(0, int(row["col_start"]) + jitter_col)
        col_stop  = col_start + self.image_size
        
        try:
            tile = zarr_array[:n_bands, row_start:row_stop, col_start:col_stop]
            if (
                tile.shape[0] == self.nbands and
                tile.shape[1] == self.image_size and
                tile.shape[2] == self.image_size
            ):                
                return tile.to_numpy()  # Convert from zarr array to numpy array for processing
            else:
                raise ValueError(f"Extracted tile has invalid shape: {tile.shape}. Expected ({self.nbands}, {self.image_size}, {self.image_size}).")
            
        except Exception as e:
            logging.error(f"Failed for DOITT_ID {row.get('DOITT_ID', '?')}: {e}")

        return None

    def _batch_resize_and_convert(self, raw_images):
        """
        Resize a list of raw (C, H, W) numpy arrays to (C, image_size, image_size)
        in a SINGLE vectorised call via torch.nn.functional.interpolate on CPU.

        Why this is faster than calling geo_utils.process_image N times in a loop:
        - Skimage/PIL resize launches a separate C routine per image; Python loop overhead
          multiplies with N (here N ~ 3,000+).
        - F.interpolate operates on the entire (N, C, H, W) tensor at once using PyTorch's
          optimised BLAS/OpenMP kernels — one C call for the whole shard.
        - Eliminating the HWC ↔ CHW transpose that process_image performs also saves
          memory bandwidth.

        Returns a list of (C, image_size, image_size) uint8 tensors, matching the
        format previously produced by process_image + permute.
        """
        image_size = self.params["image_size"]

        # Stack into (N, C, H, W) float32 — interpolate requires floating-point input
        batch = torch.stack([
            torch.from_numpy(img.astype(np.float32)) for img in raw_images
        ])  # (N, C, image_size, image_size)

        # Single batched bilinear downsample — all images in one call
        # antialias=True avoids high-frequency aliasing when downsampling by 3x (672 → 224)
        resized = F.interpolate(
            batch,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )  # (N, C, image_size, image_size)

        # Clamp + cast to uint8 — same in-memory format as the previous pipeline
        resized = resized.clamp(0, 255).to(torch.uint8)

        return list(resized)  # list of (C, image_size, image_size) uint8 tensors

    def _worker_generate(self, shard_id, show_progress=False):
        sampled_df = self.df.sample(self.shard_size, replace=False)

        valid_images = []
        valid_labels = []
        raw_chunk = []
        label_chunk = []
        CHUNK_SIZE = 8
        MAX_EXTRACT_WORKERS = 8  # tune to your disk/CPU

        rows = list(sampled_df.iterrows())

        def _extract(item):
            _, row = item
            label = row["Rel_Score"]
            # ⚠️ CRITICAL: Skip if label is NaN
            if pd.isna(label):
                return None, None
            return self._extract_raw_image(row, n_bands=self.params["nbands"]), label

        with ThreadPoolExecutor(max_workers=MAX_EXTRACT_WORKERS) as pool:
            futures = [pool.submit(_extract, item) for item in rows]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=f"Generating shard {shard_id}")

            for future in iterator:
                raw_img, label = future.result()
                if raw_img is None or label is None:
                    continue

                raw_chunk.append(raw_img)
                label_chunk.append(label)

                if len(raw_chunk) >= CHUNK_SIZE:
                    resized_tensors = self._batch_resize_and_convert(raw_chunk)
                    for tensor, lbl in zip(resized_tensors, label_chunk):
                        if tensor.max() > 0:
                            valid_images.append(tensor)
                            valid_labels.append(lbl)
                    raw_chunk.clear()
                    label_chunk.clear()

        if raw_chunk:
            resized_tensors = self._batch_resize_and_convert(raw_chunk)
            for tensor, lbl in zip(resized_tensors, label_chunk):
                if tensor.max() > 0:
                    valid_images.append(tensor)
                    valid_labels.append(lbl)

        shard_path = self.cache_dir / f"shard_{shard_id}.pt"
        if valid_images:
            labels_tensor = torch.tensor(valid_labels, dtype=torch.float32)
            
            # 🔍 Safety check: verify no NaNs in labels
            if torch.isnan(labels_tensor).any():
                nan_count = torch.isnan(labels_tensor).sum().item()
                logging.error(f"⚠️ ALERT: Shard {shard_id} contains {nan_count} NaN labels! Filtering them out.")
                # Filter out NaN labels and corresponding images
                valid_mask = ~torch.isnan(labels_tensor)
                images_tensor = torch.stack(valid_images)
                images_tensor = images_tensor[valid_mask]
                labels_tensor = labels_tensor[valid_mask]
            else:
                images_tensor = torch.stack(valid_images)
            
            torch.save({"images": images_tensor, "labels": labels_tensor}, shard_path)
        else:
            logging.warning(f"Shard {shard_id}: all extracted images were black/invalid.")
            torch.save({"images": torch.empty(0), "labels": torch.empty(0)}, shard_path)

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
 
    def start_background_generation(self):
        """Kicks off generation of the next shard on a background thread."""
        if not self._is_initialized:
            raise RuntimeError("Call build_initial_cache() before starting background generation.")
        self.bg_thread = threading.Thread(
            # False: Hide progress bar for background thread
            target=self._worker_generate, args=(self.next_shard_idx, False), daemon=True 
        )
        self.bg_thread.start()
 
    def step(self):
        """Non-blocking cache rotation."""
        if not self._is_initialized:
            raise RuntimeError("Call build_initial_cache() before calling step().")

        if not self.single_shard_mode:
            if self.bg_thread is not None and self.bg_thread.is_alive():
                return False 
    
            oldest_shard = self.active_shards.pop(0)
            oldest_shard.unlink()
    
            self.active_shards.append(self.cache_dir / f"shard_{self.next_shard_idx}.pt")
            self.next_shard_idx += 1
    
            self.bg_thread = threading.Thread(
                # False: Hide progress bar for background thread
                target=self._worker_generate, args=(self.next_shard_idx, False), daemon=True
            )
            self.bg_thread.start()
            
            return True 
        return False
 

class CyclicRAMDataset(Dataset):
    def __init__(self, cache_manager: CyclicCacheManager, transform=None):
        if not cache_manager._is_initialized:
            raise RuntimeError(
                "Call cache_manager.build_initial_cache() before instantiating CyclicRAMDataset."
            )
        self.cache_manager = cache_manager
        self.transform = transform
        self.images = None
        self.labels = None
        self.refresh()
 
    def refresh(self):
        """Drops the current in-RAM tensors and reloads from the active shards on disk."""
        if not self.cache_manager.active_shards:
            raise RuntimeError("No active shards to load. Call build_initial_cache() first.")
 
        img_list, lbl_list = [], []
        for shard_path in self.cache_manager.active_shards:
            data = torch.load(shard_path, weights_only=True)
            img_list.append(data["images"])
            lbl_list.append(data["labels"])
 
        self.images = torch.cat(img_list)
        self.labels = torch.cat(lbl_list)
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = self.labels[idx]
 
        if self.transform:
            img = self.transform(img)
 
        return img, lbl

class EvalSatelliteDataset(Dataset):
    """Static dataset for Validation, Testing, and Prediction."""
    def __init__(self, df, all_years_datasets, params, transform=None, mode="eval", fixed_year=None):
        self.df = df.reset_index(drop=True)
        self.all_years_datasets = all_years_datasets
        self.params = params
        self.transform = transform
        self.mode = mode
        self.fixed_year = fixed_year
        self.available_years = list(all_years_datasets.keys())
        
        self.total_bands = params["nbands"] * 1  # stacked_images hardcoded to [1]
        self.target_shape = (self.total_bands, params["image_size"], params["image_size"])
        
        if self.mode == "eval":
            print(f"[{self.mode.upper()}] Pre-fetching {len(self.df)} samples into RAM...")
            self.cached_images = []
            self.cached_labels = []
            self._prefetch_data()

    def _extract_image(self, row):
        batch_year = self.fixed_year if self.fixed_year is not None else random.choice(self.available_years)
        primary_dataset = self.all_years_datasets[batch_year]
        dataset_name = row.get(f"dataset_{batch_year}")
        polygon = row["geometry"]
        
        image = np.zeros(shape=(self.params["nbands"], 0, 0))
        try:
            if not pd.isna(dataset_name):
                link_dataset = primary_dataset.get(dataset_name)
                if link_dataset is not None:
                    image, _ = geo_utils.stacked_image_from_census_tract(
                        dataset=link_dataset, polygon=polygon,
                        img_size=self.params["image_size"], n_bands=self.params["nbands"],
                        stacked_images=[1],  # hardcoded
                    )
        except Exception:
            pass

        if image.shape != self.target_shape:
            processed_image = np.zeros(shape=(self.params["image_size"], self.params["image_size"], self.total_bands), dtype=np.uint8)
        else:
            processed_image = geo_utils.process_image(image, self.params["image_size"])

        return torch.tensor(processed_image, dtype=torch.uint8).permute(2, 0, 1)

    def _prefetch_data(self):
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            self.cached_images.append(self._extract_image(row))
            self.cached_labels.append(torch.tensor(row["Rel_Score"], dtype=torch.float32))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.mode == "eval":
            img = self.cached_images[idx]
            lbl = self.cached_labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, lbl

        elif self.mode == "lazy_eval":
            # Lazy load path for Validation/Testing (Saves RAM and startup time)
            row = self.df.iloc[idx]
            img = self._extract_image(row)
            lbl = torch.tensor(row["Rel_Score"], dtype=torch.float32)
            if self.transform:
                img = self.transform(img)
            return img, lbl
        
        elif self.mode == "predict":
            row = self.df.iloc[idx]
            img = self._extract_image(row)
            geoid = row["GEOID"]
            if self.transform:
                img = self.transform(img)
            return img, geoid
                                
def create_train_test_dataframes(df, savename, test_years=[], test_column=None, small_sample=False):
    """Create train and test dataframes with the IDs and xr.datasets names to use for training and testing

    Split the census tracts into train and test. The train and test dataframes contain the links and xr.datasets to use for training and
    testing.
    """
    if small_sample:
        df = df.sample(1000, random_state=825).reset_index(drop=True)

    ### Split census tracts based on train/test
    #       (the hole census tract must be in the corresponding region)
    test_bounds = build_dataset.get_test_area_from_file(filename="Test_NYC_Area.parquet")
    # val_bounds = build_dataset.get_test_area_from_file(filename="Test_NYC_Area.parquet")

    # jitter buffer = 5 meters, to prevent all the images being generated at the exact center of the buildings...
    meters_jitter = 10
    meters_per_crs_unit = geo_utils.projected_units_to_meters(1.0, epsg_code=6539)
    jitter_in_projected_units = meters_jitter / meters_per_crs_unit
    jitter_buffer = jitter_in_projected_units / 2 # The zarr pixel dimension is 0.5 EPSG 6539, s
    train_mask, test_mask, val_mask = build_dataset.split_train_test(df, test_bounds_df=test_bounds, val_bounds_df=None, test_years=test_years, test_column=test_column, jitter_buffer=jitter_buffer)

    # Keep only relevant columns for the DataLoader
    relevant_columns = [
        "DOITT_ID", "GEOID", "year",
        "Rel_Score", "Valid_Structural_Change", "score_bin",
        "dataset", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
        "row_start", "row_stop", "col_start", "col_stop",
    ]
    df = df[relevant_columns]

    # Split dataframes and shuffle them
    df_train = df[train_mask].copy().reset_index(drop=True).sample(frac=1, random_state=825, replace=False)  # Shuffle train set
    df_test = df[test_mask].copy()
    df_val = df[val_mask].copy()
    
    assert df_test.shape[0] > 0, f"Empty test dataset!"
    assert df_train.shape[0] > 0, f"Empty train dataset!"

    ### Train/Test

    test_dataframe_path = PROCESSED_DATA_DIR / "test_datasets" / f"{savename}_test_dataframe.feather"
    df_test.to_feather(test_dataframe_path)
    print("Se creó el archivo:", test_dataframe_path)

    train_dataframe_path = PROCESSED_DATA_DIR / "train_datasets" / f"{savename}_train_dataframe.feather"
    df_train.to_feather(train_dataframe_path)
    print("Se creó el archivo:", train_dataframe_path)

    val_dataframe_path = PROCESSED_DATA_DIR / "val_datasets" / f"{savename}_val_dataframe.feather"
    df_val.to_feather(val_dataframe_path)
    print("Se creó el archivo:", val_dataframe_path)

    return df_train, df_val, df_test

def setup_dataloaders(df_train, df_val, df_test, all_years_datasets, params, train_cache_manager=None, val_cache_manager=None):
    print("--- Initializing PyTorch Datasets ---")
    print(f"Train: Cyclical Cache | Val: {len(df_val)} | Test: {len(df_test)}")

    batch_size = params.get("batch_size", 32)


    # TRAIN
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToDtype(torch.float32, scale=True)
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True)
    ])

    train_dataset = CyclicRAMDataset(
        cache_manager=train_cache_manager,
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # VAL
    if df_val.shape[0] > 0:
        val_dataset = CyclicRAMDataset(
            cache_manager=val_cache_manager,
            transform=eval_transform # No random augmentations
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size*32, shuffle=False, num_workers=0)
    else:
        val_loader = None

    # TEST

    # test_dataset = EvalSatelliteDataset(
    #     df=df_test, all_years_datasets=all_years_datasets,
    #     params=params, transform=eval_transform, mode="lazy_eval"
    # )
    test_loader = None # TODO
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
   
    return train_loader, val_loader, test_loader

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
    }
    validate_parameters(params, default_params)

    # Merge default and provided hyperparameters (keep from params)
    updated_params = {**default_params, **params}
    print(updated_params)

    return updated_params


def set_model_and_loss_function(
    model_name: str, kind: str, image_size: int, bands: int = 4, weights: str = None
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
        kind=kind
    )

    # Load weights if provided
    if weights is not None and weights not in ["imagenet"]:
        # PyTorch uses state_dicts to load weights
        model.load_state_dict(torch.load(weights))
        print(f"\n--- 🚀 Successfully loaded custom weights from: {weights} ---")

    # 2. Set loss functions
    # Note: PyTorch tracks metrics during the training loop manually, not in model compilation like Keras.
    if kind == "reg":
        loss_fn = nn.MSELoss()
        
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

    print(f"Se creó {filename} con los parametros utilizados.")
    return

def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    learning_rate,
    optimizer,
    scheduler,
    epochs,
    device,
    savename,
    cache_manager,
    start_epoch=0,
    initial_best_val_loss=None,
):
    print("--- Starting PyTorch Training Loop ---")
    best_val_loss = (
        initial_best_val_loss if initial_best_val_loss is not None else float('inf')
    )
    
    # Ensure the save directory exists
    save_dir = MODELS_DIR / "models_by_epoch" / savename
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Mixed Precision Scaler
    scaler = GradScaler()
    accumulation_steps = 4  # Accumulate gradients (e.g., batch_size 8 * 4 steps = effective batch size 32)

    for epoch in range(start_epoch, epochs):
               
        # ==========================
        # 1. TRAINING PHASE
        # ==========================
        
        # Start training loop
        model.train() # Set model to training mode (enables dropout, batchnorm updates)
        running_train_loss = 0.0
        
        # Zero the gradients
        optimizer.zero_grad()
        
        train_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch [{epoch+1}/{epochs}] Train",
            leave=False   # clears the bar when done, keeping output clean
        )

        t_batch_start = time.perf_counter()  # start timer before first batch
        for batch_idx, (images, labels) in train_bar:
            t_data_end = time.perf_counter()  # data is ready; measure load time
            images, labels = images.to(device), labels.to(device)
            
            # Run forward pass in Mixed Precision (FP16)
            t_forward_start = time.perf_counter()
            with autocast(device_type='cuda'):
                outputs = model(images)
                if outputs.shape != labels.shape:
                    outputs = outputs.view(labels.shape)
                
                if outputs.shape != labels.shape:
                    raise ValueError(
                        f"\n[BROADCASTING BUG CAUGHT]\n"
                        f"Output shape: {outputs.shape} | Target shape: {labels.shape}\n"
                        f"Because these don't match EXACTLY, PyTorch is comparing every prediction to every label in the batch. "
                        f"This forces the model to predict the average, flatlining the loss. "
                        f"Fix: Add 'outputs = outputs.squeeze()' before your loss function."
                    )
                
                loss = loss_fn(outputs, labels)

                # Scale loss to account for accumulation
                loss = loss / accumulation_steps 
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Only step the optimizer every `accumulation_steps`
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Sync GPU before stopping the timer so we measure real GPU time, not just kernel launch
            torch.cuda.synchronize()
            t_forward_end = time.perf_counter()

            # ── Per-step W&B perf metrics (logged every 50 steps to reduce overhead) ──
            if batch_idx % 50 == 0:
                data_ms   = (t_data_end   - t_batch_start) * 1000
                forward_ms = (t_forward_end - t_forward_start) * 1000
                total_ms   = data_ms + forward_ms
                wandb.log({
                    "perf/data_load_ms":   data_ms,
                    "perf/forward_ms":     forward_ms,
                    # Fraction of wall-time the GPU is actually computing (higher = better)
                    "perf/gpu_util_ratio": forward_ms / total_ms if total_ms > 0 else 0,
                    "perf/samples_per_sec": images.size(0) / (total_ms / 1000) if total_ms > 0 else 0,
                })
            t_batch_start = time.perf_counter()  # reset for next batch's data-load window
            
            # De-scale loss for logging
            step_loss = loss.item() * accumulation_steps
            running_train_loss += (step_loss) * images.size(0)
            samples_seen = (batch_idx + 1) * images.size(0)
            running_avg = running_train_loss / samples_seen

            # Live loss update in the bar
            train_bar.set_postfix(
                step=f"{step_loss:.4f}",       # current batch loss (keras-style: loss per step)
                avg=f"{running_avg:.4f}"       # running epoch average (keras-style: epoch progress)
            )

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Always step scheduler at end of epoch
        if scheduler:
            scheduler.step()

        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        if val_loader is not None:
            model.eval() # Set model to eval mode (disables dropout)
            running_val_loss = 0.0
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val", leave=False)

            # Disable gradient calculation for validation (saves RAM and compute)
            with torch.no_grad():
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    if images.dtype != torch.float32 or images.max() > 10.0:
                        raise RuntimeError(
                            f"\n[NORMALIZATION BUG CAUGHT]\n"
                            f"Image dtype: {images.dtype} | Max pixel value: {images.max().item()}\n"
                            f"ScaleMAE requires float32 tensors with standard ImageNet normalization. "
                            f"If the model receives raw 0-255 uint8 arrays, its attention layers will output pure noise and refuse to learn."
                        )
                    
                    with autocast(device_type='cuda'):
                        outputs = model(images)
                        if outputs.shape != labels.shape:
                            outputs = outputs.view(labels.shape)
                        loss = loss_fn(outputs, labels)
                    running_val_loss += loss.item() * images.size(0)
                    val_bar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        else:
            epoch_val_loss = float('nan')  # No validation data

        # ==========================
        # 3. LOGGING & CHECKPOINTING
        # ==========================
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # 🎯 Log metrics directly to W&B cloud!
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        if val_loader is not None:
            log_dict["val_loss"] = epoch_val_loss

        wandb.log(log_dict)

        # Model Checkpoint logic (Replaces Keras ModelCheckpoint callback)
        if epoch_val_loss < best_val_loss:
            tqdm.write(f"⭐ Val loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving...")
            best_val_loss = epoch_val_loss
            
            model_path = save_dir / f"{savename}_best.pth"
            trainable_weights = {k: v for k, v in model.state_dict().items() if v.requires_grad}
            torch.save(trainable_weights, model_path)
                
            # Optional: Tell wandb to track this specific file
            wandb.save(str(model_path))

        # ==========================
        # 4. ROTATE CACHE FOR NEXT EPOCH
        # ==========================
        cache_updated = cache_manager.step()
        if cache_updated:
            tqdm.write("🔄 New background shard ready! Loading updated data into RAM...")
            train_loader.dataset.refresh()
        else:
            # The GPU is faster than the CPU. Re-use current RAM data.
            pass

        # Save checkpoint after each epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, save_dir / f"{savename}_last.pth")

    # Return model for caller chaining (especially if we loaded/checkpointed externally)
    return model


def predict_buildings(model, dataloader, device):
    """
    Generates predictions for a large dataset using PyTorch batching.
    Returns a Pandas DataFrame with GEOIDs and their predicted values.
    """
    model.eval()
    all_preds = []
    all_geoids = []
    
    print(f"Starting inference on {len(dataloader.dataset)} items...")
    
    # torch.no_grad() is CRITICAL here! It stops PyTorch from storing memory for backpropagation
    with torch.no_grad():
        for batch_idx, (batch_images, batch_geoids) in enumerate(dataloader):
            batch_images = batch_images.to(device)
            
            # Get predictions
            outputs = model(batch_images)
            
            # If regression, outputs are shape (Batch, 1). We flatten them to a 1D list.
            preds = outputs.view(-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_geoids.extend(batch_geoids)
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Predicted batch {batch_idx}/{len(dataloader)}")
                
    # Zip the IDs and Predictions back together beautifully
    return pd.DataFrame({"GEOID": all_geoids, "predicted_value": all_preds})

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
    tau_meters = params.get("tau_meters", 100)

    savename = generate_savename(
        model_name, image_size, learning_rate, years, extra
    )
    log_dir = f"{LOGS_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    generate_parameters_log(params, savename)
    
    print(f"========== Starting Run for {savename} ==========")

    all_years_datasets, all_years_extents, df = open_datasets(
        sat_data=sat_data, years=years, tau_meters=tau_meters
    )

    if train:

        df_train, df_val, df_test = create_train_test_dataframes(
            df, savename, test_years=test_years, test_column=test_column, small_sample=small_sample
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
            print(checkpoint_dir, " does not exist. Starting fresh training run.")

        print("Building Initial Cache...")
        if small_sample:
            num_shards = 1
            current_shard_size = len(df_train) # E.g., ~150 images per shard
        else:
            num_shards = 10
            current_shard_size = CACHE_SIZE // 10             # E.g., 20,480 images per shard

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
        )
        train_cache_manager.build_initial_cache()
        train_cache_manager.start_background_generation() # Starts generating shard 6 for Epoch 1

        print("\n[VAL] Building static validation cache...")
        if df_val.shape[0] > 0:
            val_cache_manager = CyclicCacheManager(
                df=df_val, # Or full df_val
                all_years_datasets=all_years_datasets,
                params=params,
                cache_dir=CACHE_DIR,
                num_shards=1,               # Only build one shard
                shard_size=len(df_val),     # Make it big enough for the whole val set
                single_shard_mode=True,     # IMPORTANT: This disables rotation
                type="val",
                clear_cache=not resume_cache,
            )
            val_cache_manager.build_initial_cache() # This will build and show a progress bar
        else:
            val_cache_manager = None

        #### 2. PyTorch Data Pipeline Setup
        print("Setting up data generators...")
        train_loader, val_loader, test_loader = setup_dataloaders(
            df_train=df_train, df_val=df_val, df_test=df_test,
            all_years_datasets=all_years_datasets, params=params,
            train_cache_manager=train_cache_manager, val_cache_manager=val_cache_manager
        )            

        del df, df_test, all_years_extents
        gc.collect() # Force Python to free up memory from large objects we no longer need
        
        print("Data Pipeline Ready!")

        #### 3. Model Initialization
        model, loss_fn = set_model_and_loss_function(
            model_name=model_name, 
            kind=kind,
            bands=nbands,  # stacked_images hardcoded to [1]
            image_size=image_size, 
            weights=weights,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-7
        )

        # If we found a saved checkpoint from a previous run, resume from it
        start_epoch = 0
        initial_best_val_loss = None
        if not retrain and resume_model_checkpoint is not None and resume_model_checkpoint.exists():
            print(f"➡️ Resuming model/optimizer from checkpoint: {resume_model_checkpoint}")
            checkpoint = torch.load(resume_model_checkpoint, map_location=device)
            
            # Restore all states
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and checkpoint.get("scheduler_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                
            start_epoch = checkpoint["epoch"]
            initial_best_val_loss = checkpoint.get("best_val_loss")
            print(f"✅ Resumed at epoch {start_epoch+1} with best_val_loss={initial_best_val_loss}...")

        wandb.init(
            project="urban-income-prediction", 
            name=savename, config=params
        )

        #### 4. Run PyTorch Model
        model = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=loss_fn, learning_rate=learning_rate, optimizer=optimizer, scheduler=scheduler, 
            epochs=n_epochs, device=device, savename=savename,
            cache_manager=train_cache_manager,
            start_epoch=start_epoch,
            initial_best_val_loss=initial_best_val_loss,
        )
        
        wandb.finish()
        print("Fin del entrenamiento")
    
    if generate_predictions:
        print("Generando predicciones...")
        
        # 1. Load the Best PyTorch Model
        model, _ = set_model_and_loss_function(
            model_name=model_name,
            kind=kind,
            bands=nbands * 1,  # stacked_images hardcoded to [1]
            image_size=image_size,
        )
        
        best_model_path = MODELS_DIR / "models_by_epoch" / savename / f"{savename}_best.pth"
        model.load_state_dict(torch.load(best_model_path))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 2. Setup the Evaluaton Transforms
        eval_transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True)
        ])
        
        # 3. Setup the Unified Dataset in "predict" mode
        prediction_dataset = EvalSatelliteDataset(
            df=df,  
            all_years_datasets=all_years_datasets,
            params=params,
            transform=eval_transform,
            mode="predict",       # <--- Tells the class to use lazy loading and return GEOIDs
            fixed_year=2022       # <--- Forces the class to only extract 2022 data
        )
        
        # 4. Wrap in DataLoader (Double batch size since no gradients are stored)
        prediction_loader = DataLoader(
            prediction_dataset, 
            batch_size=params.get("batch_size", 32) * 2, 
            shuffle=False, 
            num_workers=0 
        )

        # 5. Generate the Predictions DataFrame
        df_result = predict_buildings(model, prediction_loader, device)
        
        # 6. Save and merge results
        df_result.to_parquet(RESULTS_DIR / "nyc_buildings_with_predictions.parquet")

        gdf = build_dataset.load_income_dataset(tau_meters=tau_meters)
        df_result.set_index("GEOID", inplace=True) 
        gdf = gdf.join(df_result, on="GEOID", how="inner")
        
        gdf.to_parquet(RESULTS_DIR / f"2022_predictions_{savename}.parquet")

        # Fix broken geometries and dissolve by census tract
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf_by_census_tract = gdf.dissolve(
            "GEOID", 
            aggfunc={'Rel_Score': 'mean', 'predicted_value': 'mean'}
        )
        gdf_by_census_tract.to_parquet(RESULTS_DIR / f"2022_predictions_by_tract_{savename}.parquet")

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
        "batch_size": 16,
        "small_sample": False,
        "n_epochs": 200,
        "learning_rate": 0.0003,
        "sat_data": "aerial",
        "years": [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024], # Only the data inside WSL! all data is: [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
        "test_years": [2016],
        "test_column": None,
        "extra": "",  # Extra info to add to the savename (e.g. for ablation studies)
    }

    # Run full pipeline
    run(params, train=True, retrain=False, compute_loss=False, generate_predictions=False)
