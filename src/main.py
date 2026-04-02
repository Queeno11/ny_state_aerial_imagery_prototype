##############      Configuración      ##############
import os
import shutil
import threading
from xml.parsers.expat import model
from zipfile import Path
import pandas as pd
from peft import LoraConfig, get_peft_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

from tqdm import tqdm
from src import grid_predictions
from src.utils.paths import PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR, IMAGERY_ROOT
pd.set_option("display.max_columns", None)



# path_programas  = globales[7]
###############################################

import src.true_metrics as true_metrics
import src.custom_models as custom_models
import src.build_dataset as build_dataset
import src.grid_predictions as  grid_predictions
import src.geo_utils as geo_utils

import os
import sys
import json
import scipy
import random
import pandas as pd
import xarray as xr
import warnings
from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime

import wandb
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.amp import autocast, GradScaler # Add this to the top of main.py
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

os.environ['WANDB_API_KEY'] = os.getenv("WANDB_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Define a subset of the data that will comfortably fit in RAM cache
CACHE_SIZE = 2048*8 # Around 16000k images (8 batch size)
UNFREEZE_STAGE1_EPOCH = 5
UNFREEZE_STAGE2_EPOCH = 80

def generate_savename(
    model_name, image_size, learning_rate, stacked_images, years, extra
):
    years_str = "-".join(map(str, years))
    if len(stacked_images) > 1:
        stacked_str = "-".join(map(str, stacked_images))
        savename = f"{model_name}_lr{learning_rate}_size{image_size}_y{years_str}_stack{stacked_str}{extra}"
    else:
        savename = (
            f"{model_name}_lr{learning_rate}_size{image_size}_y{years_str}{extra}"
        )

    return savename


def open_datasets(sat_data="aerial", years=[2013, 2018, 2022]):

    ### Open dataframe with files and labels
    print("Reading dataset...")
    df = build_dataset.load_income_dataset()

    year_cols = []
    datasets_all_years = {}
    extents_all_years = {}
    for year in years:
        if sat_data == "aerial":
            sat_imgs_datasets, extents = build_dataset.load_satellite_datasets(
                year=year
            )
        elif sat_data == "landsat":
            raise NotImplementedError("Landsat support not implemented yet.")
            sat_imgs_datasets, extents = build_dataset.load_landsat_datasets()

        df = build_dataset.assign_datasets_to_gdf(df, extents, year=year, verbose=True)
        datasets_all_years[year] = sat_imgs_datasets
        extents_all_years[year] = extents
        year_cols += [f"dataset_{year}"]

    df = df[df[year_cols].notna().any(axis=1)]
    print("Datasets loaded!")

    return datasets_all_years, extents_all_years, df

class CyclicCacheManager:
    def __init__(self, df, all_years_datasets, params, cache_dir, num_shards=5, shard_size=20480, single_shard_mode=False, type="train"):
        self.df = df
        self.all_years_datasets = all_years_datasets
        self.params = params
        self.type = type
        self.cache_dir = cache_dir / f"{self.type}_cache"
        self.num_shards = num_shards
        self.shard_size = shard_size
        self.single_shard_mode = single_shard_mode

        self.active_shards = []
        self.next_shard_idx = 0
        self.bg_thread = None
        self._is_initialized = False
 
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for f in self.cache_dir.glob("*.pt"):
            f.unlink()
 
    def _extract_single_image(self, row):
        available_years = list(self.all_years_datasets.keys())
        batch_year = random.choice(available_years)
 
        primary_dataset = self.all_years_datasets[batch_year]
        dataset_name = row.get(f"dataset_{batch_year}")
        polygon = row["geometry"]
 
        nbands = self.params["nbands"]
        resizing_size = self.params["resizing_size"]
        total_bands = nbands * len(self.params["stacked_images"])
        target_shape = (total_bands, self.params["image_size"], self.params["image_size"])
 
        image = np.zeros(shape=(nbands, 0, 0))
        try:
            if not pd.isna(dataset_name):
                link_dataset = primary_dataset.get(dataset_name)
                if link_dataset is not None:
                    image, _ = geo_utils.stacked_image_from_census_tract(
                        dataset=link_dataset,
                        polygon=polygon,
                        img_size=self.params["image_size"],
                        n_bands=nbands,
                        stacked_images=self.params["stacked_images"],
                    )
        except Exception:
            pass
 
        if image.shape != target_shape:
            processed_image = np.zeros(
                shape=(resizing_size, resizing_size, total_bands), dtype=np.uint8
            )
        else:
            processed_image = geo_utils.process_image(image, resizing_size)
 
        # Keep as uint8 for disk/RAM efficiency — transform handles float32 later
        return torch.tensor(processed_image, dtype=torch.uint8).permute(2, 0, 1)
 
    def _worker_generate(self, shard_id, show_progress=False):
        """Runs in background: samples the dataframe and saves a new shard to disk."""
        sampled_df = self.df.sample(self.shard_size, replace=False)
        images, labels = [], []
 
        # Conditionally wrap the iterator in tqdm
        iterator = sampled_df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=sampled_df.shape[0], desc=f"Generating shard {shard_id}")
            
        for _, row in iterator:
            images.append(self._extract_single_image(row))
            labels.append(row["var"])
 
        shard_path = self.cache_dir / f"shard_{shard_id}.pt"
        torch.save(
            {
                "images": torch.stack(images),
                "labels": torch.tensor(labels, dtype=torch.float32),
            },
            shard_path,
        )
 
    def build_initial_cache(self):
        """Synchronously generates the initial num_shards shards. Call once before training."""
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
        
        self.total_bands = params["nbands"] * len(params["stacked_images"])
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
                        stacked_images=self.params["stacked_images"],
                    )
        except Exception:
            pass

        if image.shape != self.target_shape:
            processed_image = np.zeros(shape=(self.params["resizing_size"], self.params["resizing_size"], self.total_bands), dtype=np.uint8)
        else:
            processed_image = geo_utils.process_image(image, self.params["resizing_size"])

        return torch.tensor(processed_image, dtype=torch.uint8).permute(2, 0, 1)

    def _prefetch_data(self):
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            self.cached_images.append(self._extract_image(row))
            self.cached_labels.append(torch.tensor(row["var"], dtype=torch.float32))

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
            lbl = torch.tensor(row["var"], dtype=torch.float32)
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
                                
def create_train_test_dataframes(df, savename, small_sample=False):
    """Create train and test dataframes with the links and xr.datasets to use for training and testing

    Load the ICPAG dataset and assign the links to the corresponding xr.dataset, then split the census tracts
    into train and test. The train and test dataframes contain the links and xr.datasets to use for training and
    testing.
    """
    if small_sample:
        df = df.sample(1_000, random_state=825).reset_index(drop=True)

    ### Split census tracts based on train/test
    #       (the hole census tract must be in the corresponding region)
    df = build_dataset.split_train_test(df)
    df = df[
        ["GEOID", "var", "type", "geometry"]
        + [col for col in df.columns if "dataset" in col]
    ]

    ### Train/Test
    list_of_datasets = []

    df_test = df[df["type"] == "test"].copy().reset_index(drop=True)
    assert df_test.shape[0] > 0, f"Empty test dataset!"
    df_not_test = df[df["type"] == "train"].copy().reset_index(drop=True)
    assert df_not_test.shape[0] > 0, f"Empty train dataset!"

    # Shuffle everything so images are not sequential:
    df_test = df_test.sample(frac=1.0, random_state=825).reset_index(drop=True)
    df_not_test = df_not_test.sample(frac=1.0, random_state=825).reset_index(drop=True)
    
    # Generate validation dataset
    df_val = df_not_test.sample(frac=0.06667, random_state=200).reset_index(drop=True)
    df_train = df_not_test.drop(df_val.index).reset_index(drop=True)

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

    test_loader = None # TODO

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
        
    val_dataset = CyclicRAMDataset(
        cache_manager=val_cache_manager,
        transform=eval_transform # No random augmentations
    )

    
    # test_dataset = EvalSatelliteDataset(
    #     df=df_test, all_years_datasets=all_years_datasets,
    #     params=params, transform=eval_transform, mode="lazy_eval"
    # )

    batch_size = params.get("batch_size", 32)
    # Since all data is fully loaded in RAM, num_workers=0 is perfectly fast!
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*4, shuffle=False, num_workers=0)
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
    resizing_size = params["resizing_size"]
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

        if resizing_size > 1024:
            warnings.warn(
                "Warning: resizing_size greater than 1024 might encompass an area much bigger than the census tracts..."
            )

    elif sat_data == "landsat":
        if nbands > 10:
            raise ValueError("nbands for pleiades dataset must be less than 11.")

        if years != [2013]:
            raise ValueError("Landsat data only available in 2013.")

        if resizing_size > 32:
            warnings.warn(
                "Warning: resizing_size greater than 32 might encompass an area much bigger than the census tracts..."
            )

    return


def fill_params_defaults(params):

    default_params = {
        "model_name": "effnet_v2S",
        "kind": "reg",
        "weights": None,
        "image_size": 256,
        "resizing_size": 128,
        "tiles": 1,
        "nbands": 4,
        "stacked_images": [1],
        "sample_size": 5,
        "small_sample": False,
        "n_epochs": 100,
        "learning_rate": 0.0001,
        "sat_data": "pleiades",
        "years": [2013],
        "extra": "",
        "batch_size": 32
    }
    validate_parameters(params, default_params)

    # Merge default and provided hyperparameters (keep from params)
    updated_params = {**default_params, **params}
    print(updated_params)

    return updated_params


def set_model_and_loss_function(
    model_name: str, kind: str, resizing_size: int, bands: int = 4, weights: str = None
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
        resizing_size=resizing_size, 
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
    cache_manager
):
    print("--- Starting PyTorch Training Loop ---")
    best_val_loss = float('inf')
    
    # Ensure the save directory exists
    save_dir = MODELS_DIR / "models_by_epoch" / savename
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Mixed Precision Scaler
    scaler = GradScaler()
    accumulation_steps = 4  # Accumulate gradients (e.g., batch_size 8 * 4 steps = effective batch size 32)

    for epoch in range(epochs):
               
        # ==========================
        # 1. TRAINING PHASE
        # ==========================

        # ── Staged unfreezing ──────────────────────────────────────
        if epoch == UNFREEZE_STAGE1_EPOCH:
            tqdm.write(f"\n🔓 Epoch {epoch+1}: Unfreezing stage 1...")
            model.unfreeze_stage(1)
            
            new_lr = learning_rate * 0.1          # 1e-5
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            
            # Reinitialize scheduler from new LR, with remaining epochs as T_max
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=UNFREEZE_STAGE2_EPOCH - UNFREEZE_STAGE1_EPOCH,  # 40 epochs
                eta_min=1e-7
            )
            tqdm.write(f"   LR reset to {new_lr:.2e}, scheduler restarted.")

        if epoch == UNFREEZE_STAGE2_EPOCH:
            tqdm.write(f"\n🔓 Epoch {epoch+1}: Unfreezing stage 2...")
            model.unfreeze_stage(2)

            new_lr = learning_rate * 0.01         # 1e-6
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - UNFREEZE_STAGE2_EPOCH,  # 50 epochs
                eta_min=1e-7
            )
            tqdm.write(f"   LR reset to {new_lr:.2e}, scheduler restarted.")
        
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

        for batch_idx, (images, labels) in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Run forward pass in Mixed Precision (FP16)
            with autocast(device_type='cuda'):
                outputs = model(images)
                if outputs.shape != labels.shape:
                    outputs = outputs.view(labels.shape)
                
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
        scheduler.step()        

        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        model.eval() # Set model to eval mode (disables dropout)
        running_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val", leave=False)

        # Disable gradient calculation for validation (saves RAM and compute)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    if outputs.shape != labels.shape:
                        outputs = outputs.view(labels.shape)
                    loss = loss_fn(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # ==========================
        # 3. LOGGING & CHECKPOINTING
        # ==========================
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # 🎯 Log metrics directly to W&B cloud!
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Model Checkpoint logic (Replaces Keras ModelCheckpoint callback)
        if epoch_val_loss < best_val_loss:
            tqdm.write(f"⭐ Val loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving...")
            best_val_loss = epoch_val_loss
            
            model_path = save_dir / f"{savename}_best.pth"
            torch.save(model.state_dict(), model_path)
            
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
        

    print("--- Training Complete ---")
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
    resizing_size = params["resizing_size"]
    tiles = params["tiles"]
    nbands = params["nbands"]
    stacked_images = params["stacked_images"]
    sample_size = params["sample_size"]
    small_sample = params["small_sample"]
    n_epochs = params["n_epochs"]
    learning_rate = params["learning_rate"]
    sat_data = params["sat_data"]
    years = params["years"]
    extra = params["extra"]
    batch_size = params["batch_size"]

    savename = generate_savename(
        model_name, image_size, learning_rate, stacked_images, years, extra
    )
    log_dir = f"{LOGS_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    generate_parameters_log(params, savename)
    
    print(f"========== Starting Run for {savename} ==========")

    years = [2022]
    all_years_datasets, all_years_extents, df = open_datasets(
        sat_data=sat_data, years=years
    )

    if train:

        df_train, df_val, df_test = create_train_test_dataframes(
            df, savename, small_sample=small_sample
        )

        #### 1. Setup Cyclic Cache Manager
        print("Building Initial Cache...")
        if small_sample:
            num_shards = 1
            current_shard_size = len(df_train) # E.g., ~150 images per shard
        else:
            num_shards = 5
            current_shard_size = CACHE_SIZE // 5             # E.g., 20,480 images per shard

        print("\n[TRAIN] Building cyclic training cache...")
        train_cache_manager = CyclicCacheManager(
            df=df_train,
            all_years_datasets=all_years_datasets,
            params=params,
            cache_dir=CACHE_DIR,
            single_shard_mode=small_sample, # If small_sample is True, keep only one shard to speed up testing
            num_shards=num_shards,
            shard_size=current_shard_size, 
            type="train"
        )
        train_cache_manager.build_initial_cache()
        train_cache_manager.start_background_generation() # Starts generating shard 6 for Epoch 1

        print("\n[VAL] Building static validation cache...")
        val_cache_manager = CyclicCacheManager(
            df=df_val, # Or full df_val
            all_years_datasets=all_years_datasets,
            params=params,
            cache_dir=CACHE_DIR,
            num_shards=1,               # Only build one shard
            shard_size=len(df_val),     # Make it big enough for the whole val set
            single_shard_mode=True,     # IMPORTANT: This disables rotation
            type="val"
        )
        val_cache_manager.build_initial_cache() # This will build and show a progress bar

        #### 2. PyTorch Data Pipeline Setup
        print("Setting up data generators...")
        train_loader, val_loader, test_loader = setup_dataloaders(
            df_train=df_train, df_val=df_val, df_test=df_test,
            all_years_datasets=all_years_datasets, params=params,
            train_cache_manager=train_cache_manager, val_cache_manager=val_cache_manager
        )            
        print("Data Pipeline Ready!")

        #### 3. Model Initialization
        model, loss_fn = set_model_and_loss_function(
            model_name=model_name, kind=kind,
            bands=nbands * len(stacked_images),
            resizing_size=resizing_size, weights=weights,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )

        wandb.init(
            project="urban-income-prediction", 
            name=savename, config=params
        )

        #### 4. Run PyTorch Model
        model = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=loss_fn, learning_rate=learning_rate, optimizer=optimizer, scheduler=scheduler, 
            epochs=n_epochs, device=device, savename=savename,
            cache_manager=train_cache_manager
        )
        
        wandb.finish()
        print("Fin del entrenamiento")
    
    if generate_predictions:
        print("Generando predicciones...")
        
        # 1. Load the Best PyTorch Model
        model, _ = set_model_and_loss_function(
            model_name=model_name,
            kind=kind,
            bands=nbands * len(stacked_images),
            resizing_size=resizing_size,
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
        prediction_dataset = SatelliteDataset(
            df=df,  
            all_years_datasets=all_years_datasets,
            image_size=image_size,
            resizing_size=resizing_size,
            nbands=nbands,
            stacked_images=stacked_images,
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

        gdf = build_dataset.load_income_dataset()
        df_result.set_index("GEOID", inplace=True) 
        gdf = gdf.join(df_result, on="GEOID", how="inner")
        
        gdf.to_parquet(RESULTS_DIR / f"2022_predictions_{savename}.parquet")

        # Fix broken geometries and dissolve by census tract
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf_by_census_tract = gdf.dissolve(
            "GEOID", 
            aggfunc={'var': 'mean', 'predicted_value': 'mean'}
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
        "resizing_size": 224,
        "tiles": 1,
        "nbands": 3,
        "stacked_images": [1],
        "sample_size": 1,
        "batch_size": 16,
        "small_sample": True,
        "n_epochs": 200,
        "learning_rate": 0.0001,
        "sat_data": "aerial",
        "years": [2022], # Only the data inside WSL! all data is: [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
        "extra": "",  # Extra info to add to the savename (e.g. for ablation studies)
    }

    # Run full pipeline
    run(params, train=True, retrain=True, compute_loss=False, generate_predictions=False)
