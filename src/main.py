##############      Configuración      ##############
import os
import shutil
from xml.parsers.expat import model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
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
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


# Define a subset of the data that will comfortably fit in RAM cache
CACHE_SIZE = 800*128 # Around 100k images (128 batch size)

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


class SatelliteDataset(Dataset):
    def __init__(self, df, all_years_datasets, image_size, resizing_size, nbands=4, stacked_images=[1], transform=None, mode="train", fixed_year=None):
        """
        Unified PyTorch Dataset for both Training (RAM Prefetch) and Prediction (Lazy Load).
        
        Args:
            mode: "train", "eval", or "predict". 
                  - "train" and "eval" prefetch to RAM and return (image, label).
                  - "predict" loads lazily and returns (image, GEOID).
            fixed_year: Optional. If provided (e.g., for prediction), forces the dataset to only use that year.
        """
        self.df = df.reset_index(drop=True)
        self.all_years_datasets = all_years_datasets
        self.available_years = list(all_years_datasets.keys())
        self.image_size = image_size
        self.resizing_size = resizing_size
        self.nbands = nbands
        self.stacked_images = stacked_images
        self.transform = transform
        self.mode = mode
        self.fixed_year = fixed_year
        
        self.total_bands = self.nbands * len(self.stacked_images)
        self.target_shape = (self.total_bands, self.image_size, self.image_size)
        
        # If we are training or evaluating, we prefetch the data into RAM
        if self.mode in ["train", "eval"]:
            self.cached_images = []
            self.cached_labels = []
            print(f"[{self.mode.upper()}] Pre-fetching {len(self.df)} samples into RAM...")
            self._prefetch_data()
        elif self.mode == "predict":
            print(f"[PREDICT] Initialized lazy-loading dataset for {len(self.df)} samples.")

    def _extract_image(self, row):
        """
        Core geospatial extraction logic. Reused by both prefetching and lazy loading.
        """
        polygon = row["geometry"]
        
        # Pick the year (fixed for prediction, random for training robustness)
        batch_year = self.fixed_year if self.fixed_year is not None else random.choice(self.available_years)
        
        primary_dataset = self.all_years_datasets[batch_year]
        dataset_name = row.get(f"dataset_{batch_year}")
        
        image = np.zeros(shape=(self.nbands, 0, 0))
        
        try:
            if not pd.isna(dataset_name):
                link_dataset = primary_dataset.get(dataset_name)
                if link_dataset is not None:
                    image, _ = geo_utils.stacked_image_from_census_tract(
                        dataset=link_dataset,
                        polygon=polygon,
                        img_size=self.image_size,
                        n_bands=self.nbands,
                        stacked_images=self.stacked_images,
                    )
        except Exception:
            pass # Silent fail handled by shape check below

        # Validate shape and process
        if image.shape != self.target_shape:
            processed_image = np.zeros(shape=(self.resizing_size, self.resizing_size, self.total_bands), dtype=np.uint8)
        else:
            processed_image = geo_utils.process_image(image, self.resizing_size)

        # Return as PyTorch (C, H, W) tensor
        return torch.tensor(processed_image, dtype=torch.uint8).permute(2, 0, 1)

    def _prefetch_data(self):
        """Populates the RAM cache. Only called if mode is 'train' or 'eval'."""
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            
            # Extract image
            image_tensor = self._extract_image(row)
            
            # Extract label
            label_tensor = torch.tensor(row["var"], dtype=torch.float32)
            
            self.cached_images.append(image_tensor)
            self.cached_labels.append(label_tensor)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.mode in ["train", "eval"]:
            # 1. FAST PATH: Return cached image and label
            img = self.cached_images[idx]
            lbl = self.cached_labels[idx]
            
            if self.transform:
                img = self.transform(img)
                
            return img, lbl
            
        elif self.mode == "predict":
            # 2. LAZY PATH: Extract image on the fly and return GEOID
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
        df = df.sample(1000, random_state=825).reset_index(drop=True)

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
    df_val = df_not_test.sample(frac=0.066667, random_state=200).reset_index(drop=True)
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

def setup_dataloaders(df_train, df_val, df_test, all_years_datasets, params):
    """
    Initializes train, val, and test PyTorch dataloaders from pre-split dataframes.
    """
    print("--- Initializing PyTorch Datasets ---")
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # 1. Define Phase-Specific Transforms
    # Only training data gets random augmentations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToDtype(torch.float32, scale=True)
    ])
    
    # Eval transforms ONLY cast to float32 (no random augmentations for val/test)
    eval_transform = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True)
    ])

    # 2. Instantiate the Datasets
    # Each one will pre-fetch its specific subset into RAM via the _prefetch_data() method
    train_dataset = SatelliteDataset(
        df=df_train, 
        all_years_datasets=all_years_datasets, 
        image_size=params["image_size"], 
        resizing_size=params["resizing_size"],
        nbands=params["nbands"],
        stacked_images=params["stacked_images"],
        transform=train_transform,
        mode="train"
    )
    
    val_dataset = SatelliteDataset(
        df=df_val, 
        all_years_datasets=all_years_datasets,
        image_size=params["image_size"], 
        resizing_size=params["resizing_size"],
        nbands=params["nbands"],
        stacked_images=params["stacked_images"],
        transform=eval_transform,
        mode="eval"
    )
    
    test_dataset = SatelliteDataset(
        df=df_test, 
        all_years_datasets=all_years_datasets,
        image_size=params["image_size"], 
        resizing_size=params["resizing_size"],
        nbands=params["nbands"],
        stacked_images=params["stacked_images"],
        transform=eval_transform,
        mode="eval"
    )

    # 3. Wrap them in DataLoaders
    # Only shuffle the training data via PyTorch's native shuffle flag!
    batch_size = params.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
    optimizer, 
    epochs, 
    device, 
    savename
):
    print("--- Starting PyTorch Training Loop ---")
    best_val_loss = float('inf')
    
    # Ensure the save directory exists
    save_dir = MODELS_DIR / "models_by_epoch" / savename
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # ==========================
        # 1. TRAINING PHASE
        # ==========================
        model.train() # Set model to training mode (enables dropout, batchnorm updates)
        running_train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # If your labels are shape (Batch,), and output is (Batch, 1), you might need to squeeze/unsqueeze
            if outputs.shape != labels.shape:
                outputs = outputs.view(labels.shape)
                
            loss = loss_fn(outputs, labels)
            
            # Backward pass & Optimizer step
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        model.eval() # Set model to eval mode (disables dropout)
        running_val_loss = 0.0
        
        # Disable gradient calculation for validation (saves RAM and compute)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                if outputs.shape != labels.shape:
                    outputs = outputs.view(labels.shape)
                    
                loss = loss_fn(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # ==========================
        # 3. LOGGING & CHECKPOINTING
        # ==========================
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # 🎯 Log metrics directly to W&B cloud!
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Model Checkpoint logic (Replaces Keras ModelCheckpoint callback)
        if epoch_val_loss < best_val_loss:
            print(f"⭐ Val loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving...")
            best_val_loss = epoch_val_loss
            
            model_path = save_dir / f"{savename}_best.pth"
            torch.save(model.state_dict(), model_path)
            
            # Optional: Tell wandb to track this specific file
            wandb.save(str(model_path))

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

        # ### Create train and test dataframes from ICPAG
        df_train, df_val, df_test = create_train_test_dataframes(
            df, savename, small_sample=small_sample
        )

        #### PyTorch Data Pipeline Setup
        print("Setting up data generators...")
        train_loader, val_loader, test_loader = setup_dataloaders(
                df_train=df_train, 
                df_val=df_val,
                df_test=df_test,
                all_years_datasets=all_years_datasets, 
                params=params
            )            
        print("Data Pipeline Ready!")
        print(f"Batches per epoch -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

        #### Model Initialization
        model, loss_fn = set_model_and_loss_function(
            model_name=model_name,
            kind=kind,
            bands=nbands * len(stacked_images),
            resizing_size=resizing_size,
            weights=weights,
        )

        # 1. Setup Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Initialize PyTorch Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 3. Initialize Weights & Biases
        wandb.init(
            project="urban-income-prediction", # Change to your project name
            name=savename,
            config=params
        )

        # 4. Run PyTorch Model
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=n_epochs,
            device=device,
            savename=savename
        )
        
        # 5. Finish the wandb run cleanly
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
        "model_name": "small_cnn",
        "kind": "reg",
        "weights": None,
        "image_size": 224,
        "resizing_size": 224,
        "tiles": 1,
        "nbands": 3,
        "stacked_images": [1],
        "sample_size": 1,
        "small_sample": False,
        "n_epochs": 100,
        "learning_rate": 0.001,
        "sat_data": "aerial",
        "years": [2022], # Only the data inside WSL! all data is: [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
        "extra": "",  # Extra info to add to the savename (e.g. for ablation studies)
    }

    # Run full pipeline
    run(params, train=True, retrain=True, compute_loss=False, generate_predictions=False)
