##############      Configuración      ##############
from ast import Return
import os
import pickle
from tokenize import String
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from src.utils.paths import PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR, IMAGERY_ROOT
from pathlib import Path
from pyproj import CRS

pd.set_option("display.max_columns", None)

# path_programas  = globales[7]

import geopandas as gpd
import xarray as xr
import shapely.geometry as sg
import pandas as pd
import src.geo_utils as geo_utils


def load_satellite_datasets(year=2014, stretch=False, engine="zarr"):
    """Load satellite datasets and get their extents"""

    if engine=="zarr":
        file = f"nyc_{year}.zarr"
        dataset_path = IMAGERY_ROOT
        files = [file]

    elif engine=="tif":
        dataset_path = IMAGERY_ROOT / year
        files = os.listdir(dataset_path)
        files = [f for f in files if f.endswith(".tif")]
        assert all([os.path.isfile(dataset_path / f) for f in files])

    else:
        raise ValueError(f"Unknown engine: {engine}. Valid engines are: zarr, tif")

    print(f"Loading {len(files)} files from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise ValueError(f"Year {year} images not found: {dataset_path} does exist! Check they are stored in WSL!")
    datasets = {
        f: (filter_black_pixels(xr.open_dataset(dataset_path / f, engine=engine,  mask_and_scale=False)))
        for f in files
    }

    if stretch:
        datasets = {name: stretch_dataset(ds) for name, ds in datasets.items()}

    extents = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets.items()}

    return datasets, extents


# def load_landsat_datasets(stretch=False):
#     """Load satellite datasets and get their extents"""

#     files = os.listdir(rf"{path_landsat}")
#     assert os.path.isdir(rf"{path_landsat}")
#     files = [f for f in files if f.endswith(".tif")]
#     assert all([os.path.isfile(rf"{path_landsat}/{f}") for f in files])

#     datasets = {
#         f.replace(".tif", ""): (
#             normalize_landsat(xr.open_dataset(rf"{path_landsat}/{f}"))
#         )
#         for f in files
#     }
#     if stretch:
#         datasets = {name: stretch_dataset(ds) for name, ds in datasets.items()}

#     extents = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets.items()}

#     return datasets, extents


# def load_nightlight_datasets(stretch=False):
#     """Load satellite datasets and get their extents"""

#     files = os.listdir(rf"{path_nocturnas}")
#     assert os.path.isdir(rf"{path_nocturnas}")
#     files = [f for f in files if f.endswith(".tif")]
#     assert all([os.path.isfile(rf"{path_nocturnas}/{f}") for f in files])

#     datasets = {
#         f.replace(".tif", ""): (xr.open_dataset(rf"{path_nocturnas}/{f}"))
#         for f in files
#     }
#     if stretch:
#         datasets = {name: stretch_dataset(ds) for name, ds in datasets.items()}

#     extents = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets.items()}

#     return datasets, extents


def get_closest_acs_year(year, acs_years=[2009, 2014, 2019, 2024]):
    """
    Given a year and a list of panel years, return the closest panel year.
    This is used to match each building-year pair with the appropriate ACS labels.
    """
    closest_year = min(acs_years, key=lambda y: abs(y - year))
    return closest_year

def process_acs_panel():
    print("Loading and processing ACS panel data...")
    panel_path = Path(
        r"/mnt/c/Working Papers/NY State Aerial Imagery Prototype/"
        r"ny_state_aerial_imagery_prototype/data/processed/"
        r"ny_tracts_panel_2009_2014_2019_2024.feather"
    )
    panel_tract_gdf = gpd.read_feather(panel_path)
    return panel_tract_gdf


def load_building_data():
    print("Loading building footprint data from GeoJSON files...")

    BUILDINGS_DATASET_DIR = Path(
        r"/mnt/e/Datasets/Building Footprints/NYC Building Footprints"
    )
    buildings_now = gpd.read_file(
        BUILDINGS_DATASET_DIR / "BUILDING_view_8608618849432433473.geojson"
    )
    building_historical = gpd.read_file(
        BUILDINGS_DATASET_DIR / "BUILDING_HISTORIC_view_4222244593352533104.geojson"
    )

    # Keep only buildings demolished after the start of our panel
    building_historical = building_historical[
        building_historical["DEMOLITION_YEAR"] > 2009
    ]
    buildings_nyc = pd.concat([buildings_now, building_historical])

    # Log problematic duplicate IDs for inspection
    problematic_ids = buildings_nyc[
        buildings_nyc.duplicated(subset=["DOITT_ID"], keep=False)
    ]
    problematic_ids.to_parquet(
        r"/mnt/c/Working Papers/NY State Aerial Imagery Prototype/"
        r"ny_state_aerial_imagery_prototype/data/processed/"
        r"problematic_building_ids.parquet"
    )

    # Drop sentinel DOITT_ID = 0 (unusable, not a real building)
    buildings_nyc = buildings_nyc[buildings_nyc["DOITT_ID"] != 0]

    # For duplicated IDs, keep the record with the most recent demolition year
    buildings_nyc = (
        buildings_nyc
        .sort_values("DEMOLITION_YEAR", ascending=False)
        .drop_duplicates(subset=["DOITT_ID"], keep="first")
    )

    buildings_nyc = buildings_nyc[
        ["OBJECTID", "DOITT_ID", "CONSTRUCTION_YEAR", "DEMOLITION_YEAR", "geometry"]
    ]
    # Fill missing years with sentinel values:
    #   CONSTRUCTION_YEAR = 0  → building always existed before the panel
    #   DEMOLITION_YEAR   = 2999 → building still standing
    # Existence filter: CONSTRUCTION_YEAR <= year < DEMOLITION_YEAR
    buildings_nyc["CONSTRUCTION_YEAR"] = buildings_nyc["CONSTRUCTION_YEAR"].fillna(0)
    buildings_nyc["DEMOLITION_YEAR"] = buildings_nyc["DEMOLITION_YEAR"].fillna(2999)
    buildings_nyc = buildings_nyc.set_index("DOITT_ID")

    output_path = (
        r"/mnt/c/Working Papers/NY State Aerial Imagery Prototype/"
        r"ny_state_aerial_imagery_prototype/data/processed/buildings_nyc.parquet"
    )
    buildings_nyc.to_parquet(output_path)
    return buildings_nyc


def load_income_dataset(panel_years, tau_meters=50):
    """
    Produces two artifacts for the Zero-Join DataLoader:

      1. temporal_data.parquet  — flat table: one row per (building, year).
                                  Contains bbox coordinates, ACS labels, and
                                  stratification bins. No geometry column.
                                  This is the hot path the DataLoader reads.

      2. geometries.parquet     — geometry lookup: one row per building,
                                  indexed by DOITT_ID. Only used offline for
                                  visualisation and debugging; the DataLoader
                                  never touches this file at training time.
    """
    OUTPUT_DIR = Path(
        r"/mnt/c/Working Papers/NY State Aerial Imagery Prototype/"
        r"ny_state_aerial_imagery_prototype/data/processed/"
    )
    METRIC_CRS = "EPSG:6539"

    temporal_data_path = OUTPUT_DIR / f"temporal_data_t{tau_meters}_years{panel_years.min()}-{panel_years.max()}.parquet"
    geometries_path = OUTPUT_DIR / f"building_geometries_years{panel_years.min()}-{panel_years.max()}.parquet"

    # Check if output files already exist to avoid redundant processing
    if temporal_data_path.exists() and geometries_path.exists():
        print(f"Output files already exist at:\n  {temporal_data_path}\n  {geometries_path}")
        print("Loading existing datasets...")
        temporal_data_flat = pd.read_parquet(temporal_data_path)
        geometries_df = pd.read_parquet(geometries_path, index_col="DOITT_ID")
        return temporal_data_flat, geometries_df

    buildings_nyc = load_building_data()
    panel_tract_gdf = process_acs_panel()

    # ------------------------------------------------------------------ #
    # 1. CRS alignment                                                   #
    # ------------------------------------------------------------------ #
    print("1. Preparing Spatial Data and CRS...")
    if buildings_nyc.crs != METRIC_CRS:
        buildings_nyc = buildings_nyc.to_crs(METRIC_CRS)
    if panel_tract_gdf.crs != METRIC_CRS:
        panel_tract_gdf = panel_tract_gdf.to_crs(METRIC_CRS)

    # ------------------------------------------------------------------ #
    # 2. Assign buildings to 2024 census tracts                          #
    # ------------------------------------------------------------------ #
    print("2. Assigning Buildings to 2024 Census Tracts...")
    tracts_2024 = (
        panel_tract_gdf[["geoid_2024", "geometry"]]
        .rename(columns={"geoid_2024": "GEOID"})
    )
    buildings_mapped = gpd.sjoin(
        buildings_nyc, tracts_2024, how="inner", predicate="intersects"
    )
    buildings_mapped = buildings_mapped.drop(columns=["index_right"])

    # ------------------------------------------------------------------ #
    # 3. Apply tau buffer and extract bounding boxes                      #
    # ------------------------------------------------------------------ #
    print(f"3. Applying Context Spillover (tau = {tau_meters}m) and Extracting BBoxes (assuming zarr has 0.5 EPSG:6539 units per pixel)...")
    
    meters_per_crs_unit = projected_units_to_meters(1.0, 6539)    
    tau_crs_units = tau_meters / meters_per_crs_unit
    buffered_geoms = buildings_mapped.centroid.buffer(tau_crs_units)

    bounds = buffered_geoms.bounds
    buildings_mapped["bbox_minx"] = bounds["minx"]
    buildings_mapped["bbox_miny"] = bounds["miny"]
    buildings_mapped["bbox_maxx"] = bounds["maxx"]
    buildings_mapped["bbox_maxy"] = bounds["maxy"]

    # --- Artifact 1: geometry lookup (indexed by DOITT_ID) ---
    geometries_df = buildings_mapped[["geometry"]].copy()  # index = DOITT_ID

    # [FIX HERE] Now safe to drop geometry AND reset index so DOITT_ID becomes a regular column.
    # This prevents DOITT_ID from being lost during pd.concat later.
    buildings_mapped = buildings_mapped.drop(columns=["geometry"]).reset_index()

    # ------------------------------------------------------------------ #
    # 4. Unroll to (building, year) pairs                                 #
    # ------------------------------------------------------------------ #
    print("4. Unrolling Temporal Building-Year Pairs...")
    temporal_rows = []
    for year in panel_years:
        existed_mask = (
            (buildings_mapped["CONSTRUCTION_YEAR"] <= year) &
            (buildings_mapped["DEMOLITION_YEAR"] > year)
        )
        bldgs_year = buildings_mapped[existed_mask].copy()
        bldgs_year["year"] = year
        acs_year = get_closest_acs_year(year)

        # Merge ACS labels for this specific year
        tract_labels = (
            panel_tract_gdf[["geoid_2024", "Valid_Structural_Change", f"Rel_Score_{acs_year}"]]
            .copy()
            .rename(columns={
                "geoid_2024": "GEOID",
                f"Rel_Score_{acs_year}": "Rel_Score",
            })
        )
        bldgs_year = bldgs_year.merge(tract_labels, on="GEOID", how="inner")
        temporal_rows.append(bldgs_year)

    temporal_df = pd.concat(temporal_rows, ignore_index=True)

    # ------------------------------------------------------------------ #
    # 5. Stratified score bins — computed WITHIN each year               #
    # ------------------------------------------------------------------ #
    print("5. Calculating Year-Stratified Score Bins...")
    temporal_df["score_bin"] = (
        temporal_df
        .groupby("year")["Rel_Score"]
        .transform(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop")
        )
    )

    # ------------------------------------------------------------------ #
    # 6. Build the flat temporal table (DataLoader hot path)             #
    # ------------------------------------------------------------------ #
    print("6. Building Flat Temporal Table...")
    # Because we called .reset_index() earlier, DOITT_ID is now safely a column.
    hot_path_columns = [
        "DOITT_ID", "GEOID", "year",
        "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
        "Rel_Score", "Valid_Structural_Change", "score_bin",
    ]
    missing = [c for c in hot_path_columns if c not in temporal_df.columns]
    if missing:
        raise KeyError(
            f"Expected columns missing from temporal_df: {missing}\n"
            f"Available columns: {list(temporal_df.columns)}"
        )
    temporal_data_flat = temporal_df[hot_path_columns].copy()
    
    # ------------------------------------------------------------------ #
    # 7. Save                                                             #
    # ------------------------------------------------------------------ #
    print("7. Saving to Parquet...")
    temporal_data_flat.to_parquet(
        temporal_data_path, index=False
    )
    geometries_df.to_parquet(
        geometries_path, index=True  # Keep DOITT_ID as index for geometries
    )

    print(
        f"\nDone!\n"
        f"  temporal_data.parquet : {len(temporal_data_flat):,} rows "
        f"({temporal_data_flat['year'].nunique()} years × buildings)\n"
        f"  geometries.parquet    : {len(geometries_df):,} unique buildings\n"
        f"  Score bins computed within each of: {sorted(temporal_df['year'].unique())}"
    )
    return temporal_data_flat

def projected_units_to_meters(value: float, epsg_code: int) -> float:
    """
    Converts a value from the native units of a projected CRS into meters.
    
    Args:
        value (float): The distance in the native CRS units (e.g., 0.5)
        epsg_code (int): The EPSG code of the projected coordinate system
        
    Returns:
        float: The exact equivalent distance in meters.
    """
    crs = CRS.from_epsg(epsg_code)
    
    # 1. Safety check: ensure the CRS is projected (linear), not geographic (angular/degrees)
    if crs.is_geographic:
        raise ValueError(
            f"EPSG:{epsg_code} is a geographic CRS ({crs.name}). "
            "Its units are degrees, which do not have a constant meter length. "
            "Please provide a Projected CRS."
        )
        
    # 2. Fetch the exact conversion factor to meters for this specific CRS
    # axis_info[0] looks at the first spatial axis (usually Easting/X)
    conversion_factor = crs.axis_info[0].unit_conversion_factor
    unit_name = crs.axis_info[0].unit_name
    
    print(f"EPSG:{epsg_code} native unit is '{unit_name}'.")
    print(f"Conversion factor to meters: 1 {unit_name} = {conversion_factor} meters.")
    
    # 3. Calculate and return
    return value * conversion_factor
def projected_units_to_meters(value: float, epsg_code: int) -> float:
    """
    Converts a value from the native units of a projected CRS into meters.
    
    Args:
        value (float): The distance in the native CRS units (e.g., 0.5)
        epsg_code (int): The EPSG code of the projected coordinate system
        
    Returns:
        float: The exact equivalent distance in meters.
    """
    crs = CRS.from_epsg(epsg_code)
    
    # 1. Safety check: ensure the CRS is projected (linear), not geographic (angular/degrees)
    if crs.is_geographic:
        raise ValueError(
            f"EPSG:{epsg_code} is a geographic CRS ({crs.name}). "
            "Its units are degrees, which do not have a constant meter length. "
            "Please provide a Projected CRS."
        )
        
    # 2. Fetch the exact conversion factor to meters for this specific CRS
    # axis_info[0] looks at the first spatial axis (usually Easting/X)
    conversion_factor = crs.axis_info[0].unit_conversion_factor
    unit_name = crs.axis_info[0].unit_name
    
    print(f"EPSG:{epsg_code} native unit is '{unit_name}'.")
    print(f"Conversion factor to meters: 1 {unit_name} = {conversion_factor} meters.")
    
    # 3. Calculate and return
    return value * conversion_factor

def assign_datasets_to_gdf(
    df,
    extents,
    year=2013,
    verbose=True,
    save_plot=True,
):
    """Assign each geometry a dataset if the census tract falls within the extent of the dataset (images)

    Parameters:
    -----------
    df: pandas.DataFrame, must have columns "centroid_x" and "centroid_y" with the coordinates of the centroid of the census tract
    extents: dict, dictionary with the extents of the satellite datasets
    year: int, year of the satellite images
    centroid: bool, if True, the centroid of the census tract is used to assign the dataset
    select: str, method to select the dataset. Options are "first_match" or "all_matches"
    """
    import warnings
    warnings.filterwarnings("ignore")

    if "centroid_x" not in df.columns or "centroid_y" not in df.columns:
        raise ValueError("DataFrame must have 'centroid_x' and 'centroid_y' columns with the coordinates of the centroid of the census tract")  

    if year is None:
        colname = "dataset"
    else:
        colname = f"dataset_{year}"

    for name, bbox in extents.items():
        xmin, ymin, xmax, ymax = bbox
        inside_bbox = (
            (df["centroid_x"] >= xmin) &
            (df["centroid_x"] <= xmax) &
            (df["centroid_y"] >= ymin) &
            (df["centroid_y"] <= ymax)
        )
        df.loc[inside_bbox, colname] = name

    nan_links = df[colname].isna().sum()
    df = df[df[colname].notna()]

    if verbose:
        print(
            f"Links without images ({year}):", nan_links, "out of", len(df) + nan_links
        )
        print(f"Remaining links for train/test ({year}):", len(df))
    if save_plot:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["centroid_x"], df["centroid_y"]),
            crs="EPSG:6539",
        )
        gdf.plot()
        plt.savefig(rf"{PROCESSED_DATA_DIR}/links_with_images.png")

    warnings.filterwarnings("default")

    return df



def get_test_area_from_file(filename="Test_NYC_Area.parquet"):
    """
    Reads the test patches and extracts their exact bounding box coordinates.
    Since the patches are rectangular, their bounds perfectly represent their area.
    """
    test = gpd.read_parquet(RAW_DATA_DIR / filename)
    
    # .bounds returns a DataFrame with columns: ['minx', 'miny', 'maxx', 'maxy']
    # One row for each rectangular test patch.
    test_bounds_df = test.geometry.bounds
    
    return test_bounds_df

def split_train_test(
    df: pd.DataFrame, 
    test_bounds_df: pd.DataFrame, 
    val_bounds_df: pd.DataFrame = None,
    test_years: List[int] = None,
    test_column: str = "None",
    spillover_buffer: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a flat dataset into train and test sets using pure numerical bounding box comparisons
    against multiple test patches.
    
    Args:
        df: Pandas DataFrame containing the building bbox coordinates.
        test_bounds_df: DataFrame of test patch boundaries (minx, miny, maxx, maxy).
        val_bounds_df: Optional DataFrame of validation patch boundaries (minx, miny, maxx, maxy).
        test_years: Optional list of years to assign to the test set (if doing a temporal split in addition to spatial).
        test_column: Optional boolean column name in df to use for another splitting (e.g., df["Boston"]==True). If provided, 
                this column will be used to assign rows to the test set regardless of spatial location.
        spillover_buffer: Distance to expand the train building bboxes to prevent spatial leakage.
                          
    Returns:
        train_df, test_df
    """

    ##### SPATIAL SPLIT LOGIC #####

    # Initialize boolean masks
    # Test mask starts completely False. Train mask starts completely True.
    spatial_test_mask = pd.Series(False, index=df.index)
    spatial_val_mask = pd.Series(False, index=df.index)
    train_mask = pd.Series(True, index=df.index)
    
    # Iterate through each rectangular test patch
    for subset in ["Test", "Validation"]:
        mask = spatial_test_mask if subset == "Test" else spatial_val_mask
        bounds_df = test_bounds_df if subset == "Test" else val_bounds_df

        for _, test_patch in bounds_df.iterrows():
            t_minx = test_patch['minx']
            t_miny = test_patch['miny']
            t_maxx = test_patch['maxx']
            t_maxy = test_patch['maxy']
            
            # ----------------------------------------------------
            # TEST SET: Does the building fall entirely inside THIS patch?
            # ----------------------------------------------------
            inside_this_patch = (
                (df['bbox_minx'] >= t_minx) &
                (df['bbox_maxx'] <= t_maxx) &
                (df['bbox_miny'] >= t_miny) &
                (df['bbox_maxy'] <= t_maxy)
            )
            # Add to the global test mask (Logical OR)
            mask = mask | inside_this_patch
            
            # ----------------------------------------------------
            # TRAIN SET: Is the building completely outside THIS patch (with buffer)?
            # ----------------------------------------------------
            completely_outside_this_patch = (
                ((df['bbox_minx'] - spillover_buffer) > t_maxx) |
                ((df['bbox_maxx'] + spillover_buffer) < t_minx) |
                ((df['bbox_miny'] - spillover_buffer) > t_maxy) |
                ((df['bbox_maxy'] + spillover_buffer) < t_miny)
            )
            # If it intersects THIS patch, we must REMOVE it from the train set (Logical AND)
            train_mask = train_mask & completely_outside_this_patch            

        if subset == "Test":
            spatial_test_mask = mask
        else:
            spatial_val_mask = mask    
    
    ##### TIME SPLIT LOGIC #####
    time_mask = pd.Series(False, index=df.index)
    for test_year in test_years:
        time_mask = time_mask & (df["year"] == test_year)

    ##### OTHER SPLIT LOGIC (e.g., by city) #####
    other_mask = pd.Series(False, index=df.index)
    if test_column in df.columns:
        other_mask = df[test_column] == True  # Example: assign all Boston rows to test set

    ##### FINAL COMBINATION LOGIC #####
    # A row is in the test set if it meets ANY of the test criteria (spatial OR temporal OR other)
    final_test_mask = spatial_test_mask | time_mask | other_mask
    final_val_mask = spatial_val_mask  # Validation set is purely spatial in this example
    final_train_mask = ~final_test_mask & ~final_val_mask  # Train set is everything that is NOT in test or val

    # Create the final DataFrames
    test_df = df[final_test_mask].copy()
    val_df = df[final_val_mask].copy() if val_bounds_df is not None else None
    train_df = df[final_train_mask].copy()

    # Logging
    print(f"Total buildings evaluated: {len(df):,}")
    print(f"Assigned to Test Set (strictly inside patches): {len(test_df):,}")
    print(f"     - of which assigned by spatial split: {spatial_test_mask.sum():,}")
    print(f"     - of which assigned by temporal split: {time_mask.sum():,}")
    print(f"     - of which assigned by other split ({test_column}): {other_mask.sum():,}")
    print(f"     - Overlaps between criteria: {spatial_test_mask.sum() + time_mask.sum() + other_mask.sum():,}")
    if val_bounds_df is not None:
        print(f"Assigned to Validation Set (strictly inside val patches): {len(val_df):,}")
    print(f"Assigned to Train Set (strictly outside patches + buffer): {len(train_df):,}")
    print(f"Dropped (spatial moat/buffer zone): {len(df) - len(test_df) - len(train_df):,}")

    return train_df, test_df, val_df
    
from shapely.geometry import box

def assert_train_test_datapoint(bounds, test_polygon, wanted_type="train", buffer=500):
    """
    Returns True if the datapoint (defined by bounds) matches the wanted_type 
    relative to the test_polygon.
    
    Parameters:
    - bounds: A tuple/list of (min_x, min_y, max_x, max_y)
    - test_polygon: The Shapely polygon defining the test area.
    - wanted_type: "train" or "test"
    - buffer: The safety buffer distance (must match what you used in splitting).
    """
    
    # 1. Convert the bounds tuple into a Shapely geometry object
    #    The * unpacks the tuple (minx, miny, maxx, maxy)
    datapoint_geom = box(*bounds)
    
    # 2. Determine the actual type of this datapoint
    actual_type = None

    # CHECK TEST: Must be strictly inside the test polygon
    if datapoint_geom.within(test_polygon):
        actual_type = "test"
        
    # CHECK TRAIN: Must be strictly outside the (test polygon + buffer)
    # We apply the buffer here to ensure we respect the exclusion zone
    elif datapoint_geom.disjoint(test_polygon.buffer(buffer)):
        actual_type = "train"
        
    # (If it's neither, actual_type remains None, representing the buffer zone)

    # 3. Assert
    return actual_type == wanted_type


def get_dataset_for_gdf(gdf, datasets, link, year=2013, id_var="GEOID"):
    """Get dataset where the census tract is located."""
    
    # 1. Get all matches as a Series (do not squeeze)
    matches = gdf.loc[gdf[id_var] == link, f"dataset_{year}"]

    # 2. Check if we found anything
    if matches.empty:
        return None

    # 3. Take the first match. 
    # Whether there is 1 row or 100 duplicates, this safely gets the first string.
    current_ds_name = matches.iloc[0]

    # 4. Handle NaNs (if the cell was empty)
    if pd.isna(current_ds_name):
        return None

    # 5. Return the dataset
    # using .get() is safer than brackets [], but brackets are fine if you trust your data keys
    return datasets.get(current_ds_name)

def add_buffer(bounds, buffer):
    """Add buffer to bounds.

    Parameters:
    -----------
    bounds: tuple, (minx, miny, maxx, maxy)
    buffer: int, buffer to add to bounds

    Returns:
    --------
    bounds: dict, {'minx': minx-buffer, 'miny': miny-buffer, 'maxx': maxx+buffer, 'maxy': maxy+buffer}
    """
    minx, miny, maxx, maxy = bounds
    return {
        "minx": minx - buffer,
        "miny": miny - buffer,
        "maxx": maxx + buffer,
        "maxy": maxy + buffer,
    }

def crop_dataset_to_link(ds, gdf, link):
    # obtengo el poligono correspondiente al link
    gdf_sub = gdf.loc[gdf["GEOID"] == link].copy() 
    if gdf_sub.empty:
        return None

    # Try to repair invalid geometries (common fix: buffer(0) or shapely.make_valid)
    try:

        # use unary_union (avoids groupby/dissolve topology issues)
        multipolygon = gdf_sub.union_all()

        if multipolygon is None or multipolygon.is_empty:
            return None

        if not multipolygon.is_valid:
            multipolygon = multipolygon.buffer(0)

    except Exception as e:
        # Log and skip problematic geometry (caller handles None)
        print(f"Warning: invalid geometry for link {link}: {e}")
        return None

    # Get bounds of the shapefile's polygon
    bbox_img = add_buffer(multipolygon.bounds, 1000)

    # Filter dataset based on the bounds of the shapefile's polygon
    image_ds = ds.sel(
        x=slice(float(bbox_img["minx"]), float(bbox_img["maxx"])),
        y=slice(float(bbox_img["maxy"]), float(bbox_img["miny"])),
    )
    return image_ds


def get_prediction_images_for_link(
    ds,
    gdf,
    link,
    tiles,
    size,
    resizing_size,
    sample,
    n_bands=4,
    stacked_images=[1],
):
    """
    Itera sobre el bounding box del poligono del radio censal, tomando imagenes de tamño sizexsize
    Si dicha imagen se encuentra dentro del polinogo, se genera el composite con dicha imagen mas otras tiles**2 -1 imagenes
    Devuelve un array con todas las imagenes generadas, un array con los puntos centrales de cada imagen y un array con los bounding boxes de cada imagen.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    gdf: geopandas.GeoDataFrame, shapefile con los radios censales
    link: str, 9 dígitos que identifican el radio censal
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    sample: int, cantidad de imágenes a generar por box (util cuando tiles > 1)
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    images: list, lista con las imágenes generadas
    points: list, lista con los puntos centrales de cada imagen
    bounds: list, lista con los bounding boxes de cada imagen
    """

    images = []
    points = []
    bounds = []
    total_bands = len(stacked_images) * n_bands

    link_dataset = crop_dataset_to_link(ds, gdf, link)
    link_geometries = gdf.loc[gdf["GEOID"] == link, "geometry"].values
    for building_geometry in link_geometries:

        image_point = building_geometry.centroid
        point = image_point.coords[0]
        image, bound = geo_utils.stacked_image_from_census_tract(
            dataset=link_dataset,
            polygon=building_geometry,
            point=point,
            img_size=size,
            n_bands=n_bands,
            stacked_images=stacked_images,
        )

        if image.shape == (total_bands, size, size):
            # TODO: add a check to see if the image is contained in test bounds
            image = geo_utils.process_image(image, resizing_size)

            images += [image]
            bounds += [bound]

        else:
            print("Image failed")

    return images, points, bounds


def get_gridded_images_for_dataset(
    model, ds, gdf, tiles, size, resizing_size, bias, sample, to8bit
):
    """
    Itera sobre el bounding box de un dataset (raster de imagenes), tomando imagenes de tamño sizexsize
    Asigna el valor "real" del radio censal al que pertenece el centroide de la imagen.
    Devuelve un array con todas las imagenes generadas, un array con los puntos centrales de cada imagen,
    un array con los valores "reales" de los radios censales y un array con los bounding boxes de cada imagen.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    gdf: geopandas.GeoDataFrame, shapefile con los radios censales
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    sample: int, cantidad de imágenes a generar por box (util cuando tiles > 1)
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    images: list, lista con las imágenes generadas
    points: list, lista con los puntos centrales de cada imagen
    bounds: list, lista con los bounding boxes de cada imagen
    """
    import main
    from shapely.geometry import Polygon

    # FIXME: algunos radios censales no se generan bien. Ejemplo: 065150101. ¿Que pasa ahi?
    # Inicializo arrays
    batch_images = np.empty((0, resizing_size, resizing_size, 4))
    batch_link_names = np.empty((0))
    batch_predictions = np.empty((0))
    batch_real_values = np.empty((0))
    batch_bounds = np.empty((0))
    all_link_names = np.empty((0))
    all_predictions = np.empty((0))
    all_real_values = np.empty((0))
    all_bounds = np.empty((0))

    tile_size = size // tiles
    tiles_generated = 0

    # Iterate over the center points of each image:
    # - Start point is the center of the image (tile_size / 2, start_index)
    # - End point is the maximum possible center point (link_dataset.y.size)
    # - Step is the size of each image (tile_size)

    # FIXME: para mejorar la eficiencia, convendría hacer un dissolve de gdf y verificar que
    # image_point este en ese polygono y no en todo el df
    start_index = int(tile_size / 2)
    for idy in range(start_index, ds.y.size, tile_size):
        # Iterate over columns
        for idx in range(start_index, ds.x.size, tile_size):
            # Get the center point of the image
            image_point = (float(ds.x[idx]), float(ds.y[idy]))
            point_geom = sg.Point(image_point)

            # Get data for selected point
            radio_censal = gdf.loc[gdf.contains(point_geom)]
            if radio_censal.empty:
                # El radio censal no existe, es el medio del mar...
                continue

            real_value = radio_censal["var"].values[0]
            link_name = radio_censal["GEOID"].values[0]

            # Check if the centroid of the image is within the original polygon:
            #   - if it is, then generate the n images

            image, point, bound, tbound = geo_utils.random_image_from_census_tract(
                ds,
                gdf,
                link_name,
                start_point=image_point,
                tiles=tiles,
                size=size,
                bias=bias,
                to8bit=to8bit,
            )

            if image is not None:
                image = geo_utils.process_image(image, resizing_size)
                geom_bound = Polygon(
                    bound[0]
                )  # Create polygon of the shape of the image

                batch_images = np.concatenate([batch_images, np.array([image])], axis=0)
                batch_link_names = np.concatenate(
                    [batch_link_names, np.array([link_name])], axis=0
                )
                batch_real_values = np.concatenate(
                    [batch_real_values, np.array([real_value])], axis=0
                )
                batch_bounds = np.concatenate(
                    [batch_bounds, np.array([geom_bound])], axis=0
                )

                # predict with the model over the batch
                if batch_images.shape[0] == 128:
                    # predictions
                    batch_predictions = main.get_batch_predictions(
                        model, batch_images
                    )

                    # Store data
                    all_predictions = np.concatenate(
                        [all_predictions, batch_predictions], axis=0
                    )
                    all_link_names = np.concatenate(
                        [all_link_names, batch_link_names], axis=0
                    )
                    all_real_values = np.concatenate(
                        [all_real_values, batch_real_values], axis=0
                    )
                    all_bounds = np.concatenate([all_bounds, batch_bounds], axis=0)

                    # Restore batches to empty
                    batch_images = np.empty((0, resizing_size, resizing_size, 4))
                    batch_predictions = np.empty((0))
                    batch_link_names = np.empty((0))
                    batch_predictions = np.empty((0))
                    batch_real_values = np.empty((0))
                    batch_bounds = np.empty((0))

    # Creo dataframe para exportar:
    d = {
        "GEOID": all_link_names,
        "predictions": all_predictions,
        "real_value": all_real_values,
    }

    df_preds = gpd.GeoDataFrame(d, geometry=all_bounds, crs="epsg:6539")

    return df_preds


def stretch_dataset(ds, pixel_depth=32_767):
    """Stretch band data from satellite images."""
    minimum = ds.band.quantile(0.01).values
    maximum = ds.band.quantile(0.99).values
    ds = (ds - minimum) / (maximum - minimum) * pixel_depth
    ds = ds.where(ds.band > 0, 0)
    ds = ds.where(ds.band < pixel_depth, pixel_depth)
    return ds


def normalize_landsat(ds):
    band = ds.band.to_numpy()
    for band in range(band.shape[0]):
        this_band = band[band]

        vmin = np.percentile(this_band, q=2)
        vmax = np.percentile(this_band, q=98)

        # High values
        mask = this_band > vmax
        this_band[mask] = vmax

        # low values
        mask = this_band < vmin
        this_band[mask] = vmin

        # Normalize
        this_band = (this_band - vmin) / (vmax - vmin)

        band[band] = this_band * 255

    return ds


def remove_overlapping_pixels(main, to_crop):

    main_extent = geo_utils.get_dataset_extent(main)
    will_be_cropeed_extent = geo_utils.get_dataset_extent(to_crop)
    cropped_extent = will_be_cropeed_extent.difference(main_extent)

    # Crop dataset
    min_lon, min_lat, max_lon, max_lat = cropped_extent.bounds
    cropped = to_crop.sel(x=slice(min_lon, max_lon), y=slice(max_lat, min_lat))

    return cropped


def pickle_xr_dataset(ds, filename):
    import pickle

    pkl = pickle.dumps(ds, protocol=-1)
    with open(filename, "wb") as f:
        f.write(pkl)

    print("Pickled data saved to:", filename)
    return


def add_datasets_combinations(datasets):
    from shapely.geometry import box

    extents = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets.items()}
    combinations = {}
    to_remove = []

    for ds_name, ds in datasets.items():
        # Construyo lista de datasets que intersectan con ds_name
        capture_ds_name = ds_name.split("_")[1]
        ds_extent = extents[ds_name]
        buffered_extent = ds_extent.buffer(0.005).envelope
        xmin, ymin, xmax, ymax = buffered_extent.bounds

        intersecting = []
        for name, ds_extent in extents.items():
            capture_name = name.split("_")[1]
            if (
                ds_extent.intersects(buffered_extent)
                & (name != ds_name)
                & (capture_ds_name == capture_name)
            ):
                intersecting += [name]

        # Recorto datasets de intersection (buffer de 1080px):
        cropped_datasets = {}
        for intersection in intersecting:
            intersecting_ds = datasets[intersection]
            cropped_datasets[intersection] = intersecting_ds.sel(
                x=slice(xmin, xmax), y=slice(ymax, ymin)
            )

        # Armo xarray con la intersección de a pares
        for cropped_name, cropped_ds in cropped_datasets.items():

            names = [ds_name, cropped_name]
            names = [name.replace("pansharpened_", "") for name in names]
            names.sort()
            combined_name = "comb_" + "_".join(names)

            if combined_name not in combinations:
                polygon = box(
                    cropped_ds.x.min(),
                    cropped_ds.y.min(),
                    cropped_ds.x.max(),
                    cropped_ds.y.max(),
                )
                buffered_extent = polygon.buffer(0.005).envelope
                xmin, ymin, xmax, ymax = buffered_extent.bounds

                cropped_main_ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
                cropped_ds = remove_overlapping_pixels(cropped_main_ds, cropped_ds)

                # print(ds_name, cropped_name)
                # print(cropped_main_ds)
                # print(cropped_ds)

                try:
                    result_ds = xr.combine_by_coords(
                        [cropped_main_ds, cropped_ds], combine_attrs="override"
                    )

                    # Store xarray and reload to remove cross-references across objects and reduce memory usage
                    filename = rf"{PROCESSED_DATA_DIR}/tempfiles/{combined_name}.pkl"
                    pickle_xr_dataset(ds, filename)
                    with open(filename, "rb") as f:
                        result_ds = pickle.load(f)
                    to_remove += [filename]

                    combinations[combined_name] = result_ds
                except Exception as e:
                    print(e)

    all_datasets = combinations | datasets

    return all_datasets


def filter_black_pixels_over_dim(ds, dim="x"):
    if dim == "x":
        other_dim = "y"
    elif dim == "y":
        other_dim = "x"
    else:
        raise ValueError("dim must be 'x' or 'y'")

    # Selecciono la mitad de la imagen
    center = int(ds[dim].size / 2)
    edge_data = ds.isel({dim: center})

    # Busco los pixeles con al menos 50 pixeles sin datos
    has_black_pixels = (edge_data["band"] == 0).all(dim="band")
    has_black_pixels_in_row = has_black_pixels.rolling({other_dim: 50}).sum()
    valid_data = (has_black_pixels_in_row == 0) | (has_black_pixels_in_row.isnull())

    # Filtro los datos
    first_valid = valid_data.to_numpy().tolist().index(True)
    last_valid = -valid_data.to_numpy().tolist()[::-1].index(True)

    if first_valid == 0:
        first_valid = None
    if last_valid == 0:
        last_valid = None

    return ds.isel({other_dim: slice(first_valid, last_valid)})


def filter_black_pixels(ds):
    # FIXME: This function is not working properly, it requires Python 3.10-... Should add later,
    #   Technically, now (with NYC images) it is not needed because the images are already cropped.
    return ds 
    y_filtered = filter_black_pixels_over_dim(ds, "y")
    filtered = filter_black_pixels_over_dim(y_filtered, "x")
    return filtered


def generate_matrix_of_files(files):
    """Create a matrix of files to be loaded by xr.open_mfdataset.

    Files are ordered as the original tiles, where R1C3 is the first tile of the third column.
    Run xr.open_mfdataset(matrix, combine="nested", concat_dim=["x", "y"], engine="rasterio") after this.

    Parameters:
    files (list): List of files to be loaded

    Returns:
    matrix (list): List of lists of files to be loaded by xr.open_mfdataset
    """
    files.sort()

    matrix = []
    for col in range(1, 5):
        cols_files = [f for f in files if f"C{col}.tif" in f]
        if len(cols_files) > 0:
            matrix += [cols_files]
    return matrix

def generate_matrix_of_datasets(datasets):
    """Create a matrix of datasets to be merged by xr.combine_nested.

    Files are ordered as the original tiles, where R1C3 is the first tile of the third column.
    Run xr.open_mfdataset(matrix, combine="nested", concat_dim=["x", "y"], engine="rasterio") after this.

    Parameters:
    files (list): List of files to be loaded

    Returns:
    matrix (list): List of lists of files to be loaded by xr.open_mfdataset
    """
    datasets = sorted(datasets, key=lambda element: sorted(element.encoding["source"]))
    print([ds.encoding["source"] for ds in datasets])
    matrix = []
    for row in range(1, 10):
        rows_ds = [ds for ds in datasets if f"_R{row}C" in ds.encoding["source"]]
        if len(rows_ds) > 0:
            matrix += [rows_ds]
    return matrix

# Add this function to main.py or geo_utils.py
def get_gpu_augmentation_layer():
    return tf.keras.Sequential([
        # Random Flips (replaces np.flip)
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        
        # Random Contrast (replaces skimage gamma/contrast)
        tf.keras.layers.RandomContrast(0.2),
        
        # Random Brightness (replaces percentile scaling somewhat)
        tf.keras.layers.RandomBrightness(0.2),
    ])