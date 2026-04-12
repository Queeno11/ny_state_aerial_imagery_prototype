##############      Configuración      ##############
from ast import Return
import os
import math
import pickle
from tokenize import String
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

import shapely
from src.utils.paths import FIGURES_DIR, PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR, IMAGERY_ROOT
from pathlib import Path
from shapely.geometry import box

pd.set_option("display.max_columns", None)

# path_programas  = globales[7]

import geopandas as gpd
import xarray as xr
import shapely.geometry as sg
import pandas as pd
import src.geo_utils as geo_utils


def load_satellite_datasets(years,stretch=False, engine="zarr"):
    """Load satellite datasets and get their extents"""
    datasets = {}
    extents = {}
    for year in years:
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

        if not os.path.exists(dataset_path):
            raise ValueError(f"Year {year} images not found: {dataset_path} does exist! Check they are stored in WSL!")
        datasets_year = {
            f: (filter_black_pixels(xr.open_dataset(dataset_path / f, engine=engine,  mask_and_scale=False)))
            for f in files
        }

        if stretch:
            datasets_year = {(name if year in name else f"{name}_{year}"): stretch_dataset(ds) for name, ds in datasets_year.items()}

        extents_year = {name: geo_utils.get_dataset_extent(ds) for name, ds in datasets_year.items()}

        datasets.update(datasets_year)
        extents.update(extents_year)

    print(f"Loaded datasets for years {years}: {list(datasets.keys())}")

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
    panel_tract_gdf = gpd.read_feather(panel_path).to_crs(epsg=6539)
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

    temporal_data_path = OUTPUT_DIR / f"temporal_data_t{tau_meters}_years{min(panel_years)}-{max(panel_years)}.parquet"
    geometries_path = OUTPUT_DIR / f"building_geometries_years{min(panel_years)}-{max(panel_years)}.parquet"

    # Check if output files already exist to avoid redundant processing
    if temporal_data_path.exists() and geometries_path.exists():
        print(f"Preprocessed datasets already exist.\n  Temporal data: {temporal_data_path}\n  Geometries: {geometries_path}")
        print("Loading existing temporal dataset...")
        temporal_data_flat = pd.read_parquet(temporal_data_path)
        return temporal_data_flat

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

    ## Add distance to NYC economic center in kilometers as variable (used for training)
    #   NYSE is used as the economic center (40.70687862946312, -74.01126682079922)
    nyc_economic_center = gpd.GeoSeries.from_wkt(["POINT (-74.011267 40.706879)"], crs="EPSG:4326").to_crs(buildings_mapped.crs).iloc[0]
    buildings_mapped["dist_to_center"] = buildings_mapped.distance(nyc_economic_center).apply(lambda x: geo_utils.projected_units_to_meters(x, epsg_code=6539)) / 1000

    # ------------------------------------------------------------------ #
    # 3. Apply tau buffer and extract bounding boxes                      #
    # ------------------------------------------------------------------ #
    print(f"3. Applying Context Spillover (tau = {tau_meters}m) and Extracting BBoxes (assuming zarr has 0.5 EPSG:6539 units per pixel)...")
    
    meters_per_crs_unit = geo_utils.projected_units_to_meters(1.0, 6539)    
    tau_crs_units = tau_meters / meters_per_crs_unit
    buffered_geoms = buildings_mapped.centroid.buffer(tau_crs_units)

    bounds = buffered_geoms.bounds
    buildings_mapped["bbox_minx"] = bounds["minx"]
    buildings_mapped["bbox_miny"] = bounds["miny"]
    buildings_mapped["bbox_maxx"] = bounds["maxx"]
    buildings_mapped["bbox_maxy"] = bounds["maxy"]
    buildings_mapped["centroid_x"] = buildings_mapped.centroid.x
    buildings_mapped["centroid_y"] = buildings_mapped.centroid.y

    geometries_df = buildings_mapped[["geometry"]].copy()  # index = DOITT_ID + year

    # Now it's safe to drop geometry AND reset index so DOITT_ID becomes a regular column.
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
    
    # 🔍 DIAGNOSTIC: Check for NaN labels BEFORE dropping
    initial_count = len(temporal_df)
    nan_count_before = temporal_df["Rel_Score"].isna().sum()
    if nan_count_before > 0:
        print(f"⚠️  WARNING: Found {nan_count_before:,} NaN values in Rel_Score ({100*nan_count_before/initial_count:.1f}%)")
    
    temporal_df = temporal_df.dropna(subset=["Rel_Score"])
    
    # 📊 Report removal statistics
    dropped_count = initial_count - len(temporal_df)
    if dropped_count > 0:
        print(f"   → Dropped {dropped_count:,} rows with missing Rel_Score")
        print(f"   → Remaining: {len(temporal_df):,} valid rows ({100*len(temporal_df)/initial_count:.1f}%)")
    
    # ------------------------------------------------------------------ #
    # 5. Stratified score bins — computed WITHIN each year               #
    # ------------------------------------------------------------------ #
    print("5. Calculating Year-Stratified Score Bins...")
    temporal_df["score_bin"] = (
        temporal_df
        .groupby("year")["Rel_Score"]
        .transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop")
        )
    )

    # ------------------------------------------------------------------ #
    # 6. Build the flat temporal table (DataLoader hot path)             #
    # ------------------------------------------------------------------ #
    print("6. Building Flat Temporal Table...")
    # Because we called .reset_index() earlier, DOITT_ID is now safely a column.
    relevant_columns = [
        "DOITT_ID", "GEOID", "year",
        "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
        "centroid_x", "centroid_y",
        "Rel_Score", "Valid_Structural_Change", "score_bin", "dist_to_center"
    ]
    missing = [c for c in relevant_columns if c not in temporal_df.columns]
    if missing:
        raise KeyError(
            f"Expected columns missing from temporal_df: {missing}\n"
            f"Available columns: {list(temporal_df.columns)}"
        )
    temporal_data_flat = temporal_df[relevant_columns].copy()
    
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


def assign_datasets_to_gdf(
    df,
    datasets,
    extents,
    years,
    verbose=True,
    save_plot=True,
):
    """Assign each geometry a dataset if the census tract falls within the extent of the dataset (images)

    Parameters:
    -----------
    df: pandas.DataFrame, must have columns "centroid_x" and "centroid_y" with the coordinates of the centroid of the census tract
    extents: dict, dictionary with the extents of the satellite datasets
    years: list, years of the satellite images
    centroid: bool, if True, the centroid of the census tract is used to assign the dataset
    select: str, method to select the dataset. Options are "first_match" or "all_matches"
    """
    import warnings
    warnings.filterwarnings("ignore")

    if "centroid_x" not in df.columns or "centroid_y" not in df.columns:
        raise ValueError("DataFrame must have 'centroid_x' and 'centroid_y' columns with the coordinates of the centroid of the census tract")  

    colname = "dataset"
    for year in years:
        inside_year = df["year"] == year
        for name, bbox in extents.items():
            if str(year) not in name:   # ← skip datasets that don't belong to this year
                continue

            xmin, ymin, xmax, ymax = bbox.bounds
            inside_bbox = (
                (df["centroid_x"] >= xmin) &
                (df["centroid_x"] <= xmax) &
                (df["centroid_y"] >= ymin) &
                (df["centroid_y"] <= ymax)
            )
            inside_dataset = inside_bbox & inside_year

            if not inside_dataset.any():
                continue

            df.loc[inside_dataset, colname] = name

            x_values = datasets[name].x.values
            y_values = datasets[name].y.values
            boxes = df.loc[inside_dataset, ["bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]].values
            all_indices = geo_utils.precompute_all_indices(x_values, y_values, boxes)
            df.loc[inside_dataset, ["row_start", "row_stop", "col_start", "col_stop"]] = all_indices

    nan_links = df[colname].isna().sum()
    df = df[df[colname].notna()]

    if verbose:
        print(f"Buildings without images: {nan_links} out of {len(df) + nan_links}")
        print(f"Buildings for datasets (train/test/val): {len(df)}")
    if save_plot:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["centroid_x"], df["centroid_y"]),
            crs="EPSG:6539",
        )
        gdf.plot(markersize=1, figsize=(10, 10), alpha=0.5)
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



def create_stratified_tract_holdout(gdf, cluster_radius, stratify_cols, eval_fraction=0.05, exclude_mask=None):
    """
    Creates a holdout set by growing contiguous clusters of tracts.
    Uses a spatial exclude_mask to ensure clusters do not cross into restricted territories.
    """
    captured_geoids = set()
    holdout_indices = []

    # Drop any tracts that fall into the restricted exclusion zone
    if exclude_mask is not None:
        available_gdf = gdf[~exclude_mask].copy()
    else:
        available_gdf = gdf.copy()
        
    groups = available_gdf.groupby(stratify_cols, dropna=False)
    
    print(f"Stratifying across {len(groups)} unique groups...")

    for name, group in groups:
        unique_group_geoids = set(group['GEOID'].unique())
        target_count = math.ceil(len(unique_group_geoids) * eval_fraction)
        
        if target_count == 0: continue
            
        group_captured = len(unique_group_geoids.intersection(captured_geoids))
        
        while group_captured < target_count:
            # 1. Sample a random seed tract from this group
            areas = group.geometry.area
            if areas.sum() == 0: break
            
            # np.random.choice uses the global numpy random seed
            seed_idx = np.random.choice(group.index, p=areas / areas.sum())
            seed_geom = group.loc[seed_idx].geometry
            
            # 2. Capture all tracts within the radius to form a contiguous cluster
            cluster_mask = available_gdf.geometry.intersects(seed_geom.buffer(cluster_radius))
            cluster_tracts = available_gdf[cluster_mask]
            
            if cluster_tracts.empty: continue
            
            # 3. Update tracking variables
            current_geoids = set(cluster_tracts['GEOID'].unique())
            captured_geoids.update(current_geoids)
            holdout_indices.extend(cluster_tracts.index.tolist())
            
            group_captured = len(unique_group_geoids.intersection(captured_geoids))
            
    print(f"Success! Captured {len(captured_geoids)} GEOIDs for this split.")
    
    # SORT the list so that Python's hash randomization doesn't alter the output row order
    deterministic_indices = sorted(list(set(holdout_indices)))
    return gdf.loc[deterministic_indices].copy()

def assign_tracts_train_val_test(gdf, test_tracts, val_tracts, dead_zone_buffer):
    """
    Assigns the final splits and calculates the exact dead zone needed to prevent spatial leakage.
    """
    gdf['type'] = 'train' # Default to train
    
    # 1. Combine all holdout tracts to calculate a unified dead zone
    holdouts = pd.concat([test_tracts, val_tracts])
    
    # 2. Buffer the exact, irregular boundaries of the holdout tracts by tau
    dead_zone_geom = holdouts.geometry.union_all().buffer(dead_zone_buffer)
    
    # 3. Find any tract that touches this buffer
    in_dead_zone = gdf.geometry.intersects(dead_zone_geom)
    
    # 4. Apply assignment hierarchy (Holdouts override Dead Zone override Train)
    gdf.loc[in_dead_zone, 'type'] = 'dead_zone'
    
    if not val_tracts.empty:
        gdf.loc[gdf.index.isin(val_tracts.index), 'type'] = 'val'
    if not test_tracts.empty:
        gdf.loc[gdf.index.isin(test_tracts.index), 'type'] = 'test'
        
    return gdf, gpd.GeoDataFrame(geometry=[dead_zone_geom], crs=gdf.crs)

def plot_tracts_splits(gdf, dead_zone_gdf):
    ax = gdf.plot(figsize=(20, 20), color='whitesmoke', edgecolor='lightgray')
    
    if not dead_zone_gdf.empty:
        dead_zone_gdf.plot(ax=ax, color="gray", alpha=0.5)
    
    train_gdf = gdf[gdf["type"] == "train"]
    val_gdf = gdf[gdf["type"] == "val"]
    test_gdf = gdf[gdf["type"] == "test"]
    dead_gdf = gdf[gdf["type"] == "dead_zone"]

    if not train_gdf.empty: train_gdf.plot(ax=ax, color="green")
    if not val_gdf.empty: val_gdf.plot(ax=ax, color="blue")
    if not test_gdf.empty: test_gdf.plot(ax=ax, color="orange")
    if not dead_gdf.empty: dead_gdf.plot(ax=ax, color="red")
    
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='gray', alpha=0.5, label='Dead Zone (Buffer)'),
        mpatches.Patch(color='green', label='Train'),
        mpatches.Patch(color='blue', label='Validation'),
        mpatches.Patch(color='orange', label='Test'),
        mpatches.Patch(color='red', label='Dead Zone (Discarded Tracts)')
    ]
    ax.legend(handles=legend_handles)
    
    plt.title("Tract-Centric Train/Val/Test Split with Dead Zones")
    plt.savefig(FIGURES_DIR / "tract_splits_with_dead_zone.png")

    print("\n--- Tract Assignments ---")
    print(gdf['type'].value_counts().to_string())
    print("-" * 25)

def assign_buildings_train_test_val(
    df: pd.DataFrame, 
    val_polygon: shapely.geometry.Polygon = None,
    test_polygon: shapely.geometry.Polygon = None,
    test_years: List[int] = None,
    test_column: str = "None",
    jitter_buffer: float = 0.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Splits a flat dataset into train and test sets using pure numerical bounding box comparisons
    against multiple test patches.
    
    Args:
        df: Pandas DataFrame containing the building bbox coordinates.
        test_polygon: Optional shapely (Multi)Polygon representing the test area.
        val_polygon: Optional shapely (Multi)Polygon representing the validation area.
        test_years: Optional list of years to assign to the test set (if doing a temporal split in addition to spatial).
        test_column: Optional boolean column name in df to use for another splitting (e.g., df["Boston"]==True). If provided, 
            this column will be used to assign rows to the test set regardless of spatial location.
        jitter_buffer: Distance to expand the train building bboxes to avoid issues with the jitter in the imagery query augmentation. 
            This is the max jitter distance the DataLoader will use when expanding building bboxes during training. To be safe, we 
            exclude from the train set any building whose original bbox is within this distance of a test patch, to prevent any chance
            of jitter overlap between train and test sets. 
                          
    Returns:
        A tuple of three boolean Series: (train_mask, test_mask, val_mask), each indexed the same as df.
         - train_mask: True for rows assigned to the training set
         - test_mask: True for rows assigned to the test set
         - val_mask: True for rows assigned to the validation set (if val_bounds_df provided; otherwise all False)
    """

    print("\nAssigning buildings to train/test/val splits...")

    if test_years is None:
        test_years = []

    ###############################
    ##### SPATIAL SPLIT LOGIC #####
    ###############################

    # Pre-extract building arrays with Jitter Buffer applied to BBOX
    jitter_buffer_crs = geo_utils.meters_to_projected_units(jitter_buffer, epsg_code=6539)
    b_minx = df['bbox_minx'].values - jitter_buffer_crs
    b_miny = df['bbox_miny'].values - jitter_buffer_crs
    b_maxx = df['bbox_maxx'].values + jitter_buffer_crs
    b_maxy = df['bbox_maxy'].values + jitter_buffer_crs
    bboxes = gpd.GeoSeries(
        shapely.box(b_minx, b_miny, b_maxx, b_maxy), 
        crs="EPSG:6539",
        index=df.index
    )
    
    # Generate centroid arrays for point-in-polygon checks
    centroids = gpd.GeoSeries(
        gpd.points_from_xy(df['centroid_x'], df['centroid_y']), 
        crs="EPSG:6539", 
        index=df.index
    )

    # Initialize boolean masks as NumPy arrays (default False)
    spatial_test_mask = np.zeros(len(df), dtype=bool)
    spatial_val_mask = np.zeros(len(df), dtype=bool)
    train_drop_mask = np.zeros(len(df), dtype=bool) 
    

    ## Assigment logic:
    # - If a building falls entirely inside ANY test/val patch, assign it to that set
    # - If a building's BBOX expanded by jitter_buffer intersects ANY test/val patch, drop it from train
    
    # Assign to test/val based on centroids first (strict point-in-polygon)
    if test_polygon is not None:
        spatial_test_mask = centroids.within(test_polygon)
    if val_polygon is not None:
        spatial_val_mask = centroids.within(val_polygon)
    
    # Remove any building that whose image intersects with test/val from the train drop mask consideration
    if test_polygon is not None:
        train_drop_mask |= bboxes.intersects(test_polygon)
    if val_polygon is not None:
        train_drop_mask |= bboxes.intersects(val_polygon)

    ###############################    
    #####   TIME SPLIT LOGIC  #####
    ###############################

    time_mask_np = np.zeros(len(df), dtype=bool)
    time_mask_val_np = np.zeros(len(df), dtype=bool)
    for test_year in test_years:
        time_mask_np |= (df["year"].values == test_year)
        time_mask_val_np |= (df["year"].values == test_year)  # If you want val to also include these years, otherwise keep as False 

    ###############################
    #####  OTHER SPLIT LOGIC  #####
    ###############################

    other_mask_np = np.zeros(len(df), dtype=bool)
    if test_column in df.columns:
        other_mask_np = (df[test_column].values == True)

    ###################################
    ##### FINAL COMBINATION LOGIC #####
    ###################################
    # 1. val_time captures the val tracts at the year of test
    final_val_time_mask_np = time_mask_val_np & spatial_val_mask

    # 2. val_spatial captures all other val tracts NOT in the year of test
    final_val_spatial_mask_np = spatial_val_mask & ~time_mask_val_np
    
    # 3. test captures the whole test year (and other test patches), but removes those val tracts
    final_test_mask_np = (spatial_test_mask | time_mask_np | other_mask_np) & ~spatial_val_mask

    # Train set is everything NOT inside test, val, or the dropped expansion zones
    final_train_mask_np = (~train_drop_mask) & (~final_test_mask_np) & (~final_val_spatial_mask_np) & (~final_val_time_mask_np)


    # Convert back to Pandas Series with original index
    final_train_mask = pd.Series(final_train_mask_np, index=df.index)
    final_test_mask = pd.Series(final_test_mask_np, index=df.index)
    final_val_spatial_mask = pd.Series(final_val_spatial_mask_np, index=df.index)
    final_val_time_mask = pd.Series(final_val_time_mask_np, index=df.index)
    total_val_mask = final_val_spatial_mask | final_val_time_mask

    # Create dict of masks
    final_val_masks = {
        "val_spatial": final_val_spatial_mask,
    }
    if test_years:  # Only add temporal val key if test_years were provided; avoids IndexError
        final_val_masks[f"val_{test_years[0]}"] = final_val_time_mask
    
    # Compute logs
    overlaps = (spatial_test_mask & time_mask_np).sum() + (spatial_test_mask & other_mask_np).sum() + (time_mask_np & other_mask_np).sum()
    dropped = len(df) - final_test_mask.sum() - final_val_spatial_mask.sum() - final_val_time_mask.sum() - final_train_mask.sum()

    assert not (final_test_mask & total_val_mask).any(), "Error: Some buildings are assigned to both test and val sets!"
    assert not (final_test_mask & final_train_mask).any(), "Error: Some buildings are assigned to both test and train sets!"
    assert not (total_val_mask & final_train_mask).any(), "Error: Some buildings are assigned to both val and train sets!"
    assert not (final_val_time_mask & final_val_spatial_mask).any(), "Error: Some buildings are assigned to both val by time criteria and val by spatial criteria!"
    
    df.loc[final_train_mask, "type"] = "train"
    df.loc[final_test_mask, "type"] = "test"
    df.loc[final_val_spatial_mask, "type"] = "val_spatial"
    df.loc[final_val_time_mask, "type"] = "val_time"

    train_tracts = df[final_train_mask].drop_duplicates("GEOID").shape[0]
    test_tracts = df[final_test_mask].drop_duplicates("GEOID").shape[0]
    val_tracts = df[total_val_mask].drop_duplicates("GEOID").shape[0]

    # Logging
    print("\n--- Final Dataset Assignment ---")
    print(f"Total buildings evaluated: {len(df):,}")
    print(f"Assigned to Test Set (strictly inside patches/criteria): {final_test_mask.sum():,} ({test_tracts} tracts)")
    print(f"     - of which assigned by spatial split: {spatial_test_mask.sum():,}")
    print(f"     - of which assigned by temporal split: {time_mask_np.sum():,}")
    print(f"     - of which assigned by other split ({test_column}): {other_mask_np.sum():,}")
    print(f"     - Overlaps between criteria: {overlaps:,}")
    print(f"Assigned to Validation Set: {total_val_mask.sum():,} ({val_tracts} tracts)")
    print(f"     - of which assigned by spatial split: {final_val_spatial_mask.sum():,}")
    print(f"     - of which assigned by temporal split: {final_val_time_mask.sum():,}")
    print(f"Assigned to Train Set: {final_train_mask.sum():,} ({train_tracts} tracts)")
    print(f"Dropped (spatial moat/buffer zone): {dropped:,}")
    print("-" * 30)
    # Export gdf with bboxes and assigned datasets for visualization and debugging
    gdf = gpd.GeoDataFrame(
        df,
        geometry=bboxes, # Use the jitter-buffered bboxes for visualization to see the actual exclusion zones
        crs="EPSG:6539",
    )
    gdf[["DOITT_ID", "year", "type", "geometry"]].to_feather(PROCESSED_DATA_DIR / "building_splits.feather")
    return final_train_mask, final_test_mask, final_val_masks    

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

def create_train_test_dataframes(buildings_df, savename, test_years=[], test_column=None, small_sample=False, max_jitter=10):
    """Create train and test dataframes with the IDs and xr.datasets names to use for training and testing

    Split the census tracts into train and test. The train and test dataframes contain the links and xr.datasets to use for training and
    testing.
    """
    if small_sample:
        buildings_df = buildings_df.sample(1000, random_state=825).reset_index(drop=True)

    tract_panel = process_acs_panel()
    tract_panel["income_quintile"] = pd.qcut(tract_panel["Rel_Score_2024"], q=5, labels=False)
    tract_panel = tract_panel.rename(columns={"geoid_2024": "GEOID"})
    tract_panel = tract_panel.dropna(subset=["income_quintile"]).reset_index(drop=True)

    ###### Split census tracts 
    cluster_radius = geo_utils.meters_to_projected_units(300, epsg_code=6539) 
    dead_zone_buffer = geo_utils.meters_to_projected_units(100, epsg_code=6539) # Your tau parameter

    # 1. Generate TEST Holdout (5%)

    test_tracts = create_stratified_tract_holdout(
        tract_panel, 
        cluster_radius=cluster_radius, 
        stratify_cols=["income_quintile"], 
        eval_fraction=0.06
    )

    # 2. CREATE A STRICT QUARANTINE ZONE AROUND THE TEST SET
    test_restricted_geom = test_tracts.geometry.union_all().buffer(dead_zone_buffer)
    invalid_val_candidates_mask = tract_panel.geometry.intersects(test_restricted_geom)

    # 3. Generate VALIDATION Holdout (5%) 
    val_tracts = create_stratified_tract_holdout(
        tract_panel, 
        cluster_radius=cluster_radius, 
        stratify_cols=["income_quintile"], 
        eval_fraction=0.06, 
        exclude_mask=invalid_val_candidates_mask
    )

    # 4. Assign labels and compute the final combined dead zones for the Train set
    assigned_tracts, dead_zone_geom_gdf = assign_tracts_train_val_test(
        tract_panel, 
        test_tracts, 
        val_tracts, 
        dead_zone_buffer
    )

    # --- Plot and Verify ---
    plot_tracts_splits(tract_panel, dead_zone_geom_gdf)
    assigned_tracts[["GEOID", "geometry", "type"]].to_feather(PROCESSED_DATA_DIR / "tract_splits.feather", index=False)
    print(f"Created file: {PROCESSED_DATA_DIR / 'tract_splits.feather'}")

    ###### Split Buildings
    val_area = assigned_tracts[assigned_tracts["type"] == "val"].union_all()
    test_area = assigned_tracts[assigned_tracts["type"] == "test"].union_all()
    # val_bounds = get_test_area_from_file(filename="Test_NYC_Area.parquet")

    train_mask, test_mask, val_masks_dict = assign_buildings_train_test_val(buildings_df, val_area, test_area, test_years=test_years, test_column=test_column, jitter_buffer=max_jitter)

    # Keep only relevant columns for the DataLoader
    relevant_columns = [
        "DOITT_ID", "GEOID", "year",
        "Rel_Score", "Valid_Structural_Change", "score_bin",
        "dataset", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
        "row_start", "row_stop", "col_start", "col_stop", "dist_to_center"
    ]
    buildings_df = buildings_df[relevant_columns]

    # Split dataframes and shuffle them
    df_train = buildings_df[train_mask].copy().reset_index(drop=True).sample(frac=1, random_state=825, replace=False)  # Shuffle train set
    df_test = buildings_df[test_mask].copy()
    df_vals_dict = {}
    for val_name, val_mask in val_masks_dict.items():
        df_vals_dict[val_name] = buildings_df[val_mask].copy()
        assert df_vals_dict[val_name].shape[0] > 0, f"Empty val dataset for {val_name}!"
    
    assert df_test.shape[0] > 0, f"Empty test dataset!"
    assert df_train.shape[0] > 0, f"Empty train dataset!"

    ### Train/Test

    test_dataframe_path = PROCESSED_DATA_DIR / "test_datasets" / f"{savename}_test_dataframe.feather"
    df_test.to_feather(test_dataframe_path)
    print(f"Created test dataset: {test_dataframe_path}")

    train_dataframe_path = PROCESSED_DATA_DIR / "train_datasets" / f"{savename}_train_dataframe.feather"
    df_train.to_feather(train_dataframe_path)
    print(f"Created train dataset: {train_dataframe_path}")

    val_dataframe_path = PROCESSED_DATA_DIR / "val_datasets"
    for val_name, df_val_year in df_vals_dict.items():
        df_val_year.to_feather(val_dataframe_path / f"{savename}_{val_name}_val_dataframe.feather")
        print(f"Created val dataset: {val_dataframe_path}")

    return df_train, df_vals_dict, df_test

