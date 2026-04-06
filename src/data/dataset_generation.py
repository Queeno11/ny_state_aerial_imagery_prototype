import math
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from pyproj import CRS

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
    panel_gdf = gpd.read_feather(panel_path)
    return panel_gdf


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


def build_panel_datasets(buildings_nyc, panel_gdf, panel_years, tau_meters=50):
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

    # ------------------------------------------------------------------ #
    # 1. CRS alignment                                                   #
    # ------------------------------------------------------------------ #
    print("1. Preparing Spatial Data and CRS...")
    if buildings_nyc.crs != METRIC_CRS:
        buildings_nyc = buildings_nyc.to_crs(METRIC_CRS)
    if panel_gdf.crs != METRIC_CRS:
        panel_gdf = panel_gdf.to_crs(METRIC_CRS)

    # ------------------------------------------------------------------ #
    # 2. Assign buildings to 2024 census tracts                          #
    # ------------------------------------------------------------------ #
    print("2. Assigning Buildings to 2024 Census Tracts...")
    tracts_2024 = (
        panel_gdf[["geoid_2024", "geometry"]]
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
            panel_gdf[["geoid_2024", "Valid_Structural_Change", f"Rel_Score_{acs_year}"]]
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
    return temporal_data_flat, geometries_df

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

# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    PANEL_YEARS = [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    TAU_METERS = 100  # 100m spillover buffer around each building's bounding box

    buildings_nyc = load_building_data()
    acs_panel = process_acs_panel()

    temporal_df, geoms_df = build_panel_datasets(
        buildings_nyc=buildings_nyc,
        panel_gdf=acs_panel,
        panel_years=PANEL_YEARS,
        tau_meters=TAU_METERS,
    )