import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

def get_closest_acs_year(year, acs_years=[2009, 2014, 2019, 2024]):
    """
    Given a year and a list of panel years, return the closest panel year.
    This is used to match each building-year pair with the appropriate ACS labels.
    """
    closest_year = min(acs_years, key=lambda y: abs(y - year))
    return closest_year

def process_acs_panel():
    panel_path = Path(
        r"/mnt/c/Working Papers/NY State Aerial Imagery Prototype/"
        r"ny_state_aerial_imagery_prototype/data/processed/"
        r"ny_tracts_panel_2009_2014_2019_2024.feather"
    )
    panel_gdf = gpd.read_feather(panel_path)
    return panel_gdf


def load_building_data():
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


def build_training_datasets(buildings_nyc, panel_gdf, panel_years, tau_meters=50):
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
    METRIC_CRS = "EPSG:3857"

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
    print(f"3. Applying Context Spillover (tau = {tau_meters}m) and Extracting BBoxes...")
    buffered_geoms = buildings_mapped.geometry.buffer(tau_meters)
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
        OUTPUT_DIR / "temporal_data.parquet", index=False
    )
    geometries_df.to_parquet(
        OUTPUT_DIR / "geometries.parquet", index=True   # index = DOITT_ID
    )

    print(
        f"\nDone!\n"
        f"  temporal_data.parquet : {len(temporal_data_flat):,} rows "
        f"({temporal_data_flat['year'].nunique()} years × buildings)\n"
        f"  geometries.parquet    : {len(geometries_df):,} unique buildings\n"
        f"  Score bins computed within each of: {sorted(temporal_df['year'].unique())}"
    )
    return temporal_data_flat, geometries_df


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    PANEL_YEARS = [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    TAU_METERS = 50

    buildings_nyc = load_building_data()
    acs_panel = process_acs_panel()

    temporal_df, geoms_df = build_training_datasets(
        buildings_nyc=buildings_nyc,
        panel_gdf=acs_panel,
        panel_years=PANEL_YEARS,
        tau_meters=TAU_METERS,
    )