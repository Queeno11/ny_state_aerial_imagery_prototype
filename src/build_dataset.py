##############      Configuración      ##############
from tqdm import tqdm
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
from src.data import indicators
from src.data import cbsa_brackets
from src.data.pair_table import LazyPairTable, FLAT_COLUMNS, weighted_qcut
from src.data.process_acs import BASE_YEAR as ACS_BASE_YEAR, PANEL_YEARS as ACS_PANEL_YEARS


def open_datasets(sat_data="aerial", years=[2013, 2018, 2022], tau_meters=100,
                  indicator=None, footprints_source="ms_us", states=None):

    ### Open dataframe with files and labels
    print("Reading dataset...")
    indicator = indicator if indicator is not None else indicators.DEFAULT_INDICATOR
    df = load_income_dataset(years, tau_meters=tau_meters, indicator=indicator,
                             footprints_source=footprints_source, states=states)

    year_cols = []
    if isinstance(df, LazyPairTable) and sat_data != "NAIP":
        raise NotImplementedError(
            f"footprints_source='ms_us' (lazy pair table) only supports sat_data='NAIP', "
            f"got {sat_data!r}. Use footprints_source='doitt_nyc' for the legacy zarr path."
        )
    if sat_data == "aerial":
        datasets_all_years, extents_all_years = load_satellite_datasets(
            years=years
        )
        df = assign_datasets_to_gdf(df, datasets_all_years, extents_all_years, years=years, verbose=True, save_plot=False)
    elif sat_data == "NAIP":
        datasets_all_years = None
        extents_all_years = None
        if not isinstance(df, LazyPairTable):
            df["dataset"] = "NAIP"
            df["row_start"] = 0
            df["row_stop"] = 0
            df["col_start"] = 0
            df["col_stop"] = 0
        # LazyPairTable emits these constants in every materialized slice.
    elif sat_data == "landsat":
        raise NotImplementedError("Landsat support not implemented yet.")
        sat_imgs_datasets, extents = load_landsat_datasets()

    print("Datasets loaded!")

    return datasets_all_years, extents_all_years, df


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

def generate_datasets(savename, sat_data, years, small_sample=False, tau_meters=100,
                      indicator=None, footprints_source="ms_us", states=None,
                      naip_coverage_csv=None):

    indicator = indicator if indicator is not None else indicators.DEFAULT_INDICATOR
    all_years_datasets, all_years_extents, df = open_datasets(
        sat_data=sat_data, years=years, tau_meters=tau_meters,
        indicator=indicator, footprints_source=footprints_source, states=states
    )

    df_train, df_vals_dict, df_test, df_dead_zone = create_train_test_dataframes(
        df, savename, small_sample=small_sample, indicator=indicator,
        naip_coverage_csv=naip_coverage_csv
    )

    return all_years_datasets, all_years_extents, df_train, df_vals_dict, df_test, df_dead_zone
    
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


def get_closest_acs_year(year, acs_years=None):
    """
    Given a year and a list of panel years, return the closest panel year.
    This is used to match each building-year pair with the appropriate ACS labels.
    The US panel is annual (2011-2023), so imagery years inside the range match
    exactly and years outside are clamped to the nearest endpoint.
    """
    acs_years = acs_years if acs_years is not None else ACS_PANEL_YEARS
    closest_year = min(acs_years, key=lambda y: abs(y - year))
    return closest_year

# The tract key column of the US panel (base-year tract vintage), e.g. "geoid_2023".
PANEL_GEOID_COL = f"geoid_{ACS_BASE_YEAR}"

def process_acs_panel():
    print("Loading and processing ACS panel data...")
    panel_path = (
        PROCESSED_DATA_DIR
        / f"us_metros_panel_{ACS_PANEL_YEARS[0]}_{ACS_PANEL_YEARS[-1]}.feather"
    )
    panel_tract_gdf = gpd.read_feather(panel_path).to_crs(geo_utils.METRIC_CRS)
    return panel_tract_gdf


def load_building_data():
    """NYC DoITT footprints (dated construction/demolition years).

    LEGACY / NYC-EVALUATION ONLY: the training path now samples from the
    Microsoft US Building Footprints index (see :func:`load_buildings_index`).
    This loader remains the base for the Callaway-Sant'Anna event study and other
    NYC evaluation code, which need verified construction/demolition dates.
    """
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


def load_buildings_index(states=None, index_dir=None):
    """Load the Microsoft buildings_index (training hot path).

    Built offline by ``src/data/build_buildings_index.py``. Returns a plain
    DataFrame — no polygons — with columns:
      building_id (int64), centroid_x / centroid_y (EPSG:5070 meters),
      GEOID (panel tract), state, CONSTRUCTION_YEAR / DEMOLITION_YEAR sentinels.

    Microsoft footprints are a STATIC universe (no dated construction /
    demolition), so the sentinels (0 / 2999) make the temporal existence filter
    a no-op: every building exists in every panel year.
    """
    index_dir = Path(index_dir) if index_dir is not None else PROCESSED_DATA_DIR / "buildings_index"
    filters = [("state", "in", list(states))] if states else None
    print(f"Loading buildings index from {index_dir}"
          + (f" (states: {list(states)})" if states else " (all states)"))
    df = pd.read_parquet(index_dir, filters=filters)
    df["state"] = df["state"].astype(str)
    df = df.rename(columns={
        "tract_id": "GEOID", "cx": "centroid_x", "cy": "centroid_y"
    })
    # Defensive cross-state dedupe: a border building can appear in two states'
    # partitions with the same (deterministic) building_id and identical centroid.
    # build_buildings_index.dedupe_index_global normally clears these on disk, but
    # guard the training hot path in case of an incremental build that skipped it.
    n_dup = int(df.duplicated(subset="building_id").sum())
    if n_dup:
        print(f"  dropping {n_dup:,} cross-state duplicate building_ids "
              f"({100 * n_dup / len(df):.3f}%) — run `build_buildings_index "
              f"--dedupe-only` to fix the on-disk index")
        df = df.drop_duplicates(subset="building_id", keep="first").reset_index(drop=True)
    df["CONSTRUCTION_YEAR"] = 0
    df["DEMOLITION_YEAR"] = 2999
    print(f"  {len(df):,} buildings in {df['GEOID'].nunique():,} tracts")
    return df


def _cbsa_centers(panel_tract_gdf):
    """Per-CBSA center (mean of tract centroids, METRIC_CRS meters).

    A cheap proxy for each metro's economic center, replacing the NYC-only NYSE
    point; used only for the ``dist_to_center`` covariate.
    """
    cx = panel_tract_gdf.geometry.centroid
    frame = pd.DataFrame({
        "cbsa_code": panel_tract_gdf["cbsa_code"].to_numpy(),
        "x": cx.x.to_numpy(),
        "y": cx.y.to_numpy(),
    })
    return frame.groupby("cbsa_code")[["x", "y"]].mean()


# --------------------------------------------------------------------------- #
# Normalized (lazy) dataset path — ms_us / US scale                            #
#                                                                              #
# The legacy flat table is buildings × years materialized row-by-row; at US    #
# scale that is ~575M rows (150+ GB) and OOM-kills a 32 GB box. The ms_us      #
# universe is static and labels vary only at (tract, year), so we persist two  #
# small artifacts instead and let LazyPairTable synthesize flat slices on      #
# demand (see src/data/pair_table.py).                                         #
# --------------------------------------------------------------------------- #

def _states_tag(states):
    """Short deterministic tag for a states subset (artifact cache key)."""
    if not states:
        return "all"
    import hashlib
    key = "|".join(sorted(str(s) for s in states))
    return f"{len(states)}st_{hashlib.md5(key.encode()).hexdigest()[:8]}"


def _load_ms_buildings_slim(states=None, index_dir=None):
    """Slim-dtype load of the Microsoft buildings index (lazy path).

    Same source and dedupe semantics as :func:`load_buildings_index`, but
    GEOID arrives as a string category and coordinates as float32 — the
    object-string load alone was ~10 GB at 71.8M buildings. The static-universe
    sentinels are omitted entirely (existence is unconditional).
    """
    index_dir = Path(index_dir) if index_dir is not None else PROCESSED_DATA_DIR / "buildings_index"
    filters = [("state", "in", list(states))] if states else None
    print(f"Loading buildings index (slim) from {index_dir}"
          + (f" (states: {list(states)})" if states else " (all states)"))
    # No dtype_backend="pyarrow": we want numpy numerics + object strings so we
    # never hold an Arrow copy AND a str copy of the 71.8M-row columns at once.
    # pop() frees each source column the moment it is converted, keeping the
    # transient build footprint down (the whole point of the normalized path).
    df = pd.read_parquet(
        index_dir, filters=filters,
        columns=["building_id", "cx", "cy", "tract_id"],
    )
    out = pd.DataFrame({
        "building_id": df.pop("building_id").to_numpy(dtype="int64"),
        "centroid_x": df.pop("cx").to_numpy(dtype="float32"),
        "centroid_y": df.pop("cy").to_numpy(dtype="float32"),
    })
    # tract_id is already an object-string column here; astype("category")
    # factorizes it in place without a second full-width copy.
    out["GEOID"] = df.pop("tract_id").astype("category")
    del df
    n_dup = int(out.duplicated(subset="building_id").sum())
    if n_dup:
        print(f"  dropping {n_dup:,} cross-state duplicate building_ids "
              f"({100 * n_dup / len(out):.3f}%) — run `build_buildings_index "
              f"--dedupe-only` to fix the on-disk index")
        out = out.drop_duplicates(subset="building_id", keep="first").reset_index(drop=True)
    print(f"  {len(out):,} buildings in {out['GEOID'].nunique():,} tracts")
    return out


def _build_buildings_frame(panel_tract_gdf, states=None, index_dir=None):
    """Static per-building frame: id, GEOID, cbsa_code, centroid, dist_to_center.

    Category-code ``.map`` is used for every tract-level attach — merging on a
    71.8M-row string key would transiently convert it to objects (~5 GB).
    """
    buildings = _load_ms_buildings_slim(states=states, index_dir=index_dir)

    panel_geoid = panel_tract_gdf[PANEL_GEOID_COL].astype(str)
    cbsa_map = pd.Series(
        panel_tract_gdf["cbsa_code"].astype(str).to_numpy(), index=panel_geoid.to_numpy()
    )
    cbsa_map = cbsa_map[~cbsa_map.index.duplicated()]

    print("2. Computing distance to CBSA centers...")
    buildings["cbsa_code"] = buildings["GEOID"].map(cbsa_map).astype("category")
    in_panel = buildings["cbsa_code"].notna().to_numpy()
    if not in_panel.all():
        print(f"  dropping {(~in_panel).sum():,} buildings in tracts outside the ACS panel")
        buildings = buildings[in_panel].reset_index(drop=True)

    centers = _cbsa_centers(panel_tract_gdf)
    centers.index = centers.index.astype(str)
    dx = buildings["centroid_x"].to_numpy(dtype="float64")
    dx -= buildings["cbsa_code"].map(centers["x"]).to_numpy(dtype="float64")
    dy = buildings["centroid_y"].to_numpy(dtype="float64")
    dy -= buildings["cbsa_code"].map(centers["y"]).to_numpy(dtype="float64")
    meters_per_unit = geo_utils.projected_units_to_meters(1.0, geo_utils.METRIC_EPSG)
    buildings["dist_to_center"] = (np.hypot(dx, dy) * meters_per_unit / 1000.0).astype("float32")
    del dx, dy
    import gc
    gc.collect()
    return buildings


def _build_labels_frame(panel_tract_gdf, panel_years, indicator, building_counts):
    """(tract, year) label frame: Rel_Score, Valid_Structural_Change, score_bin.

    ``score_bin`` uses building-count-weighted within-year quantiles — the
    exact equivalent of the legacy ``pd.qcut`` over building-year rows (all
    rows of a tract share its score, so row quantiles ARE weighted tract
    quantiles). ``building_counts``: Series {GEOID str -> n buildings}.
    """
    vc_col = indicators.valid_change_col(indicator)
    geoids = panel_tract_gdf[PANEL_GEOID_COL].astype(str)

    frames = []
    for year in panel_years:
        acs_year = get_closest_acs_year(year)
        sc_col = indicators.score_col(indicator, acs_year)
        sub = pd.DataFrame({
            "GEOID": geoids.to_numpy(),
            "year": np.int64(year),
            "Rel_Score": panel_tract_gdf[sc_col].to_numpy(dtype="float32"),
            "Valid_Structural_Change": (
                pd.to_numeric(panel_tract_gdf[vc_col], errors="coerce")
                .fillna(0).to_numpy(dtype="int8")
            ),
        })
        sub = sub.drop_duplicates(subset="GEOID")
        counts = building_counts.reindex(sub["GEOID"]).fillna(0).to_numpy(dtype="int64")
        has_bldgs = counts > 0
        sub = sub[has_bldgs].reset_index(drop=True)
        sub["score_bin"] = weighted_qcut(
            sub["Rel_Score"].to_numpy(), counts[has_bldgs], q=5
        )
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


# Regions/tracts with no valid NAIP imagery (coverage gaps / corruption).
# Auto-consumed by the ms_us path when present so those pairs are dropped before
# any fetch. Produced by src/tests/test_naip_coverage.py.
NAIP_UNAVAILABLE_FILENAME = "naip_unavailable.feather"
_NAIP_UNAVAILABLE_COLUMNS = ["level", "key", "year"]


def naip_unavailable_path(path=None):
    """Default path of the NAIP-unavailable artifact (override with ``path``)."""
    return Path(path) if path is not None else PROCESSED_DATA_DIR / NAIP_UNAVAILABLE_FILENAME


def load_naip_unavailable(path=None):
    """Load the NAIP-unavailable table, or None if it doesn't exist.

    Returns a DataFrame with columns ``level`` ("cbsa" | "tract"), ``key``
    (cbsa_code or GEOID, str), ``year`` (int). ``.feather`` or ``.csv``.
    """
    p = naip_unavailable_path(path)
    if not p.exists():
        return None
    df = pd.read_feather(p) if p.suffix == ".feather" else pd.read_csv(p)
    missing = [c for c in _NAIP_UNAVAILABLE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{p.name} missing columns {missing}; expected {_NAIP_UNAVAILABLE_COLUMNS}")
    df = df[_NAIP_UNAVAILABLE_COLUMNS].copy()
    df["level"] = df["level"].astype(str)
    df["key"] = df["key"].astype(str)
    df["year"] = df["year"].astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def write_naip_unavailable(gaps, path=None, merge=True):
    """Persist (and by default union with any existing) NAIP-unavailable rows.

    ``gaps``: DataFrame or iterable of dicts/tuples with (level, key, year).
    Returns the path written. Used by the coverage sweep to emit the artifact
    the ms_us pipeline auto-consumes.
    """
    new = pd.DataFrame(list(gaps), columns=_NAIP_UNAVAILABLE_COLUMNS) \
        if not isinstance(gaps, pd.DataFrame) else gaps[_NAIP_UNAVAILABLE_COLUMNS].copy()
    new["level"] = new["level"].astype(str)
    new["key"] = new["key"].astype(str)
    new["year"] = new["year"].astype(int)
    p = naip_unavailable_path(path)
    if merge:
        existing = load_naip_unavailable(p)
        if existing is not None:
            new = pd.concat([existing, new], ignore_index=True)
    out = new.drop_duplicates().sort_values(_NAIP_UNAVAILABLE_COLUMNS).reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".csv":
        out.to_csv(p, index=False)
    else:
        out.to_feather(p)
    return p


def _load_income_pair_table(panel_years, tau_meters=100,
                            indicator=indicators.DEFAULT_INDICATOR, states=None,
                            unavailable_path=None):
    """LazyPairTable over the full ms_us universe (no split, no holdouts).

    Persists/reuses two parquets (vs the legacy 575M-row flat parquet):
      pair_buildings_*  — static per-building attributes (slim dtypes)
      pair_labels_*     — (tract, year) labels + weighted score_bin
    ``tau_meters`` only affects materialization, so it is not in the cache key.

    If a NAIP-unavailable artifact exists (default location, or
    ``naip_unavailable_path``), its (region, year) gaps are attached so those
    pairs materialize with NaN labels and are skipped before any fetch.
    """
    stag = _states_tag(states)
    yrs = f"years{min(panel_years)}-{max(panel_years)}"
    bpath = PROCESSED_DATA_DIR / f"pair_buildings_ms_us_epsg{geo_utils.METRIC_EPSG}_{stag}.parquet"
    lpath = PROCESSED_DATA_DIR / f"pair_labels_ms_us_{indicator}_{yrs}_{stag}.parquet"

    if bpath.exists() and lpath.exists():
        print(f"Loading normalized pair artifacts:\n  {bpath.name}\n  {lpath.name}")
        buildings = pd.read_parquet(bpath)
        labels = pd.read_parquet(lpath)
    else:
        panel_tract_gdf = process_acs_panel()
        vc_col = indicators.valid_change_col(indicator)
        needed = [PANEL_GEOID_COL, "cbsa_code", vc_col,
                  indicators.score_col(indicator, get_closest_acs_year(min(panel_years)))]
        missing = [c for c in needed if c not in panel_tract_gdf.columns]
        if missing:
            raise KeyError(
                f"Panel is missing columns for indicator '{indicator}': {missing}. "
                f"Regenerate the panel with process_acs.py (wealth flags need the VRE store)."
            )

        print("1. Building universe (ms_us, normalized)...")
        buildings = _build_buildings_frame(panel_tract_gdf, states=states)

        print(f"3. Building (tract, year) labels (indicator = {indicator})...")
        building_counts = buildings["GEOID"].value_counts()
        building_counts.index = building_counts.index.astype(str)
        labels = _build_labels_frame(
            panel_tract_gdf, panel_years, indicator, building_counts
        )

        buildings.to_parquet(bpath, index=False)
        labels.to_parquet(lpath, index=False)
        print(f"Saved normalized pair artifacts:\n  {bpath.name} ({len(buildings):,} buildings)"
              f"\n  {lpath.name} ({len(labels):,} tract-years)")

    # Parquet round-trips can degrade categories to plain strings — re-slim.
    for col in ("GEOID", "cbsa_code"):
        if not isinstance(buildings[col].dtype, pd.CategoricalDtype):
            buildings[col] = buildings[col].astype(str).astype("category")
    labels["GEOID"] = labels["GEOID"].astype(str)

    unavailable = load_naip_unavailable(unavailable_path)
    if unavailable is not None:
        print(f"  NAIP-unavailable artifact: masking {len(unavailable):,} "
              f"(region, year) gaps ({naip_unavailable_path(unavailable_path).name})")

    table = LazyPairTable(
        buildings, labels, panel_years, tau_meters, split_type="all",
        unavailable=unavailable,
    )
    print(f"  {table!r}")
    return table


def load_income_dataset(panel_years, tau_meters=100,
                        indicator=indicators.DEFAULT_INDICATOR,
                        footprints_source="ms_us", states=None):
    """
    Produces the flat table for the Zero-Join DataLoader:

      temporal_data parquet — one row per (building, year) with bbox/centroid
      coordinates (METRIC_CRS = EPSG:5070), the selected indicator's ACS label
      (as ``Rel_Score``), its structural-change flag (as
      ``Valid_Structural_Change``), ``cbsa_code``, and stratification bins.
      No geometry column: training crops are centroid + tau.

    ``indicator`` is a token from :mod:`src.data.indicators` (default W2 at
    rho=5%); the internal column names stay ``Rel_Score`` /
    ``Valid_Structural_Change`` regardless, so the loss/sampler are agnostic.
    ``footprints_source``: ``"ms_us"`` (Microsoft index, national) or
    ``"doitt_nyc"`` (legacy dated NYC footprints; also writes the geometry
    lookup parquet used by NYC evaluation).

    ms_us returns a :class:`LazyPairTable` (the building×year cross product is
    never materialized — 575M rows at US scale); doitt_nyc keeps returning the
    legacy flat DataFrame.
    """
    if footprints_source == "ms_us":
        return _load_income_pair_table(
            panel_years, tau_meters=tau_meters, indicator=indicator, states=states
        )

    OUTPUT_DIR = PROCESSED_DATA_DIR
    # CRS + indicator + source tags keep stale artifacts (old CRS or another
    # label) from being silently reused.
    tag = f"{footprints_source}_{indicator}_epsg{geo_utils.METRIC_EPSG}"
    temporal_data_path = OUTPUT_DIR / (
        f"temporal_data_{tag}_t{tau_meters}_years{min(panel_years)}-{max(panel_years)}.parquet"
    )
    geometries_path = OUTPUT_DIR / (
        f"building_geometries_{tag}_years{min(panel_years)}-{max(panel_years)}.parquet"
    )

    if temporal_data_path.exists():
        print(f"Preprocessed dataset already exists: {temporal_data_path}")
        print("Loading existing temporal dataset...")
        return pd.read_parquet(temporal_data_path)

    panel_tract_gdf = process_acs_panel()
    vc_col = indicators.valid_change_col(indicator)
    needed = [PANEL_GEOID_COL, "cbsa_code", vc_col,
              indicators.score_col(indicator, get_closest_acs_year(min(panel_years)))]
    missing = [c for c in needed if c not in panel_tract_gdf.columns]
    if missing:
        raise KeyError(
            f"Panel is missing columns for indicator '{indicator}': {missing}. "
            f"Regenerate the panel with process_acs.py (wealth flags need the VRE store)."
        )

    # ------------------------------------------------------------------ #
    # 1. Building universe -> centroids in METRIC_CRS + tract GEOID       #
    # ------------------------------------------------------------------ #
    print(f"1. Loading building universe ({footprints_source})...")
    if footprints_source == "ms_us":
        buildings_mapped = load_buildings_index(states=states)
    elif footprints_source == "doitt_nyc":
        buildings_nyc = load_building_data().to_crs(geo_utils.METRIC_CRS)
        tracts = (
            panel_tract_gdf[[PANEL_GEOID_COL, "geometry"]]
            .rename(columns={PANEL_GEOID_COL: "GEOID"})
        )
        buildings_mapped = gpd.sjoin(
            buildings_nyc, tracts, how="inner", predicate="intersects"
        ).drop(columns=["index_right"])
        centroids = buildings_mapped.centroid
        buildings_mapped["centroid_x"] = centroids.x
        buildings_mapped["centroid_y"] = centroids.y
        geometries_df = buildings_mapped[["geometry"]].copy()   # index = DOITT_ID
        geometries_df.to_parquet(geometries_path, index=True)
        buildings_mapped = (
            buildings_mapped.drop(columns=["geometry"])
            .reset_index()
            .rename(columns={"DOITT_ID": "building_id"})
        )
    else:
        raise ValueError(f"Unknown footprints_source: {footprints_source!r}")

    # Attach cbsa_code (tract -> CBSA) once, before the temporal unroll.
    tract_cbsa = (
        panel_tract_gdf[[PANEL_GEOID_COL, "cbsa_code"]]
        .rename(columns={PANEL_GEOID_COL: "GEOID"})
    )
    buildings_mapped = buildings_mapped.merge(tract_cbsa, on="GEOID", how="inner")

    # ------------------------------------------------------------------ #
    # 2. dist_to_center: km to the building's own CBSA center             #
    # ------------------------------------------------------------------ #
    print("2. Computing distance to CBSA centers...")
    centers = _cbsa_centers(panel_tract_gdf)
    ctr = centers.reindex(buildings_mapped["cbsa_code"])
    dx = buildings_mapped["centroid_x"].to_numpy() - ctr["x"].to_numpy()
    dy = buildings_mapped["centroid_y"].to_numpy() - ctr["y"].to_numpy()
    meters_per_unit = geo_utils.projected_units_to_meters(1.0, geo_utils.METRIC_EPSG)
    buildings_mapped["dist_to_center"] = np.hypot(dx, dy) * meters_per_unit / 1000.0

    # ------------------------------------------------------------------ #
    # 3. Tau bbox around the centroid                                     #
    # ------------------------------------------------------------------ #
    print(f"3. Applying Context Spillover (tau = {tau_meters}m) around centroids...")
    tau_units = geo_utils.meters_to_projected_units(tau_meters, geo_utils.METRIC_EPSG)
    buildings_mapped["bbox_minx"] = buildings_mapped["centroid_x"] - tau_units
    buildings_mapped["bbox_miny"] = buildings_mapped["centroid_y"] - tau_units
    buildings_mapped["bbox_maxx"] = buildings_mapped["centroid_x"] + tau_units
    buildings_mapped["bbox_maxy"] = buildings_mapped["centroid_y"] + tau_units

    # ------------------------------------------------------------------ #
    # 4. Unroll to (building, year) pairs with the indicator's labels     #
    # ------------------------------------------------------------------ #
    print(f"4. Unrolling Temporal Building-Year Pairs (indicator = {indicator})...")
    temporal_rows = []
    for year in panel_years:
        existed_mask = (
            (buildings_mapped["CONSTRUCTION_YEAR"] <= year) &
            (buildings_mapped["DEMOLITION_YEAR"] > year)
        )
        bldgs_year = buildings_mapped[existed_mask].copy()
        bldgs_year["year"] = year
        acs_year = get_closest_acs_year(year)
        score_col = indicators.score_col(indicator, acs_year)

        # Merge the selected indicator's labels for this specific year
        tract_labels = (
            panel_tract_gdf[[PANEL_GEOID_COL, vc_col, score_col]]
            .copy()
            .rename(columns={
                PANEL_GEOID_COL: "GEOID",
                score_col: "Rel_Score",
                vc_col: "Valid_Structural_Change",
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
    #    (scores are already z-scored within CBSA, so pooling is valid)  #
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
    relevant_columns = [
        "building_id", "GEOID", "cbsa_code", "year",
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

    print(
        f"\nDone!\n"
        f"  {temporal_data_path.name} : {len(temporal_data_flat):,} rows "
        f"({temporal_data_flat['year'].nunique()} years × buildings, "
        f"{temporal_data_flat['cbsa_code'].nunique()} CBSAs)\n"
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
            crs=geo_utils.METRIC_CRS,
        )
        gdf.plot(markersize=1, figsize=(10, 10), alpha=0.5)
        plt.savefig(rf"{PROCESSED_DATA_DIR}/links_with_images.png")

    warnings.filterwarnings("default")

    return df



def plot_city_splits(tract_panel):
    """Map of tracts colored by their city's split (whole-CBSA assignment, #28)."""
    colors = {"train": "green", "val": "blue", "test": "orange"}
    ax = tract_panel.plot(figsize=(20, 20), color='whitesmoke', edgecolor='lightgray')

    import matplotlib.patches as mpatches
    legend_handles = []
    for split, color in colors.items():
        sub = tract_panel[tract_panel["type"] == split]
        if not sub.empty:
            sub.plot(ax=ax, color=color)
        legend_handles.append(mpatches.Patch(color=color, label=split.capitalize()))
    ax.legend(handles=legend_handles)

    plt.title("Whole-City (CBSA) Train/Val/Test Split")
    plt.savefig(FIGURES_DIR / "city_splits.png")

    print("\n--- Tract Assignments (by city split) ---")
    print(tract_panel['type'].value_counts().to_string())
    print("-" * 25)

def assign_buildings_by_city(df: pd.DataFrame, city_split_df: pd.DataFrame):
    """Whole-city split (#28): every building inherits its CBSA's split.

    Train-city rows in that city's temporal-holdout year go to ``val_temporal``;
    val cities contribute ALL years to ``val_cities``; test cities contribute
    all years to test. No dead zone: train and holdout cities are different
    metros, so there is no spatial adjacency to buffer (the returned dead-zone
    mask is all-False, kept for return-arity compatibility).

    Args:
        df: flat buildings table (needs ``cbsa_code`` and ``year``).
        city_split_df: output of ``cbsa_brackets.build_city_split`` +
            ``attach_holdout_years`` (columns cbsa_code, split, holdout_year).

    Returns:
        (train_mask, test_mask, val_masks_dict, dead_zone_mask) boolean Series
        indexed like ``df``; also sets ``df["type"]``.
    """
    print("\nAssigning buildings to whole-city train/test/val splits...")

    codes = city_split_df["cbsa_code"].astype(str)
    split_map = dict(zip(codes, city_split_df["split"]))
    holdout_map = {
        code: int(year)
        for code, year, split in zip(codes, city_split_df["holdout_year"], city_split_df["split"])
        if split == "train" and pd.notna(year)
    }

    building_cbsa = df["cbsa_code"].astype(str)
    city_split = building_cbsa.map(split_map)
    unassigned = city_split.isna()
    if unassigned.any():
        print(f"⚠️ {unassigned.sum():,} building-rows belong to CBSAs outside the city split "
              f"(below MIN_METRO_POP or missing crosswalk) — excluded from every split.")

    test_mask = city_split == "test"
    val_cities_mask = city_split == "val"
    is_train_city = city_split == "train"
    holdout_year_of = building_cbsa.map(holdout_map)
    val_temporal_mask = is_train_city & (df["year"] == holdout_year_of)
    train_mask = is_train_city & ~val_temporal_mask
    dead_zone_mask = pd.Series(False, index=df.index)

    assert not (test_mask & (val_cities_mask | val_temporal_mask)).any(), "test/val overlap!"
    assert not (train_mask & (test_mask | val_cities_mask | val_temporal_mask)).any(), "train/holdout overlap!"
    assert not (val_cities_mask & val_temporal_mask).any(), "val_cities/val_temporal overlap!"

    df["type"] = "unassigned"
    df.loc[train_mask, "type"] = "train"
    df.loc[test_mask, "type"] = "test"
    df.loc[val_cities_mask, "type"] = "val_cities"
    df.loc[val_temporal_mask, "type"] = "val_temporal"

    val_masks = {"val_cities": val_cities_mask, "val_temporal": val_temporal_mask}

    print("\n--- Final Dataset Assignment (whole-city split) ---")
    print(f"Total building-rows evaluated: {len(df):,}")
    for name, mask in [("Train", train_mask),
                       ("Test (whole cities)", test_mask),
                       ("Val (whole cities)", val_cities_mask),
                       ("Val (temporal holdout year)", val_temporal_mask)]:
        n_tracts = df.loc[mask, "GEOID"].nunique()
        print(f"{name}: {mask.sum():,} rows ({n_tracts:,} tracts)")
    print("-" * 30)

    df[["building_id", "year", "cbsa_code", "type"]].reset_index(drop=True).to_feather(
        PROCESSED_DATA_DIR / "building_splits.feather"
    )
    return train_mask, test_mask, val_masks, dead_zone_mask


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

def _restricted_tract_panel(covered_geoids):
    """ACS tract panel renamed to GEOID/str keys, restricted to covered tracts.

    Keeps the split universe aligned with the buildings actually loaded (e.g. a
    states subset): holding out cities with zero buildings wastes holdout budget.
    """
    tract_panel = process_acs_panel()
    tract_panel = tract_panel.rename(columns={PANEL_GEOID_COL: "GEOID"})
    tract_panel["GEOID"] = tract_panel["GEOID"].astype(str)
    tract_panel["cbsa_code"] = tract_panel["cbsa_code"].astype(str)
    return tract_panel[tract_panel["GEOID"].isin(covered_geoids)].reset_index(drop=True)


def _city_split_and_artifacts(tract_panel, years, naip_coverage_csv, split_seed):
    """Whole-city split frame + persisted artifacts (shared legacy/lazy).

    Builds the stratified CBSA split, attaches the per-train-city temporal
    holdout year, saves ``cbsa_splits.feather`` and the tract-level
    ``tract_splits.feather`` (read by evaluation.py), and plots the split map.
    """
    tract_counts = tract_panel.groupby("cbsa_code")["GEOID"].nunique()
    pop_df = cbsa_brackets.cbsa_populations(panel=tract_panel)
    city_split_df = cbsa_brackets.build_city_split(tract_counts, pop_df, seed=split_seed)

    # Temporal holdout year per TRAIN city: imagery year nearest the middle of
    # the state's NAIP coverage window (interpolation, never extrapolation).
    coverage_df = cbsa_brackets.load_naip_coverage(naip_coverage_csv)
    state_of_cbsa = (
        tract_panel.assign(state_fips=tract_panel["GEOID"].str[:2])
        .groupby("cbsa_code")["state_fips"]
        .agg(lambda s: s.mode().iloc[0])
        .map(cbsa_brackets.FIPS_TO_STATE)
        .to_dict()
    )
    city_split_df = cbsa_brackets.attach_holdout_years(
        city_split_df, state_of_cbsa, years, coverage_df
    )
    cbsa_brackets.save_city_split(city_split_df, path=PROCESSED_DATA_DIR / "cbsa_splits.feather")

    # --- Plot and persist the tract-level split view (evaluation.py reads it) ---
    tract_panel["type"] = tract_panel["cbsa_code"].map(
        dict(zip(city_split_df["cbsa_code"].astype(str), city_split_df["split"]))
    )
    plot_city_splits(tract_panel)
    tract_panel[["GEOID", "geometry", "type"]].to_feather(PROCESSED_DATA_DIR / "tract_splits.feather", index=False)
    print(f"Created file: {PROCESSED_DATA_DIR / 'tract_splits.feather'}")
    return city_split_df


def _create_split_lazy(pair_table, savename, naip_coverage_csv, split_seed):
    """Whole-city split over a LazyPairTable — buildings level, no pair unroll.

    Same split semantics as :func:`assign_buildings_by_city`, expressed on the
    static buildings frame: split membership is per city, so it is a
    building-level property; the train table excludes its city's holdout-year
    pairs at materialization (NaN label), and val_temporal is materialized
    directly at ≤1 building/tract (matching main._subsample_val_buildings;
    labels are tract-level, so tract coverage — not buildings/tract — drives
    val-metric precision).

    Regions with NO valid NAIP in ANY requested year (per the unavailable
    artifact) are dropped from the split universe entirely — they never occupy a
    train/val/test slot — and the ~50/20/30 stratified split is computed over the
    survivors. Partially-flagged regions stay (their bad years stay NaN-masked).
    """
    covered = set(str(g) for g in pair_table.buildings["GEOID"].cat.categories)

    # Drop fully-dead regions from the split universe (issue: zero-coverage
    # metros like Honolulu were still consuming a val slot).
    dead_cbsas, dead_tracts = pair_table.fully_unavailable()
    if dead_tracts:
        covered -= dead_tracts
    tract_panel = _restricted_tract_panel(covered)
    if dead_cbsas:
        n_before = tract_panel["cbsa_code"].nunique()
        tract_panel = tract_panel[
            ~tract_panel["cbsa_code"].isin(dead_cbsas)
        ].reset_index(drop=True)
        print(f"🚫 Excluding {len(dead_cbsas)} zero-coverage CBSA(s) from the split "
              f"({n_before}→{tract_panel['cbsa_code'].nunique()} CBSAs): "
              f"{sorted(dead_cbsas)}")
    if dead_tracts:
        print(f"🚫 Excluding {len(dead_tracts):,} zero-coverage tract(s) from the split.")

    city_split_df = _city_split_and_artifacts(
        tract_panel, pair_table.years, naip_coverage_csv, split_seed
    )
    split_map = dict(zip(city_split_df["cbsa_code"].astype(str), city_split_df["split"]))
    holdout_map = cbsa_brackets.holdout_year_map(city_split_df)

    print("\nAssigning buildings to whole-city train/test/val splits (lazy)...")
    city_split = pair_table.buildings["cbsa_code"].map(split_map)
    unassigned = city_split.isna().to_numpy()
    if unassigned.any():
        print(f"⚠️ {unassigned.sum():,} buildings belong to CBSAs outside the city split "
              f"(below MIN_METRO_POP, missing crosswalk, or zero NAIP coverage) — "
              f"excluded from every split.")

    train_city_mask = (city_split == "train").to_numpy()
    val_city_mask = (city_split == "val").to_numpy()
    test_mask = (city_split == "test").to_numpy()
    assert test_mask.any(), "Empty test dataset!"
    assert train_city_mask.any(), "Empty train dataset!"

    print("\n--- Final Dataset Assignment (whole-city split, lazy) ---")
    print(f"Total buildings evaluated: {pair_table.n_buildings:,}")
    for name, mask in [("Train cities", train_city_mask),
                       ("Test (whole cities)", test_mask),
                       ("Val (whole cities)", val_city_mask)]:
        n_tracts = pair_table.buildings.loc[mask, "GEOID"].nunique()
        print(f"{name}: {mask.sum():,} buildings ({n_tracts:,} tracts)")
    print("-" * 30)

    df_train = pair_table.subset(
        train_city_mask, split_type="train", holdout_map=holdout_map
    ).shuffle_buildings(seed=825)
    df_test = pair_table.subset(test_mask, split_type="test", holdout_map={})

    df_vals_dict = {}
    df_vals_dict["val_cities"] = (
        pair_table.subset(val_city_mask, split_type="val_cities", holdout_map={})
        .sample_buildings_per_tract(1, seed=825)
        if val_city_mask.any() else pd.DataFrame(columns=FLAT_COLUMNS)
    )
    df_vals_dict["val_temporal"] = (
        pair_table.subset(train_city_mask, split_type="val_temporal", holdout_map={})
        .materialize_holdout_years(holdout_map, n_buildings_per_tract=1, seed=825)
    )
    for val_name, df_val in df_vals_dict.items():
        if df_val.shape[0] == 0:
            print(f"⚠️ Empty val dataset for {val_name} — downstream loaders will skip it.")

    val_dataframe_path = PROCESSED_DATA_DIR / "val_datasets"
    val_dataframe_path.mkdir(parents=True, exist_ok=True)
    for val_name, df_val in df_vals_dict.items():
        df_val.reset_index(drop=True).to_feather(
            val_dataframe_path / f"{savename}_{val_name}_val_dataframe.feather"
        )
        print(f"Created val dataset: {val_dataframe_path}")
    print("Lazy path: skipping train/test/dead-zone/building_splits feathers "
          "(575M-row equivalents; the split lives in cbsa_splits/tract_splits).")

    df_dead_zone = pd.DataFrame(columns=FLAT_COLUMNS)
    print(f"Train: {df_train!r}\nTest:  {df_test!r}")
    return df_train, df_vals_dict, df_test, df_dead_zone


def create_train_test_dataframes(buildings_df, savename, small_sample=False,
                                 indicator=None, naip_coverage_csv=None,
                                 split_seed=cbsa_brackets.SPLIT_SEED):
    """Whole-city (CBSA) train/val/test split (#28), Khachiyan et al. (2022)-style.

    Cities (CBSAs) are the split atoms: each is assigned wholesale to
    train/val/test (~50/20/30 in tract count, stratified by population bracket;
    the mega bracket is fixed 2 train / 1 val / 1 test), and every TRAIN city
    additionally holds out ONE year (nearest the middle of its state's NAIP
    coverage) as ``val_temporal``. There is no dead zone and no jitter buffer:
    holdout cities are entire, disjoint metros. Persists ``cbsa_splits.feather``
    (consumed by the per-bracket validation metrics, #31) and a tract-level
    ``tract_splits.feather`` (type in train/val/test) for evaluation.py.
    """
    if isinstance(buildings_df, LazyPairTable):
        if small_sample:
            # Materialize a tiny flat frame and reuse the legacy row-level path.
            buildings_df = buildings_df.sample_pairs(1000, seed=825)
        else:
            return _create_split_lazy(buildings_df, savename, naip_coverage_csv, split_seed)

    if small_sample:
        buildings_df = buildings_df.sample(min(1000, len(buildings_df)), random_state=825).reset_index(drop=True)

    indicator = indicator if indicator is not None else indicators.DEFAULT_INDICATOR

    tract_panel = _restricted_tract_panel(set(buildings_df["GEOID"].astype(str)))

    ###### Split whole cities (CBSAs)
    years = sorted(int(y) for y in buildings_df["year"].unique())
    city_split_df = _city_split_and_artifacts(
        tract_panel, years, naip_coverage_csv, split_seed
    )

    ###### Split Buildings
    train_mask, test_mask, val_masks_dict, dead_zone_mask = assign_buildings_by_city(
        buildings_df, city_split_df
    )

    # Keep only relevant columns for the DataLoader
    relevant_columns = [
        "building_id", "GEOID", "cbsa_code", "year", "type",
        "Rel_Score", "Valid_Structural_Change", "score_bin",
        "dataset", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
        "row_start", "row_stop", "col_start", "col_stop", "dist_to_center",
        "centroid_x", "centroid_y"
    ]
    buildings_df = buildings_df[relevant_columns]

    # Split dataframes and shuffle them
    df_train = buildings_df[train_mask].copy().reset_index(drop=True).sample(frac=1, random_state=825, replace=False)  # Shuffle train set
    df_test = buildings_df[test_mask].copy()
    df_dead_zone = buildings_df[dead_zone_mask].copy()
    df_vals_dict = {}
    for val_name, val_mask in val_masks_dict.items():
        df_vals_dict[val_name] = buildings_df[val_mask].copy()
        if df_vals_dict[val_name].shape[0] == 0:
            # e.g. small_sample subsets may miss every train city's holdout year
            print(f"⚠️ Empty val dataset for {val_name} — downstream loaders will skip it.")

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

    dead_zone_dataframe_path = PROCESSED_DATA_DIR / "train_datasets" / f"{savename}_dead_zone_dataframe.feather"
    df_dead_zone.reset_index(drop=True).to_feather(dead_zone_dataframe_path)

    return df_train, df_vals_dict, df_test, df_dead_zone

