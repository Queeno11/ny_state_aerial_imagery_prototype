"""NYC within-city train/val/test split — the validated notebook holdout, as code.

This is the version-controlled port of ``notebooks/split_train_test_val.ipynb``
(cell 1), so the NYC holdout that ``src.data.us_split`` consumes can be
regenerated reproducibly and never lives only in a notebook. It writes
``nyc_tract_splits.feather`` (GEOID, geometry, type in
{train, val, test, dead_zone}) — the artifact
:func:`us_split.nyc_tract_type_map` reads.

Why NYC gets its own within-city split (it is the *only* whole-city exception):
NYC is the anchor CSA city — the one metro with annual/odd-year cadence, a second
sensor, and footprints at this panel depth — so instead of assigning the whole
CBSA to one split we hold out **spatially-clustered, income-stratified** blocks of
NYC-proper tracts for validation and test, quarantined by a τ=100 m dead-zone
buffer, and keep the rest (plus the greater-NYC metro, handled in
:func:`us_split.nyc_greater_metro_overrides`) for training. Whole-city leakage
prevention is preserved because the holdout blocks are contiguous and buffered,
not random tracts interleaved with train.

**Reproducibility:** ``build_nyc_split`` sets ``np.random.seed(SEED)`` and grows
clusters with the global RNG in the exact order of the validated notebook, so it
reproduces the seed-25 split the NYC-only model was validated on. The
implementation here is intentionally verbatim (not the refactored
``us_split.grow_stratified_tract_clusters``, which is a differently-structured
generic carve for other cities and would not reproduce this bit-for-bit).

Run standalone to (re)generate the artifact::

    python -m src.data.nyc_split
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import geo_utils
from src.utils import paths
from src.utils.paths import PROCESSED_DATA_DIR

# --- Validated parameters (do not change without re-validating the NYC model) ---
SEED = 25
NYC_EVAL_EPSG = 6539                 # legacy NYC state-plane (US survey feet)
CLUSTER_RADIUS_M = 300
DEAD_ZONE_M = 100                    # τ
TEST_FRACTION = 0.06
VAL_FRACTION = 0.06
STRATIFY_COLS = ["income_quintile"]

PANEL_FILENAME = "ny_tracts_panel_2009_2014_2019_2024.feather"
BOROS_FILENAME = "Borough_Boundaries_20260131.geojson"
OUT_PATH = PROCESSED_DATA_DIR / "nyc_tract_splits.feather"


def create_stratified_tract_holdout(gdf, cluster_radius, stratify_cols,
                                    eval_fraction=0.05, exclude_mask=None):
    """Grow contiguous clusters of tracts into a holdout, stratified by group.

    Verbatim from the notebook: within each stratum, seed tracts are drawn
    area-weighted from the global ``np.random`` state and every tract within
    ``cluster_radius`` of a seed is captured, until ``eval_fraction`` of the
    stratum is held out. ``exclude_mask`` (True = restricted) keeps clusters out
    of an already-claimed quarantine zone. Returns the captured tracts (a copy).
    """
    captured_geoids = set()
    holdout_indices = []

    if exclude_mask is not None:
        available_gdf = gdf[~exclude_mask].copy()
    else:
        available_gdf = gdf.copy()

    groups = available_gdf.groupby(stratify_cols, dropna=False)
    print(f"Stratifying across {len(groups)} unique groups...")

    for name, group in groups:
        unique_group_geoids = set(group["GEOID"].unique())
        target_count = math.ceil(len(unique_group_geoids) * eval_fraction)
        if target_count == 0:
            continue

        group_captured = len(unique_group_geoids.intersection(captured_geoids))
        while group_captured < target_count:
            areas = group.geometry.area
            if areas.sum() == 0:
                break
            # np.random.choice uses the global numpy random seed
            seed_idx = np.random.choice(group.index, p=areas / areas.sum())
            seed_geom = group.loc[seed_idx].geometry

            cluster_mask = available_gdf.geometry.intersects(
                seed_geom.buffer(cluster_radius))
            cluster_tracts = available_gdf[cluster_mask]
            if cluster_tracts.empty:
                continue

            current_geoids = set(cluster_tracts["GEOID"].unique())
            captured_geoids.update(current_geoids)
            holdout_indices.extend(cluster_tracts.index.tolist())
            group_captured = len(unique_group_geoids.intersection(captured_geoids))

    print(f"Success! Captured {len(captured_geoids)} GEOIDs for this split.")
    # SORT so hash randomization doesn't alter output row order.
    deterministic_indices = sorted(list(set(holdout_indices)))
    return gdf.loc[deterministic_indices].copy()


def assign_tracts_train_val_test(gdf, test_tracts, val_tracts, dead_zone_buffer):
    """Assign final split labels and compute the dead zone (verbatim from notebook).

    Default train; tracts intersecting the ``dead_zone_buffer``-buffered union of
    all holdout tracts become ``dead_zone``; then val, then test override (test
    wins). Returns ``(gdf, dead_zone_geom_gdf)``.
    """
    import geopandas as gpd

    gdf["type"] = "train"
    holdouts = pd.concat([test_tracts, val_tracts])
    dead_zone_geom = holdouts.geometry.union_all().buffer(dead_zone_buffer)
    in_dead_zone = gdf.geometry.intersects(dead_zone_geom)

    gdf.loc[in_dead_zone, "type"] = "dead_zone"
    if not val_tracts.empty:
        gdf.loc[gdf.index.isin(val_tracts.index), "type"] = "val"
    if not test_tracts.empty:
        gdf.loc[gdf.index.isin(test_tracts.index), "type"] = "test"

    return gdf, gpd.GeoDataFrame(geometry=[dead_zone_geom], crs=gdf.crs)


def plot_final_splits(gdf, dead_zone_gdf, out_path=None):
    """Diagnostic map of the NYC within-city split (verbatim from notebook)."""
    import matplotlib.pyplot as plt

    ax = gdf.plot(figsize=(20, 20), color="whitesmoke", edgecolor="lightgray")
    dead_zone_gdf.plot(ax=ax, color="gray", alpha=0.5, label="Dead Zone (Buffer)")
    gdf[gdf["type"] == "train"].plot(ax=ax, color="green", label="Train")
    gdf[gdf["type"] == "val"].plot(ax=ax, color="blue", label="Validation")
    gdf[gdf["type"] == "test"].plot(ax=ax, color="orange", label="Test")
    gdf[gdf["type"] == "dead_zone"].plot(ax=ax, color="red", label="Dead Zone (Discarded)")
    plt.legend()
    plt.title("NYC Tract-Centric Train/Val/Test Split with Dead Zones")
    if out_path is not None:
        plt.savefig(out_path)
    print(gdf["type"].value_counts())


def _load_nyc_panel():
    """Load the NYC tract panel + borough boundaries and attach income quintiles."""
    import geopandas as gpd

    gdf = (gpd.read_feather(PROCESSED_DATA_DIR / PANEL_FILENAME)
           .to_crs(epsg=NYC_EVAL_EPSG)
           .rename(columns={"geoid_2024": "GEOID"}))
    boros = (gpd.read_file(paths.EXTERNAL_DATA_DIR / "NYC Borough Boundaries" / BOROS_FILENAME)
             .to_crs(epsg=NYC_EVAL_EPSG))
    gdf = gdf.sjoin(boros[["boroname", "geometry"]], how="left",
                    predicate="intersects").drop(columns=["index_right"])
    gdf["income_quintile"] = pd.qcut(gdf["Rel_Score_2024"], q=5, labels=False)
    return gdf.dropna(subset=["income_quintile"]).reset_index(drop=True)


def build_nyc_split(gdf=None, seed: int = SEED, save: bool = True, plot: bool = False):
    """Build (and optionally persist) the NYC within-city split.

    Reproduces the validated seed-25 notebook holdout: 6% clustered test, then 6%
    clustered val quarantined from test by a τ=100 m buffer, then the combined
    dead zone. ``gdf`` may be supplied (already carrying GEOID/geometry/
    ``income_quintile`` in EPSG:6539) for testing; otherwise it is loaded from the
    NYC panel. Returns the labelled GeoDataFrame; writes ``nyc_tract_splits.feather``.
    """
    if gdf is None:
        gdf = _load_nyc_panel()

    np.random.seed(seed)
    cluster_radius = geo_utils.meters_to_projected_units(CLUSTER_RADIUS_M, epsg_code=NYC_EVAL_EPSG)
    dead_zone_buffer = geo_utils.meters_to_projected_units(DEAD_ZONE_M, epsg_code=NYC_EVAL_EPSG)

    # 1. TEST holdout.
    test_tracts = create_stratified_tract_holdout(
        gdf, cluster_radius=cluster_radius, stratify_cols=STRATIFY_COLS,
        eval_fraction=TEST_FRACTION)

    # 2. Quarantine zone around test, so val cannot touch test.
    test_restricted_geom = test_tracts.geometry.union_all().buffer(dead_zone_buffer)
    invalid_val_candidates_mask = gdf.geometry.intersects(test_restricted_geom)

    # 3. VAL holdout, excluded from the test quarantine.
    val_tracts = create_stratified_tract_holdout(
        gdf, cluster_radius=cluster_radius, stratify_cols=STRATIFY_COLS,
        eval_fraction=VAL_FRACTION, exclude_mask=invalid_val_candidates_mask)

    # 4. Assign + compute the combined dead zone.
    gdf, dead_zone_geom_gdf = assign_tracts_train_val_test(
        gdf, test_tracts, val_tracts, dead_zone_buffer)

    if plot:
        try:
            plot_final_splits(gdf, dead_zone_geom_gdf)
        except Exception as exc:                       # pragma: no cover
            print(f"⚠️ NYC split plot skipped ({exc!r}).")

    if save:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        gdf[["GEOID", "geometry", "type"]].to_feather(OUT_PATH, index=False)
        print(f"Created file: {OUT_PATH}")
    return gdf


if __name__ == "__main__":
    build_nyc_split(plot=False)
