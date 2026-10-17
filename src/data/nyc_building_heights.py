"""Roof heights for NYC building footprints -> a small ``DOITT_ID, height`` CSV.

The footprint processing (``buildings_nyc.parquet``) is left untouched; this
module only extracts NYC's ``HEIGHT_ROOF`` (feet above ground) from the city's
Building Footprints layers so tract aggregates can weight each building's
prediction by its volume (footprint area x roof height) as a proxy for how many
people it holds.

Sources (NYC Open Data / NYC MapHub "Building" and "Building Historic"):
    BUILDING_view_*.geojson           current footprints
    BUILDING_HISTORIC_view_*.geojson  demolished / altered footprints

Heights come from the current layer when present, else from the historic one.
A zero or negative height is a missing value, not a measurement. Buildings in
the footprint table with no usable height get the mean height of their
``k`` nearest neighbours (by centroid) that have one, and are flagged
``height_imputed`` so downstream code can report how much was filled.

Run once:
    python -m src.data.nyc_building_heights \
        --current  ".../BUILDING_view_8608618849432433473.geojson" \
        --historic ".../BUILDING_HISTORIC_view_4222244593352533104.geojson"
which writes ``data/processed/nyc_building_heights.csv`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEIGHT_COLUMN = "HEIGHT_ROOF"
ID_COLUMN = "DOITT_ID"
DEFAULT_K = 5
# Tallest NYC building (One World Trade Center) is 1,776 ft; anything above
# this is a data-entry error, treated as missing and then imputed.
MAX_PLAUSIBLE_HEIGHT_FT = 2_000.0
HEIGHTS_FILENAME = "nyc_building_heights.csv"


def clean_heights(df: pd.DataFrame, id_col: str = ID_COLUMN,
                  height_col: str = HEIGHT_COLUMN) -> pd.DataFrame:
    """[DOITT_ID, height_roof] with ids > 0, heights numeric, and zero, negative
    or implausible heights set to NaN. Duplicate ids keep their largest height
    (a split/merge record should not shrink a building)."""
    out = pd.DataFrame({
        "DOITT_ID": pd.to_numeric(df[id_col], errors="coerce"),
        "height_roof": pd.to_numeric(df[height_col], errors="coerce"),
    })
    out = out[out["DOITT_ID"].notna() & (out["DOITT_ID"] > 0)].copy()
    out["DOITT_ID"] = out["DOITT_ID"].astype("int64")
    bad = (out["height_roof"] <= 0) | (out["height_roof"] > MAX_PLAUSIBLE_HEIGHT_FT)
    out.loc[bad, "height_roof"] = np.nan
    return out.groupby("DOITT_ID", as_index=False)["height_roof"].max()


def combine_layers(current: pd.DataFrame, historic: pd.DataFrame | None) -> pd.DataFrame:
    """One height per DOITT_ID: the current layer's when it has one, else the
    historic layer's."""
    if historic is None or historic.empty:
        return current.copy()
    merged = current.merge(historic, on="DOITT_ID", how="outer", suffixes=("", "_hist"))
    merged["height_roof"] = merged["height_roof"].fillna(merged["height_roof_hist"])
    return merged[["DOITT_ID", "height_roof"]]


def knn_fill(xy: np.ndarray, heights: np.ndarray, k: int = DEFAULT_K) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN heights with the mean of the ``k`` nearest non-NaN neighbours.

    ``xy`` (n, 2) are projected coordinates (metres). Returns the filled array
    and a boolean mask of the entries that were imputed. If fewer than ``k``
    buildings have a height, all of them are used; if none has, the input is
    returned unchanged (nothing to borrow from).
    """
    from scipy.spatial import cKDTree

    heights = np.asarray(heights, float).copy()
    missing = ~np.isfinite(heights)
    known = ~missing
    if not missing.any() or not known.any():
        return heights, np.zeros(len(heights), bool)
    kk = int(min(k, known.sum()))
    tree = cKDTree(xy[known])
    _, idx = tree.query(xy[missing], k=kk)
    idx = np.asarray(idx).reshape(missing.sum(), kk)
    heights[missing] = heights[known][idx].mean(axis=1)
    return heights, missing


def read_layer(path: Path) -> pd.DataFrame:
    """The id and height columns of one NYC footprint layer, no geometry."""
    import pyogrio

    df = pyogrio.read_dataframe(path, columns=[ID_COLUMN, HEIGHT_COLUMN], read_geometry=False)
    return clean_heights(df)


def footprint_centroids(footprints_path: Path, epsg: int = 5070) -> pd.DataFrame:
    """[DOITT_ID, x, y] centroids (metres) of the processed footprint table."""
    import geopandas as gpd

    fp = gpd.read_parquet(footprints_path)
    ids = fp[ID_COLUMN] if ID_COLUMN in fp.columns else pd.Series(fp.index, index=fp.index)
    c = fp.geometry.to_crs(epsg).centroid
    out = pd.DataFrame({"DOITT_ID": pd.to_numeric(ids, errors="coerce").values,
                        "x": c.x.values, "y": c.y.values})
    out = out[out["DOITT_ID"].notna()].copy()
    out["DOITT_ID"] = out["DOITT_ID"].astype("int64")
    return out.drop_duplicates("DOITT_ID")


def build_height_table(heights: pd.DataFrame, centroids: pd.DataFrame,
                       k: int = DEFAULT_K) -> pd.DataFrame:
    """Heights for every building in ``centroids``, gaps filled by k-NN.

    Returns [DOITT_ID, height_roof, height_imputed]. Buildings whose
    coordinates are not finite cannot be placed and keep NaN (none expected).
    """
    t = centroids.merge(heights, on="DOITT_ID", how="left")
    placeable = np.isfinite(t["x"]) & np.isfinite(t["y"])
    filled, imputed = knn_fill(t.loc[placeable, ["x", "y"]].to_numpy(),
                               t.loc[placeable, "height_roof"].to_numpy(), k=k)
    t["height_imputed"] = False
    t.loc[placeable, "height_roof"] = filled
    t.loc[placeable, "height_imputed"] = imputed
    return t[["DOITT_ID", "height_roof", "height_imputed"]]


def main() -> None:
    from src.utils.paths import PROCESSED_DATA_DIR

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--historic", type=Path, default=None)
    ap.add_argument("--footprints", type=Path,
                    default=PROCESSED_DATA_DIR / "buildings_nyc.parquet")
    ap.add_argument("--out", type=Path, default=PROCESSED_DATA_DIR / HEIGHTS_FILENAME)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    current = read_layer(args.current)
    historic = read_layer(args.historic) if args.historic else None
    heights = combine_layers(current, historic)
    table = build_height_table(heights, footprint_centroids(args.footprints), k=args.k)
    table.to_csv(args.out, index=False)
    n = len(table)
    print(f"current layer: {len(current):,} ids ({current['height_roof'].notna().sum():,} with height)")
    if historic is not None:
        print(f"historic layer: {len(historic):,} ids ({historic['height_roof'].notna().sum():,} with height)")
    print(f"footprints: {n:,}; imputed by {args.k}-NN: {int(table['height_imputed'].sum()):,} "
          f"({table['height_imputed'].mean():.1%}); still missing: "
          f"{int(table['height_roof'].isna().sum()):,}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
