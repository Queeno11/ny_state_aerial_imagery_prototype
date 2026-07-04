"""Build the national building sampling universe from Microsoft US Building Footprints.

Streams each state's ``{State}.geojson.zip`` (one-time download, see
https://github.com/microsoft/usbuildingfootprints) and writes two tables:

* ``buildings_index`` (HOT, training path): one row per building —
  ``[building_id int64, cx float64, cy float64, tract_id str, state str]`` with
  centroids in the national metric CRS (``geo_utils.METRIC_CRS``, EPSG:5070),
  partitioned by state and sorted by ``tract_id``. No polygon geometry: training
  crops are centroid + tau buffers, so the polygons are never touched at train time.

* ``buildings_polygons`` (COLD, offline): per-state GeoParquet of the raw polygons
  keyed by ``building_id`` — preprocessing, evaluation and visualization only.

Design notes
------------
* ``building_id`` is a deterministic packing of the centroid rounded to 0.1 m in
  EPSG:5070 (31 bits per axis -> non-negative signed int64, safe for torch tensors).
  Identical rounded centroids are duplicates by construction and are dropped. This
  happens twice: WITHIN each state during ``process_state``, then ACROSS states in a
  final ``dedupe_index_global`` pass (a border building appears in two states' files
  with the same id) so the whole on-disk index is globally unique. ``--all-states``
  runs the global pass automatically; after an incremental ``--states`` run, rerun it
  standalone with ``--dedupe-only``.
* The tract join is centroid-within against the panel's base-year tract polygons
  (``us_metros_panel_*.feather``), i.e. tract vintage = the panel's BASE_YEAR. By
  default buildings outside panel tracts (non-metro areas) are dropped
  (``--all-buildings`` keeps them with ``tract_id = None``).
* Microsoft footprints are a STATIC universe (no construction/demolition dates);
  the NYC event study keeps using the dated DoITT base (``load_building_data``).

Run (user, long job — states are processed independently and failures reported):

    python -m src.data.build_buildings_index --states Delaware NewYork
    python -m src.data.build_buildings_index --all-states

Verify afterwards with ``python -m src.data.verify_buildings_index``.
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pyarrow as pa
import pyarrow.parquet  # noqa: F401  (registers the parquet submodule)
from pyogrio.raw import open_arrow

import src.geo_utils as geo_utils
from src.utils.paths import PROCESSED_DATA_DIR

# Where the per-state Microsoft zips live (override via env for other machines).
MS_FOOTPRINTS_DIR = Path(os.getenv(
    "MS_FOOTPRINTS_DIR",
    "/mnt/e/Datasets/Building Footprints/Microsoft USBuildingFootprints",
))
DEFAULT_PANEL = PROCESSED_DATA_DIR / "us_metros_panel_2011_2023.feather"
DEFAULT_OUT = PROCESSED_DATA_DIR

# EPSG:5070 is designed for the conterminous US; these states distort badly there.
NON_CONUS_STATES = {"Alaska", "Hawaii", "PuertoRico"}

# 0.1 m grid packing: 31 bits per axis around a 2**30 offset.
_ID_OFFSET = 2 ** 30
_ID_MASK = (1 << 31) - 1


def compute_building_id(cx, cy) -> np.ndarray:
    """Deterministic int64 id from EPSG:5070 centroids rounded to 0.1 m.

    ``id = (x10 + OFFSET) << 31 | (y10 + OFFSET)`` with ``x10 = round(cx * 10)``.
    Non-negative and < 2**62, so it survives torch int64 round-trips. Two buildings
    collide only if their centroids coincide on the 0.1 m grid — treated as
    duplicates upstream.
    """
    x10 = np.rint(np.asarray(cx, dtype="float64") * 10.0).astype("int64") + _ID_OFFSET
    y10 = np.rint(np.asarray(cy, dtype="float64") * 10.0).astype("int64") + _ID_OFFSET
    if (x10 < 0).any() or (x10 > _ID_MASK).any() or (y10 < 0).any() or (y10 > _ID_MASK).any():
        raise ValueError(
            "Centroid outside the packable EPSG:5070 range (|coord| < ~1.07e8 m). "
            "Coordinates are probably not in EPSG:5070."
        )
    return (x10 << 31) | y10


def unpack_building_id(building_id) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`compute_building_id` -> (cx, cy) on the 0.1 m grid."""
    bid = np.asarray(building_id, dtype="int64")
    x10 = (bid >> 31) - _ID_OFFSET
    y10 = (bid & _ID_MASK) - _ID_OFFSET
    return x10 / 10.0, y10 / 10.0


def load_panel_tracts(panel_path=DEFAULT_PANEL) -> gpd.GeoDataFrame:
    """Panel base-year tract polygons in METRIC_CRS, keyed GEOID (+ cbsa_code)."""
    gdf = gpd.read_feather(panel_path)
    geoid_cols = [c for c in gdf.columns if c.startswith("geoid_")]
    if not geoid_cols:
        raise KeyError(f"No geoid_* column in panel {panel_path}")
    key = sorted(geoid_cols)[-1]  # geoid_{base_year}; a single geoid_* col in practice
    tracts = gdf[[key, "cbsa_code", "geometry"]].rename(columns={key: "GEOID"})
    return tracts.to_crs(geo_utils.METRIC_CRS)


def assign_tracts(centroids: gpd.GeoSeries, tracts: gpd.GeoDataFrame) -> pd.Series:
    """tract GEOID per centroid (centroid-within join); NaN when outside all tracts."""
    pts = gpd.GeoDataFrame(geometry=centroids, crs=centroids.crs)
    joined = gpd.sjoin(pts, tracts[["GEOID", "geometry"]], how="left", predicate="within")
    # A centroid on a shared boundary can match 2 tracts -> keep the first match.
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["GEOID"].reindex(pts.index)


def process_chunk(geoms_4326: gpd.GeoSeries, tracts: gpd.GeoDataFrame,
                  state: str, restrict_to_panel: bool = True):
    """One streamed batch of polygons -> (index_df, polygons_gdf).

    ``index_df``: building_id, cx, cy, tract_id, state (may be empty after the
    panel restriction). ``polygons_gdf``: building_id + polygon in METRIC_CRS for
    the SAME retained rows.
    """
    geoms = geoms_4326.to_crs(geo_utils.METRIC_CRS)
    centroids = geoms.centroid
    tract_id = assign_tracts(centroids, tracts)

    keep = tract_id.notna() if restrict_to_panel else pd.Series(True, index=geoms.index)
    geoms, centroids, tract_id = geoms[keep], centroids[keep], tract_id[keep]
    if len(geoms) == 0:
        empty_idx = pd.DataFrame({
            "building_id": pd.Series([], dtype="int64"),
            "cx": pd.Series([], dtype="float64"),
            "cy": pd.Series([], dtype="float64"),
            "tract_id": pd.Series([], dtype="object"),
            "state": pd.Series([], dtype="object"),
        })
        empty_pol = gpd.GeoDataFrame(
            {"building_id": pd.Series([], dtype="int64")},
            geometry=[], crs=geo_utils.METRIC_CRS)
        return empty_idx, empty_pol

    cx, cy = centroids.x.to_numpy(), centroids.y.to_numpy()
    building_id = compute_building_id(cx, cy)
    index_df = pd.DataFrame({
        "building_id": building_id,
        "cx": cx,
        "cy": cy,
        "tract_id": tract_id.to_numpy(),
        "state": state,
    })
    polygons_gdf = gpd.GeoDataFrame(
        {"building_id": building_id}, geometry=geoms.to_numpy(),
        crs=geo_utils.METRIC_CRS,
    )
    return index_df, polygons_gdf


def iter_state_batches(zip_path: Path, batch_size: int = 200_000):
    """Stream a Microsoft ``{State}.geojson.zip`` as GeoSeries batches (EPSG:4326).

    Single pass via pyogrio's Arrow reader; memory stays bounded by batch_size.
    """
    vsi = f"/vsizip/{zip_path}"
    with open_arrow(vsi, batch_size=batch_size, use_pyarrow=True) as (meta, reader):
        geom_col = meta["geometry_name"] or "wkb_geometry"
        crs = meta["crs"] or "EPSG:4326"
        for batch in reader:
            wkb = batch.column(geom_col)
            geoms = shapely.from_wkb(np.asarray(wkb))
            yield gpd.GeoSeries(geoms, crs=crs)


def process_state(state: str, tracts: gpd.GeoDataFrame,
                  footprints_dir: Path = MS_FOOTPRINTS_DIR,
                  out_dir: Path = DEFAULT_OUT,
                  restrict_to_panel: bool = True,
                  write_polygons: bool = True,
                  batch_size: int = 200_000) -> dict:
    """Build one state's index/polygons partitions. Returns a summary dict."""
    if state in NON_CONUS_STATES:
        warnings.warn(
            f"{state} lies outside EPSG:5070's design area (CONUS Albers); distances "
            f"and areas will be distorted. Kept, but treat with care."
        )
    zip_path = footprints_dir / f"{state}.geojson.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    index_parts = []
    poly_dir = out_dir / "buildings_polygons" / state
    if write_polygons:
        poly_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_read = 0
    for i, geoms in enumerate(iter_state_batches(zip_path, batch_size)):
        n_read += len(geoms)
        index_df, polygons_gdf = process_chunk(
            geoms, tracts, state, restrict_to_panel=restrict_to_panel
        )
        index_parts.append(index_df)
        if write_polygons and len(polygons_gdf):
            polygons_gdf.to_parquet(poly_dir / f"part_{i:04d}.parquet")
        print(f"  {state}: batch {i} — read {n_read:,}, "
              f"kept {sum(len(p) for p in index_parts):,}", flush=True)

    index = pd.concat(index_parts, ignore_index=True)

    # Identical rounded centroids are duplicates by construction (e.g. the same
    # building split across tiles); keep the first occurrence.
    n_dup = int(index.duplicated(subset="building_id").sum())
    if n_dup:
        print(f"  {state}: dropping {n_dup:,} duplicate building_ids "
              f"({100 * n_dup / len(index):.3f}%)")
        index = index.drop_duplicates(subset="building_id", keep="first")
    assert not index["building_id"].duplicated().any()

    index = index.sort_values("tract_id").reset_index(drop=True)
    idx_dir = out_dir / "buildings_index" / f"state={state}"
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_path = idx_dir / "part.parquet"
    # 'state' lives in the hive partition path; keeping it in the file too would
    # collide with the partition field when reading the whole dataset.
    index.drop(columns="state").to_parquet(idx_path, index=False)

    summary = {
        "state": state,
        "read": n_read,
        "kept": len(index),
        "dropped_duplicates": n_dup,
        "tracts": index["tract_id"].nunique(),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  {state}: DONE — {summary['kept']:,}/{summary['read']:,} buildings in "
          f"{summary['tracts']:,} tracts ({summary['seconds']}s) -> {idx_path}")
    return summary


def available_states(footprints_dir: Path = MS_FOOTPRINTS_DIR) -> list[str]:
    return sorted(p.name.removesuffix(".geojson.zip")
                  for p in footprints_dir.glob("*.geojson.zip"))


def dedupe_index_global(out_dir: Path = DEFAULT_OUT) -> dict:
    """Drop CROSS-state duplicate ``building_id``s from the on-disk buildings_index.

    :func:`process_state` only dedupes WITHIN a single state. A building sitting on
    a state border, however, appears in both neighbouring states' Microsoft files
    and lands on the same 0.1 m grid cell in each -> the *same* ``building_id`` in
    two partitions. Because the id is a deterministic packing of the rounded
    centroid, both copies carry identical ``cx``/``cy`` and (centroid-within) the
    same ``tract_id``, so either copy is interchangeable. We keep the occurrence in
    the first state alphabetically and drop the rest, rewriting ONLY the partitions
    that actually lose rows.

    Reads just the ``building_id`` column across partitions (cheap) to find the
    duplicates, so it is safe to run after every (even incremental) build. Returns
    a summary dict.
    """
    index_dir = out_dir / "buildings_index"
    parts = sorted(index_dir.glob("state=*/part.parquet"))
    if not parts:
        print(f"  dedupe: no partitions under {index_dir} — nothing to do.")
        return {"n_states": 0, "n_duplicates": 0, "states_rewritten": []}

    # (building_id, state) for every row, reading only the id column per partition.
    frames = []
    for p in parts:
        state = p.parent.name.split("=", 1)[1]
        bid = pd.read_parquet(p, columns=["building_id"])["building_id"].to_numpy()
        frames.append(pd.DataFrame({"building_id": bid, "state": state}))
    allids = pd.concat(frames, ignore_index=True)

    dup_mask = allids["building_id"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows == 0:
        print(f"  dedupe: {len(allids):,} building_ids across {len(parts)} states — "
              f"already globally unique.")
        return {"n_states": len(parts), "n_duplicates": 0, "states_rewritten": []}

    # Keeper = first state (sorted) that holds each duplicated id; every other
    # (state, id) row is a loser to be dropped from that state's partition.
    dup = allids[dup_mask]
    keeper = dup.sort_values("state", kind="stable").groupby("building_id")["state"].first()
    dup = dup.assign(keeper=dup["building_id"].map(keeper))
    losers = dup[dup["state"] != dup["keeper"]]
    n_dropped = int(len(losers))

    rewritten = []
    for state, sub in losers.groupby("state"):
        loser_ids = set(sub["building_id"].to_numpy().tolist())
        part = index_dir / f"state={state}" / "part.parquet"
        df = pd.read_parquet(part)
        before = len(df)
        df = df[~df["building_id"].isin(loser_ids)].reset_index(drop=True)
        df.to_parquet(part, index=False)
        rewritten.append(state)
        print(f"  dedupe: {state} — dropped {before - len(df):,} cross-state "
              f"duplicate building_ids ({len(df):,} kept)")

    print(f"  dedupe: removed {n_dropped:,} cross-state duplicate rows "
          f"({100 * n_dropped / len(allids):.3f}%) across {len(rewritten)} "
          f"state(s); index is now globally unique.")
    return {"n_states": len(parts), "n_duplicates": n_dropped,
            "states_rewritten": rewritten}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", default=None,
                        help="State file stems, e.g. Delaware NewYork")
    parser.add_argument("--all-states", action="store_true")
    parser.add_argument("--footprints-dir", default=str(MS_FOOTPRINTS_DIR))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--all-buildings", action="store_true",
                        help="Keep buildings outside panel tracts (tract_id=None)")
    parser.add_argument("--no-polygons", action="store_true",
                        help="Skip the cold buildings_polygons GeoParquet")
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Skip the cross-state global dedupe pass after building.")
    parser.add_argument("--dedupe-only", action="store_true",
                        help="Only run the cross-state global dedupe over the existing "
                             "on-disk index (no state processing). Use after incremental "
                             "--states runs to make the whole index globally unique.")
    args = parser.parse_args(argv)

    if args.dedupe_only:
        print("=== Cross-state global dedupe (existing index) ===")
        dedupe_index_global(Path(args.out))
        return 0

    footprints_dir = Path(args.footprints_dir)
    if args.all_states:
        states = available_states(footprints_dir)
    elif args.states:
        states = args.states
    else:
        parser.error("Pass --states ... or --all-states")

    print(f"Loading panel tracts from {args.panel} ...")
    tracts = load_panel_tracts(Path(args.panel))
    print(f"{len(tracts):,} tracts in {tracts['cbsa_code'].nunique()} CBSAs")

    summaries, failures = [], []
    for state in states:
        print(f"\n=== {state} ===")
        try:
            summaries.append(process_state(
                state, tracts,
                footprints_dir=footprints_dir,
                out_dir=Path(args.out),
                restrict_to_panel=not args.all_buildings,
                write_polygons=not args.no_polygons,
                batch_size=args.batch_size,
            ))
        except Exception as e:  # keep going: one bad state must not kill the run
            print(f"  {state}: FAILED — {e!r}")
            failures.append((state, repr(e)))

    # Border buildings appear in two states' files -> the same building_id in two
    # partitions. process_state only dedupes within a state, so make the whole
    # on-disk index globally unique here (scans building_ids across all partitions
    # present on disk, not just the ones processed this run).
    if not args.no_dedupe:
        print("\n=== Cross-state global dedupe ===")
        dedupe_index_global(Path(args.out))

    print("\n=== Summary ===")
    if summaries:
        print(pd.DataFrame(summaries).to_string(index=False))
    if failures:
        print("\nFAILED states:")
        for state, err in failures:
            print(f"  {state}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
