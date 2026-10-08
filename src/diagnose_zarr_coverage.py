# -*- coding: utf-8 -*-
"""Where the legacy NYC zarr stores have no imagery at all.

Why chunk *presence* is the right probe
---------------------------------------
The stores were written by ``notebooks/Create tifs.ipynb`` with
``write_empty_chunks=False`` and ``_FillValue: 0``. A region with no source JP2
never enters the mosaic, so zarr writes **no chunk file** for it, and a read of
that window synthesizes an array of zeros — structurally perfect, silently
black. The set of chunk keys on disk is therefore an exact map of what imagery
each store actually holds, readable from a directory listing alone: no
decompression and no pixel reads, ~7k filenames per 72 GB store.

The complement matters too. A chunk that exists but was *written* zero (a real
nodata hole inside a delivered tile) is invisible here; that case belongs to
``CyclicCacheManager._reject_for_zero_pixels``. This module answers the
coarser question — which whole areas were never ingested — which is the one
that turns a borough black.

Georeferencing: trust the coordinates, not the attribute
--------------------------------------------------------
Every store's ``spatial_ref`` carries ``GeoTransform = "997500.0 0.5 0.0
277500.0 0.0 -0.5"``, but the stored ``x`` coordinate array starts at
907500 ft — the attribute is wrong by 90,000 ft in x (a digit transposition,
9-0-7 vs 9-9-7) and was never used by the training pipeline, which indexes the
stores by row/col, not by CRS coordinate. The ``x``/``y`` arrays are
authoritative and are what this module reads. ``check_geotransform_attr``
reports the discrepancy rather than silently papering over it.

The stores also do not share an extent: ``nyc_2024`` starts at x=960000 ft,
52,500 ft (42 chunks) east of the other seven. Aligning by chunk *index* would
compare different ground across years, so stores are placed onto a common
master grid by their projected origin.

Usage
-----
    IMAGERY_ROOT=/home/abbatenicolas/data python src/diagnose_zarr_coverage.py

Prints, per year and borough, the share of the city-wide chunk footprint the
store actually covers. Legacy-zarr diagnostic only; current runs use NAIP.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The eight even years the NYC zarr panel covers (see main.py `nyc_years`).
NYC_ZARR_YEARS = (2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024)

# Coordinates in these stores were rounded to 1 decimal when the mosaic was
# built, so consecutive 0.5 ft steps read back as 0.4/0.6 alternating. Any
# tolerance well under a pixel but over that rounding is safe; units are US
# survey feet.
COORD_TOL_FT = 0.25


# ─── store geometry ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StoreGeo:
    """One store's chunk grid and where it sits on the ground.

    ``x0``/``y0`` are the *outer edge* of pixel (0, 0) — half a pixel out from
    the stored coordinate, which is a cell centre. ``dy`` is negative
    (north-up).
    """

    n_rows: int          # chunk rows
    n_cols: int          # chunk cols
    chunk_rows: int      # pixels per chunk, y
    chunk_cols: int      # pixels per chunk, x
    x0: float
    y0: float
    dx: float
    dy: float
    crs_wkt: str

    @property
    def cell_width(self) -> float:
        return self.dx * self.chunk_cols

    @property
    def cell_height(self) -> float:
        return self.dy * self.chunk_rows      # negative


def read_store_geo(zarr_path, array_name: str = "value") -> StoreGeo:
    """Chunk-grid geometry, georeferenced from the stored coordinate arrays."""
    import xarray as xr

    zarr_path = Path(zarr_path)
    with open(zarr_path / array_name / ".zarray") as fh:
        zarray = json.load(fh)
    shape, chunks = zarray["shape"], zarray["chunks"]
    if len(shape) != 3:
        raise ValueError(f"expected a (band, y, x) array, got shape {shape}")

    ds = xr.open_zarr(zarr_path, mask_and_scale=False)
    x, y = np.asarray(ds.x.values), np.asarray(ds.y.values)
    if x.size < 2 or y.size < 2:
        raise ValueError(f"{zarr_path}: degenerate coordinate arrays")

    # Average step, not first difference: the coords carry rounding noise.
    dx = (x[-1] - x[0]) / (x.size - 1)
    dy = (y[-1] - y[0]) / (y.size - 1)

    crs_wkt = ""
    sref_attrs = zarr_path / "spatial_ref" / ".zattrs"
    if sref_attrs.exists():
        with open(sref_attrs) as fh:
            crs_wkt = json.load(fh).get("crs_wkt", "")

    return StoreGeo(
        n_rows=-(-shape[1] // chunks[1]),     # ceil: a partial chunk still
        n_cols=-(-shape[2] // chunks[2]),     # occupies a grid cell
        chunk_rows=chunks[1], chunk_cols=chunks[2],
        x0=float(x[0] - dx / 2.0), y0=float(y[0] - dy / 2.0),
        dx=float(dx), dy=float(dy), crs_wkt=crs_wkt,
    )


def check_geotransform_attr(zarr_path, geo: StoreGeo, tol: float = COORD_TOL_FT):
    """Compare the stored GeoTransform against the coordinate arrays.

    Returns None when they agree, else a human-readable description of the
    disagreement. These stores all disagree; surfacing it is the point.
    """
    sref_attrs = Path(zarr_path) / "spatial_ref" / ".zattrs"
    if not sref_attrs.exists():
        return None
    with open(sref_attrs) as fh:
        raw = json.load(fh).get("GeoTransform")
    if not raw:
        return None
    gt = [float(v) for v in raw.split()]
    if len(gt) != 6:
        return f"malformed GeoTransform {raw!r}"
    if abs(gt[0] - geo.x0) > tol or abs(gt[3] - geo.y0) > tol:
        return (f"GeoTransform origin ({gt[0]:.1f}, {gt[3]:.1f}) disagrees with "
                f"coordinate arrays ({geo.x0:.1f}, {geo.y0:.1f})")
    return None


def present_chunk_grid(zarr_path, geo: StoreGeo,
                       array_name: str = "value") -> np.ndarray:
    """Boolean (n_rows, n_cols) mask: True where a chunk file exists.

    Keys are ``<band>.<row>.<col>`` (dimension_separator "."). Bands are a
    single chunk here, so any band-chunk present means the tile was ingested;
    the band index is folded away with an OR.
    """
    out = np.zeros((geo.n_rows, geo.n_cols), dtype=bool)
    for key in os.listdir(Path(zarr_path) / array_name):
        if key.startswith("."):
            continue
        parts = key.split(".")
        if len(parts) != 3:
            continue
        try:
            r, c = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if 0 <= r < geo.n_rows and 0 <= c < geo.n_cols:
            out[r, c] = True
    return out


def chunk_bounds(geo: StoreGeo, row: int, col: int) -> tuple:
    """Projected (minx, miny, maxx, maxy) of one chunk cell."""
    xa = geo.x0 + geo.cell_width * col
    xb = xa + geo.cell_width
    ya = geo.y0 + geo.cell_height * row
    yb = ya + geo.cell_height
    return (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))


def chunk_frame(geo: StoreGeo, mask: np.ndarray = None):
    """GeoDataFrame of chunk cells (row, col, geometry), optionally masked."""
    import geopandas as gpd
    from shapely.geometry import box

    if mask is None:
        mask = np.ones((geo.n_rows, geo.n_cols), bool)
    rows, cols = np.nonzero(mask)
    geoms = [box(*chunk_bounds(geo, int(r), int(c))) for r, c in zip(rows, cols)]
    return gpd.GeoDataFrame(
        {"row": rows, "col": cols}, geometry=geoms, crs=geo.crs_wkt or None,
    )


# ─── putting stores on one grid ───────────────────────────────────────────────

def assert_same_pixel_geometry(a: StoreGeo, b: StoreGeo, label: str = "",
                               rtol: float = 0.01):
    """Same pixel size and chunk size — origin and extent may differ.

    The tolerance is relative and loose because ``dx`` is recovered from
    coordinates that were rounded to one decimal: the endpoint error is
    ±0.05 ft spread over the axis. A real difference in pixel size is a
    factor, not a fraction of a percent.
    """
    if (a.chunk_rows, a.chunk_cols) != (b.chunk_rows, b.chunk_cols):
        raise ValueError(f"{label} uses a different chunk size than its peers")
    for va, vb in ((a.dx, b.dx), (a.dy, b.dy)):
        if abs(va - vb) > rtol * max(abs(va), abs(vb)):
            raise ValueError(f"{label} has a different pixel size than its peers")


def pick_crs(geos) -> str:
    """A CRS for the master grid that pyproj can actually transform with.

    nyc_2010 and nyc_2012 carry an authority-less ``NAD83 / New York Long
    Island`` WKT; reprojecting into it yields infinities rather than an error,
    which silently empties every spatial join. The other six stores carry a
    proper EPSG:6539 WKT for the same projection (NAD83 vs NAD83(2011) differs
    by well under a foot here, against 1,250 ft chunk cells), so prefer a WKT
    that resolves to an EPSG code and keep the bare one only as a fallback.
    """
    from pyproj import CRS

    fallback = ""
    for g in geos:
        if not g.crs_wkt:
            continue
        fallback = fallback or g.crs_wkt
        try:
            if CRS.from_user_input(g.crs_wkt).to_epsg():
                return g.crs_wkt
        except Exception:
            continue
    return fallback


def master_geo(geos) -> StoreGeo:
    """Smallest chunk grid, on the common origin, containing every store."""
    geos = list(geos)
    ref = geos[0]
    for g in geos[1:]:
        assert_same_pixel_geometry(ref, g, label="store")

    x0 = min(g.x0 for g in geos)
    y0 = max(g.y0 for g in geos)          # north-up: max y is the top edge
    x1 = max(g.x0 + g.cell_width * g.n_cols for g in geos)
    y1 = min(g.y0 + g.cell_height * g.n_rows for g in geos)

    n_cols = int(round((x1 - x0) / ref.cell_width))
    n_rows = int(round((y1 - y0) / ref.cell_height))
    return StoreGeo(
        n_rows=n_rows, n_cols=n_cols,
        chunk_rows=ref.chunk_rows, chunk_cols=ref.chunk_cols,
        x0=x0, y0=y0, dx=ref.dx, dy=ref.dy, crs_wkt=pick_crs(geos),
    )


def chunk_offset(geo: StoreGeo, master: StoreGeo, tol: float = COORD_TOL_FT):
    """(row, col) of this store's cell (0, 0) within ``master``.

    Rejects a store whose origin is not a whole number of chunks from the
    master origin: a fractional offset means cells do not line up and no
    index-based comparison across years is valid.
    """
    col = (geo.x0 - master.x0) / master.cell_width
    row = (geo.y0 - master.y0) / master.cell_height
    for name, v, span in (("x", col, master.cell_width),
                          ("y", row, abs(master.cell_height))):
        if abs(v - round(v)) * span > tol:
            raise ValueError(
                f"store origin is not chunk-aligned in {name} "
                f"(offset {v:.4f} cells)"
            )
    return int(round(row)), int(round(col))


def place(mask: np.ndarray, offset, master: StoreGeo) -> np.ndarray:
    """Drop a store's mask into the master grid at ``offset``, False elsewhere.

    Everything outside the store's own array is absent imagery by definition,
    which is exactly what the False padding asserts.
    """
    row, col = offset
    out = np.zeros((master.n_rows, master.n_cols), dtype=bool)
    if row < 0 or col < 0 or (row + mask.shape[0] > master.n_rows) \
            or (col + mask.shape[1] > master.n_cols):
        raise ValueError("store does not fit inside the master grid")
    out[row:row + mask.shape[0], col:col + mask.shape[1]] = mask
    return out


def load_masks(imagery_root, years=NYC_ZARR_YEARS, pattern="nyc_{year}.zarr",
               verbose: bool = True):
    """(masks_by_year, master_geo) for every store found under ``imagery_root``.

    Masks come back on a shared master grid, so ``mask[r, c]`` is the same
    ground in every year.
    """
    imagery_root = Path(imagery_root)
    raw, geos = {}, {}
    for year in years:
        path = imagery_root / pattern.format(year=year)
        if not path.exists():
            if verbose:
                print(f"  (skip {year}: {path} not found)")
            continue
        geos[year] = read_store_geo(path)
        raw[year] = present_chunk_grid(path, geos[year])
        if verbose:
            problem = check_geotransform_attr(path, geos[year])
            if problem:
                print(f"  warn {year}: {problem}")
    if not geos:
        raise FileNotFoundError(f"no zarr stores found under {imagery_root}")

    master = master_geo(geos.values())
    masks = {}
    for year, geo in geos.items():
        off = chunk_offset(geo, master)
        if verbose and off != (0, 0):
            print(f"  note: {year} origin sits at master cell {off} "
                  f"({geo.n_rows}x{geo.n_cols} cells vs master "
                  f"{master.n_rows}x{master.n_cols})")
        masks[year] = place(raw[year], off, master)
    return masks, master


# ─── coverage accounting ──────────────────────────────────────────────────────

def city_footprint(masks: dict) -> np.ndarray:
    """Union of every year's chunks — the area the panel is *supposed* to hold.

    Using the union rather than the array rectangle keeps water and
    out-of-city padding out of the denominator, so a year's shortfall is
    measured against imagery that demonstrably exists in some other year.
    """
    if not masks:
        raise ValueError("no masks given")
    out = None
    for m in masks.values():
        out = m.copy() if out is None else (out | m)
    return out


def coverage_by_region(masks: dict, geo: StoreGeo, regions,
                       region_key: str) -> pd.DataFrame:
    """Per (year, region) share of the city footprint present on disk.

    A chunk is attributed to every region it intersects, so a cell straddling
    a borough line counts for both; shares are within-region and need not sum
    across regions.
    """
    import geopandas as gpd

    footprint = city_footprint(masks)
    cells = chunk_frame(geo, footprint)
    regions = regions.to_crs(cells.crs)

    # A CRS pyproj cannot transform into produces infinities, not an error;
    # the join then comes back empty and looks like "no coverage anywhere".
    if not np.isfinite(regions.total_bounds).all():
        raise ValueError(
            "reprojecting the regions into the store CRS produced non-finite "
            "bounds — the store's WKT is not usable for transformation"
        )

    joined = gpd.sjoin(cells, regions[[region_key, "geometry"]],
                       how="inner", predicate="intersects")

    records = []
    for name, part in joined.groupby(region_key):
        cells_in = set(zip(part["row"], part["col"]))   # dedupe multi-matches
        total = len(cells_in)
        for year, mask in masks.items():
            present = sum(1 for r, c in cells_in if mask[r, c])
            records.append({
                "year": year, region_key: name,
                "chunks_total": total, "chunks_present": present,
                "chunks_missing": total - present,
                "coverage": present / total if total else np.nan,
            })
    if not records:
        raise ValueError(
            "no chunk cell intersects any region — check the CRS and the "
            "store georeferencing"
        )
    return pd.DataFrame.from_records(records).sort_values(["year", region_key])


# ─── report ───────────────────────────────────────────────────────────────────

def load_boroughs():
    import geopandas as gpd
    from src.utils.paths import EXTERNAL_DATA_DIR

    boros = gpd.read_file(
        EXTERNAL_DATA_DIR / "NYC Borough Boundaries"
        / "Borough_Boundaries_20260131.geojson"
    )
    key = next((c for c in ("boroname", "boro_name", "BoroName")
                if c in boros.columns), None)
    if key is None:
        raise KeyError(f"no borough-name column in {boros.columns.tolist()}")
    return boros, key


def save_coverage_map(masks: dict, geo: StoreGeo, regions, out_path,
                      region_key: str = None):
    """One panel per year: chunks present in grey, absent-but-expected in red.

    "The Bronx is black" is a claim about a map, so the diagnostic should
    produce one; the panel is also an independent read on the borough table.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    footprint = city_footprint(masks)
    years = sorted(masks)
    ncol = 4
    nrow = -(-len(years) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    extent = (geo.x0, geo.x0 + geo.cell_width * geo.n_cols,
              geo.y0 + geo.cell_height * geo.n_rows, geo.y0)
    regions = regions.to_crs(geo.crs_wkt) if geo.crs_wkt else regions

    for ax, year in zip(axes, years):
        # 0 = outside the city, 1 = present, 2 = expected but absent
        img = np.zeros(footprint.shape, dtype=float)
        img[footprint & masks[year]] = 1.0
        img[footprint & ~masks[year]] = 2.0
        ax.imshow(img, extent=extent, origin="upper", interpolation="nearest",
                  cmap=matplotlib.colors.ListedColormap(
                      ["white", "0.75", "crimson"]),
                  vmin=0, vmax=2)
        regions.boundary.plot(ax=ax, color="black", linewidth=0.6)
        missing = int((footprint & ~masks[year]).sum())
        ax.set_title(f"{year} — {missing} cells missing", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(years):]:
        ax.axis("off")
    fig.suptitle("NYC zarr coverage: red = no chunk written (reads as black)",
                 fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(fig_path=None):
    from src.utils.paths import IMAGERY_ROOT, FIGURES_DIR

    print(f"Reading zarr chunk keys under {IMAGERY_ROOT} ...")
    masks, geo = load_masks(IMAGERY_ROOT)
    footprint = city_footprint(masks)
    print(f"\nmaster grid: {geo.n_rows} x {geo.n_cols} cells of "
          f"{geo.chunk_rows} x {geo.chunk_cols} px @ {geo.dx} ft; "
          f"city footprint = {footprint.sum()} cells\n")

    for year in sorted(masks):
        present = (masks[year] & footprint).sum()
        print(f"  {year}: {present:5d} / {footprint.sum()} cells "
              f"({100 * present / footprint.sum():5.1f}%)")

    boros, key = load_boroughs()
    table = coverage_by_region(masks, geo, boros, key)
    print("\nCoverage by borough (share of cells that exist in ANY year):\n")
    pivot = table.pivot(index=key, columns="year", values="coverage")
    print(pivot.to_string(float_format=lambda v: f"{100 * v:5.1f}%"))

    print("\nMissing chunk cells by borough:\n")
    print(table.pivot(index=key, columns="year",
                      values="chunks_missing").to_string())

    worst = table.loc[table["coverage"].idxmin()]
    print(f"\nWorst cell: {worst[key]} {int(worst['year'])} — "
          f"{int(worst['chunks_missing'])} of {int(worst['chunks_total'])} "
          f"chunks absent ({100 * worst['coverage']:.1f}% covered)")

    out = save_coverage_map(
        masks, geo, boros,
        fig_path or FIGURES_DIR / "zarr_coverage_by_year.png")
    print(f"\nCoverage map written to {out}")
    return table


if __name__ == "__main__":
    main()
