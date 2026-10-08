# -*- coding: utf-8 -*-
"""Repair the missing-imagery holes in the legacy NYC zarr stores.

What is broken (see src/diagnose_zarr_coverage.py for the evidence)
-------------------------------------------------------------------
* ``nyc_2022.zarr`` — 191 of the 262 indexed Bronx tiles never reached the
  mosaic input folder, so 681 chunk cells were never written. Reads of that
  window return a structurally perfect array of zeros: a black Bronx.
* ``nyc_2024.zarr`` — 468 Staten Island tiles likewise never made it, *and*
  the store was built on a cropped grid whose origin is x=960,000 ft, 42
  chunks east of its peers. Most of Staten Island (913,175–960,000 ft) sits at
  negative indices in that array, so there is nowhere to write it.

The two repairs are therefore different in kind:

``append``   2022 needs no new container. The Bronx is inside the existing
             array bounds, and zarr is chunk-addressable, so writing those
             tiles rewrites only the chunks they touch and leaves the other
             ~7,400 alone.

``rebuild``  2024 needs a new container on the canonical origin. Zarr can grow
             an array at the end but not before its start, so the fix is a new
             full-extent store into which the existing chunk files are
             *relinked* under shifted keys — chunk blobs are self-contained
             and the codec is unchanged, so this is a filesystem operation on
             ~68 GB, not a re-compression — followed by writing the Staten
             Island tiles into the newly reachable western columns.

Two properties of the source tiles make this exact rather than approximate
(both verified against ``lot6_nyc_liz_06.shp``): every tile is 2500 x 2500 ft
= 5000 x 5000 px at 0.5 ft, and every tile origin is a whole multiple of
1250 ft from (907500, 277500). One tile is therefore exactly 2 x 2 chunks,
perfectly aligned — every write is whole chunks, with no partial-chunk
read-modify-write anywhere.

Quantization
------------
``notebooks/Create tifs.ipynb`` masked the low two bits of every pixel
(``mosaic & 0xFC``) before writing, and the stores confirm it: no pixel in
nyc_2022 has either low bit set. Appended tiles get the same treatment by
default, or the Bronx would carry a different quantization from the rest of
the store — and the model reads pixel values.

Prerequisite
------------
Reading .jp2 needs GDAL's JP2OpenJPEG plugin, which ``torch_geo_env`` does not
currently have (``libopenjp2`` is installed but the GDAL driver is not)::

    conda install -c conda-forge libgdal-jp2openjpeg

``ensure_jp2_support()`` checks this up front rather than failing on the first
tile. GeoTIFF inputs work without it.

Usage
-----
    # 1. dry run first — reports placements, writes nothing
    python src/repair_zarr_coverage.py append \\
        --store /home/abbatenicolas/data/nyc_2022.zarr \\
        --tiles /home/abbatenicolas/data/bronx_2022 --dry-run

    # 2. the real thing
    python src/repair_zarr_coverage.py append \\
        --store /home/abbatenicolas/data/nyc_2022.zarr \\
        --tiles /home/abbatenicolas/data/bronx_2022

    # 3. 2024: new full-extent store, then swap it in
    python src/repair_zarr_coverage.py rebuild \\
        --store /home/abbatenicolas/data/nyc_2024.zarr \\
        --out   /home/abbatenicolas/data/nyc_2024_full.zarr \\
        --tiles /home/abbatenicolas/data/staten_island_2024
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Low two bits cleared, as notebooks/Create tifs.ipynb did to every store.
QUANTIZATION_MASK = 0xFC

# Tolerance for "is this tile on the grid", in US survey feet. A tenth of a
# foot is a fifth of a pixel — far tighter than any real misplacement, loose
# enough for float noise and for coordinates rounded to one decimal.
ALIGN_TOL_FT = 0.1


# ─── the canonical grid ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridSpec:
    """Pixel grid of a store: outer edge of pixel (0, 0), step, and size.

    ``dy`` is negative (north-up). Pixel *centres* are half a step in, which
    is what gets written to the ``x``/``y`` coordinate arrays.
    """

    x0: float
    y0: float
    dx: float
    dy: float
    n_y: int
    n_x: int

    def x_coords(self) -> np.ndarray:
        return self.x0 + self.dx * (np.arange(self.n_x) + 0.5)

    def y_coords(self) -> np.ndarray:
        return self.y0 + self.dy * (np.arange(self.n_y) + 0.5)

    def geotransform(self) -> str:
        return f"{self.x0} {self.dx} 0.0 {self.y0} 0.0 {self.dy}"


# The grid the seven full-extent stores share. Reproduces nyc_2014's
# coordinate arrays exactly (x[0]=907500.25, y[0]=277499.75).
NYC_GRID = GridSpec(x0=907500.0, y0=277500.0, dx=0.5, dy=-0.5,
                    n_y=320000, n_x=325000)


def snap_grid(grid: GridSpec, tol_ft: float = 0.25) -> GridSpec:
    """Remove the coordinate-rounding noise from a recovered grid.

    The mosaic notebook rounded every coordinate to one decimal, so a store
    built on a clean 0.5 ft grid at x=907500 reads back as dx=0.5000003 and
    x0=907499.95. Left alone that propagates into every index and into any
    GeoTransform written from it.

    Two facts pin the true values: the step is a round number at millifoot
    precision, and the origin lies on a whole pixel multiple. Snapping is
    applied only when it moves the grid by less than ``tol_ft`` (a half-pixel
    by default), so a genuinely offset grid is left alone rather than dragged
    onto a lattice it was never on.
    """
    dx = round(grid.dx, 3)
    dy = round(grid.dy, 3)
    if dx == 0 or dy == 0:
        return grid
    x0 = round(grid.x0 / dx) * dx
    y0 = round(grid.y0 / abs(dy)) * abs(dy)

    if abs(x0 - grid.x0) > tol_ft or abs(y0 - grid.y0) > tol_ft:
        return grid
    if abs(dx - grid.dx) > 1e-3 or abs(dy - grid.dy) > 1e-3:
        return grid
    return GridSpec(x0=float(x0), y0=float(y0), dx=float(dx), dy=float(dy),
                    n_y=grid.n_y, n_x=grid.n_x)


def grid_from_store(zarr_path, array_name: str = "value",
                    snap: bool = True) -> GridSpec:
    """Recover a store's grid from its coordinate arrays (not its
    GeoTransform attribute, which is wrong in every one of these stores)."""
    import xarray as xr

    zarr_path = Path(zarr_path)
    with open(zarr_path / array_name / ".zarray") as fh:
        shape = json.load(fh)["shape"]

    ds = xr.open_zarr(zarr_path, mask_and_scale=False)
    x, y = np.asarray(ds.x.values), np.asarray(ds.y.values)
    dx = (x[-1] - x[0]) / (x.size - 1)
    dy = (y[-1] - y[0]) / (y.size - 1)
    grid = GridSpec(x0=float(x[0] - dx / 2), y0=float(y[0] - dy / 2),
                    dx=float(dx), dy=float(dy),
                    n_y=int(shape[1]), n_x=int(shape[2]))
    return snap_grid(grid) if snap else grid


# ─── source tiles ─────────────────────────────────────────────────────────────

@dataclass
class Tile:
    """One orthophoto tile and where its top-left pixel sits on the ground."""

    data: np.ndarray        # (bands, H, W) uint8
    x0: float
    y0: float
    dx: float
    dy: float
    name: str = ""


def has_gdal_jp2() -> bool:
    """Whether this GDAL build can decode JP2.

    Asked through rasterio, not ``osgeo``: rasterio ships its own GDAL and the
    ``osgeo`` bindings are not installed in ``torch_geo_env``.
    """
    import rasterio

    with rasterio.Env() as env:
        return "JP2OpenJPEG" in env.drivers()


def has_pillow_jp2() -> bool:
    """Whether Pillow was built with OpenJPEG."""
    try:
        from PIL import features

        return bool(features.check("jpg_2000"))
    except Exception:                                 # noqa: BLE001
        return False


def has_jp2_support() -> bool:
    return has_gdal_jp2() or has_pillow_jp2()


def ensure_jp2_support():
    """Fail early and actionably if nothing here can decode JP2."""
    if not has_jp2_support():
        raise RuntimeError(
            "Neither GDAL nor Pillow can decode JP2 in this environment.\n"
            "Install either:  conda install -c conda-forge "
            "libgdal-jp2openjpeg\n"
            "             or: conda install -c conda-forge pillow openjpeg"
        )


# ─── world files ──────────────────────────────────────────────────────────────

WORLD_SUFFIXES = {".jp2": ".j2w", ".tif": ".tfw", ".tiff": ".tfw"}


def split_vsizip(path):
    """('/abs/archive.zip', 'member') for a /vsizip/ URI, else (None, None)."""
    s = str(path)
    if not s.startswith("/vsizip/"):
        return None, None
    rest = s[len("/vsizip/"):]
    low = rest.lower()
    i = low.rfind(".zip/")
    if i < 0:
        return None, None
    return rest[:i + 4], rest[i + 5:]


def read_world_file(path):
    """Affine terms (dx, dy, x_centre_ul, y_centre_ul) from the sidecar.

    World files carry the *centre* of the upper-left pixel, not its corner —
    the half-pixel shift matters at 0.5 ft and is applied by the caller.
    Returns None when there is no sidecar.
    """
    s = str(path)
    stem, suffix = os.path.splitext(s)
    world_suffix = WORLD_SUFFIXES.get(suffix.lower(), ".wld")

    archive, member = split_vsizip(path)
    text = None
    if archive:
        import zipfile

        m_stem = os.path.splitext(member)[0]
        with zipfile.ZipFile(archive) as zf:
            for cand in (m_stem + world_suffix, m_stem + ".wld"):
                try:
                    text = zf.read(cand).decode()
                    break
                except KeyError:
                    continue
    else:
        for cand in (stem + world_suffix, stem + ".wld"):
            if os.path.exists(cand):
                with open(cand) as fh:
                    text = fh.read()
                break
    if text is None:
        return None

    vals = [float(v) for v in text.split()]
    if len(vals) != 6:
        raise ValueError(f"{tile_name(path)}: malformed world file")
    a, d, b, e, c, f = vals          # world-file order: A D B E C F
    if b or d:
        raise ValueError(f"{tile_name(path)}: rotated world file")
    return a, e, c, f


def read_tile_pillow(path, n_bands: int = 4) -> Tile:
    """Decode a JP2 with Pillow and georeference it from the world file.

    The fallback for environments without GDAL's JP2OpenJPEG plugin. Verified
    against the real stores: three Bronx tiles decoded this way reproduce
    nyc_2022's existing pixels exactly, all four bands, 100% of 25M pixels
    each — so band order (R, G, B, NIR) and the half-pixel world-file
    convention are confirmed, not assumed.
    """
    import io

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    name = tile_name(path)

    archive, member = split_vsizip(path)
    if archive:
        import zipfile

        with zipfile.ZipFile(archive) as zf:
            raw = zf.read(member)
        img = Image.open(io.BytesIO(raw))
    else:
        img = Image.open(str(path))

    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < n_bands:
        raise ValueError(
            f"{name}: decoded {arr.shape}, need at least {n_bands} bands")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name}: dtype {arr.dtype}, expected uint8")
    data = np.ascontiguousarray(np.transpose(arr[:, :, :n_bands], (2, 0, 1)))

    world = read_world_file(path)
    if world is None:
        raise ValueError(f"{name}: no world file, cannot georeference")
    dx, dy, cx, cy = world
    return Tile(data=data, x0=cx - dx / 2, y0=cy - dy / 2, dx=dx, dy=dy,
                name=name)


def tile_name(path) -> str:
    """Display name for a path that may be a /vsizip/ URI.

    ``Path()`` must not touch these: it collapses the ``//`` that separates
    the vsizip prefix from the archive's absolute path, silently producing a
    path that does not exist.
    """
    return os.path.basename(str(path).rstrip("/"))


def read_tile(path, n_bands: int = 4, reader: str = "auto") -> Tile:
    """Read a georeferenced tile as (bands, H, W) uint8.

    ``auto`` prefers rasterio, which reads the sidecar world file itself —
    including inside a zip, where the .j2w sits next to the .jp2 in the same
    archive — and falls back to Pillow for JP2 when GDAL lacks the plugin.
    A tile whose world file is missing is rejected by ``tile_placement`` (or
    directly by the Pillow path) rather than silently written to the wrong
    place.
    """
    if reader == "pillow":
        return read_tile_pillow(path, n_bands=n_bands)
    if reader == "auto" and str(path).lower().endswith(".jp2") \
            and not has_gdal_jp2():
        return read_tile_pillow(path, n_bands=n_bands)

    import rasterio

    name = tile_name(path)
    with rasterio.open(str(path)) as src:
        if src.count < n_bands:
            raise ValueError(
                f"{name}: {src.count} bands, need at least {n_bands}")
        data = src.read(list(range(1, n_bands + 1)))
        t = src.transform
        if t.b or t.d:
            raise ValueError(f"{name}: rotated transform is not supported")
        if data.dtype != np.uint8:
            raise ValueError(f"{name}: dtype {data.dtype}, expected uint8")
        return Tile(data=data, x0=t.c, y0=t.f, dx=t.a, dy=t.e, name=name)


def tile_placement(tile: Tile, grid: GridSpec, tol_ft: float = ALIGN_TOL_FT):
    """(row, col) of the tile's top-left pixel in the store, or raise.

    Validates pixel size and grid alignment before returning an index, so a
    tile from a different resolution or a shifted grid cannot be written into
    the wrong pixels — which would be invisible afterwards.
    """
    if abs(tile.dx - grid.dx) > 1e-6 or abs(tile.dy - grid.dy) > 1e-6:
        raise ValueError(
            f"{tile.name}: pixel size ({tile.dx}, {tile.dy}) does not match "
            f"the store's ({grid.dx}, {grid.dy})")

    col = (tile.x0 - grid.x0) / grid.dx
    row = (tile.y0 - grid.y0) / grid.dy
    for axis, v, step in (("x", col, abs(grid.dx)), ("y", row, abs(grid.dy))):
        if abs(v - round(v)) * step > tol_ft:
            raise ValueError(
                f"{tile.name}: not aligned to the store grid in {axis} "
                f"(offset {v:.4f} px)")
    row, col = int(round(row)), int(round(col))

    h, w = tile.data.shape[1], tile.data.shape[2]
    if row < 0 or col < 0 or row + h > grid.n_y or col + w > grid.n_x:
        raise ValueError(
            f"{tile.name}: lands outside the store at rows {row}:{row + h}, "
            f"cols {col}:{col + w} (store is {grid.n_y} x {grid.n_x}). "
            f"A store cropped away from the canonical origin needs `rebuild`, "
            f"not `append`.")
    return row, col


RASTER_SUFFIXES = (".jp2", ".tif", ".tiff")


def find_tiles(tiles_path, suffixes=RASTER_SUFFIXES):
    """Tile paths under a directory, or inside a .zip, as strings.

    Reading straight out of the archive is deliberate: the hole this repairs
    was created by a botched unzip-and-move (``move_images`` in
    ``notebooks/Create tifs.ipynb``), so removing that step removes the
    failure mode rather than inviting a repeat of it. GDAL's ``/vsizip/``
    handler reads the .jp2 and its sidecar .j2w from inside the zip.
    """
    tiles_path = Path(tiles_path)
    if not tiles_path.exists():
        raise FileNotFoundError(f"tiles path not found: {tiles_path}")

    if tiles_path.is_file():
        if tiles_path.suffix.lower() != ".zip":
            raise ValueError(
                f"{tiles_path} is a file but not a .zip — pass a directory of "
                f"tiles or a zip archive")
        import zipfile

        archive = tiles_path.resolve()
        with zipfile.ZipFile(archive) as zf:
            members = [n for n in zf.namelist()
                       if os.path.splitext(n)[1].lower() in suffixes]
        # "/vsizip/" + absolute archive path + "/" + member
        return sorted(f"/vsizip/{archive}/{m}" for m in members)

    out = set()
    for p in tiles_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.add(str(p))
    return sorted(out)


# ─── writing ──────────────────────────────────────────────────────────────────

def apply_quantization(data: np.ndarray, mask: int = QUANTIZATION_MASK):
    """Match the store's existing quantization (low two bits cleared)."""
    return data & np.uint8(mask)


def open_value_array(store_path, mode: str = "r+"):
    """The ``value`` array of a store, honouring ``write_empty_chunks=False``.

    That flag is why absent chunks are meaningful here: a genuinely blank tile
    stays absent instead of materializing a file full of zeros that the
    coverage probe would then count as imagery.
    """
    import zarr

    return zarr.open_array(str(Path(store_path) / "value"), mode=mode,
                           write_empty_chunks=False)


def append_tiles(store_path, tile_paths, grid: GridSpec = None,
                 mask_low_bits: bool = True, dry_run: bool = False,
                 n_bands: int = 4, verbose: bool = True) -> dict:
    """Write tiles into an existing store, touching only the chunks they cover."""
    store_path = Path(store_path)
    grid = grid or grid_from_store(store_path)
    # Chunk geometry comes from the store, never assumed: the cell count this
    # reports is what the coverage probe will later compare against.
    _, chunk_rows, chunk_cols = read_zarray_meta(store_path)["chunks"]
    arr = None if dry_run else open_value_array(store_path, mode="r+")

    stats = {"written": 0, "failed": 0, "errors": [], "cells": set()}
    for i, path in enumerate(tile_paths, 1):
        try:
            tile = read_tile(path, n_bands=n_bands)
            row, col = tile_placement(tile, grid)
            h, w = tile.data.shape[1], tile.data.shape[2]

            for cr in range(row // chunk_rows, (row + h - 1) // chunk_rows + 1):
                for cc in range(col // chunk_cols, (col + w - 1) // chunk_cols + 1):
                    stats["cells"].add((cr, cc))

            if not dry_run:
                data = (apply_quantization(tile.data) if mask_low_bits
                        else tile.data)
                arr[:n_bands, row:row + h, col:col + w] = data
            stats["written"] += 1
            if verbose and (i % 25 == 0 or i == len(tile_paths)):
                print(f"    {i}/{len(tile_paths)} tiles "
                      f"({'planned' if dry_run else 'written'})")
        except Exception as exc:                      # noqa: BLE001
            stats["failed"] += 1
            stats["errors"].append(f"{tile_name(path)}: {exc}")
            if verbose:
                print(f"    !! {tile_name(path)}: {exc}")

    stats["chunk_cells"] = len(stats["cells"])
    return stats


def verify_against_store(store_path, tile_paths, grid: GridSpec = None,
                         mask_low_bits: bool = True, n_bands: int = 4,
                         verbose: bool = True) -> dict:
    """Compare tiles against imagery the store *already* holds.

    A pre-flight check with real ground truth. ``boro_bronx_sp22.zip`` carries
    all 287 Bronx tiles but only 191 are missing from the store, so 96 of them
    can be re-read and checked against what is already there. If band order,
    quantization, or placement were wrong, those tiles would not match — and
    that is exactly what would otherwise corrupt good data silently.

    Tiles whose window is empty are the holes being repaired and are reported
    as such, not as failures.
    """
    store_path = Path(store_path)
    grid = grid or grid_from_store(store_path)
    arr = open_value_array(store_path, mode="r")

    out = {"match": 0, "mismatch": 0, "hole": 0, "failed": 0,
           "details": [], "errors": []}
    for path in tile_paths:
        try:
            tile = read_tile(path, n_bands=n_bands)
            row, col = tile_placement(tile, grid)
            h, w = tile.data.shape[1], tile.data.shape[2]
            existing = arr[:n_bands, row:row + h, col:col + w]
            expected = (apply_quantization(tile.data) if mask_low_bits
                        else tile.data)

            if not existing.any():
                out["hole"] += 1
                out["details"].append((tile.name, "hole", None))
                continue
            share = float((existing == expected).mean())
            key = "match" if share > 0.999 else "mismatch"
            out[key] += 1
            out["details"].append((tile.name, key, share))
            if verbose and key == "mismatch":
                print(f"    MISMATCH {tile.name}: {100 * share:.2f}% of "
                      f"pixels agree with the store")
        except Exception as exc:                      # noqa: BLE001
            out["failed"] += 1
            out["errors"].append(f"{tile_name(path)}: {exc}")
    return out


# ─── rebuilding a store on the canonical grid ─────────────────────────────────

def read_zarray_meta(store_path, array_name: str = "value") -> dict:
    with open(Path(store_path) / array_name / ".zarray") as fh:
        return json.load(fh)


def create_store_like(src_path, dst_path, grid: GridSpec,
                      crs_wkt: str = None) -> Path:
    """An empty store on ``grid`` with ``src``'s exact chunking and codec.

    Byte-identical encoding is what makes ``relocate_chunks`` valid: a chunk
    blob does not record its own index, so a chunk written by the source store
    is a legal chunk of the destination store at any key.
    """
    import zarr
    from numcodecs import get_codec

    src_path, dst_path = Path(src_path), Path(dst_path)
    meta = read_zarray_meta(src_path)
    if dst_path.exists():
        raise FileExistsError(f"{dst_path} already exists — refusing to clobber")

    n_bands = meta["shape"][0]
    compressor = get_codec(meta["compressor"]) if meta.get("compressor") else None
    filters = [get_codec(f) for f in meta["filters"]] if meta.get("filters") else None

    root = zarr.open_group(str(dst_path), mode="w")
    value = root.create_dataset(
        "value", shape=(n_bands, grid.n_y, grid.n_x),
        chunks=tuple(meta["chunks"]), dtype=meta["dtype"],
        compressor=compressor, filters=filters,
        fill_value=meta.get("fill_value", 0),
        dimension_separator=meta.get("dimension_separator", "."),
        write_empty_chunks=False,
    )
    value.attrs["_ARRAY_DIMENSIONS"] = ["band", "y", "x"]

    xs = root.create_dataset("x", data=grid.x_coords(), chunks=(grid.n_x,),
                             dtype="f8")
    xs.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    ys = root.create_dataset("y", data=grid.y_coords(), chunks=(grid.n_y,),
                             dtype="f8")
    ys.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    bands = root.create_dataset("band", data=np.arange(1, n_bands + 1),
                                chunks=(n_bands,), dtype="i8")
    bands.attrs["_ARRAY_DIMENSIONS"] = ["band"]

    # spatial_ref: keep the source's CRS, but write a GeoTransform that is
    # actually true of this store — the source's is wrong by 90,000 ft.
    sref_attrs = {}
    src_sref = src_path / "spatial_ref" / ".zattrs"
    if src_sref.exists():
        with open(src_sref) as fh:
            sref_attrs = json.load(fh)
    if crs_wkt:
        sref_attrs["crs_wkt"] = crs_wkt
        sref_attrs["spatial_ref"] = crs_wkt
    sref_attrs["GeoTransform"] = grid.geotransform()
    sref_attrs["_ARRAY_DIMENSIONS"] = []

    sref = root.create_dataset("spatial_ref", shape=(), dtype="i8")
    sref[...] = 0
    for k, v in sref_attrs.items():
        sref.attrs[k] = v

    zarr.consolidate_metadata(str(dst_path))
    return dst_path


def relocate_chunks(src_path, dst_path, row_offset: int, col_offset: int,
                    mode: str = "hardlink", array_name: str = "value",
                    verbose: bool = True) -> int:
    """Re-key every chunk of ``src`` into ``dst``, shifted by the offsets.

    ``hardlink`` is the default: it costs no space and no time, and it leaves
    the source store fully intact as a rollback. It is safe against later
    writes because zarr's DirectoryStore writes a chunk to a temp file and
    renames it over the target, which breaks the link rather than editing the
    shared inode.
    """
    src_dir = Path(src_path) / array_name
    dst_dir = Path(dst_path) / array_name
    dst_meta = read_zarray_meta(dst_path, array_name)
    n_rows = -(-dst_meta["shape"][1] // dst_meta["chunks"][1])
    n_cols = -(-dst_meta["shape"][2] // dst_meta["chunks"][2])

    moved = 0
    for key in sorted(os.listdir(src_dir)):
        if key.startswith("."):
            continue
        parts = key.split(".")
        if len(parts) != 3:
            continue
        try:
            b, r, c = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        nr, nc = r + row_offset, c + col_offset
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            raise ValueError(
                f"chunk {key} shifts to ({nr}, {nc}), outside the "
                f"{n_rows} x {n_cols} destination grid")

        target = dst_dir / f"{b}.{nr}.{nc}"
        source = src_dir / key
        if target.exists():
            raise FileExistsError(f"{target} already exists")
        if mode == "hardlink":
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)      # different filesystem
        elif mode == "copy":
            shutil.copy2(source, target)
        elif mode == "move":
            shutil.move(str(source), str(target))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        moved += 1
        if verbose and moved % 1000 == 0:
            print(f"    {moved} chunks relocated")
    return moved


def canonical_offset(src_grid: GridSpec, dst_grid: GridSpec,
                     chunk_rows: int, chunk_cols: int) -> tuple:
    """Chunk-grid offset of ``src`` inside ``dst``; must be whole chunks.

    Whole chunks is the precondition for relinking: a chunk blob can be
    re-keyed only if its pixels land exactly on a destination chunk boundary.
    A sub-chunk offset means the data must be decompressed and re-mosaicked.
    """
    col_px = (src_grid.x0 - dst_grid.x0) / dst_grid.dx
    row_px = (src_grid.y0 - dst_grid.y0) / dst_grid.dy
    for axis, v, chunk in (("x", col_px, chunk_cols),
                           ("y", row_px, chunk_rows)):
        if abs(v - round(v)) * abs(dst_grid.dx) > ALIGN_TOL_FT:
            raise ValueError(f"origin is not pixel-aligned in {axis}")
        if round(v) % chunk:
            raise ValueError(
                f"origin is {round(v)} px from the destination in {axis}, "
                f"not a whole number of {chunk}-px chunks — chunk files "
                f"cannot be relinked and the store must be re-mosaicked")
    return int(round(row_px)) // chunk_rows, int(round(col_px)) // chunk_cols


def rebuild_store(src_path, dst_path, tile_paths=(), grid: GridSpec = None,
                  mode: str = "hardlink", mask_low_bits: bool = True,
                  dry_run: bool = False, n_bands: int = 4,
                  verbose: bool = True) -> dict:
    """Full-extent copy of a cropped store, plus the tiles it was missing."""
    src_path, dst_path = Path(src_path), Path(dst_path)
    grid = grid or NYC_GRID
    src_grid = grid_from_store(src_path)
    src_meta = read_zarray_meta(src_path)
    _, chunk_rows, chunk_cols = src_meta["chunks"]
    row_off, col_off = canonical_offset(src_grid, grid, chunk_rows, chunk_cols)

    n_src_chunks = sum(1 for k in os.listdir(src_path / "value")
                       if not k.startswith("."))
    summary = {
        "src_grid": src_grid, "dst_grid": grid,
        "chunk_offset": (row_off, col_off),
        "src_chunks": n_src_chunks, "relocated": 0,
    }
    if verbose:
        print(f"  source origin ({src_grid.x0:.1f}, {src_grid.y0:.1f}), "
              f"{src_meta['shape'][1]} x {src_meta['shape'][2]} px")
        print(f"  target origin ({grid.x0:.1f}, {grid.y0:.1f}), "
              f"{grid.n_y} x {grid.n_x} px")
        print(f"  chunk offset: row {row_off}, col {col_off}")
        print(f"  {n_src_chunks} existing chunks to relocate ({mode})")
    if dry_run:
        summary["tiles"] = {"written": 0, "failed": 0, "errors": [],
                            "chunk_cells": 0}
        return summary

    create_store_like(src_path, dst_path, grid)
    summary["relocated"] = relocate_chunks(src_path, dst_path, row_off,
                                           col_off, mode=mode, verbose=verbose)
    if tile_paths:
        if verbose:
            print(f"  writing {len(tile_paths)} tiles ...")
        summary["tiles"] = append_tiles(
            dst_path, tile_paths, grid=grid, mask_low_bits=mask_low_bits,
            n_bands=n_bands, verbose=verbose)
    else:
        summary["tiles"] = {"written": 0, "failed": 0, "errors": [],
                            "chunk_cells": 0}
    return summary


# ─── metadata repair ──────────────────────────────────────────────────────────

def fix_geotransform(store_path, dry_run: bool = False) -> str:
    """Rewrite a store's GeoTransform attribute to match its coordinates.

    Inert for the current pipeline (which indexes by row/col and by the
    coordinate arrays) but wrong by 90,000 ft, so anything that ever trusts it
    silently reads the wrong ground.
    """
    import zarr

    store_path = Path(store_path)
    grid = grid_from_store(store_path)
    correct = grid.geotransform()

    attrs_file = store_path / "spatial_ref" / ".zattrs"
    with open(attrs_file) as fh:
        attrs = json.load(fh)
    old = attrs.get("GeoTransform")
    if old == correct:
        return f"{store_path.name}: already correct ({correct})"
    if not dry_run:
        attrs["GeoTransform"] = correct
        with open(attrs_file, "w") as fh:
            json.dump(attrs, fh)
        if (store_path / ".zmetadata").exists():
            zarr.consolidate_metadata(str(store_path))
    verb = "would fix" if dry_run else "fixed"
    return f"{store_path.name}: {verb}  {old}  ->  {correct}"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _report(stats: dict):
    print(f"\n  tiles written: {stats['written']}, failed: {stats['failed']}")
    print(f"  chunk cells touched: {stats.get('chunk_cells', 0)}")
    for err in stats["errors"][:20]:
        print(f"    ! {err}")
    if len(stats["errors"]) > 20:
        print(f"    ... and {len(stats['errors']) - 20} more")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    ap = sub.add_parser("append", help="write tiles into an existing store")
    ap.add_argument("--store", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N tiles (smoke test)")
    ap.add_argument("--no-mask-low-bits", action="store_true",
                    help="skip the & 0xFC the rest of the store carries")
    ap.add_argument("--allow-missing-jp2", action="store_true")

    rp = sub.add_parser("rebuild", help="full-extent copy of a cropped store")
    rp.add_argument("--store", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--tiles", default=None)
    rp.add_argument("--dry-run", action="store_true")
    rp.add_argument("--limit", type=int, default=None)
    rp.add_argument("--mode", choices=("hardlink", "copy", "move"),
                    default="hardlink")
    rp.add_argument("--no-mask-low-bits", action="store_true")
    rp.add_argument("--allow-missing-jp2", action="store_true")

    vp = sub.add_parser(
        "verify", help="check tiles against imagery the store already holds")
    vp.add_argument("--store", required=True)
    vp.add_argument("--tiles", required=True)
    vp.add_argument("--limit", type=int, default=20)
    vp.add_argument("--no-mask-low-bits", action="store_true")
    vp.add_argument("--allow-missing-jp2", action="store_true")

    gp = sub.add_parser("fix-geotransform",
                        help="correct the GeoTransform attribute of stores")
    gp.add_argument("--store", required=True, nargs="+")
    gp.add_argument("--dry-run", action="store_true")

    args = p.parse_args(argv)

    if args.command == "fix-geotransform":
        for s in args.store:
            print(" ", fix_geotransform(s, dry_run=args.dry_run))
        return 0

    tiles = []
    if getattr(args, "tiles", None):
        tiles = find_tiles(args.tiles)
        if any(str(t).lower().endswith(".jp2") for t in tiles) \
                and not args.allow_missing_jp2:
            ensure_jp2_support()
        if args.limit:
            tiles = tiles[:args.limit]
        print(f"Found {len(tiles)} tiles in {args.tiles}")

    if args.command == "verify":
        res = verify_against_store(args.store, tiles,
                                   mask_low_bits=not args.no_mask_low_bits)
        print(f"\n  match: {res['match']}   mismatch: {res['mismatch']}   "
              f"already-empty (the hole): {res['hole']}   "
              f"failed: {res['failed']}")
        for err in res["errors"][:10]:
            print(f"    ! {err}")
        if res["mismatch"] or res["failed"]:
            print("\n  Tiles that disagree with imagery already in the store "
                  "mean the reader is wrong (band order, quantization, or "
                  "placement). Do NOT run `append` until this is clean.")
            return 1
        if res["match"]:
            print("\n  Reader agrees with the store on every overlapping "
                  "tile — safe to append.")
        return 0

    if args.command == "append":
        print(f"{'DRY RUN — ' if args.dry_run else ''}appending into {args.store}")
        stats = append_tiles(args.store, tiles, dry_run=args.dry_run,
                             mask_low_bits=not args.no_mask_low_bits)
        _report(stats)
        return 1 if stats["failed"] else 0

    print(f"{'DRY RUN — ' if args.dry_run else ''}rebuilding {args.store} "
          f"-> {args.out}")
    summary = rebuild_store(args.store, args.out, tiles,
                            mode=args.mode, dry_run=args.dry_run,
                            mask_low_bits=not args.no_mask_low_bits)
    print(f"\n  chunks relocated: {summary['relocated']} "
          f"of {summary['src_chunks']}")
    _report(summary["tiles"])
    if not args.dry_run:
        print(f"\n  New store: {args.out}")
        print("  Verify, then swap it in:")
        print(f"    mv {args.store} {args.store}.old && "
              f"mv {args.out} {args.store}")
        print("  Then rebuild the dataset so pixel indices are recomputed "
              "against the new grid.")
    return 1 if summary["tiles"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
