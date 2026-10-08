# -*- coding: utf-8 -*-
"""Repairing the NYC zarr coverage holes, on synthetic data.

Tiles are real GeoTIFFs written by rasterio (JP2 needs a GDAL plugin that is
not installed; the reader path is identical either way) and the stores are
real zarr v2 groups carrying the production codec — Delta + Blosc/zstd with
BITSHUFFLE — so the chunk-relinking tests exercise genuine compressed blobs
rather than a stand-in.

The scale is shrunk: 10 x 10 px chunks instead of 2500 x 2500, tiles of 2 x 2
chunks, which is the same shape as the real thing (every NYC tile is exactly
2 x 2 chunks and grid-aligned).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src import repair_zarr_coverage as rz

CHUNK = 10
PX = 0.5
X0, Y0 = 1000.0, 500.0
NB = 4
GRID = rz.GridSpec(x0=X0, y0=Y0, dx=PX, dy=-PX, n_y=4 * CHUNK, n_x=4 * CHUNK)
WKT = "EPSG:6539"


def _make_store(root, grid=GRID, cells=(), fill=None):
    """A store with the production codec, holding only ``cells``."""
    import zarr
    from numcodecs import Blosc, Delta

    root_grp = zarr.open_group(str(root), mode="w")
    value = root_grp.create_dataset(
        "value", shape=(NB, grid.n_y, grid.n_x), chunks=(NB, CHUNK, CHUNK),
        dtype="u1", compressor=Blosc(cname="zstd", clevel=5,
                                     shuffle=Blosc.BITSHUFFLE),
        filters=[Delta(dtype="u1")], fill_value=0,
        dimension_separator=".", write_empty_chunks=False,
    )
    value.attrs["_ARRAY_DIMENSIONS"] = ["band", "y", "x"]
    xs = root_grp.create_dataset("x", data=grid.x_coords(), chunks=(grid.n_x,))
    xs.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    ys = root_grp.create_dataset("y", data=grid.y_coords(), chunks=(grid.n_y,))
    ys.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    bs = root_grp.create_dataset("band", data=np.arange(1, NB + 1),
                                 chunks=(NB,))
    bs.attrs["_ARRAY_DIMENSIONS"] = ["band"]
    sref = root_grp.create_dataset("spatial_ref", shape=(), dtype="i8")
    sref[...] = 0
    sref.attrs["_ARRAY_DIMENSIONS"] = []
    sref.attrs["crs_wkt"] = WKT
    sref.attrs["GeoTransform"] = "999999.0 0.5 0.0 500.0 0.0 -0.5"  # wrong

    for (r, c) in cells:
        v = fill if fill is not None else 100 + r * 10 + c
        value[:, r * CHUNK:(r + 1) * CHUNK, c * CHUNK:(c + 1) * CHUNK] = v
    zarr.consolidate_metadata(str(root))
    return root


def _write_tile(path, x0, y0, data, px=PX, crs=WKT):
    """A 2x2-chunk GeoTIFF at the given upper-left ground coordinate."""
    import rasterio
    from rasterio.transform import from_origin

    bands, h, w = data.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=bands,
        dtype="uint8", crs=crs, transform=from_origin(x0, y0, px, px),
    ) as dst:
        dst.write(data)
    return path


def _tile_data(value=201, bands=NB, size=2 * CHUNK):
    return np.full((bands, size, size), value, dtype=np.uint8)


def _read(store, sl=None):
    arr = rz.open_value_array(store, mode="r")
    return arr[:] if sl is None else arr[sl]


# ─── grid ─────────────────────────────────────────────────────────────────────

def test_canonical_grid_reproduces_the_real_coordinate_arrays():
    """NYC_GRID must generate nyc_2014's coords exactly, or every index the
    repair computes is off."""
    g = rz.NYC_GRID
    assert g.x_coords()[0] == pytest.approx(907500.25)
    assert g.y_coords()[0] == pytest.approx(277499.75)
    assert g.x_coords()[-1] == pytest.approx(1069999.75)
    assert g.y_coords()[-1] == pytest.approx(117500.25)
    assert g.geotransform() == "907500.0 0.5 0.0 277500.0 0.0 -0.5"


def test_grid_from_store_reads_coordinates_not_the_attribute(tmp_path):
    """The fixture's GeoTransform says x0=999999; the coords say 1000."""
    store = _make_store(tmp_path / "s")
    g = rz.grid_from_store(store)
    assert g.x0 == pytest.approx(X0)
    assert g.y0 == pytest.approx(Y0)
    assert (g.n_y, g.n_x) == (GRID.n_y, GRID.n_x)


def test_snap_removes_the_real_rounding_noise():
    """nyc_2022 reads back as x0=907499.95, dx=0.5000003 because the mosaic
    rounded every coordinate to one decimal. The true grid is exact."""
    noisy = rz.GridSpec(x0=907499.949999846, y0=277500.0500001562,
                        dx=0.5000003076932548, dy=-0.5000003125009765,
                        n_y=320000, n_x=325000)
    snapped = rz.snap_grid(noisy)
    assert snapped.x0 == pytest.approx(907500.0)
    assert snapped.y0 == pytest.approx(277500.0)
    assert snapped.dx == 0.5 and snapped.dy == -0.5
    assert snapped.geotransform() == rz.NYC_GRID.geotransform()


def test_snap_leaves_a_genuinely_offset_grid_alone():
    """A half-pixel shift is real, not noise; snapping it would invent an
    alignment the data does not have."""
    offset = rz.GridSpec(x0=1000.3, y0=500.0, dx=0.5, dy=-0.5, n_y=10, n_x=10)
    assert rz.snap_grid(offset, tol_ft=0.1).x0 == pytest.approx(1000.3)


def test_snap_preserves_a_genuinely_different_resolution():
    """Snapping cleans noise at millifoot precision; it must not drag a real
    quarter-foot grid onto the half-foot one."""
    quarter = rz.GridSpec(x0=1000.0, y0=500.0, dx=0.25, dy=-0.25,
                          n_y=10, n_x=10)
    snapped = rz.snap_grid(quarter)
    assert snapped.dx == pytest.approx(0.25)
    assert snapped.x0 == pytest.approx(1000.0)


def test_snap_can_be_disabled(tmp_path):
    store = _make_store(tmp_path / "s")
    assert rz.grid_from_store(store, snap=False).x0 == pytest.approx(X0)


# ─── placement ────────────────────────────────────────────────────────────────

def test_tile_lands_at_the_computed_index(tmp_path):
    p = _write_tile(tmp_path / "t.tif", X0 + 2 * CHUNK * PX,
                    Y0 - 1 * CHUNK * PX, _tile_data())
    assert rz.tile_placement(rz.read_tile(p), GRID) == (CHUNK, 2 * CHUNK)


def test_tile_at_the_origin_is_index_zero(tmp_path):
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data())
    assert rz.tile_placement(rz.read_tile(p), GRID) == (0, 0)


def test_misaligned_tile_is_rejected(tmp_path):
    """Half-pixel offset: writing it would smear every pixel by half a cell
    and leave no trace."""
    p = _write_tile(tmp_path / "t.tif", X0 + 0.25, Y0, _tile_data())
    with pytest.raises(ValueError, match="not aligned"):
        rz.tile_placement(rz.read_tile(p), GRID)


def test_wrong_resolution_is_rejected(tmp_path):
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data(), px=1.0)
    with pytest.raises(ValueError, match="pixel size"):
        rz.tile_placement(rz.read_tile(p), GRID)


def test_tile_outside_the_store_is_rejected_with_a_pointer_to_rebuild(tmp_path):
    """Exactly the Staten Island 2024 case: west of a cropped store's origin."""
    p = _write_tile(tmp_path / "t.tif", X0 - 10 * CHUNK * PX, Y0, _tile_data())
    with pytest.raises(ValueError, match="rebuild"):
        rz.tile_placement(rz.read_tile(p), GRID)


def test_tile_with_too_few_bands_is_rejected(tmp_path):
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data(bands=3))
    with pytest.raises(ValueError, match="bands"):
        rz.read_tile(p)


def test_find_tiles_is_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    _write_tile(tmp_path / "a.tif", X0, Y0, _tile_data())
    _write_tile(tmp_path / "sub" / "b.tif", X0, Y0, _tile_data())
    assert len(rz.find_tiles(tmp_path)) == 2


def test_find_tiles_ignores_non_raster_files(tmp_path):
    _write_tile(tmp_path / "a.tif", X0, Y0, _tile_data())
    (tmp_path / "a.j2w").write_text("0.5\n0\n0\n-0.5\n1000\n500\n")
    (tmp_path / "index.shp").write_bytes(b"")
    assert len(rz.find_tiles(tmp_path)) == 1


def test_find_tiles_errors_on_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        rz.find_tiles(tmp_path / "nope")


def test_find_tiles_rejects_a_non_zip_file(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hi")
    with pytest.raises(ValueError, match="not a .zip"):
        rz.find_tiles(p)


# ─── reading straight out of a zip ────────────────────────────────────────────

def _zip_of(tmp_path, entries):
    """A flat zip, like the boro_*.zip archives (no subdirectories)."""
    import zipfile

    z = tmp_path / "tiles.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return z


def test_find_tiles_lists_rasters_inside_a_zip(tmp_path):
    src = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data())
    z = _zip_of(tmp_path, {"000262.tif": src.read_bytes(),
                           "000262.j2w": b"0.5\n0\n0\n-0.5\n1000\n500\n",
                           "index.shp": b""})
    found = rz.find_tiles(z)
    assert len(found) == 1
    assert found[0] == f"/vsizip/{z.resolve()}/000262.tif"


def test_vsizip_path_survives_name_extraction():
    """Path() would collapse the '//' and produce a path that does not exist."""
    uri = "/vsizip//home/abbatenicolas/data/boro_bronx_sp22.zip/000262.jp2"
    assert rz.tile_name(uri) == "000262.jp2"
    from pathlib import Path as _P
    assert str(_P(uri)) != uri          # the trap this avoids


def test_tile_reads_from_inside_a_zip(tmp_path):
    """The end-to-end zip path: locate, read, place."""
    src = _write_tile(tmp_path / "t.tif", X0 + 2 * CHUNK * PX, Y0,
                      _tile_data(value=144))
    z = _zip_of(tmp_path, {"t.tif": src.read_bytes()})
    (path,) = rz.find_tiles(z)
    tile = rz.read_tile(path)
    assert tile.name == "t.tif"
    assert rz.tile_placement(tile, GRID) == (0, 2 * CHUNK)


def test_sidecar_world_file_inside_a_zip_is_honoured(tmp_path):
    """JP2 georeferencing lives in a sibling .j2w inside the same archive.
    A GeoTIFF stripped of internal georeferencing plus a .tfw exercises the
    identical GDAL sidecar-discovery path."""
    import rasterio

    plain = tmp_path / "p.tif"
    data = _tile_data(value=99)
    with rasterio.open(plain, "w", driver="GTiff", height=data.shape[1],
                       width=data.shape[2], count=NB, dtype="uint8") as dst:
        dst.write(data)                      # no crs, no transform

    x_ul, y_ul = X0 + 2 * CHUNK * PX, Y0
    tfw = f"{PX}\n0.0\n0.0\n{-PX}\n{x_ul + PX / 2}\n{y_ul - PX / 2}\n"
    z = _zip_of(tmp_path, {"p.tif": plain.read_bytes(), "p.tfw": tfw.encode()})

    (path,) = rz.find_tiles(z)
    tile = rz.read_tile(path)
    assert rz.tile_placement(tile, GRID) == (0, 2 * CHUNK)


def test_append_works_directly_from_a_zip(tmp_path):
    store = _make_store(tmp_path / "s")
    src = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data(value=180))
    z = _zip_of(tmp_path, {"t.tif": src.read_bytes()})
    stats = rz.append_tiles(store, rz.find_tiles(z), verbose=False)
    assert stats["written"] == 1 and stats["failed"] == 0
    assert (_read(store, np.s_[:, :2 * CHUNK, :2 * CHUNK]) == 180).all()


# ─── JP2 availability ─────────────────────────────────────────────────────────

def test_jp2_check_reports_the_install_command(monkeypatch):
    monkeypatch.setattr(rz, "has_gdal_jp2", lambda: False)
    monkeypatch.setattr(rz, "has_pillow_jp2", lambda: False)
    with pytest.raises(RuntimeError, match="libgdal-jp2openjpeg"):
        rz.ensure_jp2_support()


def test_pillow_alone_satisfies_the_jp2_requirement(monkeypatch):
    """This environment has no GDAL JP2 driver but Pillow does have one, so
    the repair must not demand an install it does not need."""
    monkeypatch.setattr(rz, "has_gdal_jp2", lambda: False)
    monkeypatch.setattr(rz, "has_pillow_jp2", lambda: True)
    assert rz.has_jp2_support()
    rz.ensure_jp2_support()


def test_has_jp2_support_does_not_need_osgeo():
    """osgeo is not installed in torch_geo_env; the check must go through
    rasterio's own GDAL or it raises ModuleNotFoundError instead of a
    useful message."""
    assert isinstance(rz.has_gdal_jp2(), bool)
    assert isinstance(rz.has_jp2_support(), bool)


# ─── world files and the Pillow fallback ──────────────────────────────────────

def test_split_vsizip_recovers_archive_and_member():
    uri = "/vsizip//home/abbatenicolas/data/boro_bronx_sp22.zip/000262.jp2"
    archive, member = rz.split_vsizip(uri)
    assert archive == "/home/abbatenicolas/data/boro_bronx_sp22.zip"
    assert member == "000262.jp2"


def test_split_vsizip_ignores_a_plain_path():
    assert rz.split_vsizip("/data/tiles/000262.jp2") == (None, None)


def test_world_file_is_read_from_a_plain_sidecar(tmp_path):
    p = tmp_path / "t.jp2"
    p.write_bytes(b"")
    (tmp_path / "t.j2w").write_text("0.5\n0.0\n0.0\n-0.5\n1000.25\n499.75\n")
    dx, dy, cx, cy = rz.read_world_file(p)
    assert (dx, dy) == (0.5, -0.5)
    assert (cx, cy) == (1000.25, 499.75)


def test_world_file_is_read_from_inside_a_zip(tmp_path):
    z = _zip_of(tmp_path, {
        "000262.jp2": b"",
        "000262.j2w": b"0.5\n0.0\n0.0\n-0.5\n1000.25\n499.75\n"})
    dx, dy, cx, cy = rz.read_world_file(f"/vsizip/{z.resolve()}/000262.jp2")
    assert (dx, dy, cx, cy) == (0.5, -0.5, 1000.25, 499.75)


def test_missing_world_file_returns_none(tmp_path):
    p = tmp_path / "t.jp2"
    p.write_bytes(b"")
    assert rz.read_world_file(p) is None


def test_rotated_world_file_is_rejected(tmp_path):
    p = tmp_path / "t.jp2"
    p.write_bytes(b"")
    (tmp_path / "t.j2w").write_text("0.5\n0.1\n0.1\n-0.5\n1000.0\n500.0\n")
    with pytest.raises(ValueError, match="rotated"):
        rz.read_world_file(p)


def test_malformed_world_file_is_rejected(tmp_path):
    p = tmp_path / "t.jp2"
    p.write_bytes(b"")
    (tmp_path / "t.j2w").write_text("0.5\n0.0\n")
    with pytest.raises(ValueError, match="malformed"):
        rz.read_world_file(p)


def test_pillow_reader_places_a_tile_like_rasterio_does(tmp_path):
    """Same tile, both readers, same answer — the half-pixel world-file
    convention is where these would diverge."""
    x_ul, y_ul = X0 + 2 * CHUNK * PX, Y0
    data = np.random.default_rng(7).integers(
        1, 256, (NB, 2 * CHUNK, 2 * CHUNK), dtype=np.uint8)
    p = _write_tile(tmp_path / "t.tif", x_ul, y_ul, data)
    (tmp_path / "t.tfw").write_text(
        f"{PX}\n0.0\n0.0\n{-PX}\n{x_ul + PX / 2}\n{y_ul - PX / 2}\n")

    via_rio = rz.read_tile(p)
    via_pil = rz.read_tile(p, reader="pillow")
    assert rz.tile_placement(via_pil, GRID) == rz.tile_placement(via_rio, GRID)
    np.testing.assert_array_equal(via_pil.data, via_rio.data)


def test_pillow_reader_needs_a_world_file(tmp_path):
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data())
    with pytest.raises(ValueError, match="no world file"):
        rz.read_tile(p, reader="pillow")


# ─── quantization ─────────────────────────────────────────────────────────────

def test_quantization_matches_the_store():
    a = np.array([[[0, 1, 2, 3, 253, 254, 255]]], dtype=np.uint8)
    out = rz.apply_quantization(a)
    assert out.tolist() == [[[0, 0, 0, 0, 252, 252, 252]]]
    assert not (out & 0x03).any()
    assert out.dtype == np.uint8


def test_appended_pixels_carry_the_store_quantization(tmp_path):
    store = _make_store(tmp_path / "s")
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data(value=203))
    rz.append_tiles(store, [p], verbose=False)
    got = _read(store, np.s_[:, :2 * CHUNK, :2 * CHUNK])
    assert (got == 200).all()                 # 203 & 0xFC
    assert not (got & 0x03).any()


def test_masking_can_be_switched_off(tmp_path):
    store = _make_store(tmp_path / "s")
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data(value=203))
    rz.append_tiles(store, [p], mask_low_bits=False, verbose=False)
    assert (_read(store, np.s_[:, 0, 0]) == 203).all()


# ─── append ───────────────────────────────────────────────────────────────────

def test_append_fills_a_hole_and_leaves_the_rest_untouched(tmp_path):
    """The 2022 Bronx repair in miniature."""
    existing = [(0, 0), (0, 1)]
    store = _make_store(tmp_path / "s", cells=existing, fill=64)
    before = _read(store, np.s_[:, :CHUNK, :CHUNK]).copy()

    p = _write_tile(tmp_path / "t.tif", X0 + 2 * CHUNK * PX, Y0,
                    _tile_data(value=200))
    stats = rz.append_tiles(store, [p], verbose=False)

    assert stats["written"] == 1 and stats["failed"] == 0
    assert stats["chunk_cells"] == 4                    # 2 x 2 chunks
    assert (_read(store, np.s_[:, :2 * CHUNK, 2 * CHUNK:4 * CHUNK]) == 200).all()
    np.testing.assert_array_equal(_read(store, np.s_[:, :CHUNK, :CHUNK]), before)


def test_append_creates_the_missing_chunk_files(tmp_path):
    """Coverage is measured by chunk-file presence, so the repair has to
    actually materialize them."""
    store = _make_store(tmp_path / "s")
    keys = lambda: {k for k in (store / "value").iterdir()
                    if not k.name.startswith(".")}
    assert not keys()
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data())
    rz.append_tiles(store, [p], verbose=False)
    assert {k.name for k in keys()} == {"0.0.0", "0.0.1", "0.1.0", "0.1.1"}


def test_an_all_zero_tile_stays_absent(tmp_path):
    """write_empty_chunks=False: a blank tile must not masquerade as imagery."""
    store = _make_store(tmp_path / "s")
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data(value=0))
    rz.append_tiles(store, [p], verbose=False)
    assert not [k for k in (store / "value").iterdir()
                if not k.name.startswith(".")]


def test_dry_run_writes_nothing_but_still_reports(tmp_path):
    store = _make_store(tmp_path / "s")
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data())
    stats = rz.append_tiles(store, [p], dry_run=True, verbose=False)
    assert stats["written"] == 1 and stats["chunk_cells"] == 4
    assert not [k for k in (store / "value").iterdir()
                if not k.name.startswith(".")]


def test_a_bad_tile_is_counted_not_fatal(tmp_path):
    """One unreadable tile in 191 must not abandon the other 190."""
    store = _make_store(tmp_path / "s")
    good = _write_tile(tmp_path / "g.tif", X0, Y0, _tile_data())
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a raster")
    stats = rz.append_tiles(store, [good, bad], verbose=False)
    assert stats["written"] == 1 and stats["failed"] == 1
    assert "bad.tif" in stats["errors"][0]


def test_appended_data_survives_the_codec_roundtrip(tmp_path):
    """Delta + Blosc/BITSHUFFLE with a non-uniform payload."""
    store = _make_store(tmp_path / "s")
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, (NB, 2 * CHUNK, 2 * CHUNK), dtype=np.uint8)
    p = _write_tile(tmp_path / "t.tif", X0, Y0, data)
    rz.append_tiles(store, [p], verbose=False)
    np.testing.assert_array_equal(
        _read(store, np.s_[:, :2 * CHUNK, :2 * CHUNK]),
        rz.apply_quantization(data))


# ─── verify against existing imagery ──────────────────────────────────────────

def test_verify_confirms_a_tile_already_in_the_store(tmp_path):
    """The pre-flight check: re-reading a tile the store already holds must
    reproduce it exactly, or band order / quantization is wrong."""
    store = _make_store(tmp_path / "s")
    data = np.random.default_rng(1).integers(
        1, 256, (NB, 2 * CHUNK, 2 * CHUNK), dtype=np.uint8)
    p = _write_tile(tmp_path / "t.tif", X0, Y0, data)
    rz.append_tiles(store, [p], verbose=False)

    res = rz.verify_against_store(store, [p], verbose=False)
    assert (res["match"], res["mismatch"], res["hole"]) == (1, 0, 0)
    assert res["details"][0][2] == pytest.approx(1.0)


def test_verify_flags_a_band_order_mismatch(tmp_path):
    """If the reader swapped bands, verify must catch it rather than let
    `append` overwrite good imagery."""
    store = _make_store(tmp_path / "s")
    data = np.random.default_rng(2).integers(
        1, 256, (NB, 2 * CHUNK, 2 * CHUNK), dtype=np.uint8)
    _write_tile(tmp_path / "a.tif", X0, Y0, data)
    rz.append_tiles(store, [tmp_path / "a.tif"], verbose=False)

    swapped = _write_tile(tmp_path / "b.tif", X0, Y0, data[::-1])
    res = rz.verify_against_store(store, [swapped], verbose=False)
    assert res["mismatch"] == 1 and res["match"] == 0


def test_verify_reports_a_hole_rather_than_a_failure(tmp_path):
    store = _make_store(tmp_path / "s")
    p = _write_tile(tmp_path / "t.tif", X0, Y0, _tile_data())
    res = rz.verify_against_store(store, [p], verbose=False)
    assert (res["hole"], res["match"], res["mismatch"]) == (1, 0, 0)


def test_verify_detects_unquantized_writes(tmp_path):
    """A store written with & 0xFC vs a reader that skips it: the check has to
    notice, since the difference is invisible at a glance."""
    store = _make_store(tmp_path / "s")
    data = np.full((NB, 2 * CHUNK, 2 * CHUNK), 203, dtype=np.uint8)
    p = _write_tile(tmp_path / "t.tif", X0, Y0, data)
    rz.append_tiles(store, [p], mask_low_bits=True, verbose=False)
    res = rz.verify_against_store(store, [p], mask_low_bits=False,
                                  verbose=False)
    assert res["mismatch"] == 1


def test_cli_verify_blocks_on_mismatch(tmp_path):
    store = _make_store(tmp_path / "s")
    data = np.full((NB, 2 * CHUNK, 2 * CHUNK), 100, dtype=np.uint8)
    _write_tile(tmp_path / "seed.tif", X0, Y0, data)
    rz.append_tiles(store, [tmp_path / "seed.tif"], verbose=False)

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _write_tile(tiles / "other.tif", X0, Y0,
                np.full((NB, 2 * CHUNK, 2 * CHUNK), 40, dtype=np.uint8))
    assert rz.main(["verify", "--store", str(store),
                    "--tiles", str(tiles)]) == 1


def test_cli_verify_passes_when_clean(tmp_path, capsys):
    store = _make_store(tmp_path / "s")
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    data = np.random.default_rng(3).integers(
        1, 256, (NB, 2 * CHUNK, 2 * CHUNK), dtype=np.uint8)
    _write_tile(tiles / "t.tif", X0, Y0, data)
    rz.append_tiles(store, [tiles / "t.tif"], verbose=False)
    assert rz.main(["verify", "--store", str(store),
                    "--tiles", str(tiles)]) == 0
    assert "safe to append" in capsys.readouterr().out


# ─── rebuild ──────────────────────────────────────────────────────────────────

def _cropped_grid(col_chunks=2):
    """A store shifted east by `col_chunks`, like nyc_2024's 42."""
    return rz.GridSpec(x0=X0 + col_chunks * CHUNK * PX, y0=Y0, dx=PX, dy=-PX,
                       n_y=4 * CHUNK, n_x=(4 - col_chunks) * CHUNK)


def test_canonical_offset_is_in_whole_chunks():
    assert rz.canonical_offset(_cropped_grid(2), GRID, CHUNK, CHUNK) == (0, 2)


def test_canonical_offset_matches_the_real_2024_geometry():
    """nyc_2024 sits 52,500 ft east of the canonical origin: exactly 42
    chunks of 2500 px at 0.5 ft."""
    cropped = rz.GridSpec(x0=960000.0, y0=277500.0, dx=0.5, dy=-0.5,
                          n_y=285000, n_x=220000)
    assert rz.canonical_offset(cropped, rz.NYC_GRID, 2500, 2500) == (0, 42)


def test_canonical_offset_rejects_a_non_chunk_multiple():
    odd = rz.GridSpec(x0=X0 + 3 * PX, y0=Y0, dx=PX, dy=-PX, n_y=CHUNK,
                      n_x=CHUNK)
    with pytest.raises(ValueError, match="whole number"):
        rz.canonical_offset(odd, GRID, CHUNK, CHUNK)


def test_create_store_like_clones_the_codec_and_fixes_the_geotransform(tmp_path):
    src = _make_store(tmp_path / "src")
    dst = rz.create_store_like(src, tmp_path / "dst", GRID)

    src_meta, dst_meta = rz.read_zarray_meta(src), rz.read_zarray_meta(dst)
    for field in ("chunks", "dtype", "compressor", "filters", "fill_value",
                  "dimension_separator"):
        assert src_meta[field] == dst_meta[field], field

    attrs = json.loads((dst / "spatial_ref" / ".zattrs").read_text())
    assert attrs["GeoTransform"] == GRID.geotransform()
    assert attrs["crs_wkt"] == WKT


def test_create_store_like_refuses_to_clobber(tmp_path):
    src = _make_store(tmp_path / "src")
    (tmp_path / "dst").mkdir()
    with pytest.raises(FileExistsError):
        rz.create_store_like(src, tmp_path / "dst", GRID)


def test_rebuilt_store_opens_in_xarray_with_correct_coords(tmp_path):
    import xarray as xr

    src = _make_store(tmp_path / "src")
    dst = rz.create_store_like(src, tmp_path / "dst", GRID)
    ds = xr.open_zarr(dst, mask_and_scale=False)
    assert ds["value"].dims == ("band", "y", "x")
    assert ds.x.values[0] == pytest.approx(GRID.x_coords()[0])
    assert ds.y.values[0] == pytest.approx(GRID.y_coords()[0])
    assert list(ds.band.values) == [1, 2, 3, 4]


def test_relinked_chunks_land_on_the_same_ground(tmp_path):
    """The heart of the 2024 repair: a chunk blob is index-free, so re-keying
    it must move the pixels to the same world coordinate, not the same index."""
    cropped = _cropped_grid(2)
    src = _make_store(tmp_path / "src", grid=cropped, cells=[(1, 0)], fill=77)
    dst = rz.create_store_like(src, tmp_path / "dst", GRID)
    moved = rz.relocate_chunks(src, dst, 0, 2, verbose=False)

    assert moved == 1
    # src cell (1,0) starts at x = X0 + 2*CHUNK*PX -> dst cell (1,2)
    assert (_read(dst, np.s_[:, CHUNK:2 * CHUNK, 2 * CHUNK:3 * CHUNK]) == 77).all()
    assert (_read(dst, np.s_[:, CHUNK:2 * CHUNK, :CHUNK]) == 0).all()


def test_hardlinking_leaves_the_source_intact(tmp_path):
    src = _make_store(tmp_path / "src", grid=_cropped_grid(2),
                      cells=[(0, 0)], fill=88)
    dst = rz.create_store_like(src, tmp_path / "dst", GRID)
    rz.relocate_chunks(src, dst, 0, 2, mode="hardlink", verbose=False)
    assert (src / "value" / "0.0.0").exists()
    assert (_read(src, np.s_[:, :CHUNK, :CHUNK]) == 88).all()


def test_writing_over_a_hardlinked_chunk_does_not_corrupt_the_source(tmp_path):
    """zarr replaces a chunk file rather than editing it, so the source store
    stays valid as a rollback even after the copy is written to."""
    src = _make_store(tmp_path / "src", grid=_cropped_grid(2),
                      cells=[(0, 0)], fill=88)
    dst = rz.create_store_like(src, tmp_path / "dst", GRID)
    rz.relocate_chunks(src, dst, 0, 2, mode="hardlink", verbose=False)

    p = _write_tile(tmp_path / "t.tif", X0 + 2 * CHUNK * PX, Y0,
                    _tile_data(value=160))
    rz.append_tiles(dst, [p], verbose=False)

    assert (_read(dst, np.s_[:, :CHUNK, 2 * CHUNK:3 * CHUNK]) == 160).all()
    assert (_read(src, np.s_[:, :CHUNK, :CHUNK]) == 88).all()   # untouched


def test_relocate_rejects_an_out_of_range_shift(tmp_path):
    src = _make_store(tmp_path / "src", cells=[(3, 3)], fill=5)
    dst = rz.create_store_like(src, tmp_path / "dst", GRID)
    with pytest.raises(ValueError, match="outside"):
        rz.relocate_chunks(src, dst, 0, 2, verbose=False)


def test_rebuild_end_to_end_recovers_the_missing_west(tmp_path):
    """The 2024 repair whole: cropped store + the tiles it could not hold."""
    cropped = _cropped_grid(2)
    src = _make_store(tmp_path / "src", grid=cropped, cells=[(0, 0)], fill=90)
    # a tile at the far west — outside `src` entirely
    p = _write_tile(tmp_path / "west.tif", X0, Y0, _tile_data(value=120))

    summary = rz.rebuild_store(src, tmp_path / "dst", [p], grid=GRID,
                               verbose=False)
    assert summary["chunk_offset"] == (0, 2)
    assert summary["relocated"] == 1
    assert summary["tiles"]["written"] == 1

    dst = tmp_path / "dst"
    assert (_read(dst, np.s_[:, :CHUNK, 2 * CHUNK:3 * CHUNK]) == 90).all()
    assert (_read(dst, np.s_[:, :2 * CHUNK, :2 * CHUNK]) == 120).all()


def test_rebuild_dry_run_creates_nothing(tmp_path):
    src = _make_store(tmp_path / "src", grid=_cropped_grid(2), cells=[(0, 0)])
    summary = rz.rebuild_store(src, tmp_path / "dst", [], grid=GRID,
                               dry_run=True, verbose=False)
    assert summary["chunk_offset"] == (0, 2)
    assert not (tmp_path / "dst").exists()


# ─── geotransform repair ──────────────────────────────────────────────────────

def test_fix_geotransform_rewrites_the_attribute(tmp_path):
    store = _make_store(tmp_path / "s")
    msg = rz.fix_geotransform(store)
    attrs = json.loads((store / "spatial_ref" / ".zattrs").read_text())
    assert attrs["GeoTransform"] == GRID.geotransform()
    assert "fixed" in msg
    assert "already correct" in rz.fix_geotransform(store)      # idempotent


def test_fix_geotransform_dry_run_changes_nothing(tmp_path):
    store = _make_store(tmp_path / "s")
    before = (store / "spatial_ref" / ".zattrs").read_text()
    msg = rz.fix_geotransform(store, dry_run=True)
    assert "would fix" in msg
    assert (store / "spatial_ref" / ".zattrs").read_text() == before


def test_fix_geotransform_keeps_consolidated_metadata_in_step(tmp_path):
    """xarray reads .zmetadata when present; a stale copy would undo the fix."""
    store = _make_store(tmp_path / "s")
    rz.fix_geotransform(store)
    meta = json.loads((store / ".zmetadata").read_text())["metadata"]
    assert meta["spatial_ref/.zattrs"]["GeoTransform"] == GRID.geotransform()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_append_dry_run(tmp_path, capsys):
    store = _make_store(tmp_path / "s")
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _write_tile(tiles / "t.tif", X0, Y0, _tile_data())
    rc = rz.main(["append", "--store", str(store), "--tiles", str(tiles),
                  "--dry-run"])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_cli_limit_caps_the_work(tmp_path):
    store = _make_store(tmp_path / "s")
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _write_tile(tiles / "a.tif", X0, Y0, _tile_data())
    _write_tile(tiles / "b.tif", X0 + 2 * CHUNK * PX, Y0, _tile_data())
    rz.main(["append", "--store", str(store), "--tiles", str(tiles),
             "--limit", "1"])
    present = {k.name for k in (store / "value").iterdir()
               if not k.name.startswith(".")}
    assert present == {"0.0.0", "0.0.1", "0.1.0", "0.1.1"}


def test_cli_reports_failure_in_the_exit_code(tmp_path):
    store = _make_store(tmp_path / "s")
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "bad.tif").write_bytes(b"nope")
    assert rz.main(["append", "--store", str(store), "--tiles",
                    str(tiles)]) == 1
