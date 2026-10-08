# -*- coding: utf-8 -*-
"""Chunk-presence coverage probe over synthetic zarr stores.

No real store is touched. Fixtures build a genuine zarr v2 tree with xarray,
then delete chunk files to simulate imagery that was never ingested — which is
exactly the on-disk state ``write_empty_chunks=False`` leaves behind, and the
whole reason coverage is decidable from a directory listing.

The fixtures reproduce the two quirks the real stores have: coordinates
rounded to one decimal, and a store whose origin is offset from its peers.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src import diagnose_zarr_coverage as dz

CHUNK = 10          # pixels per chunk side
PX = 0.5            # ft per pixel
CELL = CHUNK * PX   # 5 ft per chunk cell
X0, Y0 = 1000.0, 500.0
WKT = "EPSG:6539"


def _make_store(root, present_cells, n_r=4, n_c=4, x0=X0, y0=Y0,
                px=PX, round_coords=False, geotransform=None):
    """A zarr store covering `n_r` x `n_c` chunk cells, holding only
    `present_cells`. Pixel (0,0)'s outer edge is (x0, y0)."""
    import xarray as xr

    ny, nx = n_r * CHUNK, n_c * CHUNK
    x = x0 + px * np.arange(nx) + px / 2          # cell centres
    y = y0 - px * np.arange(ny) - px / 2
    if round_coords:
        x, y = np.round(x, 1), np.round(y, 1)

    da = xr.DataArray(
        np.zeros((4, ny, nx), "uint8"), dims=("band", "y", "x"),
        coords={"band": [1, 2, 3, 4], "x": x, "y": y}, name="value",
    )
    da.to_zarr(root, zarr_format=2,
               encoding={"value": {"chunks": (4, CHUNK, CHUNK)}})

    # spatial_ref, as rioxarray would have written it
    sref = root / "spatial_ref"
    sref.mkdir(exist_ok=True)
    attrs = {"crs_wkt": WKT, "_ARRAY_DIMENSIONS": []}
    if geotransform is not None:
        attrs["GeoTransform"] = geotransform
    (sref / ".zattrs").write_text(json.dumps(attrs))

    keep = {f"0.{r}.{c}" for r, c in present_cells}
    for key in list((root / "value").iterdir()):
        if not key.name.startswith(".") and key.name not in keep:
            key.unlink()
    return root


def _all_cells(n_r=4, n_c=4):
    return [(r, c) for r in range(n_r) for c in range(n_c)]


# ─── georeferencing ───────────────────────────────────────────────────────────

def test_geo_comes_from_the_coordinate_arrays(tmp_path):
    geo = dz.read_store_geo(_make_store(tmp_path / "s", []))
    assert (geo.n_rows, geo.n_cols) == (4, 4)
    assert (geo.chunk_rows, geo.chunk_cols) == (CHUNK, CHUNK)
    assert geo.x0 == pytest.approx(X0)
    assert geo.y0 == pytest.approx(Y0)
    assert geo.dx == pytest.approx(PX)
    assert geo.dy == pytest.approx(-PX)


def test_rounded_coordinates_still_give_the_true_pixel_size(tmp_path):
    """The real stores rounded coords to 1 decimal, so consecutive 0.5 ft steps
    read back as 0.4/0.6. A first difference would report the wrong pixel
    size; the average over the full axis does not."""
    geo = dz.read_store_geo(_make_store(tmp_path / "s", [], round_coords=True))
    # ±0.05 ft of endpoint error spread over a 40 px axis; on the real
    # 325,000 px stores the same rounding leaves ~3e-7 ft.
    assert geo.dx == pytest.approx(PX, abs=0.01)
    assert geo.dy == pytest.approx(-PX, abs=0.01)
    assert geo.x0 == pytest.approx(X0, abs=0.1)
    assert abs(geo.dx - PX) < abs(0.6 - PX)     # not the naive first difference


def test_a_wrong_geotransform_attribute_is_reported(tmp_path):
    """Every real store claims an origin 90,000 ft from where its coordinates
    actually are. The probe must notice rather than trust it."""
    root = _make_store(tmp_path / "s", [],
                       geotransform=f"{X0 + 90000} {PX} 0.0 {Y0} 0.0 {-PX}")
    geo = dz.read_store_geo(root)
    problem = dz.check_geotransform_attr(root, geo)
    assert problem and "disagrees" in problem
    assert geo.x0 == pytest.approx(X0)      # coords win


def test_a_correct_geotransform_attribute_is_silent(tmp_path):
    root = _make_store(tmp_path / "s", [],
                       geotransform=f"{X0} {PX} 0.0 {Y0} 0.0 {-PX}")
    assert dz.check_geotransform_attr(root, dz.read_store_geo(root)) is None


def test_chunk_bounds_are_north_up_and_contiguous(tmp_path):
    geo = dz.read_store_geo(_make_store(tmp_path / "s", []))
    assert dz.chunk_bounds(geo, 0, 0) == pytest.approx((1000.0, 495.0, 1005.0, 500.0))
    assert dz.chunk_bounds(geo, 1, 0) == pytest.approx((1000.0, 490.0, 1005.0, 495.0))
    assert dz.chunk_bounds(geo, 0, 1) == pytest.approx((1005.0, 495.0, 1010.0, 500.0))


def test_partial_trailing_chunk_still_occupies_a_cell(tmp_path):
    import xarray as xr
    root = tmp_path / "s"
    ny = nx = 35                                   # 3.5 chunks
    da = xr.DataArray(
        np.zeros((4, ny, nx), "uint8"), dims=("band", "y", "x"),
        coords={"band": [1, 2, 3, 4],
                "x": X0 + PX * np.arange(nx) + PX / 2,
                "y": Y0 - PX * np.arange(ny) - PX / 2}, name="value")
    da.to_zarr(root, zarr_format=2,
               encoding={"value": {"chunks": (4, CHUNK, CHUNK)}})
    geo = dz.read_store_geo(root)
    assert (geo.n_rows, geo.n_cols) == (4, 4)


# ─── presence mask ────────────────────────────────────────────────────────────

def test_presence_mask_reflects_files_on_disk(tmp_path):
    cells = [(0, 0), (1, 2), (3, 3)]
    root = _make_store(tmp_path / "s", cells)
    mask = dz.present_chunk_grid(root, dz.read_store_geo(root))
    assert mask.sum() == 3
    for r, c in cells:
        assert mask[r, c]
    assert not mask[2, 2]


def test_absent_chunks_are_the_signal(tmp_path):
    """A store with no chunk files reads back as all zeros — the silent
    black-imagery failure this probe exists to expose."""
    root = _make_store(tmp_path / "s", [])
    assert not dz.present_chunk_grid(root, dz.read_store_geo(root)).any()


def test_metadata_and_garbage_keys_are_ignored(tmp_path):
    root = _make_store(tmp_path / "s", [(0, 0)])
    (root / "value" / "0.1").write_bytes(b"")        # wrong arity
    (root / "value" / "0.x.1").write_bytes(b"")      # non-numeric
    (root / "value" / "0.99.1").write_bytes(b"")     # out of range
    assert dz.present_chunk_grid(root, dz.read_store_geo(root)).sum() == 1


def test_band_chunks_fold_together(tmp_path):
    root = _make_store(tmp_path / "s", [(1, 1)])
    (root / "value" / "1.1.1").write_bytes(b"")
    assert dz.present_chunk_grid(root, dz.read_store_geo(root)).sum() == 1


# ─── alignment onto a master grid ─────────────────────────────────────────────

def test_master_grid_spans_offset_stores(tmp_path):
    """nyc_2024's real situation: same pixel size, origin 42 chunks east."""
    a = dz.read_store_geo(_make_store(tmp_path / "a", [], n_c=4))
    b = dz.read_store_geo(_make_store(tmp_path / "b", [], n_c=2,
                                      x0=X0 + 2 * CELL))
    m = dz.master_geo([a, b])
    assert (m.n_rows, m.n_cols) == (4, 4)
    assert m.x0 == pytest.approx(X0)
    assert dz.chunk_offset(a, m) == (0, 0)
    assert dz.chunk_offset(b, m) == (0, 2)


def test_offset_store_lands_on_the_right_ground(tmp_path):
    """The bug this replaces: indexing by chunk number compared different
    ground across years. Cell (0,0) of an offset store is not master (0,0)."""
    a = dz.read_store_geo(_make_store(tmp_path / "a", [], n_c=4))
    b_root = _make_store(tmp_path / "b", [(0, 0)], n_c=2, x0=X0 + 2 * CELL)
    b = dz.read_store_geo(b_root)
    m = dz.master_geo([a, b])
    placed = dz.place(dz.present_chunk_grid(b_root, b),
                      dz.chunk_offset(b, m), m)
    assert placed[0, 2] and not placed[0, 0]


def test_fractional_offset_is_rejected(tmp_path):
    """Half-cell shift: no index-based comparison across years is valid."""
    a = dz.read_store_geo(_make_store(tmp_path / "a", []))
    b = dz.read_store_geo(_make_store(tmp_path / "b", [], x0=X0 + CELL / 2))
    m = dz.master_geo([a, b])
    with pytest.raises(ValueError, match="not chunk-aligned"):
        dz.chunk_offset(b, m)


def test_rounding_noise_is_not_a_misalignment(tmp_path):
    a = dz.read_store_geo(_make_store(tmp_path / "a", []))
    b = dz.read_store_geo(_make_store(tmp_path / "b", [], round_coords=True))
    assert dz.chunk_offset(b, dz.master_geo([a, b])) == (0, 0)


def test_different_pixel_size_is_rejected(tmp_path):
    a = dz.read_store_geo(_make_store(tmp_path / "a", []))
    b = dz.read_store_geo(_make_store(tmp_path / "b", [], px=1.0))
    with pytest.raises(ValueError, match="different pixel size"):
        dz.master_geo([a, b])


def test_master_prefers_a_crs_that_resolves_to_an_epsg_code(tmp_path):
    """nyc_2010/2012 carry an authority-less WKT that pyproj cannot transform
    into; picking it would silently empty every spatial join."""
    bare = ('PROJCS["NAD83 / New York Long Island",'
            'GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
            'SPHEROID["GRS 1980",6378137,298.257222101]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Lambert_Conformal_Conic_2SP"],'
            'PARAMETER["latitude_of_origin",40.1666666666667],'
            'PARAMETER["central_meridian",-74],'
            'PARAMETER["standard_parallel_1",41.0333333333333],'
            'PARAMETER["standard_parallel_2",40.6666666666667],'
            'PARAMETER["false_easting",984250],PARAMETER["false_northing",0],'
            'UNIT["us_survey_feet",0.304800609601219]]')
    good = dz.read_store_geo(_make_store(tmp_path / "good", []))
    weak = dz.read_store_geo(_make_store(tmp_path / "weak", []))
    weak = dz.StoreGeo(**{**weak.__dict__, "crs_wkt": bare})

    from pyproj import CRS
    assert CRS.from_user_input(bare).to_epsg() is None       # premise
    assert dz.pick_crs([weak, good]) == good.crs_wkt
    assert dz.master_geo([weak, good]).crs_wkt == good.crs_wkt


def test_pick_crs_falls_back_when_nothing_resolves(tmp_path):
    geo = dz.read_store_geo(_make_store(tmp_path / "s", []))
    weak = dz.StoreGeo(**{**geo.__dict__, "crs_wkt": "not a crs"})
    assert dz.pick_crs([weak]) == "not a crs"
    assert dz.pick_crs([dz.StoreGeo(**{**geo.__dict__, "crs_wkt": ""})]) == ""


def test_place_rejects_a_store_that_does_not_fit(tmp_path):
    geo = dz.read_store_geo(_make_store(tmp_path / "a", []))
    with pytest.raises(ValueError, match="does not fit"):
        dz.place(np.ones((4, 4), bool), (2, 0), geo)


def test_load_masks_aligns_years(tmp_path):
    _make_store(tmp_path / "nyc_2020.zarr", _all_cells(), n_c=4)
    _make_store(tmp_path / "nyc_2024.zarr", [(0, 0), (1, 1)], n_c=2,
                x0=X0 + 2 * CELL)
    masks, m = dz.load_masks(tmp_path, years=(2020, 2024), verbose=False)
    assert (m.n_rows, m.n_cols) == (4, 4)
    assert masks[2024].sum() == 2
    assert masks[2024][0, 2] and masks[2024][1, 3]
    assert not masks[2024][0, 0]


def test_load_masks_skips_absent_years(tmp_path):
    _make_store(tmp_path / "nyc_2020.zarr", [(0, 0)])
    masks, _ = dz.load_masks(tmp_path, years=(2020, 2022), verbose=False)
    assert list(masks) == [2020]


def test_load_masks_errors_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        dz.load_masks(tmp_path, years=(2020,), verbose=False)


# ─── coverage accounting ──────────────────────────────────────────────────────

def test_footprint_is_the_union_not_the_rectangle():
    """Water and padding are in no year; they must not dilute the denominator."""
    a = np.zeros((2, 2), bool); a[0, 0] = True
    b = np.zeros((2, 2), bool); b[0, 1] = True
    fp = dz.city_footprint({2020: a, 2022: b})
    assert fp.sum() == 2 and not fp[1, 1]


def test_city_footprint_rejects_empty():
    with pytest.raises(ValueError):
        dz.city_footprint({})


def _regions():
    """Two side-by-side 'boroughs' splitting the 4x4 grid down the middle."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"boroname": ["West", "East"]},
        geometry=[box(X0, 480, X0 + 2 * CELL - 0.1, Y0),
                  box(X0 + 2 * CELL + 0.1, 480, X0 + 4 * CELL, Y0)],
        crs=WKT,
    )


def test_coverage_isolates_the_region_and_year_with_the_hole(tmp_path):
    """The intended end-to-end result: one region-year reads as the outlier."""
    west_only = [(r, c) for r, c in _all_cells() if c < 2]
    for year, cells in [(2020, _all_cells()), (2022, west_only),
                        (2024, _all_cells())]:
        _make_store(tmp_path / f"nyc_{year}.zarr", cells)

    masks, geo = dz.load_masks(tmp_path, years=(2020, 2022, 2024),
                               verbose=False)
    table = dz.coverage_by_region(masks, geo, _regions(), "boroname")
    got = table.set_index(["boroname", "year"])["coverage"]

    assert got[("East", 2022)] == 0.0          # the black borough
    assert got[("West", 2022)] == 1.0          # unaffected, same year
    assert got[("East", 2020)] == 1.0          # same borough, other years fine
    assert got[("East", 2024)] == 1.0


def test_every_region_year_pair_is_reported(tmp_path):
    for year in (2020, 2022):
        _make_store(tmp_path / f"nyc_{year}.zarr", _all_cells())
    masks, geo = dz.load_masks(tmp_path, years=(2020, 2022), verbose=False)
    table = dz.coverage_by_region(masks, geo, _regions(), "boroname")
    assert len(table) == 4
    assert (table["chunks_present"] + table["chunks_missing"]
            == table["chunks_total"]).all()


def test_coverage_map_is_written(tmp_path):
    for year, cells in [(2020, _all_cells()),
                        (2022, [(r, c) for r, c in _all_cells() if c < 2])]:
        _make_store(tmp_path / f"nyc_{year}.zarr", cells)
    masks, geo = dz.load_masks(tmp_path, years=(2020, 2022), verbose=False)
    out = tmp_path / "fig" / "coverage.png"
    dz.save_coverage_map(masks, geo, _regions(), out)
    assert out.exists() and out.stat().st_size > 0


def test_disjoint_regions_raise_rather_than_report_nothing(tmp_path):
    """An empty join means the georeferencing is wrong; returning an empty
    frame would look like a clean result."""
    import geopandas as gpd
    from shapely.geometry import box

    _make_store(tmp_path / "nyc_2020.zarr", _all_cells())
    masks, geo = dz.load_masks(tmp_path, years=(2020,), verbose=False)
    far = gpd.GeoDataFrame({"boroname": ["Elsewhere"]},
                           geometry=[box(9e5, 9e5, 9e5 + 10, 9e5 + 10)],
                           crs=WKT)
    with pytest.raises(ValueError, match="no chunk cell intersects"):
        dz.coverage_by_region(masks, geo, far, "boroname")
