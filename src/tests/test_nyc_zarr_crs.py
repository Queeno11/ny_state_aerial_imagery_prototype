"""Synthetic-data tests for the CRS plumbing of the legacy NYC zarr path.

The NYC zarr store is indexed on EPSG:6539 (US survey feet, 0.5 units/pixel)
while the US-scale pipeline builds every table on EPSG:5070 (meters). Feeding
5070 coordinates to the 6539 raster matches zero buildings, which used to
surface as a bare ``KeyError: 'dataset'`` deep inside ``assign_datasets_to_gdf``.

These tests cover: the ``epsg`` plumbing through ``open_datasets`` /
``load_income_dataset``, the raster-CRS check, and the explicit no-match guard.
No real data, no network.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
import pytest
import xarray as xr
from pyproj import CRS
from shapely.geometry import box

import src.geo_utils as geo_utils
import src.build_dataset as bd

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

NYC_EPSG = geo_utils.NYC_EVAL_EPSG           # 6539, US survey feet
FT_TO_M = geo_utils.projected_units_to_meters(1.0, NYC_EPSG)
YEARS = [2016, 2018]


@pytest.fixture(autouse=True)
def offline_proj():
    """NAD83 → NAD83(2011) (5070 ↔ 6539) makes PROJ reach for a shift grid from
    cdn.proj.org; with PROJ_NETWORK=ON and no network it returns inf instead of
    raising. Force the ballpark transform so these tests never depend on that.
    """
    was_enabled = pyproj.network.is_network_enabled()
    pyproj.network.set_network_enabled(False)
    yield
    pyproj.network.set_network_enabled(was_enabled)


# ── synthetic NYC panel + DoITT footprints ────────────────────────────────────

def _make_nyc_panel():
    """4 lon/lat tracts over Manhattan, delivered in EPSG:5070 like the real one."""
    rows, geoms = [], []
    for t in range(4):
        lon = -74.02 + 0.01 * t
        geoms.append(box(lon, 40.74, lon + 0.01, 40.75))
        row = {bd.PANEL_GEOID_COL: f"3606100{t:04d}", "cbsa_code": "35620",
               "valid_change_W2_r5": bool(t % 2 == 0)}
        for y in range(2011, 2024):
            row[f"Rel_Score_W2_i_r5pct_{y}"] = float(t) - 1.5
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326").to_crs(geo_utils.METRIC_CRS)


def _make_doitt(panel, n_per_tract=3):
    """DoITT-shaped footprints (index=DOITT_ID, dated) inside the panel tracts."""
    tracts = panel.to_crs("EPSG:4326")
    rows, geoms = [], []
    did = 1
    for _, tract in tracts.iterrows():
        minx, miny, maxx, maxy = tract.geometry.bounds
        for k in range(n_per_tract):
            f = (k + 1) / (n_per_tract + 1)
            cx, cy = minx + f * (maxx - minx), miny + f * (maxy - miny)
            geoms.append(box(cx, cy, cx + 0.0002, cy + 0.0002))
            rows.append({"DOITT_ID": did, "OBJECTID": did,
                         "CONSTRUCTION_YEAR": 0, "DEMOLITION_YEAR": 2999})
            did += 1
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326").set_index("DOITT_ID")


@pytest.fixture
def nyc_env(tmp_path, monkeypatch):
    panel = _make_nyc_panel()
    doitt = _make_doitt(panel)
    monkeypatch.setattr(bd, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(bd, "process_acs_panel", lambda: panel.copy())
    monkeypatch.setattr(bd, "load_building_data", lambda: doitt.copy())
    return {"panel": panel, "doitt": doitt, "tmp": tmp_path}


# ── load_income_dataset: coordinates land on the requested grid ───────────────

def test_doitt_table_is_built_in_requested_crs(nyc_env):
    tau_m = 100.0
    flat = bd.load_income_dataset(YEARS, tau_meters=tau_m, indicator="W2_r5",
                                  footprints_source="doitt_nyc", epsg=NYC_EPSG)
    assert len(flat) == len(nyc_env["doitt"]) * len(YEARS)

    # EPSG:6539 over Manhattan: x ~ 9.7e5-1.1e6 ftUS, y ~ 1.9e5-2.8e5 ftUS.
    assert flat["centroid_x"].between(9.0e5, 1.2e6).all()
    assert flat["centroid_y"].between(1.5e5, 3.0e5).all()

    # tau is metres on every grid: the bbox side must be 2*tau converted to feet.
    side = (flat["bbox_maxx"] - flat["bbox_minx"]).to_numpy()
    np.testing.assert_allclose(side, 2 * tau_m / FT_TO_M, rtol=1e-9)
    np.testing.assert_allclose(
        (flat["bbox_maxy"] - flat["bbox_miny"]).to_numpy(), side, rtol=1e-9)

    # dist_to_center is km: feet-based deltas must be converted, not passed through.
    ctr_x, ctr_y = flat["centroid_x"].mean(), flat["centroid_y"].mean()
    approx_km = np.hypot(flat["centroid_x"] - ctr_x, flat["centroid_y"] - ctr_y) * FT_TO_M / 1000
    assert flat["dist_to_center"].max() < 50           # Manhattan, not 1e5 "km"
    np.testing.assert_allclose(flat["dist_to_center"], approx_km, atol=2.0)


def test_artifacts_are_tagged_by_epsg(nyc_env):
    bd.load_income_dataset(YEARS, tau_meters=100.0, indicator="W2_r5",
                           footprints_source="doitt_nyc", epsg=NYC_EPSG)
    names = {p.name for p in nyc_env["tmp"].glob("*.parquet")}
    assert any(f"epsg{NYC_EPSG}" in n and n.startswith("temporal_data_") for n in names)
    assert any(f"epsg{NYC_EPSG}" in n and n.startswith("building_geometries_") for n in names)
    assert not any(f"epsg{geo_utils.METRIC_EPSG}" in n for n in names)
    # main.py finds the geometry artifact through the same tag helper.
    assert bd.temporal_data_tag("doitt_nyc", "W2_r5", NYC_EPSG) == f"doitt_nyc_W2_r5_epsg{NYC_EPSG}"


def test_metric_and_nyc_grids_do_not_share_a_cache_file(nyc_env):
    nyc = bd.load_income_dataset(YEARS, tau_meters=100.0, indicator="W2_r5",
                                 footprints_source="doitt_nyc", epsg=NYC_EPSG)
    us = bd.load_income_dataset(YEARS, tau_meters=100.0, indicator="W2_r5",
                                footprints_source="doitt_nyc", epsg=geo_utils.METRIC_EPSG)
    # Same buildings, genuinely different coordinates (no stale-artifact reuse).
    assert len(nyc) == len(us)
    assert abs(nyc["centroid_x"].mean() - us["centroid_x"].mean()) > 1e5
    # 5070 is metres: the bbox side is 2*tau exactly.
    np.testing.assert_allclose(
        (us["bbox_maxx"] - us["bbox_minx"]).to_numpy(), 200.0, rtol=1e-9)


def test_non_finite_reprojection_is_caught(nyc_env, monkeypatch):
    """A failed datum-grid fetch yields inf coords, not an exception, from PROJ."""
    broken = nyc_env["doitt"].copy()
    broken["geometry"] = gpd.GeoSeries(
        [box(np.inf, np.inf, np.inf, np.inf)] * len(broken), crs="EPSG:4326"
    )
    monkeypatch.setattr(bd, "load_building_data", lambda: broken)
    with pytest.raises(ValueError, match="non-finite coordinates"):
        bd.load_income_dataset(YEARS, tau_meters=100.0, indicator="W2_r5",
                               footprints_source="doitt_nyc", epsg=NYC_EPSG)


def test_ms_us_rejects_non_metric_epsg():
    with pytest.raises(NotImplementedError, match="national grid"):
        bd.load_income_dataset(YEARS, indicator="W2_r5",
                               footprints_source="ms_us", epsg=NYC_EPSG)


# ── open_datasets: default epsg follows the imagery ───────────────────────────

def _spy_open_datasets(monkeypatch, sat_data):
    seen = {}

    def fake_load_income_dataset(years, **kw):
        seen["epsg"] = kw.get("epsg")
        return pd.DataFrame({"centroid_x": [0.0], "centroid_y": [0.0], "year": [years[0]]})

    monkeypatch.setattr(bd, "load_income_dataset", fake_load_income_dataset)
    monkeypatch.setattr(bd, "load_satellite_datasets", lambda years: ({}, {}))
    monkeypatch.setattr(bd, "_check_datasets_crs", lambda datasets, epsg: None)
    monkeypatch.setattr(bd, "assign_datasets_to_gdf",
                        lambda df, *a, **kw: seen.setdefault("assign_epsg", kw.get("epsg")) or df)
    bd.open_datasets(sat_data=sat_data, years=[2018], indicator="W2_r5",
                     footprints_source="doitt_nyc")
    return seen


def test_open_datasets_defaults_to_the_zarr_grid_for_aerial(monkeypatch):
    seen = _spy_open_datasets(monkeypatch, "aerial")
    assert seen["epsg"] == NYC_EPSG
    assert seen["assign_epsg"] == NYC_EPSG


def test_open_datasets_defaults_to_the_national_grid_for_naip(monkeypatch):
    seen = _spy_open_datasets(monkeypatch, "NAIP")
    assert seen["epsg"] == geo_utils.METRIC_EPSG


# ── raster CRS check ──────────────────────────────────────────────────────────

def _fake_zarr(x0=1000.0, y0=5000.0, n=64, epsg=NYC_EPSG):
    """Tiny raster on a 0.5-unit grid (x ascending, y descending), like the zarr."""
    x = x0 + 0.5 * np.arange(n)
    y = y0 - 0.5 * np.arange(n)
    ds = xr.Dataset(
        {"value": (("band", "y", "x"), np.zeros((3, n, n), dtype="uint8"))},
        coords={"band": [1, 2, 3], "y": y, "x": x},
    )
    if epsg is not None:
        ds = ds.assign_coords(spatial_ref=0)
        ds["spatial_ref"].attrs["crs_wkt"] = CRS.from_epsg(epsg).to_wkt()
    return ds


def test_check_datasets_crs_accepts_matching_grid():
    bd._check_datasets_crs({"nyc_2018.zarr": _fake_zarr(epsg=NYC_EPSG)}, NYC_EPSG)


def test_check_datasets_crs_rejects_mismatched_grid():
    with pytest.raises(ValueError, match=f"EPSG:{NYC_EPSG}"):
        bd._check_datasets_crs({"nyc_2018.zarr": _fake_zarr(epsg=NYC_EPSG)},
                               geo_utils.METRIC_EPSG)


def test_check_datasets_crs_tolerates_missing_crs(capsys):
    bd._check_datasets_crs({"nyc_2018.zarr": _fake_zarr(epsg=None)}, NYC_EPSG)
    assert "No CRS recorded" in capsys.readouterr().out


def test_dataset_epsg_reads_and_defaults():
    assert geo_utils.dataset_epsg(_fake_zarr(epsg=NYC_EPSG)) == NYC_EPSG
    assert geo_utils.dataset_epsg(_fake_zarr(epsg=None)) is None


# ── assign_datasets_to_gdf ────────────────────────────────────────────────────

def _grid_df(ds, year, half_width_units=4.0, n=6):
    """Buildings centred exactly on grid points → deterministic pixel windows."""
    xs = ds.x.values[10:10 + n]
    ys = ds.y.values[10:10 + n]
    return pd.DataFrame({
        "building_id": np.arange(n),
        "year": year,
        "centroid_x": xs, "centroid_y": ys,
        "bbox_minx": xs - half_width_units, "bbox_maxx": xs + half_width_units,
        "bbox_miny": ys - half_width_units, "bbox_maxy": ys + half_width_units,
    })


def test_assign_datasets_to_gdf_links_and_indexes():
    ds = _fake_zarr()
    datasets = {"nyc_2018.zarr": ds}
    extents = {"nyc_2018.zarr": geo_utils.get_dataset_extent(ds)}
    df = _grid_df(ds, 2018)

    out = bd.assign_datasets_to_gdf(df, datasets, extents, years=[2018],
                                    verbose=False, save_plot=False, epsg=NYC_EPSG)
    assert len(out) == len(df)
    assert (out["dataset"] == "nyc_2018.zarr").all()
    for col in ("row_start", "row_stop", "col_start", "col_stop"):
        assert out[col].dtype == np.dtype("int64")
    # 8-unit-wide window on a 0.5-unit grid: 16 pixel steps → 17 inclusive samples.
    assert ((out["col_stop"] - out["col_start"]) == 17).all()
    assert ((out["row_stop"] - out["row_start"]) == 17).all()


def test_assign_datasets_to_gdf_drops_only_unmatched_years():
    ds = _fake_zarr()
    datasets = {"nyc_2018.zarr": ds}
    extents = {"nyc_2018.zarr": geo_utils.get_dataset_extent(ds)}
    df = pd.concat([_grid_df(ds, 2018), _grid_df(ds, 2016)], ignore_index=True)

    out = bd.assign_datasets_to_gdf(df, datasets, extents, years=[2016, 2018],
                                    verbose=False, save_plot=False, epsg=NYC_EPSG)
    assert set(out["year"]) == {2018}


def test_assign_datasets_to_gdf_raises_on_crs_mismatch():
    """The regression: 5070 metres against a 6539 raster used to KeyError."""
    ds = _fake_zarr()
    datasets = {"nyc_2018.zarr": ds}
    extents = {"nyc_2018.zarr": geo_utils.get_dataset_extent(ds)}
    df = _grid_df(ds, 2018)
    df[["centroid_x", "bbox_minx", "bbox_maxx"]] += 1.5e6   # "another CRS"

    with pytest.raises(ValueError) as exc:
        bd.assign_datasets_to_gdf(df, datasets, extents, years=[2018],
                                  verbose=False, save_plot=False,
                                  epsg=geo_utils.METRIC_EPSG)
    msg = str(exc.value)
    assert "not on the same grid" in msg
    assert f"EPSG:{NYC_EPSG}" in msg and "2018" in msg


def test_assign_datasets_to_gdf_reports_empty_extents():
    ds = _fake_zarr()
    with pytest.raises(ValueError, match="no dataset extents"):
        bd.assign_datasets_to_gdf(_grid_df(ds, 2018), {}, {}, years=[2018],
                                  verbose=False, save_plot=False, epsg=NYC_EPSG)


# ── end-to-end: the exact path that used to raise KeyError('dataset') ─────────

def _coords_only_zarr(bounds, pad=2000.0, epsg=NYC_EPSG):
    """Raster grid (0.5 units/pixel) covering `bounds`; coords only, no pixels —
    assign_datasets_to_gdf reads x/y and the CRS, never the data."""
    minx, miny, maxx, maxy = bounds
    ds = xr.Dataset(coords={
        "x": np.arange(minx - pad, maxx + pad, 0.5),
        "y": np.arange(maxy + pad, miny - pad, -0.5),
    })
    ds = ds.assign_coords(spatial_ref=0)
    ds["spatial_ref"].attrs["crs_wkt"] = CRS.from_epsg(epsg).to_wkt()
    return ds


def test_open_datasets_aerial_links_every_building(nyc_env, monkeypatch):
    image_size = 16
    tau, step = geo_utils.calculate_exact_tau(20.0, image_size, epsg_code=NYC_EPSG)
    ds = _coords_only_zarr(nyc_env["panel"].to_crs(f"EPSG:{NYC_EPSG}").total_bounds)
    datasets = {f"nyc_{y}.zarr": ds for y in YEARS}
    extents = {n: geo_utils.get_dataset_extent(d) for n, d in datasets.items()}
    monkeypatch.setattr(bd, "load_satellite_datasets", lambda years: (datasets, extents))

    _, _, df = bd.open_datasets(sat_data="aerial", years=YEARS, tau_meters=tau,
                                indicator="W2_r5", footprints_source="doitt_nyc")

    # Used to die here with KeyError('dataset'): nothing matched across CRSs.
    assert len(df) == len(nyc_env["doitt"]) * len(YEARS)
    assert set(df["dataset"]) == {f"nyc_{y}.zarr" for y in YEARS}
    assert (df["row_start"] >= 0).all() and (df["col_start"] >= 0).all()
    # Windows are exactly step*image_size raw pixels → subsample lands on image_size.
    assert ((df["row_stop"] - df["row_start"]) == step * image_size).all()
    assert ((df["col_stop"] - df["col_start"]) == step * image_size).all()


# ── main.run_nyc_zarr_validation_predictions wiring ───────────────────────────

def test_nyc_pass_hands_the_zarr_grid_downstream(monkeypatch, tmp_path):
    """The pass must re-snap tau/subsample onto the 0.5 ftUS grid and ask for
    EPSG:6539 — the national params it inherits describe a different raster."""
    main = pytest.importorskip("src.main")
    from src import prediction
    from src.data import us_split

    params = {"tau_meters": 112.0, "subsample_step": 2, "image_size": 224,
              "indicator": "W2_r5", "batch_size": 8, "nbands": 3}
    expected_tau, expected_step = geo_utils.calculate_exact_tau(
        112.0, 224, epsg_code=NYC_EPSG)

    df = pd.DataFrame({"building_id": [1, 2], "GEOID": ["36061000100"] * 2,
                       "year": [2018, 2018], "Rel_Score": [0.1, -0.2]})
    seen = {}

    def fake_open_datasets(**kw):
        seen["open"] = kw
        return {"nyc_2018.zarr": None}, None, df.copy()

    def fake_predict(**kw):
        seen["predict"] = kw
        out = kw["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"building_id": [1], "GEOID": ["36061000100"],
                      "Rel_Score": [0.1], "predicted_value": [0.3]}).to_csv(out, index=False)

    monkeypatch.setattr(main.build_dataset, "open_datasets", fake_open_datasets)
    monkeypatch.setattr(main, "predict_buildings_chunked", fake_predict)
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(main, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(us_split, "nyc_tract_type_map", lambda: {"36061000100": "test"})
    monkeypatch.setattr(prediction, "select_prediction_rows", lambda d, p: d)

    main.run_nyc_zarr_validation_predictions(None, "cpu", None, "run_test", params)

    assert seen["open"]["epsg"] == NYC_EPSG
    assert seen["open"]["footprints_source"] == "doitt_nyc"
    assert seen["open"]["tau_meters"] == pytest.approx(expected_tau)
    # The crop the model sees: raw window / step == image_size.
    assert seen["predict"]["params"]["subsample_step"] == expected_step
    assert seen["predict"]["params"]["tau_meters"] == pytest.approx(expected_tau)
    # …and the caller's national params are left untouched.
    assert params["tau_meters"] == 112.0 and params["subsample_step"] == 2


# ── tau snapping on the two grids ─────────────────────────────────────────────

@pytest.mark.parametrize("epsg", [geo_utils.METRIC_EPSG, NYC_EPSG])
def test_exact_tau_snaps_to_whole_tiles_on_either_grid(epsg):
    image_size = 224
    tau, step = geo_utils.calculate_exact_tau(112.0, image_size, epsg_code=epsg)
    units_per_m = 1.0 / geo_utils.projected_units_to_meters(1.0, epsg)
    raw_pixels = (2 * tau * units_per_m) / 0.5
    assert raw_pixels == pytest.approx(step * image_size, rel=1e-9)


def test_nyc_grid_tau_stays_near_the_training_footprint():
    """The zarr crop must cover roughly the same ground as the NAIP training crop."""
    train_tau, _ = geo_utils.calculate_exact_tau(100.0, 224)              # 5070, 0.5 m/px
    nyc_tau, nyc_step = geo_utils.calculate_exact_tau(train_tau, 224, epsg_code=NYC_EPSG)
    assert nyc_step > 1
    assert 0.9 < nyc_tau / train_tau < 1.1
