"""Synthetic-data unit tests for src/data/build_buildings_index.py."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

import src.geo_utils as geo_utils
from src.data import build_buildings_index as bbi


# ── building_id packing ───────────────────────────────────────────────────────

def test_building_id_deterministic_and_int64():
    cx = np.array([1500000.123, -2000000.456, 0.0])
    cy = np.array([2500000.789, 100.0, 3200000.0])
    a = bbi.compute_building_id(cx, cy)
    b = bbi.compute_building_id(cx, cy)
    assert (a == b).all()
    assert a.dtype == np.int64
    assert (a >= 0).all()          # torch int64 safe
    assert (a < 2 ** 62).all()


def test_building_id_round_trip_within_grid():
    rng = np.random.default_rng(42)
    cx = rng.uniform(-2.4e6, 2.3e6, 1000)   # EPSG:5070 CONUS x-range
    cy = rng.uniform(0.0, 3.3e6, 1000)      # EPSG:5070 CONUS y-range
    ids = bbi.compute_building_id(cx, cy)
    rx, ry = bbi.unpack_building_id(ids)
    assert np.abs(rx - cx).max() <= 0.05 + 1e-9   # 0.1 m grid → ≤ 5 cm error
    assert np.abs(ry - cy).max() <= 0.05 + 1e-9


def test_building_id_unique_for_distinct_grid_points():
    xs, ys = np.meshgrid(np.arange(100) * 0.1, np.arange(100) * 0.1)
    ids = bbi.compute_building_id(xs.ravel(), ys.ravel())
    assert len(np.unique(ids)) == ids.size


def test_building_id_out_of_range_raises():
    with pytest.raises(ValueError):
        bbi.compute_building_id([1e9], [0.0])   # not a plausible 5070 coordinate


# ── synthetic tract/building fixtures ────────────────────────────────────────

@pytest.fixture
def tracts():
    """Two 1 km tracts side by side at plausible EPSG:5070 coordinates."""
    x0, y0 = 1_500_000.0, 2_000_000.0
    return gpd.GeoDataFrame(
        {
            "GEOID": ["36001000100", "36001000200"],
            "cbsa_code": ["10580", "10580"],
        },
        geometry=[
            box(x0, y0, x0 + 1000, y0 + 1000),
            box(x0 + 1000, y0, x0 + 2000, y0 + 1000),
        ],
        crs=geo_utils.METRIC_CRS,
    )


def _square(cx, cy, half=5.0):
    return Polygon([
        (cx - half, cy - half), (cx + half, cy - half),
        (cx + half, cy + half), (cx - half, cy + half),
    ])


@pytest.fixture
def buildings_4326(tracts):
    """5 buildings: 2 in tract 1, 2 in tract 2, 1 outside both."""
    x0, y0 = 1_500_000.0, 2_000_000.0
    centers = [
        (x0 + 200, y0 + 200),
        (x0 + 800, y0 + 700),
        (x0 + 1200, y0 + 300),
        (x0 + 1900, y0 + 900),
        (x0 + 5000, y0 + 5000),   # outside the panel tracts
    ]
    geoms = gpd.GeoSeries([_square(cx, cy) for cx, cy in centers],
                          crs=geo_utils.METRIC_CRS)
    return geoms.to_crs("EPSG:4326")


# ── process_chunk ─────────────────────────────────────────────────────────────

def test_process_chunk_tract_assignment(tracts, buildings_4326):
    index_df, polygons = bbi.process_chunk(buildings_4326, tracts, "TestState")
    assert len(index_df) == 4                      # outside building dropped
    assert list(index_df["tract_id"][:2]) == ["36001000100"] * 2
    assert list(index_df["tract_id"][2:]) == ["36001000200"] * 2
    assert (index_df["state"] == "TestState").all()
    assert len(polygons) == 4
    assert set(polygons["building_id"]) == set(index_df["building_id"])


def test_process_chunk_all_buildings_keeps_outside(tracts, buildings_4326):
    index_df, _ = bbi.process_chunk(
        buildings_4326, tracts, "TestState", restrict_to_panel=False
    )
    assert len(index_df) == 5
    assert index_df["tract_id"].isna().sum() == 1


def test_process_chunk_centroid_round_trip(tracts, buildings_4326):
    """Stored cx/cy must recover the true 5070 centroid within the 0.1 m grid."""
    index_df, _ = bbi.process_chunk(buildings_4326, tracts, "TestState")
    true_centroids = buildings_4326.to_crs(geo_utils.METRIC_CRS).centroid
    rx, ry = bbi.unpack_building_id(index_df["building_id"].to_numpy())
    assert np.abs(index_df["cx"].to_numpy() - true_centroids.x.to_numpy()[:4]).max() < 1e-6
    assert np.abs(rx - index_df["cx"].to_numpy()).max() <= 0.05 + 1e-9
    assert np.abs(ry - index_df["cy"].to_numpy()).max() <= 0.05 + 1e-9


def test_process_chunk_determinism(tracts, buildings_4326):
    a, _ = bbi.process_chunk(buildings_4326, tracts, "TestState")
    b, _ = bbi.process_chunk(buildings_4326, tracts, "TestState")
    pd.testing.assert_frame_equal(a, b)


def test_process_chunk_empty_after_restriction(tracts):
    far = gpd.GeoSeries([_square(3_000_000.0, 500_000.0)], crs=geo_utils.METRIC_CRS)
    index_df, polygons = bbi.process_chunk(far.to_crs("EPSG:4326"), tracts, "S")
    assert len(index_df) == 0
    assert len(polygons) == 0


# ── chunked equals single-shot ────────────────────────────────────────────────

def test_chunked_matches_single_shot(tracts, buildings_4326):
    whole, _ = bbi.process_chunk(buildings_4326, tracts, "S")
    parts = [
        bbi.process_chunk(buildings_4326.iloc[i:i + 2].reset_index(drop=True),
                          tracts, "S")[0]
        for i in range(0, len(buildings_4326), 2)
    ]
    stitched = pd.concat(parts, ignore_index=True)
    pd.testing.assert_frame_equal(
        whole.sort_values("building_id").reset_index(drop=True),
        stitched.sort_values("building_id").reset_index(drop=True),
    )


# ── cross-state global dedupe ─────────────────────────────────────────────────

def _write_partition(out_dir, state, building_ids):
    """Write a minimal buildings_index partition for ``state``."""
    part_dir = out_dir / "buildings_index" / f"state={state}"
    part_dir.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(building_ids, dtype="int64")
    cx, cy = bbi.unpack_building_id(ids)
    pd.DataFrame({
        "building_id": ids,
        "cx": cx,
        "cy": cy,
        "tract_id": ["36001000100"] * len(ids),
    }).to_parquet(part_dir / "part.parquet", index=False)


def test_dedupe_index_global_drops_cross_state_duplicates(tmp_path):
    # NewYork and NewJersey share ids 100 and 101 (border buildings); 102/103 and
    # 200/201 are unique to their state.
    _write_partition(tmp_path, "NewJersey", [100, 101, 200, 201])
    _write_partition(tmp_path, "NewYork", [100, 101, 102, 103])

    summary = bbi.dedupe_index_global(tmp_path)
    assert summary["n_duplicates"] == 2                  # ids 100, 101 dropped once
    assert summary["states_rewritten"] == ["NewYork"]    # keeper = first sorted state

    full = pd.read_parquet(tmp_path / "buildings_index")
    assert not full["building_id"].duplicated().any()
    assert len(full) == 6                                # 8 rows - 2 duplicates
    # The keeper copies live in the alphabetically-first state (NewJersey).
    nj = pd.read_parquet(tmp_path / "buildings_index" / "state=NewJersey")
    ny = pd.read_parquet(tmp_path / "buildings_index" / "state=NewYork")
    assert {100, 101} <= set(nj["building_id"])
    assert {100, 101}.isdisjoint(set(ny["building_id"]))


def test_dedupe_index_global_noop_when_unique(tmp_path):
    _write_partition(tmp_path, "Delaware", [100, 101])
    _write_partition(tmp_path, "Maryland", [200, 201])
    summary = bbi.dedupe_index_global(tmp_path)
    assert summary["n_duplicates"] == 0
    assert summary["states_rewritten"] == []


def test_dedupe_index_global_no_partitions(tmp_path):
    summary = bbi.dedupe_index_global(tmp_path)
    assert summary == {"n_states": 0, "n_duplicates": 0, "states_rewritten": []}
