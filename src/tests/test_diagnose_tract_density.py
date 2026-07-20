"""Synthetic-data tests for src/diagnose_tract_density.py (pure pieces only)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.diagnose_tract_density import (
    build_tract_table,
    cap_savings,
    count_buildings_per_tract,
    population_bin_report,
    sparse_tract_report,
    summarize,
)


@pytest.fixture
def tracts():
    """Three 5070-style tracts: 1 km^2, 4 km^2, 25 km^2."""
    geoms = [box(0, 0, 1_000, 1_000),
             box(10_000, 0, 12_000, 2_000),
             box(50_000, 0, 55_000, 5_000)]
    return gpd.GeoDataFrame(
        {"GEOID": ["36061000100", "36061000200", "36061000300"]},
        geometry=geoms, crs="EPSG:5070")


@pytest.fixture
def building_geoids():
    # 10 buildings in tract 1, 4 in tract 2, none in tract 3,
    # plus 2 in a tract outside the split (must not leak in).
    return pd.Series(
        ["36061000100"] * 10 + ["36061000200"] * 4 + ["99999999999"] * 2,
        dtype="category")


def test_count_buildings_per_tract(building_geoids):
    counts = count_buildings_per_tract(building_geoids)
    assert counts.loc["36061000100"] == 10
    assert counts.loc["36061000200"] == 4
    # unused categorical levels / zero-count tracts don't appear
    assert "36061000300" not in counts.index
    assert counts.index.dtype == object  # plain str, not categorical


def test_build_tract_table(tracts, building_geoids):
    counts = count_buildings_per_tract(building_geoids)
    pop = pd.Series({"36061000100": 5_000.0, "36061000200": 800.0})
    df = build_tract_table(tracts, counts, pop).set_index("GEOID")

    assert df.loc["36061000100", "n_buildings"] == 10
    assert df.loc["36061000300", "n_buildings"] == 0        # left join fill
    assert "99999999999" not in df.index                    # outside split
    np.testing.assert_allclose(
        df["area_km2"].to_numpy(), [1.0, 4.0, 25.0])
    np.testing.assert_allclose(
        df["bldg_density"].to_numpy(), [10.0, 1.0, 0.0])
    np.testing.assert_allclose(
        df.loc["36061000100", "pop_density"], 5_000.0)
    assert np.isnan(df.loc["36061000300", "pop_density"])   # no ACS row


def test_summarize_shape(tracts, building_geoids):
    counts = count_buildings_per_tract(building_geoids)
    df = build_tract_table(tracts, counts, pd.Series(dtype=float))
    stats = summarize(df)
    assert list(stats.columns) == ["n_buildings", "area_km2",
                                   "bldg_density", "pop_density"]
    assert stats.loc["max", "n_buildings"] == 10


def test_cap_savings():
    counts = np.array([10, 4, 0, 100])
    out = cap_savings(counts, caps=(5, 50))
    # cap 5: 5 + 4 + 0 + 5 = 14 of 114
    assert out.loc[5, "fetches"] == 14
    assert out.loc[5, "tracts_binding"] == 2
    np.testing.assert_allclose(out.loc[5, "pct_saved"], 100 * 100 / 114)
    # cap 50: 10 + 4 + 0 + 50 = 64, binds only on the 100-building tract
    assert out.loc[50, "fetches"] == 64
    assert out.loc[50, "tracts_binding"] == 1


def test_cap_savings_all_zero():
    out = cap_savings(np.zeros(3, dtype=int), caps=(10,))
    assert out.loc[10, "fetches"] == 0
    assert np.isnan(out.loc[10, "pct_saved"])


def test_population_bin_report():
    pop = pd.Series([0.0, 0.0, 5.0, 10.0, 15.0, 45.0, 50.0, 51.0, 3_000.0,
                     np.nan])
    rep = population_bin_report(pop)
    assert rep.loc["== 0", "tracts"] == 2
    assert rep.loc["(0, 10]", "tracts"] == 2      # 5 and 10 (right-closed)
    assert rep.loc["(10, 20]", "tracts"] == 1     # 15
    assert rep.loc["(20, 30]", "tracts"] == 0
    assert rep.loc["(30, 40]", "tracts"] == 0
    assert rep.loc["(40, 50]", "tracts"] == 2     # 45 and 50
    assert rep.loc["> 50", "tracts"] == 2         # 51 and 3000
    assert rep.loc["NaN", "tracts"] == 1
    assert rep["tracts"].sum() == len(pop)        # every tract accounted for
    np.testing.assert_allclose(rep.loc["== 0", "pct_tracts"], 20.0)


def test_sparse_tract_report(tracts, building_geoids):
    counts = count_buildings_per_tract(building_geoids)
    df = build_tract_table(tracts, counts, pd.Series(dtype=float))
    rep = sparse_tract_report(df, thresholds=(0.5, 2.0))
    # density < 0.5: only the empty 25 km^2 tract
    assert rep.loc[0.5, "tracts"] == 1
    np.testing.assert_allclose(rep.loc[0.5, "pct_buildings"], 0.0)
    np.testing.assert_allclose(rep.loc[0.5, "pct_area"], 100 * 25 / 30)
    # density < 2.0: adds the 4-building tract (density 1.0)
    assert rep.loc[2.0, "tracts"] == 2
    np.testing.assert_allclose(rep.loc[2.0, "pct_buildings"], 100 * 4 / 14)
