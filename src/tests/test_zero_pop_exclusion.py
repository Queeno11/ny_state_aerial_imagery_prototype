"""Synthetic tests for the zero-population tract exclusion (build_dataset)."""

import numpy as np
import pandas as pd
import pytest

from src.build_dataset import drop_zero_population_tracts, zero_population_geoids
from src.data.pair_table import LazyPairTable

YEARS = [2012, 2018]


def _panel(rows):
    """tract_panel-shaped frame: GEOID + total_population_{2011,2023}."""
    return pd.DataFrame(rows, columns=["GEOID", "total_population_2011",
                                       "total_population_2023"])


def _pair_table(geoids_per_building):
    n = len(geoids_per_building)
    buildings = pd.DataFrame({
        "building_id": np.arange(1, n + 1),
        "GEOID": pd.Categorical(geoids_per_building),
        "cbsa_code": pd.Categorical(["10001"] * n),
        "centroid_x": np.zeros(n, dtype="float32"),
        "centroid_y": np.zeros(n, dtype="float32"),
        "dist_to_center": np.zeros(n, dtype="float32"),
    })
    labels = pd.DataFrame({
        "GEOID": sorted(set(geoids_per_building)) * len(YEARS),
        "year": np.repeat(YEARS, len(set(geoids_per_building))),
        "Rel_Score": np.float32(0.5),
        "Valid_Structural_Change": np.int8(0),
        "score_bin": np.int8(1),
    })
    return LazyPairTable(buildings, labels, YEARS, tau_meters=100.0)


def test_zero_population_geoids_all_years_zero_or_nan():
    panel = _panel([
        ("T_zero", 0.0, 0.0),        # zero every year -> excluded
        ("T_nan", np.nan, np.nan),   # never observed -> excluded
        ("T_mixed", 0.0, np.nan),    # zero/NaN mix -> excluded
        ("T_depop", 120.0, 0.0),     # populated once -> KEPT
        ("T_pop", 500.0, 610.0),     # populated -> KEPT
    ])
    out = zero_population_geoids(panel, years=[2011, 2023])
    assert out == {"T_zero", "T_nan", "T_mixed"}


def test_zero_population_geoids_ignores_missing_year_columns():
    panel = _panel([("T_zero", 0.0, 0.0), ("T_pop", 10.0, 10.0)])
    # 2017 has no column — silently skipped, the present years decide
    out = zero_population_geoids(panel, years=[2011, 2017, 2023])
    assert out == {"T_zero"}


def test_zero_population_geoids_no_columns_raises():
    panel = _panel([("T", 1.0, 1.0)])
    with pytest.raises(KeyError):
        zero_population_geoids(panel, years=[1999])


def test_drop_zero_population_tracts():
    table = _pair_table(["A", "A", "B", "C", "C", "C"])
    filtered, n_dropped = drop_zero_population_tracts(table, {"C", "Z_not_present"})
    assert n_dropped == 3
    assert filtered.n_buildings == 3
    kept = set(filtered.buildings["GEOID"].astype(str))
    assert kept == {"A", "B"}
    # pair count shrinks accordingly (years preserved)
    assert len(filtered) == 3 * len(YEARS)
    # labels frame is left alone — orphan label rows are harmless
    assert set(filtered.labels["GEOID"]) == {"A", "B", "C"}


def test_drop_zero_population_tracts_noop():
    table = _pair_table(["A", "B"])
    same, n = drop_zero_population_tracts(table, set())
    assert same is table and n == 0
    same2, n2 = drop_zero_population_tracts(table, {"NOPE"})
    assert same2 is table and n2 == 0
